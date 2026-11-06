"""Bounded client reads with cleanup-only cancellation precedence.

Reuse predecessor path/metadata checks, but track read-body errors separately
from close-only failures. Unknown descriptor closure is quarantined, never retried.
"""
import os,sys
import ranked_input_gate as base


class PrivateReader(base.BorrowedFixtures):
    def __init__(self):
        super().__init__({});self._read_body_failures=[];self._file_close_failures=[]

    def _read(self,path,parent_fd,expected=None):
        fd=None;body=None
        try:
            visible=self._file_state(os.stat(path.name,dir_fd=parent_fd,follow_symlinks=False))
            if expected is not None and visible!=expected:raise ValueError('private file changed before open')
            fd=os.open(path.name,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW,dir_fd=parent_fd)
            first=self._file_state(os.fstat(fd))
            if first!=visible or expected is not None and first!=expected:raise ValueError('private file replaced while opening')
            pieces=[];remaining=(16<<20)+1
            while remaining:
                piece=os.read(fd,min(1<<20,remaining))
                if not piece:break
                pieces.append(piece);remaining-=len(piece)
            raw=b''.join(pieces);last=self._file_state(os.fstat(fd));now=self._file_state(os.stat(path.name,dir_fd=parent_fd,follow_symlinks=False))
            if first!=last or first!=now or len(raw)!=first[2]:raise ValueError('private file changed during authenticated read')
            if expected is None:self.states[path]=('file',parent_fd,first)
            return raw
        except BaseException as error:
            body=error;self._read_body_failures.append(error);raise
        finally:
            if fd is not None:
                try:os.close(fd)
                except BaseException as error:
                    self.uncertain_file_descriptors.append(fd);self.close_errors.append(type(error).__name__)
                    self._file_close_failures.append(error);base._FAILURE_QUARANTINE.append(self)
                    if body is None:raise
                    body.add_note('private file close: '+type(error).__name__)

    def close(self,primary=None):
        # Only a known file-close-only exception is eligible to be superseded
        # by cleanup cancellation. A body exception retains its exact identity.
        close_only=primary is not None and any(primary is e for e in self._file_close_failures) and not any(primary is e for e in self._read_body_failures)
        body=None if close_only else primary
        failure=primary if close_only else None
        pending,self.handles=self.handles,[]
        for fd in reversed(pending):
            try:os.close(fd)
            except BaseException as error:
                self.uncertain_directory_descriptors.append(fd);self.close_errors.append(type(error).__name__)
                if body is not None:body.add_note('private directory close: '+type(error).__name__)
                elif failure is None:failure=error
                elif not isinstance(error,Exception) and isinstance(failure,Exception):
                    error.add_note('earlier private close failure: '+type(failure).__name__);failure=error
                else:failure.add_note('additional private directory close: '+type(error).__name__)
        self.closed=not self.uncertain_directory_descriptors and not self.uncertain_file_descriptors
        if not self.closed:base._FAILURE_QUARANTINE.append(self)
        if body is None and failure is not None and failure is not primary:raise failure
