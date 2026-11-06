"""Dedicated child/session ownership for component evidence collectors.

Healthy leader wait has no deadline. Only cancellation/failure or an exited
leader with a surviving group triggers bounded TERM/KILL cleanup. Group absence
is distinct from direct-child wait evidence; grandchildren are never claimed
reaped here. Escaped sessions and malicious same-user PID reuse are outside the
owned-session contract. No process-wide subreaper setting is changed.
"""
import os
import signal
import subprocess
import time
from cleanup import cleanup_signals


class OwnedChild:
    def __init__(self, command, *, cwd, stdout, stderr=subprocess.STDOUT, env=None,
                 term_grace=5., kill_grace=5.):
        if (type(term_grace) is not float or type(kill_grace) is not float
                or not 0 < term_grace <= 30 or not 0 < kill_grace <= 30):
            raise ValueError('finite positive bounded cleanup grace required')
        self.term_grace=term_grace;self.kill_grace=kill_grace;self._closing=False;self._wait_started=False
        self._process=subprocess.Popen(command,cwd=cwd,stdout=stdout,stderr=stderr,env=env,
            start_new_session=True,umask=0o077)
        self.pid=self._process.pid
        self.record=dict(pid=self.pid,process_group=self.pid,direct_child_reaped=False,
            direct_returncode=None,group_empty_verified=False,cleanup_completed=False,
            cleanup_errors=[],signals_sent=[],scope='direct child wait and owned process-group absence; no descendant-reap claim')

    def _wait_direct(self, timeout=None):
        code=self._process.wait() if timeout is None else self._process.wait(timeout=timeout)
        if type(code) is not int:raise RuntimeError('direct-child wait did not return an integer status')
        self.record.update(direct_child_reaped=True,direct_returncode=code)
        return code

    def _group_empty(self):
        self.record['group_empty_verified']=False
        try:os.killpg(self.pid,0)
        except ProcessLookupError:
            self.record['group_empty_verified']=True
            return True
        self.record['group_empty_verified']=False
        return False

    def _await_group_empty(self, grace):
        end=time.monotonic()+grace
        while True:
            if self._group_empty():return True
            remaining=end-time.monotonic()
            if remaining<=0:return False
            time.sleep(min(.01,remaining))

    def _signal(self, number):
        try:os.killpg(self.pid,number)
        except ProcessLookupError:return
        self.record['signals_sent'].append(int(number))

    def wait(self):
        if self._closing or self._wait_started:raise RuntimeError('direct child wait cannot reenter or repeat')
        self._wait_started=True
        try:
            code=self._wait_direct() # no healthy timeout
            if not self._group_empty():
                raise RuntimeError('direct child exited with a surviving owned process group')
            self.record['cleanup_completed']=True
            return code
        except BaseException as primary:
            self.close(primary=primary)
            raise

    def close(self, primary=None):
        """Failure/cancellation cleanup; never a deadline for healthy execution."""
        if self._closing:raise RuntimeError('child cleanup cannot reenter')
        if self.record['direct_child_reaped'] and self.record['group_empty_verified']:return
        self._closing=True;errors=[]
        def attempt(label, action):
            try:return action()
            except BaseException as error:
                errors.append((label,error));return None
        try:
            try:
                with cleanup_signals():
                    term_end=time.monotonic()+self.term_grace
                    empty=attempt('check owned group',self._group_empty)
                    if empty is not True:attempt('terminate owned group',lambda:self._signal(signal.SIGTERM))
                    if not self.record['direct_child_reaped']:
                        try:self._wait_direct(max(0.,term_end-time.monotonic()))
                        except subprocess.TimeoutExpired:pass
                        except BaseException as error:errors.append(('wait after TERM',error))
                    empty=attempt('group TERM grace',lambda:self._await_group_empty(max(0.,term_end-time.monotonic())))
                    if empty is not True or not self.record['direct_child_reaped']:
                        kill_end=time.monotonic()+self.kill_grace
                        attempt('kill owned group',lambda:self._signal(signal.SIGKILL))
                        if not self.record['direct_child_reaped']:
                            attempt('wait after KILL',lambda:self._wait_direct(max(0.,kill_end-time.monotonic())))
                        attempt('group KILL grace',lambda:self._await_group_empty(max(0.,kill_end-time.monotonic())))
                    # Final observation must not borrow an earlier absence flag.
                    attempt('final owned group check',self._group_empty)
                    if not self.record['direct_child_reaped'] or not self.record['group_empty_verified']:
                        errors.append(('incomplete child cleanup',RuntimeError('direct wait or group absence remains unproven')))
            except BaseException as error:errors.append(('cleanup signal scope',error))
        finally:
            self._closing=False
            self.record['cleanup_errors'].extend(label+': '+repr(error) for label,error in errors)
            self.record['cleanup_completed']=bool(self.record['direct_child_reaped'] and
                self.record['group_empty_verified'] and not self.record['cleanup_errors'])
        if primary is not None:
            for label,error in errors:
                if error is not primary:primary.add_note(label+': '+repr(error))
            return
        if errors:
            chosen=next((error for _,error in errors if not isinstance(error,Exception)),errors[0][1])
            for label,error in errors:
                if error is not chosen:chosen.add_note(label+': '+repr(error))
            raise chosen
