"""Unapplied source-diff construction only; no imports of producer/NumPy."""
from pathlib import Path
import hashlib,json,difflib,ast
HERE=Path(__file__).resolve().parent
root=Path('/tmp/qwen-key-generation-producer-20261104')
path=root/'qwen_key_store.py';old=path.read_text();sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(old.encode())=='1ebd2db1f82d07816e0931517ab225ca286bf47efa38ba1d2fa70a3940f0581b'
helper='''class _NpySink:
    """Synchronous borrowed-fd writes only; no buffer or close authority."""
    __slots__=('_owner','_fd','_limit','_written')
    def __init__(self,owner,limit):
        if type(owner.fd)is not int or owner.fd<0 or type(limit)is not int or limit<=0:
            raise ValueError('live owned descriptor and positive extent bound required')
        self._owner=owner;self._fd=owner.fd;self._limit=limit;self._written=0
    def _live(self):
        if self._owner is None or self._fd is None or self._owner.fd!=self._fd:
            raise ValueError('NPY sink descriptor has been retired')
    def write(self,data):
        self._live();view=memoryview(data).cast('B');size=view.nbytes
        if size>self._limit-self._written:raise ValueError('NPY write exceeds bounded file extent')
        offset=0
        while offset<size:
            self._live();part=view[offset:offset+CHUNK]
            count=os.write(self._fd,part)
            if type(count)is not int or not 0<count<=len(part):raise OSError('invalid NPY write progress')
            offset+=count;self._written+=count
        return size
    def invalidate(self):
        self._owner=None;self._fd=None

'''
anchor='class _StoredFile(_Descriptor):\n';assert old.count(anchor)==1
before="            file=self._open(name,spec);state=_CloseState()\n            with _stream(file.fd,'wb',state)as stream:np.save(stream,a,allow_pickle=False);stream.flush()\n"
after="            file=self._open(name,spec);state=_CloseState();sink=_NpySink(file,spec.raw_bytes+4096)\n            try:np.save(sink,a,allow_pickle=False)\n            finally:state.cleanup('NPY sink invalidation',sink.invalidate,sink)\n"
assert old.count(before)==1
new=old.replace(anchor,helper+anchor).replace(before,after)
ast.parse(new)  # syntax only; do not import/execute the proposed module
patch=''.join(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),fromfile='held64/qwen_key_store.py',tofile='proposed-successor/qwen_key_store.py'))
(HERE/'proposed.patch.txt').write_text(patch)
refs={}
for p in [path,root/'qwen_key_producer.py',root/'key_descriptor_io.py',root/'qwen_key_readers.py',root/'cleanup_origin.py',Path('/tmp/qwen-key-producer-security-review-20261104/review.json'),Path('/tmp/bert-toy4-checkpoint-component-v4-20261104/stage/client_keys.py'),Path('/home/jianming_gatech/miniconda3/envs/jaxite/lib/python3.13/site-packages/numpy/lib/_npyio_impl.py'),Path('/home/jianming_gatech/miniconda3/envs/jaxite/lib/python3.13/site-packages/numpy/lib/_format_impl.py')]:refs[str(p)]=sha(p.read_bytes())
expected=json.loads(Path('/tmp/qwen-key-producer-review-sources-v1-20261104.json').read_bytes())
assert {p.relative_to(root).as_posix():sha(p.read_bytes())for p in root.rglob('*')if p.is_file()}==expected
(HERE/'source_refs.json').write_text(json.dumps(dict(scope='Source-only proposed unapplied delta; no tests/runtime imports/backend/body reads',held64_map_sha256=sha(Path('/tmp/qwen-key-producer-review-sources-v1-20261104.json').read_bytes()),refs=refs,old_store_sha256=sha(old.encode()),proposed_store_sha256=sha(new.encode()),patch_sha256=sha(patch.encode()),selected_producer_unchanged=True,shared_stream_unchanged=True,held64_current_unchanged=True),sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(proposed_store_sha256=sha(new.encode()),patch_sha256=sha(patch.encode()),held64_unchanged=True,implemented=False,tests_executed=False)))
