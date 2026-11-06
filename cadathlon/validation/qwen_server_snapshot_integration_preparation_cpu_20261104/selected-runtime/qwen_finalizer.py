"""One shared cleanup-origin policy across file, engine and key-loader layers."""
from cleanup_origin import mark,select,protect_body

class Finalizer:
    def __init__(self,report,primary=None):
        self.report=report;self.primary=protect_body(primary);self.failure=None;self._errors=[]
    def attempt(self,label,fn):
        try:return fn()
        except BaseException as error:
            # Primary identity is protected before any action can rethrow it.
            self.report.setdefault('cleanup_errors',[]).append(dict(operation=label,type=type(error).__name__,message=str(error)))
            self._errors.append((label,mark(error)))
            self.failure=select(None,self._errors)
    def shielded(self,shield,fn):
        entered=False
        try:
            with shield():entered=True;fn()
        except BaseException as error:self.attempt('cleanup signal scope',lambda:(_ for _ in ()).throw(error))
        if not entered:self.attempt('cleanup after shield installation failure',fn)
    def selected(self):return select(self.primary,self._errors)
    def raise_if_failed(self):
        chosen=self.selected()
        if chosen is not None:raise chosen
