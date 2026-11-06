"""Exact supervise/accumulate AST; synthetic resources and no backend/private IO."""
from pathlib import Path
import ast,json,os,sys,time,unittest
ROOT=Path(__file__).resolve().parent

def run_case(kind):
 events=[];namespace={'time':time}
 exec(compile((ROOT/'cleanup_origin.py').read_bytes(),'cleanup_origin.py','exec'),namespace)
 tree=ast.parse((ROOT/'checker_process.py').read_bytes())
 body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in('supervise','accumulate')]
 assert len(body)==2
 exec(compile(ast.Module(body=body,type_ignores=[]),'checker_process.py','exec'),namespace)
 genuine=ValueError('genuine synthetic body')
 failure=None if kind is None else kind('synthetic shield-construction failure')
 class SignalScope:
  def __init__(self,state):self.state=state
  def __enter__(self):return self
  def __exit__(self,*args):
   if self.state is not None:self.state['restored']=True
   return False
 def signals(state=None,shield=False):
  if shield and failure is not None:raise failure
  return SignalScope(state)
 class Output:
  path='/synthetic-owned-output';owner=(0,1)
  def __init__(self,publication=False):self.publication=publication
  def close(self):events.append('publication_close'if self.publication else'owned_close')
  def json(self,*args):events.append('publish')
  @classmethod
  def reopen(cls,*args):return cls(True)
 def source_end(*args):events.append('source_check');return {}
 def action():raise genuine
 def cleanup():events.append('resource_cleanup')
 namespace.update(signals=signals,Output=Output,source_end=source_end)
 report={'passed':True,'candidate_validation_admissible':True,'failed_attempt_terminal_candidate':True}
 try:namespace['supervise'](Output(),report,action,[cleanup],Path('/synthetic'),{})
 except BaseException as error:observed=error
 else:raise AssertionError('genuine body was not rethrown')
 return events,report,observed,genuine

class Tests(unittest.TestCase):
 def check(self,kind):
  events,report,observed,genuine=run_case(kind)
  self.assertIs(observed,genuine)
  self.assertEqual(events.count('resource_cleanup'),1,events)
  self.assertEqual(events.count('source_check'),1,events)
  self.assertEqual(events.count('owned_close'),1,events)
  self.assertEqual(events.count('publish'),1,events)
  self.assertEqual(events.count('publication_close'),1,events)
  self.assertIs(report['passed'],False)
  if kind is not None:self.assertIs(report['cleanup_completed'],False)
 def test_memoryerror_construct(self):self.check(MemoryError)
 def test_keyboardinterrupt_construct(self):self.check(KeyboardInterrupt)
 def test_systemexit_construct(self):self.check(SystemExit)
 def test_normal_control(self):self.check(None)

if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 Path(sys.argv[1]).write_text(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),'pid':os.getpid(),'pgid':os.getpgrp()},indent=2)+'\n')
 raise SystemExit(0 if result.wasSuccessful()else 1)
