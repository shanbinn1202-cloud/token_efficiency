from pathlib import Path
import sys,unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
D=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(D/'selective-prefix'),str(D/'selective-prefix/source-overlay')]
import test_gemma_e2e as GE
import test_gemma_chunked as C
original=GE.Harness.__init__
def fixture(self,*a,**kw):
 original(self,*a,**kw)
 self.core.scheduler.scheduler_config=NS(async_scheduling=False)
with patch.object(GE.Harness,'__init__',fixture):
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(C.ChunkedEndToEndTests))
 sys.exit(not result.wasSuccessful())
