import asyncio,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_controls import one
async def main():
 await one('portfolio-optimization','oracle')
 await one('portfolio-optimization','nop')
asyncio.run(main())
