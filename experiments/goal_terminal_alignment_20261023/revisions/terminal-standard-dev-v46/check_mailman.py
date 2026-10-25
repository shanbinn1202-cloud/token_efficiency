import asyncio,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_controls import one
async def main():
 await one('mailman','oracle')
 await one('mailman','nop')
asyncio.run(main())
