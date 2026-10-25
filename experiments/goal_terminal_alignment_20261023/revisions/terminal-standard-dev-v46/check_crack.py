import asyncio,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_controls import one
async def main():
 await one('crack-7z-hash','oracle')
 await one('crack-7z-hash','nop')
asyncio.run(main())
