"""Recoverable uninstall of one isolated release. Original environment unchanged."""
import argparse
import json
from pathlib import Path
from install import uninstall
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir',required=True,type=Path)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--processes-stopped',action='store_true');args=parser.parse_args()
    print(json.dumps(uninstall(args.release_dir,apply=args.apply,processes_stopped=args.processes_stopped),sort_keys=True))
if __name__=='__main__':main()
