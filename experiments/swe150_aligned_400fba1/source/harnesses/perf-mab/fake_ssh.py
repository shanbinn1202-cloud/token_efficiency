"""Local transport probe; never creates a network connection."""
import argparse
import base64
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--record', required=True)
parser.add_argument('--code', type=int, default=0)
parser.add_argument('--probe', default='')
args, rest = parser.parse_known_args()
raw = sys.stdin.buffer.read()
Path(args.record).write_text(json.dumps(dict(argv=rest, probe=args.probe,
    payload_base64=base64.b64encode(raw).decode()), sort_keys=True), encoding='utf-8')
sys.exit(args.code)
