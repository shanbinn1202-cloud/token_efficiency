"""Deprecated import path; use scripts/run.sh --method agentkv."""
import sys
from tebench.run import main
if __name__ == "__main__":
    main(["--method", "agentkv", *sys.argv[1:]])
