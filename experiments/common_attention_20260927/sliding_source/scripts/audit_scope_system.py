"""Estimate duplicated system-instruction tokens on fixed recorded trajectories.

This is a text-token audit, not a counterfactual online performance measurement.
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    marker = "\n\nSCOPE METADATA\n"
    rows = []
    for path in sorted((args.run / "trajectories").glob("*.json")):
        data = json.loads(path.read_text())
        system = next(m for m in data["messages"] if m["role"] == "system")
        base, *chunks = system["content"].split(marker)
        cumulative, excess = base, 0
        for chunk in chunks:
            cumulative += marker + chunk
            excess += len(tokenizer.encode(cumulative).ids) - len(
                tokenizer.encode(base + marker + chunk).ids
            )
        selections = data["memory"]["selections"]
        rows.append({
            "instance_id": path.stem,
            "total_tokens": sum(
                (request.get("usage") or {}).get(key, 0) or 0
                for request in data["memory"]["requests"]
                for key in ("prompt_tokens", "completion_tokens")
            ),
            "duplicate_system_tokens_estimate": excess,
            "system_instruction_copies": len(chunks),
            "kept_fraction": sum(len(s["kept"]) for s in selections)
            / max(1, sum(s["available"] for s in selections)),
        })
    output = {
        "note": "Tokenizer estimate of cumulative system-text duplication with actions held fixed; not an online token saving measurement.",
        "cases": rows,
        "total_tokens": sum(r["total_tokens"] for r in rows),
        "duplicate_system_tokens_estimate": sum(
            r["duplicate_system_tokens_estimate"] for r in rows
        ),
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
