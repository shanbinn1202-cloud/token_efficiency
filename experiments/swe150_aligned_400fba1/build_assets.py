"""Reproduce the two SWE-150 input files and verify their frozen SHA-256 digests."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

DATASET = "princeton-nlp/SWE-Bench_Verified"
REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
COUNT = 150
FIELDS = (
    "repo", "instance_id", "base_commit", "patch", "test_patch", "problem_statement",
    "hints_text", "created_at", "version", "FAIL_TO_PASS", "PASS_TO_PASS",
    "environment_setup_commit", "difficulty",
)
HASHES = {
    "dataset-first150.json": "98c611294f8aca7e1180635f1c81be70da872b1a48f1cbc83386d69ae32e9661",
    "tasks/tasks-pool.json": "55a03cd1c6e48fa2bfbfacf2b2d2c5b19d706134ed2932427de2eff993e87ffe",
}


def load_rows(source):
    if source is None:
        from datasets import load_dataset
        dataset = load_dataset(DATASET, split="test", revision=REVISION)
        return list(dataset.select(range(COUNT)))
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < COUNT:
        raise ValueError(f"Expected a JSON array containing at least {COUNT} ordered test rows")
    return data[:COUNT]


def encode(obj, indent):
    return (json.dumps(obj, ensure_ascii=True, indent=indent) + "\n").encode("utf-8")


def check_or_write(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing file differs from the frozen input: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            os.link(tmp.name, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Concurrent write produced a different input: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="destination assets directory inside the experiment bundle")
    parser.add_argument("--dataset-json", type=Path,
                        help="offline JSON array of original test rows in source order")
    parser.add_argument("--roster", type=Path,
                        help="400fba1 source/examples/swebench/verified-150.txt to cross-check")
    args = parser.parse_args()
    rows = [{field: row[field] for field in FIELDS} for row in load_rows(args.dataset_json)]
    ids = [row["instance_id"] for row in rows]
    if args.roster is not None:
        roster = [line.split()[0] for line in args.roster.read_text().splitlines() if line.strip()]
        if ids != roster:
            raise ValueError("Dataset order or IDs differ from verified-150.txt")
    tasks = [{"instance_id": row["instance_id"], "problem_statement": row["problem_statement"]}
             for row in rows]
    payloads = {"dataset-first150.json": encode(rows, 2), "tasks/tasks-pool.json": encode(tasks, 1)}
    for name, data in payloads.items():
        actual = hashlib.sha256(data).hexdigest()
        if actual != HASHES[name]:
            raise ValueError(f"Input digest mismatch for {name}: {actual}")
    for name, data in payloads.items():
        check_or_write(args.output_dir / name, data)
        print(f"{HASHES[name]}  {args.output_dir / name}")


if __name__ == "__main__":
    main()
