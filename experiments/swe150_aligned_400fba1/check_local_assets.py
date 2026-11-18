import collections
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    rows=json.loads((ROOT/"assets/dataset-first150.json").read_text())
    pool=json.loads((ROOT/"assets/tasks/tasks-pool.json").read_text())
    pins=[line.split() for line in (ROOT/"source/examples/swebench/verified-150.txt").read_text().splitlines() if line.strip()]
    images=json.loads((ROOT/"local_images/manifest.json").read_text())["images"]
    ids=[r["instance_id"] for r in rows]
    assert len(ids)==150 and len(set(ids))==150 and ids==[k for k,_ in pins]
    assert pool==[{"instance_id":r["instance_id"],"problem_statement":r["problem_statement"]} for r in rows]
    assert all(Path(images[k]["image"]).is_file() and Path(images[k]["image"]).stat().st_size>0 for k in ids)
    audit=json.loads((ROOT/"local_images/audit.json").read_text())
    assert audit["passed"]==150 and not audit["failed"] and audit["architecture"]=="arm64"
    for row in rows:
        iid=row["instance_id"]
        receipt=json.loads((ROOT/"local_images/receipts"/(iid+".json")).read_text())
        image=images[iid]
        assert receipt["passed"] and receipt["image"]==image["image"]
        assert image["base_commit"]==receipt["base_commit"]==row["base_commit"]
        assert Path(image["image"]).stat().st_size==receipt["image_bytes"]
    assert shutil.which("apptainer") and shutil.which("unsquashfs")
    assert (ROOT/"source/examples/swebench/swebench_prompt_testbed.yaml").is_file()
    print(json.dumps({"instances":150,"local_sifs":150,"platform":"ARM64 Apptainer",
                      "repos":dict(collections.Counter(r["repo"] for r in rows)),"ec2_required":False}))

if __name__=="__main__":
    main()
