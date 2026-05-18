"""Export C4 Arrow dataset (from save_to_disk) to gzipped JSONL shards for indexing.py."""

import gzip
import json
import os
from pathlib import Path

from datasets import load_from_disk

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data1/joshw/hugging_face"))
DATASET_DIR = DATA_ROOT / "datasets" / "c4_en_full"
OUTPUT_DIR = Path(os.environ.get("C4_JSONL_DIR", "/data1/joshw/hugging_face/c4_jsonl"))

RECORDS_PER_SHARD = 400_000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset from {DATASET_DIR} ...")
ds = load_from_disk(str(DATASET_DIR))

for split_name in ("train", "validation"):
    if split_name not in ds:
        continue
    split = ds[split_name]
    n = len(split)
    print(f"Exporting split={split_name}  rows={n:,}")

    shard_idx = 0
    for start in range(0, n, RECORDS_PER_SHARD):
        end = min(start + RECORDS_PER_SHARD, n)
        out_path = OUTPUT_DIR / f"c4-{split_name}-{shard_idx:05d}.jsonl.gz"

        if out_path.exists():
            print(f"  skip existing {out_path.name}")
            shard_idx += 1
            continue

        batch = split.select(range(start, end))
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            for row in batch:
                rec = {
                    "text": row.get("text", ""),
                    "timestamp": row.get("timestamp", ""),
                    "url": row.get("url", ""),
                }
                f.write(json.dumps(rec) + "\n")

        print(f"  wrote {out_path.name}  rows={end - start:,}")
        shard_idx += 1

print("Export complete.")
