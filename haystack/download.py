import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data1/joshw/hugging_face"))
CACHE_DIR = DATA_ROOT / "cache"
DATASET_DIR = DATA_ROOT / "datasets"
HF_HOME = DATA_ROOT / "hf_home"

for d in (CACHE_DIR, DATASET_DIR, HF_HOME):
    d.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_HOME)

from datasets import load_dataset

dataset = load_dataset(
    "allenai/c4",
    "en",
    cache_dir=str(CACHE_DIR),
)
dataset.save_to_disk(str(DATASET_DIR / "c4_en_full"))
