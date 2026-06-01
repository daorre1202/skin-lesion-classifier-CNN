"""
I/O utilities: logging, image indexing, directory preparation.
"""

import glob
import os
import shutil
import sys
from pathlib import Path


def log(msg: str) -> None:
    """Print with immediate flush (works correctly with Tee)."""
    print(msg, flush=True)


class Tee:
    """
    Redirect sys.stdout to both the terminal and a log file simultaneously.
    Ensures the full console output is preserved in Drive even if the
    Colab session disconnects.
    """

    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log_file = open(filepath, 'w', encoding='utf-8', buffering=1)

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        if not self.log_file.closed:
            self.log_file.close()


def index_images(folder: str) -> dict:
    """
    Build a {image_id: file_path} mapping for all images under a folder.

    IDs are sorted lexicographically to ensure the same ordering across
    sessions and operating systems, which is required for a reproducible
    dataset split.
    """
    per_id = {}
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for path in glob.glob(os.path.join(folder, "**", ext), recursive=True):
            img_id = os.path.splitext(os.path.basename(path))[0]
            if img_id not in per_id:
                per_id[img_id] = path
    return dict(sorted(per_id.items()))


def prepare_directories(
    prep_train:  dict,
    prep_val:    dict,
    prep_test:   dict,
    classes_used: list,
    prep_dir:    str,
) -> None:
    """
    Copy images into a flat directory tree for fast DataLoader access.

    Images are copied to RAM (e.g. /content/data/) to avoid per-batch
    network latency when reading from Google Drive. Existing contents
    are cleared to prevent cross-run contamination.

    Structure:
        prep_dir/
            train/<class>/<image>.jpg
            val/<class>/<image>.jpg
            test/<class>/<image>.jpg
    """
    for split_name, split_data in [
        ('train', prep_train),
        ('val',   prep_val),
        ('test',  prep_test),
    ]:
        for cls in classes_used:
            target = Path(prep_dir) / split_name / cls
            target.mkdir(parents=True, exist_ok=True)
            for f in target.glob("*"):
                f.unlink()
            for src in split_data.get(cls, []):
                shutil.copy(src, target)
    log("✓ Images copied to directory structure")
