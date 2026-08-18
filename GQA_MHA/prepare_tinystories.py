"""Self-contained TinyStories downloader/flattener for the experiment."""

import argparse
import os

from datasets import load_dataset


def prepare(output_path: str, num_stories: int | None = None, split: str = "train"):
    ds = load_dataset("roneneldan/TinyStories", split=split)
    if num_stories is not None:
        ds = ds.select(range(min(num_stories, len(ds))))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Writing {len(ds)} stories to {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(row["text"].strip())
            f.write("\n<|endoftext|>\n")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/tinystories.txt")
    parser.add_argument("--num-stories", type=int, default=500000)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    prepare(args.output, args.num_stories, args.split)
