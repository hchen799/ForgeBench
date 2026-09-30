#!/usr/bin/env python3
"""Write/merge rows (file, size_bytes, sha256, zenodo_path) into release/MANIFEST_FILES.csv.

Usage: python release/make_manifest.py [--out release/MANIFEST_FILES.csv] [--category NAME] PATH [PATH ...]
Directories are walked recursively. Existing rows with the same `file` are replaced.
`zenodo_path` is left as TODO until the human uploads the bundle.
"""
import argparse, csv, hashlib, os, sys

HEADER = ["file", "category", "size_bytes", "sha256", "zenodo_path"]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def walk(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in sorted(files):
                    yield os.path.join(root, fn)
        elif os.path.isfile(p):
            yield p
        else:
            print(f"warning: {p} not found, skipped", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "MANIFEST_FILES.csv"))
    ap.add_argument("--category", default="")
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()

    rows = {}
    if os.path.exists(a.out):
        with open(a.out, newline="") as f:
            rows = {r["file"]: r for r in csv.DictReader(f)}
    for p in walk(a.paths):
        rel = os.path.relpath(p).replace(os.sep, "/")
        old = rows.get(rel, {})
        rows[rel] = {"file": rel, "category": a.category or old.get("category", ""),
                     "size_bytes": os.path.getsize(p), "sha256": sha256(p),
                     "zenodo_path": old.get("zenodo_path", "TODO")}
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, lineterminator="\n")
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
