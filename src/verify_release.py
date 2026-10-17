#!/usr/bin/env python3
"""Verify an extracted FORMIC release: per-file SHA-256 checksums, sizes, and row counts.

Usage:
    python verify_release.py PATH_TO_RELEASE_ROOT            # print summary
    python verify_release.py ROOT --write MANIFEST.csv       # write a manifest
    python verify_release.py ROOT --check MANIFEST.csv       # verify against a manifest

PATH_TO_RELEASE_ROOT is the extracted dataset directory, i.e. the folder that
contains index.csv and the 16 scenario folders.
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(root):
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if p.suffix == ".csv":
            with open(p, "rb") as f:
                n_rows = sum(1 for _ in f) - 1  # minus header
        else:
            n_rows = ""
        rows.append({"path": rel, "bytes": p.stat().st_size,
                     "csv_rows": n_rows, "sha256": file_sha256(p)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="extracted release root (contains index.csv)")
    ap.add_argument("--write", type=Path, help="write manifest CSV to this path")
    ap.add_argument("--check", type=Path, help="verify release against this manifest CSV")
    args = ap.parse_args()

    if not (args.root / "index.csv").exists():
        sys.exit(f"error: {args.root} does not look like a release root (index.csv missing)")

    rows = collect(args.root)
    if args.write:
        with open(args.write, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "bytes", "csv_rows", "sha256"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} entries to {args.write}")
    if args.check:
        with open(args.check, encoding="utf-8") as f:
            ref = {r["path"]: r for r in csv.DictReader(f)}
        have = {r["path"] for r in rows}
        mismatches = [r["path"] for r in rows if ref.get(r["path"], {}).get("sha256") != r["sha256"]]
        missing = [k for k in ref if k not in have]
        if mismatches or missing:
            for m in mismatches:
                print(f"CHECKSUM MISMATCH: {m}")
            for m in missing:
                print(f"MISSING FILE: {m}")
            sys.exit(1)
        print(f"OK: {len(rows)} files verified")
    if not args.write and not args.check:
        for r in rows[:5]:
            print(f"{r['path']}  {r['bytes']:,} B  {r['sha256'][:16]}...")
        print(f"... {len(rows)} files total (use --write / --check)")


if __name__ == "__main__":
    main()
