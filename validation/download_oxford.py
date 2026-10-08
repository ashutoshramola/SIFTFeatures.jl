#!/usr/bin/env python3
"""Fetch the eight Oxford photo sequences with pinned archive and file hashes."""
import argparse
import json
from pathlib import Path
import tarfile
import urllib.request

from common import sha256


def download(args):
    manifest = json.loads(Path(__file__).with_name("oxford-sources.json").read_text())
    cache = args.archives or args.data_dir / "archives"
    cache.mkdir(parents=True, exist_ok=True)
    for sequence in manifest["sequences"]:
        name = sequence["name"]
        destination = args.data_dir / name
        files = sequence["files_sha256"]
        if all((destination / f).is_file() and sha256(destination / f) == digest
               for f, digest in files.items()):
            print(name + ": verified existing files", flush=True)
            continue
        archive = cache / (name + ".tar.gz")
        if not archive.exists():
            temporary = archive.with_suffix(".partial")
            with urllib.request.urlopen(sequence["url"], timeout=90) as source, temporary.open("wb") as sink:
                while chunk := source.read(1024 * 1024):
                    sink.write(chunk)
            temporary.replace(archive)
        if sha256(archive) != sequence["sha256"]:
            raise ValueError(name + ": archive checksum mismatch")
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as package:
            for filename, digest in files.items():
                # Only named regular files are copied; tar paths are never extracted.
                member = package.getmember(filename)
                if not member.isfile():
                    raise ValueError(name + ": expected a regular file: " + filename)
                content = package.extractfile(member).read()
                target = destination / filename
                target.write_bytes(content)
                if sha256(target) != digest:
                    raise ValueError(name + ": file checksum mismatch: " + filename)
        print(name + ": downloaded and verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--archives", type=Path)
    download(parser.parse_args())
