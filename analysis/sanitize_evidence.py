"""Produce a sanitized COPY of an evidence tree, for sharing outside the project.

The published evidence under ``results/published/`` is the forensic source of truth and
is never edited: it is exactly what the tools wrote, identifiers included. When a copy
has to leave the project, this script writes a redacted one somewhere else and proves
the redaction worked.

Redacted by default:

* machine hostnames (``--host``, repeatable);
* GPU UUIDs (``GPU-xxxxxxxx-...``) and NVML/`nvidia-smi` serial numbers;
* POSIX and Windows home directories, which carry account names;
* anything else given as ``--replace OLD=NEW``.

IPv4 addresses are **reported, not rewritten**: library versions are indistinguishable
from addresses by shape alone (``libcublas.so.13.8.0.4`` and ``cuDNN 9.24.0.43`` both
parse as IPv4), so the script lists candidates and a human decides, passing any real
address through ``--replace``.

Binary files (e.g. ``outputs.npz``) hold arrays, not text, and are copied byte for byte.

Usage:
    python analysis/sanitize_evidence.py <source-dir> <destination-dir> [--host NAME]...
    python analysis/sanitize_evidence.py --help

The destination must not already exist. Nothing under the source is ever written to.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

BINARY_SUFFIXES = {".npz", ".npy", ".png", ".gz", ".tar", ".zip", ".pt", ".onnx"}

# (name, pattern, replacement). Applied to text files in this order.
RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("gpu-uuid", re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F-]{27}"), "GPU-<redacted-uuid>"),
    (
        "serial",
        re.compile(r"(?i)(serial\s*(?:number)?\s*[:=]\s*)([0-9A-Za-z-]{4,})"),
        r"\1<redacted-serial>",
    ),
    ("posix-home", re.compile(r"/home/[A-Za-z0-9._-]+"), "/home/<redacted-user>"),
    (
        "windows-home",
        re.compile(r"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9._-]+"),
        r"C:\\Users\\<redacted-user>",
    ),
]

# Reported, never rewritten: by shape alone a version is an address.
# "libcublas.so.13.8.0.4" and "cuDNN 9.24.0.43" both parse as IPv4.
IPV4_SHAPED = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
)


def build_rules(
    hosts: list[str], replacements: list[str]
) -> list[tuple[str, re.Pattern[str], str]]:
    rules = [
        (f"host:{h}", re.compile(re.escape(h), re.IGNORECASE), "<redacted-host>") for h in hosts
    ]
    for item in replacements:
        old, _, new = item.partition("=")
        if not old or not new:
            raise SystemExit(f"--replace needs OLD=NEW, got {item!r}")
        rules.append((f"replace:{old}", re.compile(re.escape(old)), new))
    return rules + RULES


def sanitize_text(
    text: str, rules: list[tuple[str, re.Pattern[str], str]]
) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for name, pattern, replacement in rules:
        text, n = pattern.subn(replacement, text)
        if n:
            counts[name] = counts.get(name, 0) + n
    return text, counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--host", action="append", default=[], help="hostname to redact (repeatable)"
    )
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="literal substitution, e.g. an instance IP (repeatable)",
    )
    args = parser.parse_args()

    source: Path = args.source.resolve()
    destination: Path = args.destination.resolve()
    if not source.is_dir():
        print(f"source is not a directory: {source}", file=sys.stderr)
        return 2
    if destination.exists():
        print(f"destination already exists, refusing to overwrite: {destination}", file=sys.stderr)
        return 2
    if destination.is_relative_to(source):
        print("destination must be outside the source tree", file=sys.stderr)
        return 2

    rules = build_rules(args.host, args.replace)
    totals: dict[str, int] = {}
    text_files = binary_files = 0
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in BINARY_SUFFIXES:
            shutil.copy2(path, target)
            binary_files += 1
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            shutil.copy2(path, target)
            binary_files += 1
            continue
        cleaned, counts = sanitize_text(original, rules)
        target.write_text(cleaned, encoding="utf-8")
        text_files += 1
        for name, n in counts.items():
            totals[name] = totals.get(name, 0) + n

    print(f"source      : {source}")
    print(f"destination : {destination}")
    print(f"text files rewritten: {text_files} | binary files copied verbatim: {binary_files}")
    print("redactions:")
    for name, n in sorted(totals.items()):
        print(f"  {name:16} {n}")
    if not totals:
        print("  (none matched)")

    # Verification: the redacted values must not survive anywhere in the copy.
    leaks: list[str] = []
    for path in sorted(p for p in destination.rglob("*") if p.is_file()):
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern, replacement in rules:
            if replacement.startswith("<redacted") and pattern.search(text):
                leaks.append(f"{path.relative_to(destination)}: {name}")
    if leaks:
        print("\nFAILED: redacted patterns still present:", file=sys.stderr)
        for leak in leaks[:20]:
            print(f"  {leak}", file=sys.stderr)
        return 1
    print("\nverified: no redacted pattern remains in the copy")

    # Reported for human review, never rewritten: most matches are version strings.
    candidates: dict[str, set[str]] = {}
    for path in sorted(p for p in destination.rglob("*") if p.is_file()):
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for match in IPV4_SHAPED.findall(text):
            candidates.setdefault(match, set()).add(str(path.relative_to(destination)))
    if candidates:
        print("\nIPv4-shaped strings, for review (use --replace for any real address):")
        for value, files in sorted(candidates.items()):
            print(f"  {value:20} in {len(files)} file(s), e.g. {sorted(files)[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
