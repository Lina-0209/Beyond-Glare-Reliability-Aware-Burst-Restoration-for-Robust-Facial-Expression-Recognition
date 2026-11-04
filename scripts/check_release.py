#!/usr/bin/env python3
"""Conservative source-tree check before a first public commit (not a secret audit guarantee)."""
from pathlib import Path
import hashlib
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}
PRIVATE_DIRS = {"data", "prepared_data", "outputs", "checkpoints", "assets", "third_party"}
SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".txt"}
# Explicitly registered documentation diagrams/plots, not a general image allowance.
DOCUMENTATION_FIGURES = {
    "docs/figures/fig01_pipeline.png": "3b75891fc2f738c4a36a4ba28cbb42eaad6c79f568126c1bcfce5f8486c421aa",
    "docs/figures/fig01_pipeline.svg": "0a43bfeb7ac32ac9b2ac85956fc7ea7682b5041b06c330142b22bd20d0dbf8dd",
    "docs/figures/fig06d_rafdb_four_paths_two_fer.png": "beda1a4dcc6937e557ba486b4d3d85678fa646daf7117da895343c54c4d00255",
    "docs/figures/fig06e_rafdb_strong_utility_heatmap.png": "29383f7243b19ebc9160e8be297bf4be2e3506f40f4abcec5c4014e8d9b739c3",
}


def main():
    findings = []
    private_dirs = set()
    checked = 0
    checked_figures = 0
    patterns = [
        r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----",
        r"sk-[A-Za-z0-9_-]{20,}",
        r"(?i)(?:api_key|password|secret)\s*[:=]\s*[\"'](?!replace|your|example|test|secret|CHANGE)[A-Za-z0-9_+/=-]{24,}[\"']",
    ]
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(p in SKIP_DIRS for p in relative.parts):
            continue
        if relative.parts[0] in PRIVATE_DIRS:
            if path.is_file():
                private_dirs.add(relative.parts[0])
            continue
        if path.is_symlink():
            findings.append((str(relative), "symlink must not be included in a release archive"))
            continue
        if not path.is_file():
            continue
        if path.name.startswith("._") or path.name == ".DS_Store":
            findings.append((str(relative), "macOS metadata"))
            continue
        if relative.as_posix() in DOCUMENTATION_FIGURES:
            if hashlib.sha256(path.read_bytes()).hexdigest() != DOCUMENTATION_FIGURES[relative.as_posix()]:
                findings.append((str(relative), "documentation figure differs from recorded provenance"))
            checked_figures += 1
            continue
        if path.suffix not in SUFFIXES and path.name not in {"LICENSE", ".gitignore", "Dockerfile", ".dockerignore", ".env.example"}:
            findings.append((str(relative), "unexpected binary/file type"))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append((str(relative), "non-UTF8/binary content"))
            continue
        checked += 1
        if any(re.search(pattern, text) for pattern in patterns):
            findings.append((str(relative), "potential embedded credential"))
        if re.search(r"/(?:Users|home|data)/[A-Za-z0-9_-]+/", text):
            findings.append((str(relative), "private absolute path"))
        if re.search(r"\b(?:10\.120\.17\.133|139\.199\.17\.201)\b", text):
            findings.append((str(relative), "private deployment host"))
    for path, reason in findings:
        print(f"FAIL: {path}: {reason}")
    if private_dirs:
        print("NOTE: local-only ignored assets/outputs exist; exclude them from archives:", ", ".join(sorted(private_dirs)))
    print(f"Checked {checked} text files and {checked_figures} documentation figures; findings={len(findings)}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
