#!/usr/bin/env python3
"""
Generate docstrings.h from docstrings.json.

Usage:
    python tools/generate_docstrings.py
    python tools/generate_docstrings.py --check  # Verify header is up-to-date
"""

import json
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
JSON_FILE = PROJECT_ROOT / "python" / "data" / "docstrings.json"
HEADER_FILE = PROJECT_ROOT / "python" / "data" / "docstrings.h"

LICENSE_HEADER = """/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * AUTO-GENERATED from docstrings.json - DO NOT EDIT DIRECTLY
 * Regenerate with: python tools/generate_docstrings.py
 */
"""


def escape_c_string(s: str) -> str:
    """Escape a string for C string literal with line continuation."""
    result = s.replace('\\', '\\\\')
    result = result.replace('"', '\\"')
    # Split into lines and format with proper continuation
    lines = result.split('\n')
    formatted = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            formatted.append(f'"{line}\\n"')
        else:
            formatted.append(f'"{line}"')
    return ' \\\n    '.join(formatted)


def generate_header(docstrings: dict) -> str:
    """Generate C header content from docstrings dict."""
    content = LICENSE_HEADER + """
#ifndef SLICOT_DOCSTRINGS_H
#define SLICOT_DOCSTRINGS_H

"""
    for name in sorted(docstrings.keys()):
        doc = docstrings[name]
        escaped = escape_c_string(doc)
        macro_name = f"DOC_{name.upper()}"
        content += f'#define {macro_name} {escaped}\n\n'

    content += "#endif /* SLICOT_DOCSTRINGS_H */\n"
    return content


def main():
    parser = argparse.ArgumentParser(description='Generate docstrings.h from JSON')
    parser.add_argument('--check', action='store_true',
                        help='Check if header is up-to-date (for CI)')
    args = parser.parse_args()

    if not JSON_FILE.exists():
        print(f"ERROR: {JSON_FILE} not found")
        print("Run: python tools/split_files.py --extract-docstrings")
        sys.exit(1)

    with open(JSON_FILE) as f:
        docstrings = json.load(f)

    new_content = generate_header(docstrings)

    if args.check:
        if not HEADER_FILE.exists():
            print(f"ERROR: {HEADER_FILE} not found")
            sys.exit(1)

        with open(HEADER_FILE) as f:
            existing = f.read()

        if existing != new_content:
            print("ERROR: docstrings.h is out of date")
            print("Run: python tools/generate_docstrings.py")
            sys.exit(1)

        print("OK: docstrings.h is up-to-date")
        return

    HEADER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HEADER_FILE, 'w') as f:
        f.write(new_content)

    print(f"Generated {HEADER_FILE.relative_to(PROJECT_ROOT)} ({len(docstrings)} docstrings)")


if __name__ == '__main__':
    main()
