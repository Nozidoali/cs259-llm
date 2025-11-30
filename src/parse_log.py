#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

PREFIX_RE = re.compile(r'^llama_perf_context_print:\s+total time\s*=')
NUM_RE = re.compile(r'[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?')


def parse_line_numbers(line: str):
    floats, ints = [], []
    for m in NUM_RE.finditer(line):
        s = m.group(0)
        if ('.' in s) or ('e' in s.lower()):
            try:
                floats.append(float(s))
            except ValueError:
                pass
        else:
            try:
                ints.append(int(s))
            except ValueError:
                pass
    return floats, ints


def main():
    parser = argparse.ArgumentParser(description="Parse llama.cpp performance logs")
    parser.add_argument("logfile", type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    if not args.logfile.is_file():
        raise SystemExit(f"File not found: {args.logfile}")

    records = []
    with args.logfile.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            if PREFIX_RE.match(line):
                floats, ints = parse_line_numbers(line)
                records.append({
                    "line_number": lineno,
                    "floats": floats,
                    "ints": ints,
                    "line": line.rstrip("\n"),
                })

    if not records:
        print("No performance records found")
        return

    avg_tok_speed = 0
    for rec in records:
        if rec['ints'] and rec['floats']:
            token_per_second = rec['ints'][0] / rec['floats'][0] * 1000
            avg_tok_speed += token_per_second
    
    avg_tok_speed /= len(records)
    print(f"Average token speed: {avg_tok_speed:.2f} tokens/s")


if __name__ == "__main__":
    main()
