#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path


def detect_status(text: str) -> str:
    if '>>>>> All Tests Passed' in text or '>>>>> Tests Passed' in text:
        return 'PASS'
    if '>>>>> Some Tests Failed' in text or '>>>>> Tests Failed' in text:
        return 'FAIL'
    if 'Apply patch failed (pred)' in text:
        return 'APPLY_FAIL'
    if 'Reset Failed' in text:
        return 'RESET_FAIL'
    return 'UNKNOWN'


def main() -> int:
    parser = argparse.ArgumentParser(description='Summarize SWE-bench eval log statuses.')
    parser.add_argument('log_dir', help='Directory containing *.eval.log files')
    args = parser.parse_args()

    root = Path(args.log_dir)
    logs = sorted(root.glob('django__django-*.eval.log'))
    if not logs:
        print(f'no eval logs found in {root}')
        return 1

    counts = Counter()
    for path in logs:
        text = path.read_text(errors='ignore')
        status = detect_status(text)
        counts[status] += 1
        print(f'{path.name}\t{status}')

    print('\nsummary')
    for key in ['PASS', 'FAIL', 'APPLY_FAIL', 'RESET_FAIL', 'UNKNOWN']:
        if counts[key]:
            print(f'{key}\t{counts[key]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
