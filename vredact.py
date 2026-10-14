#!/usr/bin/env python3
import sys

from src.vredact.apply import main

if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("Interrupted; rerun to resume")
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

