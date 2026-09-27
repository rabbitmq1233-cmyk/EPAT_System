"""Enable ``python -m epat``."""

from __future__ import annotations

import sys

from epat.cli import main

if __name__ == "__main__":
    sys.exit(main())
