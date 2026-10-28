"""Download the IBM Telco Customer Churn CSV into data/raw/ and verify its checksum.

Usage:
    python scripts/download_data.py [--force] [--no-verify]
"""

from __future__ import annotations

import argparse
import logging

from churn.config import DATA_URL, RAW_DATA_PATH, RAW_DATA_SHA256
from churn.data import download_raw_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = download_raw_data(
        url=DATA_URL,
        dest=RAW_DATA_PATH,
        expected_sha256=None if args.no_verify else RAW_DATA_SHA256,
        force=args.force,
    )
    logging.info("Raw data ready at %s", path)


if __name__ == "__main__":
    main()
