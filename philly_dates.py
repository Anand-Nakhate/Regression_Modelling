#!/usr/bin/env python3
"""
Fetch the Survey of Professional Forecasters release dates from the Philadelphia Fed
and convert to a clean CSV.
"""
import argparse
import logging
import sys

import requests
import pandas as pd

DEFAULT_URL = (
    "https://www.philadelphiafed.org/-/media/"
    "FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/"
    "spf-release-dates.txt"
    "?sc_lang=en&hash=CE16E2057464DBD7A139FFE6188B48EC"
)


def fetch_spf_release_dates(url: str) -> pd.DataFrame:
    """
    Download and parse the SPF release dates text file into a DataFrame.

    Returns columns:
      - Survey               (e.g. "1990 Q2")
      - True Deadline Date   (ISO YYYY-MM-DD)
      - News Release Date    (ISO YYYY-MM-DD)
    """
    logging.info(f"Downloading SPF file from {url}")
    resp = requests.get(url)
    resp.raise_for_status()

    # Keep only non-blank lines
    lines = [ln for ln in resp.text.splitlines() if ln.strip()]

    # The one line starting with "Survey " contains both headers and all data tokens
    data_line = next((ln for ln in lines if ln.startswith("Survey ")), None)
    if data_line is None:
        raise RuntimeError("Unable to locate the 'Survey …' line in the downloaded file.")

    tokens = data_line.split()
    # First 7 tokens = ["Survey","True","Deadline","Date","News","Release","Date"]
    data_tokens = tokens[7:]
    if len(data_tokens) % 4 != 0:
        raise RuntimeError(f"Expected data tokens in multiples of 4, got {len(data_tokens)}")

    records = []
    for i in range(0, len(data_tokens), 4):
        year, quarter = data_tokens[i], data_tokens[i+1]
        raw_deadline = data_tokens[i+2].rstrip("*")
        raw_release  = data_tokens[i+3].rstrip("*")
        survey_period = f"{year} {quarter}"

        # Parse two-digit years correctly (’90 -> 1990, ’25 -> 2025)
        deadline_dt = pd.to_datetime(raw_deadline, format="%m/%d/%y")
        release_dt  = pd.to_datetime(raw_release,  format="%m/%d/%y")

        records.append({
            "Survey": survey_period,
            "True Deadline Date":   deadline_dt.date().isoformat(),
            "News Release Date":    release_dt.date().isoformat(),
        })

    df = pd.DataFrame.from_records(records)
    logging.info(f"Parsed {len(df)} survey entries.")
    return df


def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch SPF release dates and write to CSV"
    )
    p.add_argument(
        "--url", type=str, default=DEFAULT_URL,
        help="URL of the SPF release-dates text file"
    )
    p.add_argument(
        "--output", "-o", type=str, default="spf_release_dates.csv",
        help="Path for the output CSV"
    )
    p.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],
        help="Logging verbosity"
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    try:
        df = fetch_spf_release_dates(args.url)
        df.to_csv(args.output, index=False)
        logging.info(f"CSV successfully written to {args.output}")
    except Exception:
        logging.exception("Failed to download or convert SPF release dates")
        sys.exit(1)


if __name__ == "__main__":
    main()
