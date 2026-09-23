"""Homework 3 (Stretch, ~15 min): salary analyzer -> CLI tool.

Usage:
    py jobs_tool.py jobs.csv
    py jobs_tool.py jobs.csv --city Bangalore
    py jobs_tool.py jobs.csv --city Bangalore --out report.json

Reads a CSV of job postings (title, company, city, salary_lpa, skills),
skips rows whose salary is not a number, counts the top skills, and prints a
JSON report. Optionally filters by city first, and/or writes the report to a
JSON file. Run with `py jobs_tool.py --help` for the full usage.
"""

import argparse
import csv
import json
from collections import Counter


def load(path):
    """Return a list of row dicts with float salary and a set of skills.

    Rows whose `salary_lpa` cannot be parsed as a float (e.g. "Not disclosed")
    are skipped and counted, so one bad row never stops the whole pipeline.
    """
    rows, skipped = [], 0
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):                       # each row -> dict
            try:
                r["salary_lpa"] = float(r["salary_lpa"])  # bad value -> ValueError
            except ValueError:
                skipped += 1
                continue
            # Split "python;sql;aws" into a set of lowercase skill names.
            r["skills"] = {s.strip().lower() for s in r["skills"].split(";")}
            rows.append(r)
    print(f"loaded {len(rows):,} rows, skipped {skipped}")
    return rows


def report(rows, top_n=5):
    """Summarize rows: total job count + the most common skills."""
    counts = Counter(s for r in rows for s in r["skills"])
    return {"n_jobs": len(rows), "top_skills": counts.most_common(top_n)}


if __name__ == "__main__":
    # `argparse` builds the --help text and parses argv for us.
    parser = argparse.ArgumentParser(description="Salary-vs-skills analyzer")
    parser.add_argument("csv", help="path to a jobs CSV file")
    parser.add_argument("--city", help="only count jobs in this city")
    parser.add_argument("--out", help="write the JSON report to this path")
    args = parser.parse_args()

    rows = load(args.csv)

    # Optional city filter (case-insensitive) applied before reporting.
    if args.city:
        city = args.city.lower()
        rows = [r for r in rows if r.get("city", "").lower() == city]

    rep = report(rows)
    print(json.dumps(rep, indent=2))

    # Bonus: persist the report to disk with json.dump.
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        print(f"report written to {args.out}")
