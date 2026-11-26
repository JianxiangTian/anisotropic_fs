"""Reproduce the figures, statistics, torque validation, and Sec. S7."""
from pathlib import Path
import argparse
import csv
import hashlib
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "outputs" / "tables"


def run(script):
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / script)], cwd=ROOT, check=True)


def verify_data():
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf8").splitlines():
        expected, name = line.split("  ", 1)
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch in {name}")


def verify_statistics():
    for source in sorted((ROOT / "results").glob("*.csv")):
        def read(path):
            with path.open(encoding="utf8", newline="") as stream:
                return list(csv.DictReader(stream))
        actual, expected = read(TABLES/source.name), read(source)
        assert len(actual) == len(expected) and actual[0].keys() == expected[0].keys(), source.name
        for key in expected[0]:
            left, right = ([row[key] for row in rows] for rows in (actual, expected))
            try:
                x, y = np.array(left, dtype=float), np.array(right, dtype=float)
            except ValueError:
                assert left == right, (source.name, key)
            else:
                np.testing.assert_allclose(x, y, rtol=1e-8, atol=1e-12, err_msg=source.name+" "+key)
        print(f"Verified {len(expected)} rows in {source.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["all","figures","statistics","torque","s7"], nargs="?", default="all")
    task = parser.parse_args().task
    verify_data()
    if task in ("statistics","all"):
        for script in ("analyse.py","check_resolution.py","check_intervals.py","check_dimer.py"):
            run(script)
        verify_statistics()
    if task in ("figures","all"):
        for script in ("figures.py","resolution_figures.py"):
            run(script)
    if task in ("torque","all"):
        run("torque_example.py")
    if task in ("all","figures","statistics"):
        run("paper_tables.py")
    if task in ("s7", "all"):
        TABLES.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-B", str(ROOT / "s7" / "reproduce_s7.py"),
             "--output", str(TABLES / "s7_reproduced.csv")],
            cwd=ROOT, check=True,
        )


if __name__ == "__main__":
    main()

