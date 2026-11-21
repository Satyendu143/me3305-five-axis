"""Recompute coverage and equivalent radius from the saved fit-level records."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

RESULTS = Path(__file__).resolve().parent / "results"


def seed_means(filename, group_fields):
    """Weight each truth seed equally, preserving within-seed repeated fits."""
    grouped = defaultdict(lambda: defaultdict(list))
    with (RESULTS / filename).open(newline="") as file:
        for row in csv.DictReader(file):
            group = tuple(row[field] for field in group_fields)
            grouped[group][row["truth_seed"]].append(
                (float(row["observed_coverage"]), float(row["mean_radius_um"]))
            )
    return {
        group: tuple(mean(mean(v[i] for v in trials) for trials in seeds.values())
                     for i in range(2))
        for group, seeds in grouped.items()
    }


if __name__ == "__main__":
    for filename, fields in [
        ("audit_region_trials.csv", ["family", "noise_law", "method"]),
        ("audit_shift_trials.csv", ["family", "noise_law", "state"]),
    ]:
        print(f"\n{filename}\nFamily      Noise       Method or state                Coverage   Radius (um)")
        for (family, law, method), (coverage, radius) in seed_means(filename, fields).items():
            print(f"{family:11} {law:11} {method:30} {coverage:8.2%} {radius:12.3f}")
