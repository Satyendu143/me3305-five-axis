"""Regenerate the course report figure from the two fit-level CSV files."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from summarize import seed_means


def main():
    regions = seed_means("audit_region_trials.csv", ["family", "noise_law", "method"])
    shifts = seed_means("audit_shift_trials.csv", ["family", "noise_law", "state"])
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.1), layout="constrained")
    fig.set_facecolor("white")
    for ax in axes:
        ax.set_ylim(0, 105)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_ylabel("Observed coverage (%)")
        ax.axhline(95, color=".45", linewidth=.9, linestyle="--", zorder=0)
    families = ["matched", "harmonic", "localized"]
    for offset, method, label, color in [
        (-.18, "joint_noise_aware", "Joint Gaussian", "#195571"),
        (.18, "conformal_spherical", "Calibrated sphere", "#bb6126"),
    ]:
        values = [100 * regions[(f, "iid", method)][0] for f in families]
        axes[0].bar([i + offset for i in range(3)], values, width=.34, label=label, color=color)
    axes[0].set_xticks(range(3), ["Matched", "Harmonic", "Localized"])
    axes[0].set_title("(a) Effect of omitted geometry", loc="left", fontsize=11)
    axes[0].legend(loc="lower left", fontsize=8)
    states = ["before_shift", "shift_stale_calibration", "shift_recalibrated"]
    values = [100 * shifts[("matched", "iid", state)][0] for state in states]
    axes[1].bar(range(3), values, color=["#195571", "#973f3d", "#bb6126"], width=.57)
    for i, value in enumerate(values):
        axes[1].text(i, value - 7 if value > 20 else value + 2, f"{value:.2f}%",
                     ha="center", fontsize=9, color="white" if value > 20 else "black")
    axes[1].set_xticks(range(3), ["Original", "Shift, stale", "Shift, fresh"])
    axes[1].set_title("(b) Response to a 3 µm X shift", loc="left", fontsize=11)
    fig.savefig(Path(__file__).resolve().parent / "course-report" / "coverage.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
