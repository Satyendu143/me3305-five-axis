# Five-axis machine tool calibration and compensation

Course project for **ME3305 · Manufacturing Technology – II**, Indian Institute of Technology Madras.

How accurately can a calibrated XYZ–AC machine follow a multi-turn path, and how reliable is its prediction of a new tool-position measurement? This repository contains the synthetic machine, calibration and compensation algorithms, Monte Carlo studies, response-inspection notebook, frozen numerical records, and course report.

[Read the course report](course-report/main.pdf) · [Edit the LaTeX source](course-report/main.tex) · [Open the inspection notebook](ME3305.ipynb) · [Browse all saved outputs](OUTPUTS.md)

## Quick start

From the repository root, run the following with Python 3. The summary uses the standard library and reads the frozen CSV files directly.

```bash
python summarize.py
```

Look for the `joint_noise_aware` rows under `iid`. Coverage is 94.98% for matched geometry, 53.11% for harmonic mismatch and 47.95% for localized mismatch. Coverage is the fraction of new noisy XYZ vectors inside a region. The mean volume-equivalent radius is reported in micrometres.

For simulation, plotting and the interactive notebook, use Python 3.12 and install the scientific dependencies. Run these commands in Bash on Linux or macOS.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab ME3305.ipynb
```

The notebook reads files relative to this checkout. Its first cell reproduces the summary. The inspection panel lets you select a saved case, vary the injected offset, select A/C angle ranges, and export residuals in µm. The final panel explores the earlier uncertainty study. Units inside the forward model are mm and rad.

## Development and evidence

| Stage | Implementation | Saved evidence |
| --- | --- | --- |
| Ordered five-axis kinematics, inverse solution, calibration and iterative compensation | `fiveaxis.py`, `calibration.py`, `experiments.py`, `remedies.py` | `results/helix_full.npz`, `results/eq29_audit.npz` |
| Independent machine, periodic rotary maps and pose selection | `independent_plant.py`, `periodic_model.py`, `research_experiments.py` | `results/research_models.json`, `results/research_design.npz` |
| Static representation and dynamic tracking comparisons | `publication_studies.py`, `dynamic_sim.py` | Report Table 1 uses `results/publication_representation_summary.csv`. The 30 s physical-clock comparison uses `results/publication_dynamic_ablations.csv`. |
| Equal-capacity Fourier/spline comparison and shared auxiliary measurement covariance | `validation_plant.py`, `journal_models.py`, `journal_studies.py` | `results/journal_representation_trials.csv`, `results/journal_tcp_covariance.npz` |
| Seven uncertainty methods and state-shift experiment | `audit_models.py`, `audit_studies.py` | `results/audit_region_trials.csv`, `results/audit_shift_trials.csv`, `results/audit_nonlinear_MC.csv` |
| Sensor inspection and reporting | `inspection_dashboard.py`, `summarize.py`, `report_figure.py` | `inspection/inspection_example.csv`, `course-report/coverage.pdf` |

The final audit uses 48 training poses with three displacement and three separately simulated angular channels, 96 response-calibration commands, and 64 assessment commands per seed. Each fit contributes seven region rows. The 7,200 fits share 12 error seeds across the study. Seed-level averaging preserves the repeated-measures design. The earlier trajectory study uses its own training protocol. All observations and command records are synthetic.

## Verification and regeneration

Run all six suites from the repository root. They exercise the kinematics, independent plant, covariance propagation, scoring, inspection logic and Python widget callbacks. Each suite writes its verification record under `results/`.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
for suite in verify research_verify publication_verify journal_verify audit_verify inspection_verify
do
    python "tests/${suite}.py" || exit 1
done
```

The following commands regenerate the final audit, print its results, rebuild the course figure and compile the report. They overwrite their respective generated records and figures. Use a separate checkout to preserve an untouched copy of the frozen values.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python audit_studies.py --seeds 12 --trials 100 --mc-draws 1024
python summarize.py
python report_figure.py
(cd course-report && latexmk -pdf main.tex)
```

Expected full-audit output includes 50,400 region rows, 21,600 shift rows and 72 cell-level Monte Carlo records representing 73,728 refits. Figure 1 reads the two trial CSVs. Table 1 and the prose report the frozen protocol; if you change the protocol, revise those values in the TeX source. For a short pipeline run use `--seeds 2 --trials 3 --mc-draws 16` in a separate checkout. The reduced run exercises the pipeline and gives unstable estimates.

Earlier stages have independent regeneration commands:

```bash
python experiments.py
python research_experiments.py --trials 24 --points 1501
python publication_studies.py --seeds 12 --repeats 8
python journal_studies.py --seeds 12 --trials 100
python inspection_dashboard.py
```

The final command writes the replay example, CSV and metrics to `inspection/`. Study scripts write their own diagnostic figures to `report/figures/`. To edit in Overleaf, upload the provided Overleaf ZIP as a new project and compile `main.tex` with pdfLaTeX. The source folder includes the summary plot, all 51 research figures, the inspection example, numerical tables, code excerpts, references and the source-equation index. The complete report compiles offline with a standard TeX Live installation.

## Team and provenance

| Name | Roll number |
| --- | --- |
| Satyendu Kar | ME24B059 |
| Prateek Kathare | ME24B050 |
| Rishit Khandelwal | ME24B054 |
| Shriansh Krishna Dasi | ME24B014 |
| Sarvesh Sasikumar | ME24B058 |

The course edition adapts the supplied five-axis numerical research package credited to Satyendu Kar. The original package disclosed AI assistance with mathematical exposition, implementation, numerical studies, figures and drafting. The current packaging and report preparation also used AI assistance. The institute crest on the course cover comes from the team-supplied report template. The kinematic model follows Xiang and Altintas (2016), DOI 10.1016/j.ijmachtools.2015.11.006. The response-calibration construction follows Lei et al. (2018), cited in the report. The repository does not grant a software or data license.

The full focused audit was rerun for the expanded course report. [REPRODUCTION.json](REPRODUCTION.json) records row counts and maximum absolute differences from the frozen numerical CSVs for every numeric column.
