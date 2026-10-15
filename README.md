# Dynamic Extinction of Polar Marine Food Webs

This repository contains the figure generator and the food-web methods used to calculate metrics and simulate dynamic extinctions for the Potter Cove (Antarctic) and Kongsfjorden (Arctic) webs.

## Files

- `figures_main.py` generates Figures 1–4, Figure S1, and Table S1.
- `trophic_web_methods.py` contains food-web loading, metrics, and extinction procedures.
- `Potter_FW.txt` is the Antarctic input network.
- `Arctic_Food_Web_2016.xlsx` is the Arctic input network.

## Setup

Use Python 3.10 and install the dependencies:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Generate figures and table

Run the full generator from any working directory:

```powershell
.venv\Scripts\python.exe C:\path\to\repo\figures_main.py
```

The script previews each figure in a Matplotlib window; close a window to save it and continue. Add `--no-show` to save without opening preview windows:

```powershell
.venv\Scripts\python.exe C:\path\to\repo\figures_main.py --no-show
```

Useful targeted modes:

- `--figure4-only` generates Figure 4.
- `--figure-s1-only` generates Figure S1.
- `--table-s1-only` generates Table S1.

Outputs are written beside the script. The loader also creates `Potter_FW.csv` and `Arctic_Food_Web_2016.csv` as intermediate files. The current shared methods module imports additional packages used elsewhere in that module, so all entries in `requirements.txt` are needed for this unmodified version.

The repository-level `LICENSE` covers code. Check the original terms and attribution requirements for the included food-web data before redistributing or reusing those data separately.
