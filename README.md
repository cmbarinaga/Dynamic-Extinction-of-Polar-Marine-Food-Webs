# Dynamic Extinction of Polar Marine Food Webs

This repository contains the figure generator and the food-web methods used to calculate metrics and simulate dynamic extinctions for the Potter Cove (Antarctic) and Kongsfjorden (Arctic) food webs.

## Included files

- `figures_main.py` generates Figures 1–4, Figure S1, and Table S1.
- `trophic_web_methods.py` contains food-web loading, metrics, and extinction procedures.
- `Potter_FW.txt` is the Antarctic input network.
- `Arctic_Food_Web_2016.xlsx` is the Arctic input network.

## Initialize Python

Use Python 3.10. From the repository directory, create and activate a virtual environment, then install the dependencies.

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Generate figures and table

From the repository directory, with the environment activated:

```bash
python figures_main.py
```

The script previews each figure in a Matplotlib window; close a window to save it and continue. This interactive mode requires a desktop Matplotlib backend (Tk on Windows). To generate outputs without opening preview windows:

```bash
python figures_main.py --no-show
```

Outputs are written beside the script. The loader also creates `Potter_FW.csv` and `Arctic_Food_Web_2016.csv` as intermediate files. The current shared methods module imports additional packages used elsewhere in that module, so all entries in `requirements.txt` are needed for this unmodified version.

The repository-level `LICENSE` covers code. Check the original terms and attribution requirements for the included food-web data before redistributing or reusing those data separately.
