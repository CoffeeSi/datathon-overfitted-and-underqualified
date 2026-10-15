# Overfitted and Underqualified

Datathon workspace for the **Overfitted and Underqualified** team.

## Quick start (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

If PowerShell blocks environment activation, run the commands below from a terminal where Python is available:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\jupyter.exe lab
```

## Project layout

```text
data/
  raw/          # Original data; contents are ignored by Git
  processed/    # Generated data; contents are ignored by Git
notebooks/      # Exploration and experiments
reports/figures/ # Charts and exported figures
src/overfitted_underqualified/ # Reusable project code
```

Start with `notebooks/01_eda.ipynb`. Put reusable functions in `src/overfitted_underqualified/` once they are useful in more than one notebook. Keep notebooks focused, restart and run them top to bottom before sharing results, and record important findings and data sources here or in the notebook.