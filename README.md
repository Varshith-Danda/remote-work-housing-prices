# Remote Work and Urban Housing Prices

ASU DAT 490 capstone, Team 10: Sai Krishna Varshith Danda, Gayathri Kota, Eshitha Chowdary Kaka, Zain Khan.

When the pandemic made remote work common, homes farther from downtown started gaining value faster than homes close in.
We asked whether that shift lasted through 2025 in eight Sun Belt metros (Phoenix, Austin, Atlanta, Dallas, Tampa, Charlotte,
Nashville and Las Vegas), using ZIP-level Zillow home values and rents joined to Census data.

## What we found

- Across the eight metros, farther ZIP codes gained more after 2019 and the shift kept growing through 2025, but that average is
  carried mostly by Atlanta, the only metro where the result holds under every check.
- On ZIP codes that also have rent data, rents moved outward first (2020), prices passed them by 2022, and that did not reverse
  after mortgage rates rose.
- Areas with more work-from-home before 2020 did not grow in a consistent way afterward: the sign flips once income and home
  value are controlled for.
- ZIP codes group by the shape of their price paths only weakly, and the groups follow the metro much more than distance.

These are descriptive patterns, not causal estimates. The reports explain why.

## Reports

| Stage | Report | Notebook |
|---|---|---|
| Project plan and literature review | `docs/Team10_Project_Plan.pdf`, `docs/Team10_Lit_Review.pdf` | |
| Exploratory data analysis | `report/Team10_EDA.pdf` | `notebooks/01_eda.ipynb` |
| Methodology and first results | `report/Team10_Methodology.pdf` | `notebooks/02_methods.ipynb` |
| Results, implications and challenges | `report/Team10_Results.pdf` | `notebooks/03_results.ipynb` |

## Repository layout

| Folder | Contents |
|---|---|
| `src/` | Data pipeline (steps `01` to `08`, run by `run_pipeline.py`), analysis code (`methods.py`, `results.py`) and figure code (`viz*.py`) |
| `notebooks/` | The three analysis notebooks |
| `data/reference/` | City-hall coordinates used as metro centers, with a source for each |
| `data/clean/` | Cleaned panels and ZIP-level tables produced by the pipeline |
| `data/raw/` | Raw downloads (not in the repository; see `docs/SOURCES.md`) |
| `outputs/` | Figures, result tables and the pipeline's data-quality logs |
| `report/` | LaTeX sources, tables and figures for the three reports |
| `docs/` | Project plan, literature review and `SOURCES.md` (where every file came from) |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name capstone --display-name "Python 3 (capstone venv)"
```

Download the raw Zillow and Census files into `data/raw/` using the links in `docs/SOURCES.md`. The ACS step pulls tables from
the Census API and needs a free key (https://api.census.gov/data/key_signup.html) set as an environment variable:

```bash
export CENSUS_API_KEY="your key"
```

## Reproducing the results

```bash
.venv/bin/python src/run_pipeline.py          # clean data, about 20 seconds
.venv/bin/python src/make_figures.py          # EDA figures and tables
.venv/bin/python src/make_method_results.py   # methodology results, about 10 minutes
.venv/bin/python src/make_results.py          # final results, about 1 minute
```

To rebuild a report, regenerate its tables (`report/make_tables.py` for the EDA, `report/make_method_tables.py` for the
methodology) and run `xelatex` twice on the `.tex` file inside `report/`. Each report has a script that recomputes every number
it quotes and confirms it matches the text: `src/check_report_numbers.py`, `src/check_method_numbers.py` and
`src/check_results_numbers.py`.

## Data

Zillow Home Value Index and Zillow Observed Rent Index (Zillow Research), American Community Survey 5-year estimates
(U.S. Census Bureau), Census Gazetteer files, the 2023 CBSA delineation and 2020 cartographic ZCTA boundaries. Full sources,
versions and download dates are in `docs/SOURCES.md`.
