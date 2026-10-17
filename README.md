# Culex nigripalpus habitat suitability, Florida (MSc dissertation)

Code for my MSc Data Science and AI dissertation at the University of Liverpool: weekly machine-learning models of habitat suitability for *Culex nigripalpus*, a mosquito vector of St. Louis encephalitis and West Nile virus, across Florida.

The pipeline extracts satellite-derived climate, vegetation and land-cover predictors with Google Earth Engine, joins them to mosquito observation records on a regular grid, trains XGBoost, random forest and MaxEnt models, and scores each one against simple reference forecasts under spatial and spatiotemporal block cross-validation.

## Pipeline

| Stage | Folder | What it does |
|---|---|---|
| 1. Data extraction | `Data_Extraction/` | Earth Engine extraction of Daymet climate, MODIS EVI/NDWI and NLCD land cover onto the grid; observation download and counting |
| 2. Static model | `Static_Model/` | Builds the master Parquet tables, cleans observations and fits a static XGBoost baseline, including a diagnostic run showing how far a static model transfers to unseen regions |
| 3. Weekly model | `Weekly_XGModel/` | Weekly aggregation of presence and absence, lagged features, block cross-validation, nested isotonic calibration and statewide suitability surfaces |
| 4. MaxEnt | `MaxEnt_Model/` | MaxEnt (elapid) with standard and target-group background, seed sensitivity and bootstrap uncertainty |
| 5. Nowcast | `Nowcast_Model/` | Prevalence, climatology and persistence baselines, paired block-bootstrap comparisons, the 2018 validation and 2025 nowcast maps |
| 6. Interpretation | `SHAPS/` | SHAP attribution by predictor group, with collinearity checks |
| 7. Figures | `Added_Graphs/` | Results figures: ROC, PR-lift and Brier skill panels with bootstrap CIs, Boyce index, calibration and map panels |

## Evaluation

Each model is scored with ROC-AUC, PR-AUC, the Brier skill score and the continuous Boyce index, with block-bootstrap 95% confidence intervals. Skill is always reported relative to the constant-prevalence, climatology and persistence forecasts, so a model has to beat a sensible baseline to count.

The diagnostic results in `spatial_roc_diagnosis.csv` show the central finding: tree models reach a ROC-AUC of about 0.85 under spatiotemporal cross-validation but about 0.66 when whole regions are held out. They capture the timing of mosquito activity well but generalise less well to unseen areas.

## Setup

```bash
pip install -r requirements.txt     # exact versions: requirements-lock.txt
```

Earth Engine extraction needs a Google account with Earth Engine access. `Data_Extraction/authentification.py` opens a browser login and initialises the project. Raw downloads are excluded by `.gitignore`.

## Tools

Python, pandas, NumPy, PyArrow, scikit-learn, XGBoost, elapid, SHAP, SciPy, Matplotlib, imageio, Google Earth Engine (`earthengine-api`, `geemap`).
