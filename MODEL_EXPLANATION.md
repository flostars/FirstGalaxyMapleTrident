# Scientific Rationale Behind the Classifier

## Objective
The application classifies exoplanet candidates into disposition categories (e.g., confirmed planet, false positive) by merging three NASA Exoplanet Archive catalogues:
- TESS: `tfopwg_disp`
- Kepler cumulative: `disposition` / `koi_disposition`
- K2: `disposition` / `koi_disposition`

These catalogues provide labelled examples of how follow-up analyses judged previously discovered candidates. Training on them lets the model learn empirical decision boundaries among orbital, planetary, and stellar parameters that historically distinguish bona fide exoplanets from false alarms.

## Feature Selection
We focus on eleven physical parameters that are: (1) available across all catalogues, and (2) physically informative for planet validation.

| Feature | Description | Relevance |
|---------|-------------|-----------|
| `pl_orbper` | Orbital period (days) | Planets and false positives favor characteristic period ranges (e.g., eclipsing binaries often short-period). |
| `pl_orbsmax` | Semi-major axis (AU) | Encodes orbital distance; combined with stellar properties estimates incident flux. |
| `pl_rade` | Planet radius (Earth radii) | Distinguishes inflated giants, compact terrestrial planets, or stellar companions. |
| `pl_bmasse` | Planet mass (Earth masses) | Higher masses can indicate transiting brown dwarfs or stars vs. planets. |
| `pl_eqt` | Equilibrium temperature (K) | Proxy for irradiation; extreme temperatures correlate with certain false-positive mechanisms. |
| `pl_insol` | Insolation relative to Earth | Tracks stellar flux received; helps separate habitable-zone candidates from hot binaries. |
| `st_teff` | Stellar effective temperature (K) | Stellar type affects lightcurve depth and noise characteristics. |
| `st_rad` | Stellar radius (Solar radii) | Combined with transit depth gives radius ratio constraints. |
| `st_mass` | Stellar mass (Solar masses) | Physically linked with gravitational environment; influences derived planetary parameters. |
| `st_met` | Stellar metallicity (dex) | Correlates with planet occurrence rates; metal-poor stars host fewer giant planets. |
| `st_logg` | Stellar surface gravity (cgs) | Helps discriminate dwarfs from giants; giants yield higher false-positive rates. |

All features are numeric and undergo median imputation to handle missing values. Median (instead of mean) is robust against skewed distributions common in astrophysical data.

## Learning Algorithms
Two ensemble choices are available:
- **RandomForestClassifier (baseline)** – 300 estimators, balanced class weights, excellent out-of-the-box performance on tabular, moderately sized datasets.
- **XGBoost (optional)** – gradient-boosted trees (`XGBClassifier`) with 400 estimators, learning rate 0.05, and histogram-based tree building for efficiency.

Selecting an existing model checkpoint (.pkl joblib payload) reuses its hyperparameters. For XGBoost, the booster itself is passed back into `fit`, enabling warm-started training on your new dataset.

### Why Tree Ensembles?
1. **Non-linear decision boundaries**: Planet validation depends on complex interactions (e.g., high `pl_rade` + short `pl_orbper` often implies eclipsing binary). Tree ensembles capture such interactions without explicit feature engineering.
2. **Robust to outliers and heterogeneous scales**: Different catalogues provide parameters spanning orders of magnitude; tree-based models handle this naturally.
3. **Handles class imbalance**: Random Forest leverages `balanced_subsample`; XGBoost optimises the gradient objective while still allowing class weighting or sampling tweaks.
4. **Built-in uncertainty averaging**: Averaging across trees (Random Forest) or sequential boosting (XGBoost) mitigates variance and overfitting.

### Training Pipeline
1. **Load catalogues** while treating `#` lines as comments (NASA metadata headers).
2. **Harmonise target labels** into a unified `label` column across datasets.
3. **Coerce features to numeric** and drop rows lacking dispositions.
4. **Impute medians** on each feature to maintain consistent dimensionality.
5. **Concatenate datasets** into a single training frame.
6. **Train/test split (80/20)** stratified by label when multiple classes exist.
7. **Fit the selected ensemble** (RandomForest with balanced subsampling or XGBoost with gradient boosting hyperparameters) using `random_state=42` for reproducibility.
8. **Evaluate** via accuracy, macro recall, and macro F1 scores.

Macro-averaged metrics ensure each disposition class contributes equally regardless of support, aligning with scientific interest in retaining sensitivity to rare but important outcomes (e.g., true planets).

## Scientific Interpretation
The trained model acts as a data-driven proxy for the vetting heuristics used by astronomers:
- Large radii or short orbital periods often indicate stellar binaries; forests learn to associate those regions with false positives.
- Consistency between stellar surface gravity and claimed planetary size helps confirm true planets.
- Insolation and equilibrium temperature trends help flag likely systematic artifacts (e.g., hot, shallow signals near the detection threshold).

Although the classifier is empirical, its decisions are grounded in observed correlations from vetted catalogues. It should be viewed as a triage tool, highlighting candidates that resemble previously confirmed planets vs. past false alarms, rather than providing definitive astrophysical validation.

## Limitations & Future Work
- **Label imbalance**: Confirmed planets outnumber false positives in some catalogues, potentially biasing metrics; further tuning or resampling could help.
- **Feature coverage**: Restricting to shared numeric columns excludes potentially informative flags (e.g., centroid vetting). Future iterations could engineer survey-specific features followed by domain adaptation.
- **Temporal drift**: As new observations update catalogues, retraining is essential. The provided `/retrain` endpoint addresses this by merging new labelled data.
- **Interpretability**: Feature importance plots or SHAP analysis could quantify which parameters drive individual predictions, aiding scientific trust and follow-up prioritisation.

## Usage in the Application
- **FastAPI `/predict`**: Applies the same preprocessing to user CSVs and feeds them into the saved forest.
- **Streamlit UI**: Lets analysts upload new lightcurve-derived parameters, inspect predictions, and monitor class distributions.
- **Retraining**: Incorporates new labelled datasets to keep pace with evolving survey data, ensuring the model reflects current vetting standards.

## Verification View
The Streamlit dashboard now includes a 2D scatter plot of the combined verification dataset (the 20% hold-out fold used for evaluation). The plot shows `pl_orbper` (orbital period) versus `pl_rade` (planet radius) with points coloured by disposition. Hover tooltips expose additional parameters such as equilibrium temperature, insolation, stellar temperature, and radius, making it easy to inspect clusters that correspond to confirmed planets, candidates, or false positives.

A complementary star-map view projects all survey targets onto Right Ascension/Declination coordinates from the same catalogues. Marker size reflects the reported planet radius, colour encodes discovery year (where available), and tooltips include disposition and survey metadata, allowing quick spatial vetting of crowded sky regions.

## Model Architecture
The default configuration trains a RandomForest ensemble with 300 trees. When the XGBoost option is selected, the model becomes a gradient-boosted tree stack with 400 boosting rounds. Both approaches are tree-based, so they do not use neural network layers—the depth of each tree adapts during training. Uploading a previously saved checkpoint reuses its hyperparameters, and for XGBoost the booster parameters warm-start continued training on your data.
