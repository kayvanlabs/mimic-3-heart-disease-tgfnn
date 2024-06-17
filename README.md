# mimic-3-heart-disease-tgfnn
Code used to train a tropical geometry fuzzy neural network to predict adverse cardiovascular events in patients with chronic heart disease in the intensive care unit (MIMIC-III database). Note, to replicate this you will need to adjust hard-coded paths in the scripts we provide.

## Worflow
1. Request access for and download the [MIMIC-3 database](https://mimic.mit.edu/). For this workflow, we unzipped the tables so we could query the CSVs directly using Pandas in Python
2. **Somewhat optional** Clone the [FIDDLE experiments repo](https://github.com/MLD3/FIDDLE-experiments) and create the conda environment for it and run `1_data_extraction
/extract_data.py`. Our `1_get_cohort.py` script uses some intermediate files from this (`prep/icustays_MV.csv`, `prep/procedureevents_mv.p`, `prep/items_table.csv`) however this isn't necessary. You could use the original MIMIC-III files and then filter them as you'd like
3. Run `1_get_cohort.py` to generate the ICU stay IDs for the cohort in `HF_cohort.csv`
4. Run `2_make_labels.py` to do some additional filtering to create `HF_24.0h.csv`
5. Run `3_preprocess.ipynb` to do all the data preprocessing, formatting, feature selection, and imputation to input into machine learning models
6. Run `4_cohort_stats.ipynb` to generate overview information about the cohort
7. Run `5_run_cv.sh` to submit a SLURM job to hyperparameter tune, train, and evaluate a logistic regression, random forest, and TGFNN model
8. Run `6_get_rules_figure.ipynb` to generate the TGFNN rule heatmap

## Required Libraries
- If using Fiddle, see the [FIDDLE experiments repo](https://github.com/MLD3/FIDDLE-experiments) for requirements
- numpy
- sklearn
- scipy
- matplotlib
- seaborn
- pandas
- PyTorch
- icd9cms
- mrmr
