# This script filters and formats the cohort table to prep it for FIDDLE.

import pandas as pd
import numpy as np

cohort_file_path = 'fiddle/data/HF_cohort.csv'
labels_file_path = 'fiddle/data/population/HF_24.0h.csv'

# Load and filter data
hf_cohort = pd.read_csv(cohort_file_path)
hf_cohort = hf_cohort.groupby('SUBJECT_ID').first() # keep only first visit for each patient
hf_cohort = hf_cohort.reset_index()
hf_cohort = hf_cohort[hf_cohort['AGE'] <= 90] # remove patients older than 90 because their age is obfuscated to 300
hf_cohort = hf_cohort[hf_cohort['LOS'] <= 10] # remove patients with an ICU stay longer than 10 days because they are probably outliers
hf_cohort['HF_LABEL'] = hf_cohort['HF_LABEL'].astype(str)

# Get relevant columns, convert time to hours, make label column
df = hf_cohort[['ICUSTAY_ID', 'TIME_TIL_DEATH', 'TIME_TIL_SERIOUS']]
df['HF_ONSET_HOUR'] = df[['TIME_TIL_DEATH','TIME_TIL_SERIOUS']].min(axis=1) * 8760
df = df.drop(['TIME_TIL_DEATH', 'TIME_TIL_SERIOUS'], axis=1)
df['HF_LABEL'] = df['HF_ONSET_HOUR'].notnull().astype(int)
df = df.rename(columns={'ICUSTAY_ID' : 'ID'})

# if the serious/death event is more than a year after the start of the ICU stay, mark it as control.
df.loc[df['HF_ONSET_HOUR'] > (365 * 24), 'HF_LABEL'] = 0
df.loc[df['HF_LABEL'] == 0, 'HF_ONSET_HOUR'] = np.nan

# write to file
df.to_csv(labels_file_path, index=False)