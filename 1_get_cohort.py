import pandas as pd
import numpy as np



# data paths
data_path = 'fiddle/data/'
mimic_path = 'mimic3/'
csv_path = mimic_path + 'csv/'




# load files
df_icd = pd.read_csv(csv_path + 'DIAGNOSES_ICD.csv')
df_picd = pd.read_csv(csv_path + 'PROCEDURES_ICD.csv')
df_Dicd = pd.read_csv(csv_path + 'D_ICD_DIAGNOSES.csv')
df_Dpicd = pd.read_csv(csv_path + 'D_ICD_PROCEDURES.csv')

patients = pd.read_csv(csv_path + 'PATIENTS.csv')
admissions = pd.read_csv(csv_path + 'ADMISSIONS.csv')
icustays = pd.read_csv(data_path + 'prep/icustays_MV.csv')
procedures_mv = pd.read_pickle(data_path + 'prep/procedureevents_mv.p')
items_table = pd.read_csv(data_path + 'prep/items_table.csv')


# patient diagnoses with descriptions
df_icddetail = pd.merge(df_icd, df_Dicd, on='ICD9_CODE', how='inner')


# patient procedures with descriptions
df_picddetail = pd.merge(df_picd, df_Dpicd, on='ICD9_CODE', how='inner')
mv_subjects = np.sort(pd.unique(icustays.SUBJECT_ID))
df_picd_detail_mv = df_picddetail[df_picddetail.SUBJECT_ID.isin(mv_subjects)].sort_values(by=['SUBJECT_ID','HADM_ID','SEQ_NUM'], ascending=True)
procedures_mv['ITEMID'] = procedures_mv['ITEMID'].astype(int)
df_procedures = pd.merge(procedures_mv, items_table, on='ITEMID', how='inner')



# heart failure ICD 9 diagnosis code
HF = [
    '4280',  # congestive heart failure
    '41401', # atherosclerosis of coronary artery
]

# serious cardiac procedure ICD 9 codes
# https://www.findacode.com/code-set.php?set=ICD9V3&i=3070
# 3500 - 3528  :  heart valve operations
# 3603 - 3699  :  heart vessel operations
# 3751         :  heart transplant
# 3961 - 3966  :  extracorporeal circulation, auxiliary procedures for open heart surgery


# find all subjects with heart failure ICD9 diagnosis
# predict if these subjects will have a serious readmission within one year
# or die within one year after initial heart failure diagnosis
# serious readmission is defined as having a serious cardiac procedure ICD 9 ICU visit
# patients must have data from metavision

# patient has heart failure ICD 9 diagnosis

# patients with HF ICD 9 diagnosis and are in metavision icustays
mv_admissions = np.sort(pd.unique(icustays.HADM_ID))
df_icd_mv = df_icd[df_icd.HADM_ID.isin(mv_admissions)].sort_values(by=['SUBJECT_ID'])
df_icd_mv_hf = df_icd_mv[df_icd_mv.ICD9_CODE.isin(HF)].sort_values(by=['SUBJECT_ID'])


# check number of patients with heart failure or atherosclerosis
df_hf_cohort = df_icd_mv_hf.drop_duplicates(['HADM_ID'], keep='first')
co_subjects = pd.unique(df_hf_cohort.SUBJECT_ID)
# co_admissions = pd.unique(df_hf_cohort.HADM_ID) 


# patient admissions list with date of death
patients = pd.read_csv(csv_path + 'PATIENTS.csv', parse_dates=['DOB', 'DOD'], usecols=['SUBJECT_ID', 'DOB', 'DOD'])
admissions = pd.read_csv(csv_path + 'ADMISSIONS.csv', parse_dates=['ADMITTIME','DISCHTIME','DEATHTIME'], usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG'])

df_admissions = pd.merge(admissions, patients, on=['SUBJECT_ID'], how='left')
df_admissions = df_admissions[df_admissions.SUBJECT_ID.isin(co_subjects)].sort_values(by=['SUBJECT_ID','HADM_ID'])


# find patient serious readmissions
df_picd_co = df_picddetail[df_picddetail.SUBJECT_ID.isin(co_subjects)].sort_values(by=['SUBJECT_ID','HADM_ID','SEQ_NUM'], ascending=True)

# serious cardiac procedure ICD 9 codes
# https://www.findacode.com/code-set.php?set=ICD9V3&i=3070
# 3500 - 3528  :  heart valve operations
# 3603 - 3699  :  heart vessel operations
# 3751         :  heart transplant
# 3961 - 3966  :  extracorporeal circulation, auxiliary procedures for open heart surgery
# 9671 - 9672  :  continuous mechanical ventilation
codes = df_picd_co.ICD9_CODE.astype(int)
serious_mask = ((codes >= 3500) & (codes <= 3528)) | (
    (codes >= 3603) & (codes <= 3699)) | (codes == 3751) | (
    (codes >= 3961) & (codes <= 3966)) | (
    (codes >= 9671) & (codes <= 9672))
df_picd_serious = df_picd_co[serious_mask]


# serious admissions
df_serious_hadm = df_picd_serious.loc[:, ['HADM_ID']]
df_serious_hadm['SERIOUS'] = np.ones(len(df_serious_hadm))


# metavision admissions
df_mv_hadm = icustays.loc[:, ['HADM_ID', 'ICUSTAY_ID', 'LOS']]
df_mv_hadm['ICUSTAYS_MV'] = np.ones(len(df_mv_hadm))


# HF ICD 9 admissions
df_hf_hadm = df_icd[df_icd.ICD9_CODE.isin(HF)]
df_hf_hadm = df_hf_hadm.loc[:, ['HADM_ID']]
df_hf_hadm['HF'] = np.ones(len(df_hf_hadm))
df_hf_hadm.head()


# merge labels together
df_co_hadm = pd.merge(df_admissions, df_serious_hadm, on=['HADM_ID'], how='left')
df_co_hadm.drop_duplicates(subset=['HADM_ID'], keep='first', inplace=True, ignore_index=True)

df_co_hadm = pd.merge(df_co_hadm, df_mv_hadm, on=['HADM_ID'], how='left')
df_co_hadm.drop_duplicates(subset=['HADM_ID'], keep='first', inplace=True, ignore_index=True)

df_co_hadm = pd.merge(df_co_hadm, df_hf_hadm, on=['HADM_ID'], how='left')
df_co_hadm.drop_duplicates(subset=['HADM_ID'], keep='first', inplace=True, ignore_index=True)

df_co_hadm = df_co_hadm.sort_values(by=['SUBJECT_ID', 'ADMITTIME'], ascending=True)


# find first usable admission for each patient (admission with HF diagnosis and MV icustay and non-serious)
df_usable = df_co_hadm[(df_co_hadm.ICUSTAYS_MV == 1) & (df_co_hadm.HF == 1) & (df_co_hadm.SERIOUS != 1)].copy()
# calculate time until death (in years) if death time is recorded
df_usable['TIME_TIL_DEATH'] = df_usable.apply(lambda x: (x['DOD'].to_pydatetime() - x['DISCHTIME'].to_pydatetime()).total_seconds(), axis=1) / 3600 / 24 / 365.25


# time until serious admission
from IPython.core.debugger import set_trace

time_til_serious = []
for hadmid in df_usable.HADM_ID:
    hadm_info = df_usable[df_usable.HADM_ID == hadmid]
    sid = hadm_info.SUBJECT_ID.iloc[0]
    disch = hadm_info.DISCHTIME.iloc[0].to_pydatetime()
    serious_hadm = df_co_hadm[(df_co_hadm.SUBJECT_ID == sid) & (df_co_hadm.SERIOUS == 1)]
    if serious_hadm.empty:
        time_til_serious.append(np.nan)
    else:
        # find nearest serious admission
        time_til = serious_hadm.apply(lambda x: (x['ADMITTIME'].to_pydatetime() - disch).total_seconds(), axis=1) / 3600 / 24 / 365.25
        time_til[time_til <= 0] = np.nan
        time_til_serious.append(np.min(time_til))

# add result to table
df_usable['TIME_TIL_SERIOUS'] = time_til_serious


# remove icustays where patients died in hospital
df_final_co = df_usable[(df_usable.TIME_TIL_DEATH > 0) | (np.isnan(df_usable.TIME_TIL_DEATH))].copy()

# remove icustays with age < 18 and age > 90
df_final_co['AGE'] = df_final_co.apply(lambda x: (x['ADMITTIME'].to_pydatetime() - x['DOB'].to_pydatetime()).total_seconds(), axis=1) / 3600 / 24 / 365.25
min_age = 18
max_age = 90 # exclude patients with age 300+
df_final_co = df_final_co[(df_final_co.AGE >= min_age) & (df_final_co.AGE <= max_age)]
print('Excluded non-adults, remaining ', df_final_co['ICUSTAY_ID'].nunique())

# remove icustays shorter than T 
T = 12  # 12 hours of icu data

# Remove LOS < cutoff hour
df_final_co = df_final_co[df_final_co['LOS']*24 >= T]
print('Excluded short ICU stays, remaining ', df_final_co['ICUSTAY_ID'].nunique())

# create labels
df_final_co['HF_LABEL'] = (df_final_co.TIME_TIL_DEATH < 1) | (df_final_co.TIME_TIL_SERIOUS < 1)
df_final_co['HF_ONSET_YEAR'] = np.min(df_final_co.loc[:,['TIME_TIL_DEATH','TIME_TIL_SERIOUS']],axis=1)
df_final_co = df_final_co.drop(columns=['ICUSTAYS_MV','HF','HOSPITAL_EXPIRE_FLAG','DEATHTIME'])

# save cohort
df_final_co.to_csv(data_path + 'HF_cohort.csv', index=False)


# cohort statistics
num_subjects = len(pd.unique(df_final_co.SUBJECT_ID))
print("num subjects: ", num_subjects)
num_admissions = len(pd.unique(df_final_co.HADM_ID))
print('num admissions: ', num_admissions)
num_positive = sum(df_final_co.HF_LABEL)
num_control = sum(df_final_co.HF_LABEL == False)
print('positive: ', num_positive, ' | control: ', num_control)
print('positive %: ', num_positive/num_admissions, ' | control %: ', num_control/num_admissions)
pos_death = sum((df_final_co.HF_LABEL == True) & (df_final_co.TIME_TIL_DEATH < 1) & ~(df_final_co.TIME_TIL_SERIOUS < 1))
pos_serious = sum((df_final_co.HF_LABEL == True) & (df_final_co.TIME_TIL_SERIOUS < 1) & ~(df_final_co.TIME_TIL_DEATH < 1))
pos_both = sum((df_final_co.HF_LABEL == True) & (df_final_co.TIME_TIL_SERIOUS < 1) & (df_final_co.TIME_TIL_DEATH < 1))
print('positive bc death within 1 year: ', pos_death)
print('positive bc serious readmission within 1 year: ', pos_serious)
print('positive bc both within 1 year: ', pos_both)