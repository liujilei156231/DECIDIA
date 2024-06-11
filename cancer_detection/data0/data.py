import pandas as pd
import random

random.seed(123)

trn=pd.read_csv('../../CRCv2/size500_ver2_nofilter_no_val/Fold_0/trn.csv.gz')
test=pd.read_csv('../../CRCv2/size500_ver2_nofilter_no_val/Fold_0/test.csv.gz')

df = pd.concat([trn, test])
df.reset_index(inplace=True)
del df['Unnamed: 0']


patients = df.patient.unique().tolist()

val_patients = random.sample(patients, 200)
train_patients = random.sample(set(patients) - set(val_patients), 1000)
test_patients = set(patients) - set(train_patients) - set(val_patients)


trn = df[df.patient.isin(train_patients)]
val = df[df.patient.isin(val_patients)]
test = df[df.patient.isin(test_patients)]

trn.reset_index(inplace=True)
test.reset_index(inplace=True)
val.reset_index(inplace=True)


trn.to_csv('trn.csv.gz', index=False)
test.to_csv('test.csv.gz', index=False)
val.to_csv('val.csv.gz', index=False)

