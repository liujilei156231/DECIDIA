import pandas as pd
import random


x = pd.read_csv('../data.csv.gz')
df=x[x.label==1]
d={'crc':0, 'hcc':1, 'lung_cancer':2}
df=df.assign(label=[d[k] for k in df.y])

a = df.patient.value_counts()
selected_patients = a.index[a == 1000]

df = df[df.patient.isin(selected_patients)]

patients = df.patient.unique().tolist()

val_patients = random.sample(patients, 300)
train_patients = random.sample(set(patients) - set(val_patients), 2308)
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

