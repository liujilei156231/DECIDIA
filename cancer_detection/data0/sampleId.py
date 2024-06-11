import pandas as pd


trn=pd.read_csv('trn.csv.gz')
val=pd.read_csv('val.csv.gz')
test=pd.read_csv('test.csv.gz')

d=pd.concat([trn, val, test])


sampleIds = d.patient.unique()

with open('sampleId_CRC.txt','w') as f:
    for e in sampleIds:print(e, file=f)
