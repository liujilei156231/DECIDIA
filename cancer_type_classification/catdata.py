import pandas as pd


trn=pd.read_csv('../CRCv2/size500_ver2_nofilter_no_val/Fold_0/trn.csv.gz')
test=pd.read_csv('../CRCv2/size500_ver2_nofilter_no_val/Fold_0/test.csv.gz')
CRC = pd.concat([trn, test])
del CRC['read_type']

HCC=pd.read_csv("../LIHC/HCC_1000reads_length75.csv.gz")
HCC.insert(4, 'y', [['normal','hcc'][i] for i in HCC.label])


COAD=pd.read_csv('../LIHC/COAD_1000reads_length75.csv.gz')
COAD.columns=['Unnamed: 0', 'seq', 'patient', 'y']
COAD=COAD.assign(y='crc')
COAD=COAD.assign(label=1)


LUNG=pd.read_csv('../LIHC/lung_cancer_1000reads_length100.csv.gz')
LUNG.columns=['Unnamed: 0', 'seq', 'patient', 'y']
LUNG=LUNG.assign(label=1)


h=['seq','patient','label','y']
x=pd.concat([LUNG[h], COAD[h], HCC[h], CRC[h]])

x.to_csv('data.csv.gz', index=False)


#Ca=x[x.label==1]
#d={'crc':0, 'hcc':1, 'lung_cancer':2}
#Ca=Ca.assign(label=[d[k] for k in Ca.y])





