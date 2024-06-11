import pandas as pd

df=pd.read_csv('val.csv.gz')
val=df.sample(2000)
val.to_csv('val_subset.csv.gz', index=False)
