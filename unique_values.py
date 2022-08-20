import pandas as pd

for i in range(4):
    # data = pd.read_csv(('UNSW-NB15-modified_{}.csv').format(i+1))
    data = pd.read_csv(('UNSW-NB15-modified_{}.csv').format(i+1))
    print(f"\nFile {i+1}\n")
    print('Unique values for proto')
    print(data['proto'].unique().tolist())
    print(data['proto'].value_counts().sort_values(ascending=False).head(20))
    print(data['proto'].nunique())
    print('Unique values for state')
    print(data['state'].unique().tolist())
    print(data['state'].nunique())
    print('Unique values for service')
    print(data['service'].unique().tolist())
    print(data['service'].nunique())
    print('Unique values for attack_cat')
    print(data['attack_cat'].unique().tolist())
    print(data['attack_cat'].nunique())


