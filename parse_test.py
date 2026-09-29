import pandas as pd
import json

df = pd.read_csv('../../test_datasets/docs/batch_1.csv')
print(df.columns)
data = json.loads(df['json_data'].iloc[0])
print(json.dumps(data, indent=2))
