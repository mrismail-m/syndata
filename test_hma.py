from sdv.metadata import MultiTableMetadata
from sdv.multi_table import HMASynthesizer
import pandas as pd

data = {'T1': pd.DataFrame({'a': [1,2,3]}), 'T2': pd.DataFrame({'b': [4,5,6]})}
meta = MultiTableMetadata()
meta.detect_from_dataframes(data)

print(meta.relationships)
try:
    synth = HMASynthesizer(meta)
    synth.fit(data)
    synth.sample(scale=1)
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
