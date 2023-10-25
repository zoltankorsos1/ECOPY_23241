import pandas as pd

file_path = "../data/sp500.parquet"
df = pd.read_parquet(file_path, engine="fastparquet")


file_path = "../data/ff_factors.parquet"
df = pd.read_parquet(file_path, engine="fastparquet")


sp500_df = pd.read_parquet("../data/sp500.parquet", engine="fastparquet")
ff_df = pd.read_parquet("../data/ff_factors.parquet", engine="fastparquet")
merged_df = sp500_df.merge(ff_df, on='Date', how='left')


merged_df['Excess Return'] = merged_df['Monthly Returns'] - merged_df['RF']


merged_df = merged_df.sort_values(by=['Symbol', 'Date'])
merged_df['ex_ret_1'] = merged_df.groupby('Symbol')['Excess Return'].shift(-1)


merged_df = merged_df.dropna(subset=['ex_ret_1'])
merged_df = merged_df.dropna(subset=['HML'])
