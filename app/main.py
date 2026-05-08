import pandas as pd

# df = pd.read_parquet('/mnt/storage/data/code/whathappenednow/data/aligned_premarket_news.parquet')
# df.to_csv('/mnt/storage/data/code/whathappenednow/data/aligned_premarket_news.csv', index=False)

df = pd.read_parquet('/mnt/storage/data/code/whathappenednow/data/features_9am_snapshot.parquet')
print(df.head(1))
print(df.tail(1))
print(df.columns)
print(df.shape)

# print(df.head(1))
# print(df.tail(1))
# print(df.columns)
# print(df.shape)
