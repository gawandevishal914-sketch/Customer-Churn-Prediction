from preprocess import load_and_preprocess

df = load_and_preprocess()

print(df.head())

print("\nDataset Shape:")
print(df.shape)