import pandas as pd

df = pd.read_csv("train.csv")[['trip_duration']]
print("Full dataset shape:", df.shape)

# Scale 
df_25 = df.iloc[:int(0.25 * len(df))]
df_50 = df.iloc[:int(0.50 * len(df))]
df_75 = df.iloc[:int(0.75 * len(df))]
df_100 = df  

print("25% shape:", df_25.shape)
print("50% shape:", df_50.shape)
print("75% shape:", df_75.shape)
print("100% shape:", df_100.shape)


