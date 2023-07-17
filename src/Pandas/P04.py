import pandas as pd

df = pd.read_csv("P04_data.csv")
print(df)

print()

print(df.loc[0, :])  # rows, column

print()

print(df.loc[[0, 1, 2], :])
# or df.loc[0:2,:]

print()

df.loc[0, "Country"] = "JUPYTER"
print(df.loc[0, "Country"])

