import pandas as pd

df = pd.read_csv("parado.csv", header=None)
print(df.shape)  # (num_filas, num_columnas)