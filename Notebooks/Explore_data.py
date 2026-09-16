import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data/support_tickets_sample.csv')


print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
print(df.duplicated().sum())
category_counts = df["category"].value_counts()
print(df["subject"].head)
print(df["description"].head)
print(df["category"].head)

plt.bar(category_counts.index, category_counts.values)
plt.xticks(rotation=45)
plt.show()