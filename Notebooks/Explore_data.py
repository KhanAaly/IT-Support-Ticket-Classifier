import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('Data/support_tickets_sample.csv')


print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
print(df.duplicated().sum())
category_counts = df["category"].value_counts()
print(df["subject"].head())
print(df["description"].head())
print(df["category"].head())

plt.bar(category_counts.index, category_counts.values)
plt.xticks(rotation=45)


x = df[["subject", "description"]]
y = df["category"]



x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

x_train_test = x_train["subject"] + " " + x_train["description"]
x_test_test = x_test["subject"] + " " + x_test["description"]

vectorizer = TfidfVectorizer()
vectorizer.fit(x_train_test)
x_train_tfidf = vectorizer.transform(x_train_test)
x_test_tfidf = vectorizer.transform(x_test_test)
model = LogisticRegression(max_iter=1000)
model.fit(x_train_tfidf, y_train)
model_predictions = model.predict(x_test_tfidf)


most_common_category = y_train.value_counts().index[0]


baseline_predictions = np.full(len(y_test), most_common_category)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)
print("baseline accuracy:", baseline_accuracy)

model_accuracy = accuracy_score(y_test, model_predictions)
print("model accuracy:", model_accuracy)
#plt.show()