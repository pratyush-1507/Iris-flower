
import numpy as np  # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore


url = "dataset.csv"

columns = [
    "sepal_length", "sepal_width",
    "petal_length", "petal_width",
    "species"
]

iris = pd.read_csv(url, names=columns)

iris.dropna(inplace=True)

le = LabelEncoder()
iris["species"] = le.fit_transform(iris["species"])

X = iris.drop("species", axis=1)
y = iris["species"]

print("Encoded Species Mapping:")
for i, cls in enumerate(le.classes_):
    print(f"{i}: {cls}")
print()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,random_state=42, stratify=y
)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

models = {
    "Decision Tree": dt,
    "Random Forest": rf,
    "Logistic Regression": lr
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))
    print(f"===== {name} =====")
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred), "\n")

results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])

plt.figure(figsize=(7, 5))
sns.barplot(
    x="Model",
    y="Accuracy",
    data=results_df,
    hue="Model",
    palette="cool",
    legend=False
)
plt.title("Model Comparison on Iris Dataset (UCI Version)")
plt.ylim(0.8, 1.0)
plt.show()

depths = [2, 3, 4, 5, 7, None]
rf_scores = []

for d in depths:
    model = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    rf_scores.append(acc)

plt.figure(figsize=(7, 5))
plt.plot([str(d) for d in depths], rf_scores, marker='o', linestyle='-', color='orange')
plt.title("Random Forest Accuracy vs Tree Depth (UCI Iris)")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.show()

print("🔹 Final Accuracy Comparison:")
print(results_df)