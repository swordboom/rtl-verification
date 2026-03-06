import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("rtl_regression_dataset.csv")

le_module = LabelEncoder()
le_severity = LabelEncoder()
le_priority = LabelEncoder()

df["module"] = le_module.fit_transform(df["module"])
df["severity"] = le_severity.fit_transform(df["severity"])
df["priority"] = le_priority.fit_transform(df["priority"])

X = df[[
    "module",
    "severity",
    "coverage_drop",
    "frequency",
    "recurrence",
    "fix_time"
]]

y = df["priority"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))