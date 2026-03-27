import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression, LinearRegression

warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv("data.csv")

# Convert Pass/Fail to numeric
data["Result"] = data["Result"].map({"Fail": 0, "Pass": 1})

# Features and targets
X = data[["StudyHours", "Attendance", "PreviousMarks", "Assignment", "SleepHours"]]
y_class = data["Result"]
y_marks = data["FinalMarks"]

# Models
model_class = LogisticRegression()
model_marks = LinearRegression()

# Train models
model_class.fit(X, y_class)
model_marks.fit(X, y_marks)

# Sample input
sample = pd.DataFrame([[5, 80, 65, 1, 6]],
columns=["StudyHours", "Attendance", "PreviousMarks", "Assignment", "SleepHours"])

# Predictions
result = model_class.predict(sample)[0]
marks = model_marks.predict(sample)[0]

# Output
print("Pass/Fail (1=Pass, 0=Fail):", result)
print("Predicted Marks:", round(marks, 2))
