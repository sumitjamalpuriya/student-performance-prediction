import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

# load dataset
data = pd.read_csv("data.csv")

# convert Pass/Fail to numeric
data["Result"] = data["Result"].map({"Fail": 0, "Pass": 1})

# features
X = data[["StudyHours", "Attendance", "PreviousMarks", "Assignment", "SleepHours"]]

# targets
y_class = data["Result"]
y_marks = data["FinalMarks"]

# models
model_class = LogisticRegression()
model_marks = LinearRegression()

# train
model_class.fit(X, y_class)
model_marks.fit(X, y_marks)

# sample prediction
sample = [[5, 80, 65, 1, 6]]

print("Pass/Fail Prediction (1=Pass, 0=Fail):", model_class.predict(sample))
print("Predicted Marks:", model_marks.predict(sample))
