from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = Flask(__name__)

# -------------------- Load Dataset --------------------
data = pd.read_csv("career_dataset.csv")

# Standardize column names
data.columns = data.columns.str.strip().str.lower()

# Optional: verify dataset
print(data.columns)
print(data.head())

# Features (14 skills) and target
X = data.drop("course", axis=1)
y = data["course"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Helper function to get 0/1 from checkbox
def get_value(name):
    return 1 if request.form.get(name) else 0

# -------------------- Routes --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = [
            get_value("coding"),
            get_value("creativity"),
            get_value("communication"),
            get_value("logical"),
            get_value("design"),
            get_value("research"),
            get_value("math"),
            get_value("science"),
            get_value("leadership"),
            get_value("problem"),
            get_value("writing"),
            get_value("speaking"),
            get_value("teamwork"),
            get_value("sports")
        ]

        prob = model.predict_proba([user_input])[0]
        classes = model.classes_

        top_indices = np.argsort(prob)[::-1][:3]
        top_careers = [classes[i] for i in top_indices]

        return render_template("result.html", careers=top_careers)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)



