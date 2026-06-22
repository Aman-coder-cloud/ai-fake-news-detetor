from flask import Flask, render_template, request, redirect
import pickle
import sqlite3
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load Model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- DATABASE ---------------- #

def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        news TEXT,
        result TEXT,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------------- HOME ---------------- #

@app.route("/")
def home():
    return render_template("home.html")

# ---------------- DETECTOR ---------------- #

@app.route("/detector")
def detector():

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT result
    FROM predictions
    ORDER BY id DESC
    LIMIT 5
    """)

    history = [row[0] for row in cursor.fetchall()]

    conn.close()

    return render_template(
        "detector.html",
        history=history
    )

# ---------------- PREDICT ---------------- #

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    transformed_news = vectorizer.transform([news])

    prediction = model.predict(transformed_news)[0]

    probability = model.predict_proba(transformed_news)[0]

    confidence_score = round(max(probability) * 100, 2)

    if prediction == 1:
        result = "✅ Real News"
        result_class = "real"
    else:
        result = "❌ Fake News"
        result_class = "fake"

    # Save to database

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions(news,result,confidence)
    VALUES(?,?,?)
    """, (news, result, confidence_score))

    conn.commit()

    cursor.execute("""
    SELECT result
    FROM predictions
    ORDER BY id DESC
    LIMIT 5
    """)

    history = [row[0] for row in cursor.fetchall()]

    conn.close()

    return render_template(
        "detector.html",
        result=result,
        result_class=result_class,
        confidence_score=confidence_score,
        history=history
    )

# ---------------- CLEAR HISTORY ---------------- #

@app.route("/clear-history")
def clear_history():

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions")

    conn.commit()
    conn.close()

    return redirect("/detector")

# ---------------- DASHBOARD ---------------- #

@app.route("/dashboard")
def dashboard():

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    # Total Predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    # Real News Count
    cursor.execute("""
    SELECT COUNT(*)
    FROM predictions
    WHERE result='✅ Real News'
    """)
    real_count = cursor.fetchone()[0]

    # Fake News Count
    cursor.execute("""
    SELECT COUNT(*)
    FROM predictions
    WHERE result='❌ Fake News'
    """)
    fake_count = cursor.fetchone()[0]

    # Recent Logs
    cursor.execute("""
    SELECT result, confidence, created_at
    FROM predictions
    ORDER BY id DESC
    LIMIT 10
    """)

    logs = cursor.fetchall()

    conn.close()

    # Generate Pie Chart

    if total > 0:

        plt.figure(figsize=(5,5))

        plt.pie(
            [real_count, fake_count],
            labels=["Real News", "Fake News"],
            autopct="%1.1f%%"
        )

        plt.title("Prediction Distribution")

        plt.savefig("static/pie_chart.png")

        plt.close()

    return render_template(
        "dashboard.html",
        total=total,
        real_count=real_count,
        fake_count=fake_count,
        logs=logs
    )

# ---------------- ABOUT ---------------- #

@app.route("/about")
def about():
    return render_template("about.html")

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True)