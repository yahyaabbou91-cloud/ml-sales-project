from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# 
MODEL_PATH = "outputs/final_model.pkl"
model = joblib.load(MODEL_PATH)
print("✅ Modèle chargé :", MODEL_PATH)

#
ALL_FEATURES = [
    "Store", "DayOfWeek", "Customers", "Promo",
    "Open",
    "Promo2", "PromoInterval", "Promo2SinceWeek", "Promo2SinceYear",
    "StoreType", "Assortment",
    "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "StateHoliday", "SchoolHoliday"
]

# 
DEFAULTS = {
    "Store": 1,
    "DayOfWeek": 1,
    "Customers": 0,
    "Promo": 0,

    "Open": 1,

    "Promo2": 0,
    "PromoInterval": "",        # و
    "Promo2SinceWeek": 0,
    "Promo2SinceYear": 0,

    "StoreType": "a",           # 
    "Assortment": "a",          #
    "CompetitionDistance": 0,
    "CompetitionOpenSinceMonth": 0,
    "CompetitionOpenSinceYear": 0,

    "StateHoliday": "0",        # "0"ا "a"/"b"/"c"
    "SchoolHoliday": 0
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # 
        row = DEFAULTS.copy()

        # 
        row["Store"] = int(data.get("Store", row["Store"]))
        row["DayOfWeek"] = int(data.get("DayOfWeek", row["DayOfWeek"]))
        row["Customers"] = int(data.get("Customers", row["Customers"]))
        row["Promo"] = int(data.get("Promo", row["Promo"]))

        # 
        df = pd.DataFrame([row], columns=ALL_FEATURES)

        y = model.predict(df)[0]
        return jsonify({"prediction": round(float(y), 2)})

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Lancement du serveur Flask...")
    app.run(host="127.0.0.1", port=5000, debug=True)