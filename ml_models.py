import os
import numpy as np
import pandas as pd
import joblib  

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("Debut ML (4 modeles + MAE/MSE/RMSE/R2)")

# ======================
# 1) Chemins
# ======================
BASE_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# 2) Charger le dernier dataset_final_*.csv
# ======================
files = [f for f in os.listdir(OUT_DIR) if f.startswith("dataset_final") and f.endswith(".csv")]
if not files:
    raise FileNotFoundError("Aucun fichier dataset_final_*.csv trouve dans outputs/")

files.sort()
data_path = os.path.join(OUT_DIR, files[-1])
print("Fichier utilise :", data_path)

# Forcer StateHoliday en string (evite mix int/str)
df = pd.read_csv(data_path, low_memory=False, dtype={"StateHoliday": "string"})
print("Dataset charge :", df.shape)

# ======================
# 3) Cible + Features
# ======================
target = "Sales"
if target not in df.columns:
    raise ValueError("La colonne cible 'Sales' n'existe pas dans le dataset.")

y = df[target]
X = df.drop(columns=[target])

# Retirer Date (optionnel)
if "Date" in X.columns:
    X = X.drop(columns=["Date"])

# Colonnes cat / num
cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

print("Nb colonnes numeriques :", len(num_cols))
print("Nb colonnes categorielles :", len(cat_cols))

# ======================
# 4) Prétraitement
# ======================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

# ======================
# 5) Split Train/Test (chronologique)
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ======================
# 6) Modèles (3 sur dataset complet)
# ======================
models = {
    "Regression Lineaire": LinearRegression(),
    "Arbre de Decision": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=120, max_depth=15, random_state=42, n_jobs=-1
    ),
}

# ======================
# 6.5) Suivi du meilleur modèle (RMSE min)
# ======================
best_rmse = float("inf")
best_name = None
best_pipe = None

# ======================
# 7) Entrainement + Evaluation
# ======================
results = []

for name, model in models.items():
    print(f"\nEntrainement : {name}")

    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({"Modele": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})
    print(f"{name} -> MAE={mae:.4f} | MSE={mse:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")

    
    if rmse < best_rmse:
        best_rmse = rmse
        best_name = name
        best_pipe = pipe

# ======================
# 8) SVM sur sous-échantillon
# ======================
print("\nPreparation SVM (echantillon) ...")
svm_sample_size = 50000
df_svm = df.sample(n=svm_sample_size, random_state=42)

y_svm = df_svm[target]
X_svm = df_svm.drop(columns=[target])

if "Date" in X_svm.columns:
    X_svm = X_svm.drop(columns=["Date"])

cat_cols_svm = X_svm.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols_svm = [c for c in X_svm.columns if c not in cat_cols_svm]

numeric_transformer_svm = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer_svm = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess_svm = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_svm, num_cols_svm),
        ("cat", categorical_transformer_svm, cat_cols_svm),
    ]
)

X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    X_svm, y_svm, test_size=0.2, shuffle=False
)

print("Entrainement : SVM (echantillon)")
svm_pipe = Pipeline(steps=[
    ("prep", preprocess_svm),
    ("model", SVR(kernel="rbf", C=10, epsilon=0.1))
])

svm_pipe.fit(X_train_svm, y_train_svm)
y_pred_svm = svm_pipe.predict(X_test_svm)

mae_svm = mean_absolute_error(y_test_svm, y_pred_svm)
mse_svm = mean_squared_error(y_test_svm, y_pred_svm)
rmse_svm = np.sqrt(mse_svm)
r2_svm = r2_score(y_test_svm, y_pred_svm)

results.append({"Modele": "SVM (echantillon)", "MAE": mae_svm, "MSE": mse_svm, "RMSE": rmse_svm, "R2": r2_svm})
print(f"SVM (echantillon) -> MAE={mae_svm:.4f} | MSE={mse_svm:.4f} | RMSE={rmse_svm:.4f} | R2={r2_svm:.4f}")


if rmse_svm < best_rmse:
    best_rmse = rmse_svm
    best_name = "SVM (echantillon)"
    best_pipe = svm_pipe

# ======================
# 9) Résultats finaux + sauvegarde
# ======================
results_df = pd.DataFrame(results).sort_values("RMSE")
print("\nComparaison finale :")
print(results_df)

out_csv = os.path.join(OUT_DIR, "ml_comparaison_resultats.csv")
results_df.to_csv(out_csv, index=False)
print("\nResultats sauvegardes :", out_csv)

# ======================
# 10) Sauvegarder le meilleur modèle (final_model.pkl)
# ======================
final_model_path = os.path.join(OUT_DIR, "final_model.pkl")
joblib.dump(best_pipe, final_model_path)
print(f"\nModele final sauvegarde : {final_model_path}  (Best = {best_name}, RMSE = {best_rmse:.4f})")

print("Fin ML")
