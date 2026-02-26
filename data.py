import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print("Debut du script")

# =========================
# 1) Configuration
# =========================
BASE_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

train_path = os.path.join(BASE_DIR, "train.csv")
stores_path = os.path.join(BASE_DIR, "stores.csv")
if not os.path.exists(stores_path):
    alt = os.path.join(BASE_DIR, "store.csv")
    if os.path.exists(alt):
        stores_path = alt

# =========================
# 2) Chargement des données
# =========================
print("Lecture de train.csv ...")
train = pd.read_csv(train_path, low_memory=False)
print("Train charge :", train.shape)

stores = None
if os.path.exists(stores_path):
    print("Lecture de stores.csv ...")
    stores = pd.read_csv(stores_path, low_memory=False)
    print("Stores charge :", stores.shape)
else:
    print("stores.csv introuvable, on continue avec train seulement")

# Nettoyage des colonnes
train.columns = train.columns.astype(str).str.strip().str.replace(" ", "_")
if stores is not None:
    stores.columns = stores.columns.astype(str).str.strip().str.replace(" ", "_")

# Date
if "Date" in train.columns:
    train["Date"] = pd.to_datetime(train["Date"], errors="coerce")

# =========================
# 3) Fusion
# =========================
print("Fusion ...")
if stores is not None and "Store" in train.columns and "Store" in stores.columns:
    df = train.merge(stores, on="Store", how="left")
else:
    df = train.copy()
print("Dataset apres fusion :", df.shape)

# =========================
# 4) Filtrage Open = 1
# =========================
print("Filtrage Open=1 ...")
if "Open" in df.columns:
    df = df[df["Open"] == 1].copy()
print("Dataset apres filtrage :", df.shape)

# =========================
# 5) Valeurs manquantes
# =========================
print("Calcul des valeurs manquantes ...")
missing_table = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_rate_%": (df.isna().mean() * 100).round(4)
}).sort_values("missing_rate_%", ascending=False)

missing_path = os.path.join(OUT_DIR, f"missing_values_{stamp}.csv")
missing_table.to_csv(missing_path)
print("Missing sauvegarde :", missing_path)

# =========================
# 6) Statistiques descriptives
# =========================
print("Stats descriptives ...")
desc_path = os.path.join(OUT_DIR, f"statistiques_descriptives_{stamp}.csv")
df.describe(include="all").to_csv(desc_path)
print("Stats sauvegardees :", desc_path)

# =========================
# 7) Visualisations EDA
# =========================
target = "Sales" if "Sales" in df.columns else None

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=300, bbox_inches="tight")
    plt.close()

def grid():
    plt.grid(True, alpha=0.3)

# Nettoyage catégories
for col in ["StoreType", "Assortment"]:
    if col in df.columns:
        df[col] = df[col].astype("string").fillna("Unknown")

# ---------- Distribution ----------
if target:
    print("Graphique distribution ...")

    s = df[target].dropna()
    q1, q99 = s.quantile([0.01, 0.99])
    s_plot = s[(s >= q1) & (s <= q99)]

    plt.figure(figsize=(9, 5))
    plt.hist(s_plot, bins=60, color="#1f77b4", edgecolor="black", alpha=0.85)

    mean_val = s_plot.mean()
    median_val = s_plot.median()
    plt.axvline(mean_val, color="red", linestyle="--", label=f"Moyenne = {mean_val:.0f}")
    plt.axvline(median_val, color="green", linestyle=":", label=f"Mediane = {median_val:.0f}")

    plt.title("Distribution des ventes (hors valeurs extrêmes 1%-99%)")
    plt.xlabel("Sales")
    plt.ylabel("Frequence")
    plt.legend()
    grid()

    savefig("distribution_Sales.png")

# ---------- Série temporelle ----------
if target and "Date" in df.columns:
    print("Graphique serie temporelle ...")

    ts = df.dropna(subset=["Date"]).groupby("Date")[target].sum()

    plt.figure(figsize=(12, 4))
    plt.plot(ts.index, ts.values, color="#2ca02c", linewidth=1.3)

    plt.title("Evolution temporelle des ventes (somme quotidienne)")
    plt.xlabel("Date")
    plt.ylabel("Sales")

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    grid()

    savefig("timeseries_Sales.png")

# ---------- Sales par Assortment ----------
if target and "Assortment" in df.columns:
    print("EDA : Sales par Assortment ...")

    g = df.groupby("Assortment")[target].mean().sort_values(ascending=False)

    plt.figure(figsize=(7, 4))
    bars = plt.bar(g.index.astype(str), g.values, color="#9467bd")

    plt.title("Moyenne des ventes par Assortment")
    plt.xlabel("Assortment")
    plt.ylabel("Sales moyennes")
    grid()

    for b in bars:
        plt.text(b.get_x() + b.get_width()/2, b.get_height(),
                 f"{b.get_height():.0f}", ha="center", va="bottom")

    savefig("eda_sales_by_assortment.png")

# ---------- Sales par jour de semaine (COULEUR GRISE) ----------
if target and "Date" in df.columns:
    print("EDA : Sales par jour de semaine ...")

    df["DayOfWeek"] = df["Date"].dt.dayofweek
    g = df.groupby("DayOfWeek")[target].mean()

    labels = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(
        labels,
        g.reindex(range(7)).values,
        color="#4d4d4d"   # ✅ couleur grise au lieu de orange
    )

    plt.title("Moyenne des ventes selon le jour de la semaine")
    plt.xlabel("Jour")
    plt.ylabel("Sales moyennes")
    grid()

    for b in bars:
        plt.text(b.get_x() + b.get_width()/2, b.get_height(),
                 f"{b.get_height():.0f}", ha="center", va="bottom")

    savefig("eda_sales_by_dayofweek.png")

# ---------- Total mensuel + moyenne mobile ----------
if target and "Date" in df.columns:
    print("EDA : Total sales over time (mensuel) ...")

    ts_daily = df.groupby("Date")[target].sum()
    ts_month = ts_daily.resample("ME").sum()   # compat pandas
    ts_roll = ts_month.rolling(3, min_periods=1).mean()

    plt.figure(figsize=(12, 4))
    plt.plot(ts_month.index, ts_month.values, label="Somme mensuelle")
    plt.plot(ts_roll.index, ts_roll.values, linewidth=2,
             label="Moyenne mobile (3 mois)")

    plt.title("Evolution mensuelle des ventes")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    grid()

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    savefig("eda_total_sales_over_time.png")

# =========================
# 8) Sauvegarde dataset final
# =========================
final_path = os.path.join(OUT_DIR, f"dataset_final_{stamp}.csv")
df.to_csv(final_path, index=False)
print("Dataset final sauvegarde :", final_path)

print("Fin du script")
