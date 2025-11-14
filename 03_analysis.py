# -*- coding: utf-8 -*-
"""
03_analysis.py
Versão revisada — com Naïve Rolling/Static, seeds fixos e ACF/PACF.

Modelos:
    - Naïve (Rolling e Static)
    - Linear Regression
    - SVR
    - Random Forest
    - XGBoost
    - LSTM
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, random
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Seeds fixos
np.random.seed(42); random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# ---------------- Config ----------------
PREP_DIR = Path("./prepared")
RES_DIR = Path("./results"); RES_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "inadimpl_cartao_total"; DATE_COL = "data"

# ---------------- Métricas ----------------
def smape(y_true, y_pred, eps=1e-9):
    return (100/len(y_true))*np.sum(2*np.abs(y_pred - y_true)/(np.abs(y_true)+np.abs(y_pred)+eps))

def direction_acc(y_true, y_pred):
    if len(y_true) < 2: return np.nan
    return ((np.diff(y_true)>0)==(np.diff(y_pred)>0)).mean()*100

def evaluate(y_true, y_pred, model_name):
    return dict(Model=model_name,
                MAE=mean_absolute_error(y_true,y_pred),
                RMSE=np.sqrt(mean_squared_error(y_true,y_pred)),
                R2=r2_score(y_true,y_pred),
                SMAPE=smape(y_true,y_pred),
                DA=direction_acc(y_true,y_pred))

# ---------------- Modelos ----------------
def make_lstm_model(input_shape):
    inp = Input(shape=input_shape)
    x = LSTM(64, activation='tanh')(inp)
    x = Dense(32, activation='relu')(x)
    out = Dense(1)(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------- Funções ----------------
def plot_real_pred(dates, y_true, y_pred, title, outpath):
    plt.figure(figsize=(12,6))
    plt.plot(dates, y_true, label="Real", linewidth=2)
    plt.plot(dates, y_pred, label="Predito", linestyle="--", linewidth=2)
    plt.title(title); plt.xlabel("Data"); plt.ylabel("Taxa de Inadimplência (%)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight"); plt.close()

def plot_feature_importance(importances, features, title, outpath):
    idx = np.argsort(importances)[::-1][:10]
    plt.figure(figsize=(8,5))
    plt.barh(np.array(features)[idx][::-1], np.array(importances)[idx][::-1])
    plt.title(title); plt.xlabel("Importância Relativa")
    plt.tight_layout(); plt.savefig(outpath, dpi=300, bbox_inches="tight"); plt.close()

def analyse_acf_pacf(y, title, outprefix):
    fig, ax = plt.subplots(2,1,figsize=(10,6))
    plot_acf(y, lags=24, ax=ax[0]); ax[0].set_title(f"{title} - ACF (autocorrelação)")
    plot_pacf(y, lags=24, ax=ax[1]); ax[1].set_title(f"{title} - PACF (autocorrelação parcial)")
    plt.tight_layout(); plt.savefig(RES_DIR/f"{outprefix}_ACF_PACF.png", dpi=300); plt.close()

# ---------------- Pipeline ----------------
def run_analysis(tag, file):
    df = pd.read_parquet(file)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    feats = [c for c in df.columns if c not in [DATE_COL, TARGET]]
    X = df[feats].values; y = df[TARGET].values; dates = df[DATE_COL].values
    cut = int(len(df)*0.8)
    Xtr, Xte = X[:cut], X[cut:]; ytr, yte = y[:cut], y[cut:]; dte = dates[cut:]

    analyse_acf_pacf(y, tag, tag)  # salva ACF/PACF do alvo

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    results = []; preds = {}

    # --- Naïve Rolling e Static ---
    y_shift = df[TARGET].shift(1).values
    naive_roll = y_shift[cut:]
    mask = ~np.isnan(naive_roll)
    assert np.isfinite(yte[mask]).all() and np.isfinite(naive_roll[mask]).all()
    results.append(evaluate(yte[mask], naive_roll[mask], "Naive_Rolling"))
    preds["Naive_Rolling"] = naive_roll

    last_train = ytr[-1]
    naive_static = np.full_like(yte, fill_value=last_train, dtype=float)
    results.append(evaluate(yte, naive_static, "Naive_Static"))
    preds["Naive_Static"] = naive_static

    # --- Linear Regression ---
    lr = LinearRegression().fit(Xtr_s, ytr)
    preds["Linear"] = lr.predict(Xte_s)
    results.append(evaluate(yte, preds["Linear"], "Linear"))

    # --- SVR ---
    svr = SVR(kernel="rbf", C=1.0, gamma="scale").fit(Xtr_s, ytr)
    preds["SVR"] = svr.predict(Xte_s)
    results.append(evaluate(yte, preds["SVR"], "SVR"))

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=600, random_state=42).fit(Xtr, ytr)
    preds["RandomForest"] = rf.predict(Xte)
    results.append(evaluate(yte, preds["RandomForest"], "RandomForest"))

    # --- XGBoost ---
    xgb = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=5,
                       subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb.fit(Xtr, ytr)
    preds["XGBoost"] = xgb.predict(Xte)
    results.append(evaluate(yte, preds["XGBoost"], "XGBoost"))

    # --- LSTM ---
    Xtr_r = Xtr_s.reshape((Xtr_s.shape[0], 1, Xtr_s.shape[1]))
    Xte_r = Xte_s.reshape((Xte_s.shape[0], 1, Xte_s.shape[1]))
    lstm = make_lstm_model((1, Xtr_s.shape[1]))
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    lstm.fit(Xtr_r, ytr, validation_split=0.2, epochs=200, batch_size=8, verbose=0, callbacks=[es])
    preds["LSTM"] = lstm.predict(Xte_r).flatten()
    results.append(evaluate(yte, preds["LSTM"], "LSTM"))

    # --- Resultado final ---
    dfres = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)
    dfres.to_csv(RES_DIR/f"results_{tag}_final.csv", index=False)

    best = dfres.iloc[0]["Model"]
    plot_real_pred(dte, yte, preds[best], f"{tag} - Real vs Predito ({best})",
                   RES_DIR/f"{tag}_{best}_real_vs_pred.png")

    # Importância (RF e XGB)
    plot_feature_importance(rf.feature_importances_, feats, f"{tag} - RF Importância",
                            RES_DIR/f"{tag}_RF_importance.png")
    plot_feature_importance(xgb.feature_importances_, feats, f"{tag} - XGB Importância",
                            RES_DIR/f"{tag}_XGB_importance.png")

    print(f"\n{tag} - Resultados finais:")
    print(dfres.to_string(index=False, float_format="%.4f"))
    return dfres

def main():
    for tag in ["FULL", "EXCL"]:
        path = PREP_DIR / f"prepared_{tag}.parquet"
        if path.exists():
            run_analysis(tag, path)
        else:
            print(f"Arquivo não encontrado: {path}")

if __name__ == "__main__":
    main()
