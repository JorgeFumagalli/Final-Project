# -*- coding: utf-8 -*-
"""
03_analysis.py
Versão alinhada ao TCC – Modelagem preditiva (regressão)

Modelos:
    - Regressão Linear
    - SVR
    - XGBoost
    - MLP
    - LSTM

Métricas:
    - MSE
    - R²
    - MAPE
    - Directional Accuracy (DA)

Cenários:
    - FULL: todos os dados disponíveis (2015–2025)
    - EXCL: exclui o período de instabilidade fiscal (2019–2021)

Saídas:
    - CSV com métricas por cenário (results_FULL_final.csv, results_EXCL_final.csv)
    - Gráficos Real vs Predito por modelo e cenário
    - Gráficos de barras com métricas por cenário
    - Gráficos FULL vs EXCL por métrica
    - Painel consolidado FULL vs EXCL (PANEL_FULL_EXCL_metrics.png)
"""

#%% Importações
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

#%% Seeds fixos (reprodutibilidade)
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

#%% Configurações gerais
PREP_DIR = Path("./prepared")
RES_DIR  = Path("./results")
RES_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = "inadimpl_cartao_total"
DATE_COL = "data"

#%% Métricas
def mape(y_true, y_pred, eps=1e-9):
    """Mean Absolute Percentage Error em %."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def direction_acc(y_true, y_pred):
    """Directional Accuracy: % de acertos na direção da variação."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    return ((true_diff > 0) == (pred_diff > 0)).mean() * 100


def evaluate(y_true, y_pred, model_name):
    """Retorna dicionário com métricas para um modelo."""
    return dict(
        Model = model_name,
        MSE   = mean_squared_error(y_true, y_pred),
        R2    = r2_score(y_true, y_pred),
        MAPE  = mape(y_true, y_pred),
        DA    = direction_acc(y_true, y_pred)
    )

#%% Modelos Keras (MLP e LSTM)
def make_mlp_model(input_dim: int) -> Model:
    inp = Input(shape=(input_dim,))
    x   = Dense(64, activation='relu')(inp)
    x   = Dense(32, activation='relu')(x)
    out = Dense(1)(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model


def make_lstm_model(input_shape) -> Model:
    inp = Input(shape=input_shape)
    x   = LSTM(64, activation='tanh')(inp)
    x   = Dense(32, activation='relu')(x)
    out = Dense(1)(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

#%% Função de estilo ABNT para gráficos
def format_abnt_axes(ax, xlabel: str = "", ylabel: str = "", show_legend: bool = True):
    # Títulos dos eixos
    if xlabel:
        ax.set_xlabel(xlabel,
                      fontfamily="Arial",
                      fontsize=11,
                      color="black")
    if ylabel:
        ax.set_ylabel(ylabel,
                      fontfamily="Arial",
                      fontsize=11,
                      color="black")

    # Ticks dos eixos
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")
        label.set_fontsize(9)
        label.set_color("black")

    # Sem grade
    ax.grid(False)

    # Fundo branco
    ax.set_facecolor("white")
    if ax.figure is not None:
        ax.figure.set_facecolor("white")

    # Remover bordas superior e direita
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Eixos principal horizontal e vertical: linha preta 1,5 pt
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color("black")

    # Legenda sem moldura
    if show_legend:
        ax.legend(frameon=False,
                  prop={"family": "Arial", "size": 9})

#%% Funções de gráficos básicos
def plot_real_pred(dates, y_true, y_pred, title, outpath):
    """Gráfico Real vs Predito para um modelo/período (padrão ABNT)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(dates, y_true, label="Real", linewidth=2)
    ax.plot(dates, y_pred, label="Predito", linestyle="--", linewidth=2)

    # Título do gráfico não é usado (vai na legenda do TCC)
    format_abnt_axes(ax,
                     xlabel="Data",
                     ylabel="Taxa de inadimplência (%)",
                     show_legend=True)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bars(dfres: pd.DataFrame, tag: str, outdir: Path):
    """
    Gera quatro gráficos de barras comparando os modelos:
        - MSE
        - R²
        - MAPE
        - DA
    para um dado período (tag), no padrão ABNT.
    """
    metrics = ["MSE", "R2", "MAPE", "DA"]

    for m in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(dfres["Model"], dfres[m])

        ylabel = m if m != "R2" else "R²"
        format_abnt_axes(ax,
                         xlabel="Modelo",
                         ylabel=ylabel,
                         show_legend=False)

        plt.tight_layout()
        outpath = outdir / f"{tag}_metric_{m}.png"
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)

#%% Gráficos comparativos FULL vs EXCL
def plot_full_vs_excl_series(df_full: pd.DataFrame,
                             df_excl: pd.DataFrame,
                             date_col: str,
                             target: str,
                             outdir: Path):
    """Compara a série de inadimplência FULL vs EXCL em um único gráfico (padrão ABNT)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df_full[date_col], df_full[target], label="FULL", linewidth=2)
    ax.plot(df_excl[date_col], df_excl[target], label="EXCL (sem 2019–2021)", linewidth=2)

    format_abnt_axes(ax,
                     xlabel="Data",
                     ylabel="Taxa de inadimplência (%)",
                     show_legend=True)

    plt.tight_layout()
    outpath = outdir / "FULL_vs_EXCL_inadimplencia.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_compare_metric_full_excl(df_full_res: pd.DataFrame,
                                  df_excl_res: pd.DataFrame,
                                  metric: str,
                                  outdir: Path):
    """
    Compara uma métrica (MSE, R2, MAPE, DA) entre FULL e EXCL para cada modelo.
    Gera gráfico de barras agrupadas (padrão ABNT).
    """
    mf = df_full_res[["Model", metric]].rename(columns={metric: f"{metric}_FULL"})
    me = df_excl_res[["Model", metric]].rename(columns={metric: f"{metric}_EXCL"})
    merged = mf.merge(me, on="Model", how="inner")

    x = np.arange(len(merged["Model"]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, merged[f"{metric}_FULL"], width, label="FULL")
    ax.bar(x + width/2, merged[f"{metric}_EXCL"], width, label="EXCL")

    ax.set_xticks(x)
    ax.set_xticklabels(merged["Model"])

    ylabel = metric if metric != "R2" else "R²"
    format_abnt_axes(ax,
                     xlabel="Modelo",
                     ylabel=ylabel,
                     show_legend=True)

    plt.tight_layout()
    outpath = outdir / f"COMPARE_FULL_EXCL_{metric}.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

#%% Painel consolidado FULL vs EXCL
def plot_panel_full_excl(df_full_res: pd.DataFrame,
                         df_excl_res: pd.DataFrame,
                         outdir: Path):
    """
    Gera um painel (2x2 subplots) comparando FULL vs EXCL
    para as quatro métricas (MSE, R2, MAPE, DA).
    """
    metrics = ["MSE", "R2", "MAPE", "DA"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        mf = df_full_res[["Model", metric]].rename(columns={metric: f"{metric}_FULL"})
        me = df_excl_res[["Model", metric]].rename(columns={metric: f"{metric}_EXCL"})
        merged = mf.merge(me, on="Model", how="inner")

        x = np.arange(len(merged["Model"]))
        width = 0.35

        ax.bar(x - width/2, merged[f"{metric}_FULL"], width, label="FULL")
        ax.bar(x + width/2, merged[f"{metric}_EXCL"], width, label="EXCL")
        ax.set_xticks(x)
        ax.set_xticklabels(merged["Model"])

        ylabel = metric if metric != "R2" else "R²"
        format_abnt_axes(ax,
                         xlabel="Modelo",
                         ylabel=ylabel,
                         show_legend=False)

    # Legenda no topo
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center",
               ncol=2,
               frameon=False,
               prop={"family": "Arial", "size": 10},
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    outpath = outdir / "PANEL_FULL_EXCL_metrics.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

#%% Função de análise para um único cenário
def run_period_analysis(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """
    Executa o pipeline de modelagem para um cenário específico (FULL ou EXCL):
        - split temporal 80/20
        - scaler
        - Regressão Linear, SVR, XGBoost, MLP, LSTM
        - métricas
        - gráficos Real vs Predito
        - gráficos de métricas
    Retorna DataFrame de resultados.
    """
    print(f"\n=== Análise {tag} ===")

    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Features e alvo
    feats = [c for c in df.columns if c not in [DATE_COL, TARGET]]
    X = df[feats].values
    y = df[TARGET].values
    dates = df[DATE_COL].values

    # Split temporal 80/20
    cut = int(len(df) * 0.8)
    Xtr, Xte = X[:cut], X[cut:]
    ytr, yte = y[:cut], y[cut:]
    dte = dates[cut:]

    # Escalonamento (para Linear, SVR, MLP, LSTM)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    results = []
    preds   = {}

    # -------- Regressão Linear --------
    lin = LinearRegression()
    lin.fit(Xtr_s, ytr)
    preds["Linear"] = lin.predict(Xte_s)
    results.append(evaluate(yte, preds["Linear"], "Linear"))

    # -------- SVR --------
    svr = SVR(kernel="rbf", C=1.0, gamma="scale")
    svr.fit(Xtr_s, ytr)
    preds["SVR"] = svr.predict(Xte_s)
    results.append(evaluate(yte, preds["SVR"], "SVR"))

    # -------- XGBoost --------
    xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb.fit(Xtr, ytr)
    preds["XGBoost"] = xgb.predict(Xte)
    results.append(evaluate(yte, preds["XGBoost"], "XGBoost"))

    # -------- MLP --------
    mlp = make_mlp_model(Xtr_s.shape[1])
    es_mlp = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    mlp.fit(
        Xtr_s, ytr,
        validation_split=0.2,
        epochs=200,
        batch_size=8,
        verbose=0,
        callbacks=[es_mlp]
    )
    preds["MLP"] = mlp.predict(Xte_s).flatten()
    results.append(evaluate(yte, preds["MLP"], "MLP"))

    # -------- LSTM --------
    Xtr_r = Xtr_s.reshape((Xtr_s.shape[0], 1, Xtr_s.shape[1]))
    Xte_r = Xte_s.reshape((Xte_s.shape[0], 1, Xte_s.shape[1]))
    lstm = make_lstm_model((1, Xtr_s.shape[1]))
    es_lstm = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    lstm.fit(
        Xtr_r, ytr,
        validation_split=0.2,
        epochs=200,
        batch_size=8,
        verbose=0,
        callbacks=[es_lstm]
    )
    preds["LSTM"] = lstm.predict(Xte_r).flatten()
    results.append(evaluate(yte, preds["LSTM"], "LSTM"))

    # -------- Resultados finais --------
    dfres = pd.DataFrame(results).sort_values(by="MSE").reset_index(drop=True)
    out_csv = RES_DIR / f"results_{tag}_final.csv"
    dfres.to_csv(out_csv, index=False)

    print(f"\n{tag} - Resultados finais:")
    print(dfres.to_string(index=False, float_format="%.4f"))

    best = dfres.iloc[0]["Model"]
    print(f"\nMelhor modelo ({tag}): {best}")

    # ----- Gráficos Real vs Predito -----
    for model_name, y_pred in preds.items():
        out_png = RES_DIR / f"{tag}_{model_name}_real_vs_pred.png"
        titulo  = f"{tag} - Real vs Predito ({model_name})"
        plot_real_pred(dte, yte, y_pred, titulo, out_png)

    # ----- Gráficos de métricas (barras) -----
    plot_metrics_bars(dfres, tag, RES_DIR)

    return dfres

#%% Função principal
def main():
    # Carrega dataset FULL
    path = PREP_DIR / "prepared_FULL.parquet"
    if not path.exists():
        print(f"Arquivo não encontrado: {path}")
        return

    df_full = pd.read_parquet(path)
    df_full[DATE_COL] = pd.to_datetime(df_full[DATE_COL])
    df_full = df_full.sort_values(DATE_COL).reset_index(drop=True)

    # Define cenário EXCL: exclui 2019-01-01 até 2021-12-01
    mask_excl = (df_full[DATE_COL] < "2019-01-01") | (df_full[DATE_COL] > "2021-12-01")
    df_excl = df_full.loc[mask_excl].reset_index(drop=True)

    print(f"Observações FULL: {len(df_full)}")
    print(f"Observações EXCL (sem 2019–2021): {len(df_excl)}")

    # Roda análise para FULL e EXCL
    dfres_full = run_period_analysis(df_full, "FULL")
    dfres_excl = run_period_analysis(df_excl, "EXCL")

    # Gráfico da série FULL vs EXCL
    plot_full_vs_excl_series(df_full, df_excl, DATE_COL, TARGET, RES_DIR)

    # Gráficos comparando métricas FULL vs EXCL (por métrica)
    for metric in ["MSE", "R2", "MAPE", "DA"]:
        plot_compare_metric_full_excl(dfres_full, dfres_excl, metric, RES_DIR)

    # Painel consolidado FULL vs EXCL (todas as métricas)
    plot_panel_full_excl(dfres_full, dfres_excl, RES_DIR)

#%% Execução
if __name__ == "__main__":
    main()
