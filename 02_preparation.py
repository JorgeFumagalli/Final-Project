# -*- coding: utf-8 -*-
"""
02_preparation.py
TCC – Preparação e PRÉ-PROCESSAMENTO (versão revisada)

Responsabilidades:
    • Carregar dados consolidados (01_extraction.py)
    • Limpar e imputar apenas preditores
    • Criar defasagens (lags) das variáveis
    • Incluir TARGET_lag1 nas features
    • Gerar dois conjuntos: FULL (tudo) e EXCL (sem 01/2019–12/2021)
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------- Config ----------------------
DATA_DIR = Path("./data")
PREP_DIR = Path("./prepared")
PREP_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / r"D:/Degas/Documents/Pós Gradução - Data Science and Analytics/TCC/Final-Project/data/dados_consolidados_macro_credito.parquet"
TARGET = "inadimpl_cartao_total"
DATE_COL = "data"
LAGS = [1, 3, 6, 12]

EXCLUDE_START = "2019-01-01"
EXCLUDE_END   = "2021-12-31"
MIN_NONNA_RATIO = 0.70  # mantém colunas com >=70% não-nulos

# ---------------------- Funções ---------------------
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    keep = [DATE_COL] + [c for c in df.columns if c != DATE_COL and df[c].notna().mean() >= MIN_NONNA_RATIO]
    df = df[keep]

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
    df[num_cols] = df[num_cols].ffill()
    return df

def _make_lags(df: pd.DataFrame, lags, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for L in lags:
            out[f"{c}_lag{L}"] = out[c].shift(L)
    return out

def _finalize(df: pd.DataFrame, feature_cols) -> pd.DataFrame:
    final = df.dropna(subset=[TARGET] + feature_cols).copy()
    final = final[[DATE_COL, TARGET] + feature_cols]
    return final

# ---------------------- Pipeline --------------------
def prepare_dataset(scope: str) -> pd.DataFrame:
    print(f"\n=== Preparando dataset: {scope} ===")
    if not INPUT_FILE.exists():
        raise SystemExit(f"Arquivo não encontrado: {INPUT_FILE}")

    base = pd.read_parquet(INPUT_FILE)
    base = _clean_df(base)

    if scope == "EXCL":
        base = base.loc[~base[DATE_COL].between(EXCLUDE_START, EXCLUDE_END)].reset_index(drop=True)

    predictors = [c for c in base.select_dtypes(include=[np.number]).columns if c != TARGET]
    base_lag = _make_lags(base, LAGS, predictors + [TARGET])

    # features = todos os lags das variáveis + TARGET_lag1 (essencial)
    all_lag_feats = [c for c in base_lag.columns if any(c.endswith(f"lag{L}") for L in LAGS)]
    target_lags = [f"{TARGET}_lag1"]  # pode incluir mais se desejar
    feature_cols = sorted(list((set(all_lag_feats) | set(target_lags)) - {TARGET}))

    final = _finalize(base_lag, feature_cols)
    print(f"Observações finais: {len(final)} | Features: {len(feature_cols)}")
    print(f"Primeira data: {final[DATE_COL].min().date()} | Última data: {final[DATE_COL].max().date()}")
    return final

def main():
    full = prepare_dataset("FULL")
    excl = prepare_dataset("EXCL")

    out_full = PREP_DIR / "prepared_FULL.parquet"
    out_excl = PREP_DIR / "prepared_EXCL.parquet"
    full.to_parquet(out_full, index=False)
    excl.to_parquet(out_excl, index=False)

    print("\nArquivos gerados em ./prepared:")
    print(f" - {out_full.name}")
    print(f" - {out_excl.name}")

if __name__ == "__main__":
    main()
