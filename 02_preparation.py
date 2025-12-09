# -*- coding: utf-8 -*-
"""
02_preparation.py
TCC – Preparação e PRÉ-PROCESSAMENTO (versão alinhada ao texto)

Responsabilidades:
    • Carregar dados consolidados (01_extraction.py)
    • Ordenar e padronizar datas
    • Selecionar variáveis numéricas
    • Criar defasagens (lags) das variáveis
    • Incluir TARGET_lag1 nas features
    • Gerar um único conjunto: prepared_FULL.parquet
"""

#%% Importações

from pathlib import Path
import pandas as pd
import numpy as np

#%% ---------------------- Config ----------------------
DATA_DIR = Path("./data")
PREP_DIR = Path("./prepared")
PREP_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "dados_consolidados_macro_credito.parquet"
TARGET     = "inadimpl_cartao_total"
DATE_COL   = "data"
LAGS       = [1, 3, 6, 12]

MIN_NONNA_RATIO = 0.70  # mantém colunas com >=70% não-nulos (por segurança)

#%% ---------------------- Funções ---------------------
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Seleciona colunas com boa completude (na prática, todas as suas)
    keep = [DATE_COL] + [
        c for c in df.columns
        if c != DATE_COL and df[c].notna().mean() >= MIN_NONNA_RATIO
    ]
    df = df[keep]

    # NÃO faz forward fill – dataset sem ausências, conforme TCC
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
def prepare_full_dataset() -> pd.DataFrame:
    print("\n=== Preparando dataset FULL ===")
    if not INPUT_FILE.exists():
        raise SystemExit(f"Arquivo não encontrado: {INPUT_FILE}")

    base = pd.read_parquet(INPUT_FILE)
    base = _clean_df(base)

    # Preditores numéricos (exceto TARGET)
    predictors = [
        c for c in base.select_dtypes(include=[np.number]).columns
        if c != TARGET
    ]

    # Cria lags para preditores + TARGET
    base_lag = _make_lags(base, LAGS, predictors + [TARGET])

    # Features = todos os lags das variáveis + TARGET_lag1
    all_lag_feats = [
        c for c in base_lag.columns
        if any(c.endswith(f"lag{L}") for L in LAGS)
    ]
    target_lags = [f"{TARGET}_lag1"]
    feature_cols = sorted(list((set(all_lag_feats) | set(target_lags)) - {TARGET}))

    final = _finalize(base_lag, feature_cols)

    print(f"Observações finais: {len(final)} | Features: {len(feature_cols)}")
    print(f"Primeira data: {final[DATE_COL].min().date()} | Última data: {final[DATE_COL].max().date()}")
    return final

#%%

def main():
    full = prepare_full_dataset()
    out_full = PREP_DIR / "prepared_FULL.parquet"
    full.to_parquet(out_full, index=False)

    print("\nArquivo gerado em ./prepared:")
    print(f" - {out_full.name}")

if __name__ == "__main__":
    main()
