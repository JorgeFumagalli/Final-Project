# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 11:47:44 2025

@author: jorge
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

DATA_DIR = Path(r"C:\Users\jorge\Downloads")
OUTPUT_DIR = Path(r"C:\Users\jorge\Downloads\model_results")
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_RATE_COLUMNS = [
    "inadimpl_cartao_total",    # Inadimplência cartão total
]

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR, SVC
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow não disponível. Modelos de Deep Learning serão pulados.")
    TENSORFLOW_AVAILABLE = False

def load_prepared_data(target_variable):
    """Carrega dados preparados do arquivo Excel gerado pelo script anterior"""
    print(f"\n{'='*80}")
    print(f"CARREGANDO DADOS PREPARADOS PARA: {target_variable}")
    print(f"{'='*80}")
    
    input_file = OUTPUT_DIR / f'prepared_data_{target_variable}.xlsx'
    
    if not input_file.exists():
        print(f"Arquivo {input_file} não encontrado.")
        print(f"Execute o script 02_prepare_data.py primeiro!")
        return None
    
    # Carregar dados do Excel
    feature_df = pd.read_excel(input_file, sheet_name='feature_df')
    metadata = pd.read_excel(input_file, sheet_name='metadata')
    feature_cols_df = pd.read_excel(input_file, sheet_name='feature_cols')
    
    # Extrair metadados
    metadata_dict = dict(zip(metadata['parameter'], metadata['value']))
    split_idx = int(metadata_dict['split_idx'])
    sequence_length = int(metadata_dict['sequence_length'])
    feature_cols = feature_cols_df['feature_cols'].tolist()
    
    print(f"Dados carregados: {len(feature_df)} observações")
    print(f"Features: {len(feature_cols)} colunas")
    
    # Reconstruir arrays
    X = feature_df[feature_cols].values
    y = feature_df[target_variable].values
    dates = pd.to_datetime(feature_df['data']).values
    
    # Split temporal
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Preparar targets de classificação
    def create_classification_target(y_values):
        y_diff = np.diff(y_values)
        return (y_diff > 0).astype(int)
    
    y_train_class = create_classification_target(y_train)
    y_test_class = create_classification_target(y_test)
    
    X_train_class = X_train_scaled[1:]
    X_test_class = X_test_scaled[1:]
    dates_test_class = dates_test[1:]
    
    # Preparar sequências para LSTM
    def prepare_sequences(data, target_col, seq_length):
        feature_data = data[feature_cols].values
        target_data = data[target_col].values
        
        X_seq, y_seq = [], []
        for i in range(seq_length, len(data)):
            X_seq.append(feature_data[i-seq_length:i])
            y_seq.append(target_data[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    X_seq, y_seq = prepare_sequences(feature_df, target_variable, sequence_length)
    
    split_seq = int(len(X_seq) * 0.8)
    X_train_seq, X_test_seq = X_seq[:split_seq], X_seq[split_seq:]
    y_train_seq, y_test_seq = y_seq[:split_seq], y_seq[split_seq:]
    
    # Normalizar sequências
    scaler_seq = MinMaxScaler()
    X_train_seq_scaled = scaler_seq.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[-1])).reshape(X_train_seq.shape)
    X_test_seq_scaled = scaler_seq.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)
    
    dates_lstm = dates[split_seq + sequence_length:]
    
    print(f"Dados reconstruídos com sucesso!")
    print(f"Treino: {len(X_train)} observações")
    print(f"Teste: {len(X_test)} observações")
    
    return {
        'target_variable': target_variable,
        'feature_df': feature_df,
        'feature_cols': feature_cols,
        
        # Dados de regressão
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'dates_test': dates_test,
        
        # Dados de classificação
        'X_train_class': X_train_class,
        'X_test_class': X_test_class,
        'y_train_class': y_train_class,
        'y_test_class': y_test_class,
        'dates_test_class': dates_test_class,
        
        # Dados para LSTM
        'X_train_seq': X_train_seq_scaled,
        'X_test_seq': X_test_seq_scaled,
        'y_train_seq': y_train_seq,
        'y_test_seq': y_test_seq,
        'dates_lstm': dates_lstm,
        'sequence_length': sequence_length,
        
        # Scalers
        'scaler': scaler,
        'scaler_seq': scaler_seq
    }

# =============================================================================
# SEÇÃO 1: MODELOS SVM (SUPPORT VECTOR MACHINE)
# =============================================================================

class SVMPredictor:
    """Modelo baseado em Support Vector Machine para regressão"""
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.model = SVR(kernel=kernel, C=C, gamma=gamma)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class SVMClassifier:
    """Modelo de classificação baseado em Support Vector Machine"""
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

def train_svm_models(data_dict):
    """Treina modelos SVM para regressão e classificação"""
    print("\n=== SEÇÃO 1: MODELOS SVM (SUPPORT VECTOR MACHINE) ===")
    
    target_variable = data_dict['target_variable']
    results = {'regression': [], 'classification': []}
    
    # SVM Regressão
    print("Treinando SVM para Regressão...")
    svm_reg = SVMPredictor(kernel='rbf', C=1.0)
    svm_reg.fit(data_dict['X_train'], data_dict['y_train'])
    y_pred_reg = svm_reg.predict(data_dict['X_test'])
    
    reg_result = evaluate_regression_model(data_dict['y_test'], y_pred_reg, 'SVM Regression')
    results['regression'].append(reg_result)
    plot_predictions(data_dict['y_test'], y_pred_reg, f'SVM_Regression_{target_variable}', data_dict['dates_test'])
    
    # SVM Classificação
    print("Treinando SVM para Classificação...")
    svm_class = SVMClassifier(kernel='rbf', C=1.0)
    svm_class.fit(data_dict['X_train_class'], data_dict['y_train_class'])
    y_pred_class = svm_class.predict(data_dict['X_test_class'])
    y_pred_proba = svm_class.predict_proba(data_dict['X_test_class'])
    
    class_result = evaluate_classification_model(data_dict['y_test_class'], y_pred_class, y_pred_proba, 'SVM Classification')
    results['classification'].append(class_result)
    
    return results

# =============================================================================
# SEÇÃO 2: MODELOS DE MACHINE LEARNING TRADICIONAIS
# =============================================================================

def get_ml_regression_models():
    """Retorna dicionário com modelos de ML para regressão"""
    return {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

def get_ml_classification_models():
    """Retorna dicionário com modelos de ML para classificação"""
    return {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

def train_ml_models(data_dict):
    """Treina modelos de Machine Learning tradicionais"""
    print("\n=== SEÇÃO 2: MODELOS DE MACHINE LEARNING TRADICIONAIS ===")
    
    target_variable = data_dict['target_variable']
    results = {'regression': [], 'classification': []}
    
    # Modelos de Regressão
    print("Treinando modelos de Regressão...")
    ml_models = get_ml_regression_models()
    
    for name, model in ml_models.items():
        print(f"   Treinando {name}...")
        try:
            model.fit(data_dict['X_train'], data_dict['y_train'])
            y_pred = model.predict(data_dict['X_test'])
            
            result = evaluate_regression_model(data_dict['y_test'], y_pred, name)
            results['regression'].append(result)
            
            # Plot para os 3 primeiros modelos
            if len(results['regression']) <= 3:
                plot_predictions(data_dict['y_test'], y_pred, f"{name}_{target_variable}", data_dict['dates_test'])
                
        except Exception as e:
            print(f"   Erro ao treinar {name}: {e}")
    
    # Modelos de Classificação
    print("Treinando modelos de Classificação...")
    classification_models = get_ml_classification_models()
    
    for name, model in classification_models.items():
        print(f"   Treinando {name}...")
        try:
            model.fit(data_dict['X_train_class'], data_dict['y_train_class'])
            y_pred_class = model.predict(data_dict['X_test_class'])
            
            # Probabilidades
            try:
                y_pred_proba = model.predict_proba(data_dict['X_test_class'])
            except:
                y_pred_proba = None
            
            result = evaluate_classification_model(data_dict['y_test_class'], y_pred_class, y_pred_proba, name)
            results['classification'].append(result)
                
        except Exception as e:
            print(f"   Erro ao treinar {name}: {e}")
    
    return results

# =============================================================================
# SEÇÃO 3: MODELOS DE DEEP LEARNING
# =============================================================================

def create_lstm_model(input_shape, units=50):
    """Cria modelo LSTM"""
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def create_dnn_model(input_dim):
    """Cria modelo DNN (Deep Neural Network)"""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_deep_learning_models(data_dict):
    """Treina modelos de Deep Learning"""
    print("\n=== SEÇÃO 3: MODELOS DE DEEP LEARNING ===")
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow não disponível. Pulando modelos de Deep Learning.")
        return {'regression': [], 'classification': []}
    
    target_variable = data_dict['target_variable']
    results = {'regression': [], 'classification': []}
    
    # DNN
    print("Treinando DNN (Deep Neural Network)...")
    try:
        dnn_model = create_dnn_model(data_dict['X_train'].shape[1])
        dnn_model.fit(data_dict['X_train'], data_dict['y_train'], 
                     epochs=100, batch_size=32, validation_split=0.2, verbose=0)
        
        y_pred_dnn = dnn_model.predict(data_dict['X_test'], verbose=0).flatten()
        result = evaluate_regression_model(data_dict['y_test'], y_pred_dnn, 'DNN')
        results['regression'].append(result)
        plot_predictions(data_dict['y_test'], y_pred_dnn, f'DNN_{target_variable}', data_dict['dates_test'])
        
    except Exception as e:
        print(f"   Erro ao treinar DNN: {e}")
    
    # LSTM
    print("Treinando LSTM...")
    try:
        lstm_model = create_lstm_model((data_dict['X_train_seq'].shape[1], data_dict['X_train_seq'].shape[2]))
        lstm_model.fit(data_dict['X_train_seq'], data_dict['y_train_seq'], 
                      epochs=50, batch_size=16, validation_split=0.2, verbose=0)
        
        y_pred_lstm = lstm_model.predict(data_dict['X_test_seq'], verbose=0).flatten()
        result = evaluate_regression_model(data_dict['y_test_seq'], y_pred_lstm, 'LSTM')
        results['regression'].append(result)
        
        plot_predictions(data_dict['y_test_seq'], y_pred_lstm, f'LSTM_{target_variable}', data_dict['dates_lstm'])
        
    except Exception as e:
        print(f"   Erro ao treinar LSTM: {e}")
    
    return results

# =============================================================================
# FUNÇÕES DE AVALIAÇÃO E VISUALIZAÇÃO
# =============================================================================

def evaluate_regression_model(y_true, y_pred, model_name):
    """Calcula métricas de avaliação para regressão"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2
    }

def evaluate_classification_model(y_true, y_pred, y_pred_proba, model_name):
    """Calcula métricas de classificação"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    
    # Matriz de confusão
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Taxas de falsos positivos e negativos
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # ROC e AUC
    if y_pred_proba is not None:
        fpr_roc, tpr_roc, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr_roc, tpr_roc)
    else:
        roc_auc = None
        fpr_roc, tpr_roc = None, None
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'FPR': fpr,
        'FNR': fnr,
        'AUC': roc_auc,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'fpr_roc': fpr_roc,
        'tpr_roc': tpr_roc
    }

def plot_predictions(y_true, y_pred, model_name, dates=None):
    """Plota predições vs valores reais"""
    plt.figure(figsize=(12, 6))
    
    if dates is not None:
        plt.plot(dates, y_true, label='Real', alpha=0.7)
        plt.plot(dates, y_pred, label='Predição', alpha=0.7)
        plt.xlabel('Data')
    else:
        plt.plot(y_true, label='Real', alpha=0.7)
        plt.plot(y_pred, label='Predição', alpha=0.7)
        plt.xlabel('Observação')
    
    plt.ylabel(model_name.split('_')[-1])
    plt.title(f'{model_name} - Predições vs Real')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{model_name.replace(" ", "_")}_predictions.png', dpi=300)
    plt.show()

def plot_roc_curves(classification_results, target_variable):
    """Plota curvas ROC para todos os modelos"""
    plt.figure(figsize=(10, 8))
    
    for result in classification_results:
        if result['fpr_roc'] is not None and result['tpr_roc'] is not None:
            plt.plot(result['fpr_roc'], result['tpr_roc'], 
                    label=f"{result['Model']} (AUC = {result['AUC']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title(f'Curvas ROC - {target_variable}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'roc_curves_{target_variable}.png', dpi=300)
    plt.show()

def plot_confusion_matrices(classification_results, target_variable):
    """Plota matrizes de confusão para todos os modelos"""
    n_models = len(classification_results)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(classification_results):
        row, col = i // cols, i % cols
        
        cm = np.array([[result['TN'], result['FP']], 
                      [result['FN'], result['TP']]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Pred 0', 'Pred 1'],
                   yticklabels=['Real 0', 'Real 1'],
                   ax=axes[row, col])
        axes[row, col].set_title(f'{result["Model"]}')
    
    # Remove subplots vazios
    for i in range(n_models, rows * cols):
        row, col = i // cols, i % cols
        fig.delaxes(axes[row, col])
    
    plt.suptitle(f'Matrizes de Confusão - {target_variable}')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'confusion_matrices_{target_variable}.png', dpi=300)
    plt.show()

def generate_comparison_reports(all_results, target_variable):
    """Gera relatórios comparativos dos modelos"""
    print(f"\n=== RELATÓRIOS COMPARATIVOS PARA {target_variable} ===")
    
    # Resultados de Regressão
    if all_results['regression']:
        results_df = pd.DataFrame(all_results['regression'])
        results_df = results_df.sort_values('RMSE')
        
        print(f"\n=== RANKING DOS MODELOS DE REGRESSÃO (por RMSE) ===")
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Salvar resultados
        results_df.to_csv(OUTPUT_DIR / f'regression_results_{target_variable}.csv', index=False)
        
        # Plot comparativo
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.barh(results_df['Model'], results_df['RMSE'])
        plt.xlabel('RMSE')
        plt.title(f'Root Mean Square Error - {target_variable}')
        plt.gca().invert_yaxis()
        
        plt.subplot(2, 2, 2)
        plt.barh(results_df['Model'], results_df['R²'])
        plt.xlabel('R²')
        plt.title(f'Coeficiente de Determinação - {target_variable}')
        plt.gca().invert_yaxis()
        
        plt.subplot(2, 2, 3)
        plt.barh(results_df['Model'], results_df['MAE'])
        plt.xlabel('MAE')
        plt.title(f'Mean Absolute Error - {target_variable}')
        plt.gca().invert_yaxis()
        
        plt.subplot(2, 2, 4)
        plt.barh(results_df['Model'], results_df['MSE'])
        plt.xlabel('MSE')
        plt.title(f'Mean Square Error - {target_variable}')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'regression_metrics_{target_variable}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n=== MELHOR MODELO DE REGRESSÃO ===")
        best_model = results_df.iloc[0]
        print(f"Modelo: {best_model['Model']}")
        print(f"RMSE: {best_model['RMSE']:.4f}")
        print(f"R²: {best_model['R²']:.4f}")
        print(f"MAE: {best_model['MAE']:.4f}")
    
    # Resultados de Classificação
    if all_results['classification']:
        class_df = pd.DataFrame([{k: v for k, v in result.items() 
                               if k not in ['fpr_roc', 'tpr_roc']} 
                               for result in all_results['classification']])
        class_df = class_df.sort_values('AUC', ascending=False)
        
        print(f"\n=== RANKING DOS MODELOS DE CLASSIFICAÇÃO (por AUC) ===")
        print(class_df.to_string(index=False, float_format='%.4f'))
        
        # Salvar resultados
        class_df.to_csv(OUTPUT_DIR / f'classification_results_{target_variable}.csv', index=False)
        
        # Plots de classificação
        plot_roc_curves(all_results['classification'], target_variable)
        plot_confusion_matrices(all_results['classification'], target_variable)
        
        # Plot comparativo de métricas
        plt.figure(figsize=(15, 10))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'AUC', 'FPR', 'FNR']
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 3, i)
            plt.barh(class_df['Model'], class_df[metric])
            plt.xlabel(metric)
            plt.title(f'{metric} por Modelo - {target_variable}')
            plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'classification_metrics_{target_variable}.png', dpi=300)
        plt.show()
        
        print(f"\n=== MELHOR MODELO DE CLASSIFICAÇÃO ===")
        best_class_model = class_df.iloc[0]
        print(f"Modelo: {best_class_model['Model']}")
        print(f"AUC: {best_class_model['AUC']:.4f}")
        print(f"Acurácia: {best_class_model['Accuracy']:.4f}")
        print(f"Precisão: {best_class_model['Precision']:.4f}")
        print(f"Recall: {best_class_model['Recall']:.4f}")
        print(f"Taxa de Falsos Positivos: {best_class_model['FPR']:.4f}")
        print(f"Taxa de Falsos Negativos: {best_class_model['FNR']:.4f}")

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main():
    print("=== ANÁLISE COMPARATIVA DE MODELOS ===")
    print("=== SVM vs MACHINE LEARNING vs DEEP LEARNING ===")
    print("=== ANÁLISE DAS TAXAS DE INADIMPLÊNCIA ===\n")
    
    for target_variable in DEFAULT_RATE_COLUMNS:
        print(f"\n{'='*80}")
        print(f"ANALISANDO: {target_variable}")
        print(f"{'='*80}")
        
        data_dict = load_prepared_data(target_variable)
        if data_dict is None:
            continue
        
        # Treinar todos os modelos
        svm_results = train_svm_models(data_dict)
        ml_results = train_ml_models(data_dict)
        dl_results = train_deep_learning_models(data_dict)
        
        # Consolidar resultados
        all_results = {
            'regression': svm_results['regression'] + ml_results['regression'] + dl_results['regression'],
            'classification': svm_results['classification'] + ml_results['classification'] + dl_results['classification']
        }
        
        # Gerar relatórios comparativos
        generate_comparison_reports(all_results, target_variable)
    
    print(f"\n{'='*80}")
    print("ANÁLISE COMPLETA FINALIZADA")
    print(f"Resultados salvos em: {OUTPUT_DIR}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
