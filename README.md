# Final Project

\# 📈 Previsão da Inadimplência de Cartões de Crédito no Brasil



> Trabalho de Conclusão de Curso (TCC) do MBA em Data Science \& Analytics (USP/ESALQ): Análise comparativa de modelos de Machine Learning e Deep Learning para previsão de inadimplência, avaliando performance em diferentes regimes econômicos.



---



\## 📊 \*\*Visão Geral\*\*



Este projeto foi desenvolvido como Trabalho de Conclusão de Curso (TCC) do MBA em Data Science \& Analytics da USP/ESALQ, analisando a previsão de inadimplência total de cartões de crédito no Brasil utilizando variáveis macroeconômicas mensais entre janeiro de 2015 e julho de 2025.



\### 🎯 \*\*Objetivos do Projeto\*\*



\- Comparar performance de 5 modelos supervisionados: \*\*Linear Regression, SVR, XGBoost, MLP e LSTM\*\*

\- Avaliar impacto de choques estruturais (pandemia 2019-2021) no desempenho dos modelos

\- Identificar qual arquitetura é mais adequada para diferentes regimes econômicos

\- Fornecer subsídios práticos para seleção de técnicas em gestão de risco de crédito



\### 🏆 \*\*Principais Contribuições\*\*



1\. \*\*Análise Dual de Cenários\*\*: Comparação entre série completa (FULL) vs período estável (EXCL)

2\. \*\*Descoberta Metodológica\*\*: LSTM superior em alta volatilidade, SVR em estabilidade

3\. \*\*Aplicação Prática\*\*: Orientação para seleção de modelos conforme contexto econômico

4\. \*\*Rigor Acadêmico\*\*: Metodologia completa com validação temporal e múltiplas métricas



---



\## 🚀 \*\*Principais Resultados\*\*



\### ✅ \*\*Cenário FULL (Série Completa 2015-2025)\*\*



Inclui período de instabilidade fiscal 2019-2021.



| Modelo | MSE | R² | MAPE (%) | DA (%) | Destaque |

|--------|-----|-----|----------|---------|----------|

| \*\*LSTM\*\* ⭐ | \*\*0.0179\*\* | \*\*0.7050\*\* | \*\*1.83\*\* | 40.00 | Melhor para alta volatilidade |

| Linear Regression | 0.0210 | 0.6542 | 2.05 | 44.00 | Baseline competitivo |

| XGBoost | 0.0228 | 0.6242 | 2.13 | 44.00 | Bom equilíbrio |

| SVR | 0.0572 | 0.0594 | 3.10 | 56.00 | Maior acerto direcional |

| MLP | 14.9447 | -244.79 | 56.59 | 48.00 | Overfitting severo |



> \*\*💡 Insight Chave:\*\* LSTM captura dependências temporais complexas em ambientes de alta volatilidade, explicando 70% da variância da inadimplência.



\### ✅ \*\*Cenário EXCL (Excluindo 2019-2021)\*\*



Remove período de instabilidade para analisar performance em ambiente estável.



| Modelo | MSE | R² | MAPE (%) | DA (%) | Destaque |

|--------|-----|-----|----------|---------|----------|

| \*\*SVR\*\* ⭐ | \*\*0.0295\*\* | \*\*0.3559\*\* | \*\*2.26\*\* | 35.29 | Melhor para estabilidade |

| Linear Regression | 0.0370 | 0.1924 | 2.57 | 47.06 | Consistente |

| XGBoost | 0.1422 | -2.1029 | 5.40 | 41.18 | Perde generalização |

| LSTM | 0.2194 | -3.7858 | 7.50 | 47.06 | Requer mais dados |

| MLP | 0.9264 | -19.2102 | 12.36 | 41.18 | Inadequado |



> \*\*💡 Descoberta:\*\* SVR supera LSTM em ambiente estável, revelando que padrões não-lineares suaves são melhor capturados por kernels RBF sem necessidade de memória temporal complexa.



---



\## 💡 \*\*Principais Descobertas\*\*



\### 🎯 Descoberta 1: Contexto Econômico > Complexidade do Modelo



\*\*No cenário FULL (alta volatilidade):\*\*

\- \*\*LSTM:\*\* R² = 0.70, MAPE = 1.83%

\- Capacidade de capturar dependências temporais durante choques macroeconômicos

\- Volatilidade extrema da pandemia exige memória de longo prazo



\*\*No cenário EXCL (estabilidade):\*\*

\- \*\*SVR:\*\* R² = 0.36, MAPE = 2.26%

\- Padrões não-lineares mais suaves favorecem kernel RBF

\- Modelos mais simples suficientes sem choques estruturais



\*\*Implicação Prática:\*\* A escolha do modelo deve considerar o regime econômico vigente, não apenas métricas de treino.



\### 🎯 Descoberta 2: Trade-off entre Complexidade e Volume de Dados



\- \*\*MLP:\*\* Performance ruim em ambos cenários

\- Séries temporais curtas (126 meses) insuficientes para deep learning complexo

\- LSTM funciona por ter arquitetura especializada em sequências

\- \*\*Lição:\*\* Deep learning requer > 200-300 observações para generalizar bem



\### 🎯 Descoberta 3: Baseline Linear Surpreendentemente Competitivo



\- \*\*Linear Regression:\*\* R² = 0.65 (FULL), 0.19 (EXCL)

\- 65% da inadimplência explicada por relações aproximadamente lineares

\- Modelos simples podem ser suficientes para interpretabilidade

\- \*\*Lição:\*\* Sempre compare com baseline antes de usar modelos complexos



\### 🎯 Descoberta 4: Acerto Direcional ≠ Magnitude do Erro



\- \*\*SVR:\*\* Maior DA (56%) no FULL, mas maior MAPE

\- Prevê corretamente direção do movimento (alta/baixa)

\- Mas erra na magnitude exata

\- \*\*Aplicação:\*\* Ideal para decisões estratégicas (tendência futura)



\### 🎯 Descoberta 5: Períodos de Crise Contêm Informação Valiosa



\- Exclusão 2019-2021 resulta em piora geral de performance

\- Choques estruturais revelam relações funcionais importantes

\- Modelos treinados com crises são mais robustos

\- \*\*Lição:\*\* Não exclua outliers antes de avaliar seu valor informacional



---



\## 📊 \*\*Dados e Variáveis\*\*



\### Fonte dos Dados



\- \*\*Banco Central do Brasil\*\* - Sistema Gerenciador de Séries Temporais (SGS)

\- \*\*IBGE\*\* - Índice Nacional de Preços ao Consumidor Amplo (IPCA)

\- \*\*Período:\*\* Janeiro/2015 a Julho/2025 (126 observações mensais)



\### Variáveis Preditoras



| Variável | Descrição | Fonte |

|----------|-----------|-------|

| \*\*Taxa Selic\*\* | Taxa básica de juros da economia brasileira | BCB |

| \*\*IBC-Br Dessazonalizado\*\* | Índice de Atividade Econômica (proxy do PIB) | BCB |

| \*\*IPCA\*\* | Inflação mensal oficial | IBGE |

| \*\*Comprometimento de Renda\*\* | % da renda comprometida com dívidas | BCB |

| \*\*Endividamento das Famílias\*\* | Nível total de endividamento em relação à renda | BCB |



\### Variável Target



\- \*\*Inadimplência Total de Cartão de Crédito\*\* (% do saldo total inadimplente)

\- Fonte: Banco Central do Brasil

\- Série oficial mensal



\### Feature Engineering



\*\*Features Criadas:\*\*

```python

\# Temporal

\- lag\_1\_target: Valor anterior da inadimplência

&nbsp; (única feature derivada mantida após análise de colinearidade)



\# Variáveis originais

\- IBC-Br dessazonalizado (melhor performance vs versão original)

\- Remoção de lags de variáveis independentes (evitar multicolinearidade)

```



\*\*Decisões de Engenharia:\*\*

\- Testados múltiplos lags → aumentaram colinearidade e pioraram R²

\- Testado IBC-Br original vs dessazonalizado vs ambos → dessazonalizado venceu

\- Lag 1 da target altamente informativo (consistente com literatura de persistência)



---



\## 🔬 \*\*Metodologia\*\*



\### Cenários de Análise



\*\*CENÁRIO FULL (Completo)\*\*

\- \*\*Período:\*\* Jan/2015 a Jul/2025

\- \*\*N:\*\* 126 observações

\- \*\*Características:\*\* Inclui instabilidade fiscal 2019-2021

\- \*\*Objetivo:\*\* Avaliar capacidade de lidar com volatilidade extrema



\*\*CENÁRIO EXCL (Exclusão)\*\*

\- \*\*Período:\*\* Jan/2015 a Dez/2018 + Jan/2022 a Jul/2025

\- \*\*N:\*\* 90 observações

\- \*\*Características:\*\* Remove choques da pandemia

\- \*\*Objetivo:\*\* Avaliar performance em ambiente estável



\*\*Justificativa da Exclusão 2019-2021:\*\*

\- Medidas fiscais extraordinárias durante pandemia

\- Postergação de despesas obrigatórias

\- Deterioração acentuada de indicadores fiscais

\- Ruptura estrutural documentada (TCU, 2021; FGV IBRE, 2022)



\### Pré-processamento



```python

\# Padronização

\- StandardScaler (média 0, desvio padrão 1)

\- Necessário para SVR, MLP e LSTM



\# Divisão Temporal

\- Train: 80% das observações

\- Test: 20% das observações

\- Respeita ordem cronológica (evita data leakage)



\# Dados

\- Nenhum valor faltante identificado

\- Séries completas no período analisado

```



\### Modelos Implementados



\#### 1. \*\*Linear Regression (Baseline)\*\*



```python

from sklearn.linear\_model import LinearRegression



\# Modelo paramétrico simples (OLS)

model = LinearRegression()

```



\*\*Por que usar:\*\*

\- Estabelece baseline para comparação

\- Avalia presença de padrões lineares

\- Máxima interpretabilidade



\#### 2. \*\*Support Vector Regression (SVR)\*\*



```python

from sklearn.svm import SVR



model = SVR(

&nbsp;   kernel='rbf',

&nbsp;   C=100,

&nbsp;   gamma='scale',

&nbsp;   epsilon=0.1

)

```



\*\*Características:\*\*

\- Captura relações não-lineares via kernel RBF

\- Eficiente em datasets pequenos-médios

\- Robusto a outliers



\#### 3. \*\*XGBoost\*\*



```python

import xgboost as xgb



model = xgb.XGBRegressor(

&nbsp;   n\_estimators=1000,

&nbsp;   max\_depth=7,

&nbsp;   learning\_rate=0.01,

&nbsp;   subsample=0.8,

&nbsp;   colsample\_bytree=0.8,

&nbsp;   objective='reg:squarederror',

&nbsp;   early\_stopping\_rounds=50

)

```



\*\*Características:\*\*

\- State-of-the-art para dados tabulares

\- Regularização built-in (menos overfitting)

\- Feature importance nativa



\#### 4. \*\*Multilayer Perceptron (MLP)\*\*



```python

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout



model = Sequential(\[

&nbsp;   Dense(64, activation='relu'),

&nbsp;   Dropout(0.2),

&nbsp;   Dense(32, activation='relu'),

&nbsp;   Dropout(0.2),

&nbsp;   Dense(16, activation='relu'),

&nbsp;   Dense(1)

])

```



\*\*Características:\*\*

\- Rede neural feedforward

\- Captura relações não-lineares complexas

\- Requer mais dados para treinar bem



\#### 5. \*\*Long Short-Term Memory (LSTM)\*\*



```python

from tensorflow.keras.layers import LSTM



model = Sequential(\[

&nbsp;   LSTM(128, return\_sequences=True, input\_shape=(lookback, n\_features)),

&nbsp;   Dropout(0.2),

&nbsp;   LSTM(64, return\_sequences=False),

&nbsp;   Dropout(0.2),

&nbsp;   Dense(32, activation='relu'),

&nbsp;   Dense(1)

])



\# Otimização

model.compile(

&nbsp;   optimizer='adam',

&nbsp;   loss='mse',

&nbsp;   metrics=\['mae']

)



\# Early Stopping

early\_stop = EarlyStopping(patience=20, restore\_best\_weights=True)

```



\*\*Características:\*\*

\- Especializada em séries temporais

\- Captura dependências de longo prazo

\- Memória de curto e longo prazo



\*\*Técnicas Aplicadas:\*\*

\- Early Stopping (patience=20)

\- Dropout para regularização

\- Validation split interno



\### Métricas de Avaliação



\*\*MSE (Mean Squared Error):\*\*

```

MSE = (1/n) × Σ(y\_i - ŷ\_i)²

```

\- Penaliza erros grandes

\- Sensível a outliers



\*\*R² (Coeficiente de Determinação):\*\*

```

R² = 1 - Σ(y\_i - ŷ\_i)² / Σ(y\_i - ȳ)²

```

\- Proporção da variância explicada

\- Valores < 0 indicam performance pior que média



\*\*MAPE (Mean Absolute Percentage Error):\*\*

```

MAPE = (100/n) × Σ|((y\_i - ŷ\_i) / y\_i)|

```

\- Erro percentual médio

\- Interpretável em termos relativos



\*\*DA (Directional Accuracy):\*\*

```

DA = (1/(n-1)) × Σ 𝟙\[(y\_i - y\_{i-1})(ŷ\_i - ŷ\_{i-1}) > 0]

```

\- Acerto da direção do movimento

\- Crucial para decisões estratégicas



---



\## 📈 \*\*Análise Comparativa Detalhada\*\*



\### Desempenho por Cenário



\#### \*\*CENÁRIO FULL: Alta Volatilidade Favorece LSTM\*\*



\*\*Ranking de Performance:\*\*

1\. 🥇 \*\*LSTM\*\*: MSE=0.0179, R²=0.70, MAPE=1.83%

2\. 🥈 \*\*Linear\*\*: MSE=0.0210, R²=0.65, MAPE=2.05%

3\. 🥉 \*\*XGBoost\*\*: MSE=0.0228, R²=0.62, MAPE=2.13%

4\. \*\*SVR\*\*: MSE=0.0572, R²=0.06, MAPE=3.10%

5\. \*\*MLP\*\*: MSE=14.94, R²=-244.79, MAPE=56.59%



\*\*Análise:\*\*

\- LSTM explica 70% da variância

\- Erro percentual de apenas 1.83%

\- Superior em capturar choques da pandemia

\- Linear Regression surpreendentemente competitiva (R²=0.65)



\#### \*\*CENÁRIO EXCL: Estabilidade Favorece SVR\*\*



\*\*Ranking de Performance:\*\*

1\. 🥇 \*\*SVR\*\*: MSE=0.0295, R²=0.36, MAPE=2.26%

2\. 🥈 \*\*Linear\*\*: MSE=0.0370, R²=0.19, MAPE=2.57%

3\. 🥉 \*\*XGBoost\*\*: MSE=0.1422, R²=-2.10, MAPE=5.40%

4\. \*\*LSTM\*\*: MSE=0.2194, R²=-3.79, MAPE=7.50%

5\. \*\*MLP\*\*: MSE=0.9264, R²=-19.21, MAPE=12.36%



\*\*Análise:\*\*

\- SVR único com R² positivo

\- LSTM perde performance sem volatilidade

\- XGBoost sofre com redução de amostra

\- Padrões não-lineares suaves favorecem kernel RBF



\### Comparação Visual



\*\*Mudança de Performance (FULL → EXCL):\*\*



| Modelo | Δ MSE | Δ R² | Δ MAPE | Interpretação |

|--------|-------|------|--------|---------------|

| LSTM | +1125% | -530% | +310% | Grande degradação |

| XGBoost | +523% | -430% | +154% | Sensível a amostra |

| Linear | +76% | -71% | +25% | Mais robusto |

| SVR | -48% | +500% | -27% | \*\*Melhora!\*\* |

| MLP | -94% | +92% | -78% | Melhora relativa |



\*\*Conclusão:\*\* SVR é o único modelo que \*\*melhora\*\* com a remoção dos choques, enquanto modelos complexos degradam significativamente.



---



\## 🎯 \*\*Recomendações Práticas\*\*



\### Quando Usar Cada Modelo



\#### \*\*LSTM (Long Short-Term Memory)\*\*



\*\*✅ Use quando:\*\*

\- Séries com alta volatilidade e choques estruturais

\- Disponibilidade de dados históricos longos (>200 observações idealmente)

\- Recursos computacionais suficientes (GPU recomendada)

\- Necessidade de capturar dependências de longo prazo

\- Contexto: crises econômicas, mudanças estruturais



\*\*❌ Evite quando:\*\*

\- Séries curtas (<100 observações)

\- Ambiente econômico estável

\- Necessidade de máxima interpretabilidade

\- Restrições computacionais



\*\*Exemplo:\*\* Previsão durante crises (COVID-19, crise 2008)



---



\#### \*\*SVR (Support Vector Regression)\*\*



\*\*✅ Use quando:\*\*

\- Ambiente econômico estável

\- Padrões não-lineares suaves

\- Datasets pequenos-médios (50-500 observações)

\- Necessidade de robustez a outliers

\- Recursos computacionais limitados



\*\*❌ Evite quando:\*\*

\- Séries muito longas (>1000 obs) - custo computacional alto

\- Padrões predominantemente lineares

\- Necessidade de interpretabilidade total



\*\*Exemplo:\*\* Previsão de curto prazo em períodos normais



---



\#### \*\*XGBoost\*\*



\*\*✅ Use quando:\*\*

\- Bom equilíbrio entre complexidade e performance

\- Necessidade de interpretabilidade (feature importance)

\- Produção com baixa latência

\- Interações não-lineares entre variáveis

\- Prioridade para robustez



\*\*❌ Evite quando:\*\*

\- Padrões lineares são suficientes

\- Séries muito curtas (<50 observações)

\- Dependências temporais de longo prazo são cruciais



\*\*Exemplo:\*\* Sistemas de decisão em tempo real



---



\#### \*\*Linear Regression\*\*



\*\*✅ Use quando:\*\*

\- Baseline rápido necessário

\- Relações predominantemente lineares

\- Máxima interpretabilidade necessária

\- Recursos computacionais muito limitados

\- Compliance e auditoria (explicabilidade total)



\*\*❌ Evite quando:\*\*

\- Padrões claramente não-lineares

\- Interações complexas entre variáveis

\- Performance é prioridade absoluta



\*\*Exemplo:\*\* Relatórios regulatórios, explicações para executivos



---



\### Aplicação em Gestão de Risco de Crédito



\#### \*\*Cenário 1: Períodos Normais (Estabilidade)\*\*

```

Modelo recomendado: SVR ou Linear Regression

Justificativa: Estabilidade + interpretabilidade

Frequência de retreino: Trimestral

Threshold de alerta: MAPE > 3%

```



\#### \*\*Cenário 2: Períodos de Crise (Alta Volatilidade)\*\*

```

Modelo recomendado: LSTM

Justificativa: Captura choques e dependências complexas

Frequência de retreino: Mensal

Threshold de alerta: MAPE > 2.5%

```



\#### \*\*Cenário 3: Produção (Real-time)\*\*

```

Modelo recomendado: XGBoost

Justificativa: Equilíbrio performance/latência

Frequência de retreino: Bimestral

Threshold de alerta: R² < 0.5

```



\#### \*\*Cenário 4: Relatórios Regulatórios\*\*

```

Modelo recomendado: Linear Regression

Justificativa: Máxima transparência

Frequência: Anual

Documentação: Completa com coeficientes interpretáveis

```



---



\## 🛠️ \*\*Tecnologias Utilizadas\*\*



\### Core Libraries

```python

pandas>=1.5.0          # Manipulação de dados

numpy>=1.23.0          # Computação numérica

scikit-learn>=1.0.0    # Machine Learning tradicional

xgboost>=1.7.0         # Gradient Boosting

tensorflow>=2.10.0     # Deep Learning

keras>=2.10.0          # Interface DL

```



\### Analysis \& Visualization

```python

matplotlib>=3.6.0      # Visualizações

seaborn>=0.12.0        # Gráficos estatísticos

statsmodels>=0.13.0    # Análise de séries temporais

```



---



\## 📁 \*\*Estrutura do Projeto\*\*



```

inadimplencia-cartoes-ml/

│

├── data/

│   ├── raw/                    # Dados do BCB e IBGE

│   └── processed/              # Dados processados

│

├── notebooks/

│   ├── 01\_coleta\_dados.ipynb

│   ├── 02\_eda\_series\_temporais.ipynb

│   ├── 03\_feature\_engineering.ipynb

│   ├── 04\_baseline\_models.ipynb

│   ├── 05\_ml\_models.ipynb

│   ├── 06\_deep\_learning.ipynb

│   └── 07\_comparacao\_final.ipynb

│

├── src/

│   ├── data\_preprocessing.py

│   ├── feature\_engineering.py

│   ├── models.py

│   └── evaluation.py

│

├── models/

│   ├── lstm\_full.h5

│   ├── svr\_excl.pkl

│   └── model\_comparison.csv

│

├── reports/

│   ├── TCC\_Final.pdf

│   └── figures/

│

└── README.md

```



---



\## 🎯 \*\*Como Usar\*\*



\### 1. Instalação



```bash

git clone https://github.com/JorgeFumagalli/Final-Project.git

cd Final-Project



python -m venv venv

source venv/bin/activate



pip install -r requirements.txt

```



\### 2. Coleta de Dados



```python

\# Os dados podem ser obtidos do SGS do Banco Central

\# Links disponíveis no notebook 01\_coleta\_dados.ipynb



\# Ou use dados já processados em data/processed/

```



\### 3. Reproduzir Análises



```bash

\# Execute notebooks na ordem

jupyter notebook notebooks/



\# Ou rode pipeline completo

python src/run\_pipeline.py --scenario full

python src/run\_pipeline.py --scenario excl

```



\### 4. Fazer Previsões



```python

from tensorflow.keras.models import load\_model

import joblib



\# Carregue modelo apropriado

lstm\_model = load\_model('models/lstm\_full.h5')  # Para alta volatilidade

svr\_model = joblib.load('models/svr\_excl.pkl')  # Para estabilidade



\# Prepare dados (mesmo preprocessing do treino)

import pandas as pd

new\_data = pd.DataFrame({

&nbsp;   'Selic': \[10.5],

&nbsp;   'IBC-Br': \[135.2],

&nbsp;   'IPCA': \[0.45],

&nbsp;   'Comprometimento': \[28.5],

&nbsp;   'Endividamento': \[50.2],

&nbsp;   'lag\_1\_target': \[5.2]  # Inadimplência do mês anterior

})



\# Padronize

from sklearn.preprocessing import StandardScaler

scaler = joblib.load('models/scaler.pkl')

X\_scaled = scaler.transform(new\_data)



\# Previsão

pred\_lstm = lstm\_model.predict(X\_scaled.reshape(1, 1, -1))

pred\_svr = svr\_model.predict(X\_scaled)



print(f"Previsão LSTM: {pred\_lstm\[0]\[0]:.2f}%")

print(f"Previsão SVR: {pred\_svr\[0]:.2f}%")

```



---



\## 🔮 \*\*Trabalhos Futuros\*\*



\### Melhorias Planejadas

\- \[ ] Incorporar variáveis microeconômicas (renda per capita, desemprego por região)

\- \[ ] Testar modelos híbridos (ensemble ML + DL)

\- \[ ] Implementar detecção automática de quebras estruturais

\- \[ ] Sistema de seleção automática de modelo baseado em volatilidade

\- \[ ] Previsão probabilística (intervalos de confiança)



\### Extensões Acadêmicas

\- \[ ] Análise de outras modalidades de crédito (consignado, veículos)

\- \[ ] Comparação internacional (Brasil vs outros emergentes)

\- \[ ] Análise de causalidade (Granger, VAR)

\- \[ ] Incorporar variáveis de política monetária



---



\## 📚 \*\*Referências\*\*



\### Principais Referências do TCC



\*\*Metodologia:\*\*

\- Hochreiter \& Schmidhuber (1997) - Long Short-Term Memory

\- Chen \& Guestrin (2016) - XGBoost: A Scalable Tree Boosting System

\- Cortes \& Vapnik (1995) - Support-vector networks

\- Hyndman \& Athanasopoulos (2018) - Forecasting: Principles and Practice



\*\*Aplicações em Finanças:\*\*

\- Barboza et al. (2017) - Machine learning models and bankruptcy prediction

\- Alonso \& Carbó (2020) - Machine learning in credit risk

\- Wang \& Zhang (2024) - Credit risk prediction using deep learning



\*\*Contexto Brasileiro:\*\*

\- Sicsú et al. (2022) - Crédito, crescimento e estabilidade financeira no Brasil

\- Banco Central do Brasil (2025) - Sistema Gerenciador de Séries Temporais

\- TCU (2021) - Relatório das Contas do Governo da República



\*\*Veja referências completas no TCC (reports/TCC\_Final.pdf)\*\*



---



\## 👤 \*\*Autor\*\*



\*\*Jorge Luiz Fumagalli\*\*



\*\*Formação:\*\*

\- 🎓 MBA em Data Science \& Analytics - USP/ESALQ (2024-2026)

\- 🎓 Engenharia de Produção - UFTM

\- 🎓 Técnico em Informática - ETEC



\*\*Orientador do TCC:\*\*

\- Prof. Me. Diego Pedroso dos Santos



\*\*Contato:\*\*

\- 💼 LinkedIn: \[linkedin.com/in/jorge-fumagalli-bb8975121](https://www.linkedin.com/in/jorge-fumagalli-bb8975121/)

\- 📧 Email: jorgefumagalli@yahoo.com.br

\- 🐙 GitHub: \[github.com/JorgeFumagalli](https://github.com/JorgeFumagalli)



---



\## 📄 \*\*Licença\*\*



Este projeto está sob a licença MIT.



---



\## 🙏 \*\*Agradecimentos\*\*



\- Prof. Diego Pedroso dos Santos pela orientação

\- USP/ESALQ pelo programa de MBA em Data Science \& Analytics

\- Banco Central do Brasil pela disponibilização dos dados

\- Comunidades open-source de Machine Learning e Deep Learning



---



\## 📖 \*\*Citação\*\*



Se este trabalho foi útil para sua pesquisa, considere citar:



```bibtex

@mastersthesis{fumagalli2026,

&nbsp; author  = {Fumagalli, Jorge Luiz},

&nbsp; title   = {Previsão da Inadimplência de Cartões de Crédito no Brasil com Modelos de Aprendizado de Máquina},

&nbsp; school  = {USP/ESALQ - MBA em Data Science \& Analytics},

&nbsp; year    = {2026},

&nbsp; type    = {Trabalho de Conclusão de Curso}

}

```



---



\## ⭐ \*\*Se este projeto foi útil, considere dar uma estrela!\*\*



---



\*\*💡 Dúvidas? Sugestões? Feedbacks são sempre bem-vindos!\*\*



\[Abrir Issue](https://github.com/JorgeFumagalli/Final-Project/issues) | \[Pull Requests](https://github.com/JorgeFumagalli/Final-Project/pulls)

