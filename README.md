
# Projeto: Modelagem de Risco de Inadimplência e Decisão de Política de Risco (PD-based)
Em problemas de crédito, o desafio não se limita à previsão de inadimplência, mas à tomada de decisão sob risco.

A aplicação direta de probabilidades estimadas sem uma política clara de decisão pode levar a: perda de retorno potencial, exposição excessiva ao risco e políticas instáveis ao longo do tempo.

Este projeto tem como objetivo desenvolver, avaliar e comparar modelos de classificação binária aplicados a um problema de **risco de crédito**, com foco especial na **definição do threshold de decisão** orientado por métricas econômicas e de negócio.

O trabalho vai além da comparação tradicional de métricas estatísticas, explorando o impacto direto das decisões de corte sobre inadimplência, aprovação de crédito e retorno esperado.

## 📸 Destaques Visuais

Os gráficos abaixo representam os principais achados do projeto:

### Impacto da variação do threshold sobre o retorno esperado
![Impacto do threshold no retorno esperado](https://github.com/jooarantes/credit-default-prediction/blob/main/reports/graphs/roe-vs-threshold.png)

### Relação entre faixa de risco, inadimplência e volume aprovado
![Distribuição de PDs por faixa de risco e Bad Rate](https://github.com/jooarantes/credit-default-prediction/blob/main/reports/graphs/pd-por-faixa-de-risco.png)

### Sensibilidade das métricas estatísticas e econômicas ao Threshold
![Impacto do Threshold nas métricas](https://github.com/jooarantes/credit-default-prediction/blob/main/reports/graphs/sensibilidade-threshold.png)



##  🎯 Objetivos do Projeto

- Compreender os fatores associados ao aumento do risco de taxa de default;
- Desenvolver modelos preditivos aplicados a risco de crédito;
- Comparar modelos interpretáveis e modelos de maior flexibilidade;
- Avaliar modelos sob métricas estatísticas e métricas econômicas;
- Estudar o impacto do threshold na política de concessão de crédito;
- Propor uma política final de decisão alinhada a risco, retorno e estabilidade;
## 🧠 Principais Aprendizados

- Construção e Validação de modelos WOE-Based;
- Elaboração de métricas personalizadas alinhadas com o contexto do Negócio;
- Estratégias de controle de Overfitting como EarlyStopping e Poda;
- Calibração das probabilidades;
- Análise de Trade-off com Thresholds;
- Interpretação dos SHAP values;
- Elaboração de Política de Decisão de Risco (PD-based). 


## 📂 Conteúdo do Repositório

O repositório está organizado para facilitar a navegação entre análises, resultados e implementação, permitindo que diferentes perfis de leitores explorem o projeto conforme seu interesse.

### 📓 Notebooks Analíticos

Os notebooks representam o fluxo principal do projeto e devem ser lidos de forma sequencial:

- **[01_eda.ipynb](https://github.com/jooarantes/credit-default-prediction/blob/main/notebooks/01_eda.ipynb)**  
  Análise exploratória dos dados, distribuição do target e análise inferencial de risco.

- **[02_model_interpretaveis.ipynb](https://github.com/jooarantes/credit-default-prediction/blob/main/notebooks/02_model_interpret%C3%A1veis.ipynb)**  
  Modelos interpretáveis, análise de coeficientes, estabilidade e coerência econômica.

- **[03_model_performance_gbm.ipynb](https://github.com/jooarantes/credit-default-prediction/blob/main/notebooks/03_model_performance_gbm.ipynb)**  
  Modelos de maior flexibilidade (GBM) avaliados sob métricas estatísticas como diagnóstico.

- **[04_faixa_de_risco_e_decisao.ipynb](https://github.com/jooarantes/credit-default-prediction/blob/main/notebooks/04_faixa_de_risco_e_decisao.ipynb)**  
  Definição de faixas de risco, estudo do threshold e política final de decisão.

---

### 📊 Reports e Resultados

A pasta `reports/` contém os principais artefatos gerados ao longo do projeto, permitindo acesso direto a resultados sem a necessidade de executar os notebooks:

- **[Gráficos](https://github.com/jooarantes/credit-default-prediction/tree/main/reports/graphs)** utilizados na análise final;
- **[Tabelas](https://github.com/jooarantes/credit-default-prediction/tree/main/reports/tables)** resumo;
- **[Figuras](https://github.com/jooarantes/credit-default-prediction/tree/main/reports/figures)** consolidadas para comunicação dos resultados.

---

### 🗂️ Dados

A pasta `data/` está organizada em:

- **`raw/`**: dados brutos, conforme disponibilizados na origem  
- **`processed/`**: dados tratados e artefatos intermediários gerados ao longo do pipeline analítico  

---

### 🧠 Código Fonte (`src/`)

A pasta `src/` contém a implementação modular utilizada nos notebooks:

- **`evaluation/`**  
  Implementação das métricas de avaliação, incluindo métricas econômicas utilizadas na definição da política de decisão.

- **`utils/`**  
  Funções auxiliares reutilizáveis ao longo do projeto (pré-processamento, visualizações e helpers).

## ▶️ Como Reproduzir as Análises

As etapas abaixo descrevem como reproduzir integralmente as análises deste projeto em um ambiente local.

### 1. Clonar o repositório
Clone o repositório para sua máquina local:

```bash
git clone https://github.com/jooarantes/credit-default-prediction.git
cd credit-default-prediction
```
### 2. Criar e Ativar o ambiente virtual
```bash
conda env create -f environment.yml
conda activate credit-default-prediction
```
### 3. Executar os notebooks
Os notebooks **devem ser executados sequencialmente**, respeitando a ordem abaixo, pois cada etapa gera artefatos utilizados nas etapas seguintes:
- `01_eda.ipynb`
- `02_model_interpretaveis.ipynb`
- `03_model_performance_gbm.ipynb`
- `04_faixa_de_risco_e_decisao.ipynb`

Durante a execução, são gerados dados processados, métricas e artefatos intermediários utilizados nas análises finais.

## 📖 Contexto do Problema de Negócio

Uma fintech de crédito iniciou sua operação de concessão de empréstimo pessoal e acompanhou a performance de pagamento dos seus clientes durante 1 ano. O grande problema foi a alta taxa de inadimplência observada, cerca de 30%. A partir disso, um estudo foi conduzido para identificar os fatores de risco que estão mais associados à inadimplência de forma a conseguir aprovar mais contratos com a menor taxa de inadimplência possível.
## ⚙️ Metodologia

A metodologia segue um fluxo analítico estruturado:

- Análise exploratória com foco na distribuição do risco;
- Construção de modelos interpretáveis como baseline;
- Avaliação de modelos mais flexíveis como benchmark de performance;
- Uso de métricas estatísticas como ferramentas de diagnóstico;
- Consolidação da decisão via definição de faixas de risco e threshold;

Cada etapa é documentada em um notebook específico, refletindo o encadeamento lógico do projeto.
## 📐 Métricas de Avaliação

O projeto utiliza dois grupos de métricas:

**Métricas Estatísticas**

- ROC AUC
- Precision-Recall
- KS

Utilizadas principalmente para **diagnóstico e comparação técnica.**

**Métricas Econômicas**

- Prejuízo esperado (Expected Loss)
- Matriz de Payoff
- ROE

A decisão final **não é baseada exclusivamente em métricas estatísticas.**
## 📊 Principais Resultados

- O ranking econômico preliminar foi: **Regressão Logística > LGBM > XGBM**;
- O modelo de regressão logística apresentou um **ganho percentual de 39%** em relação ao baseline econômico (approve all transactions policy);
- As distribuições de PD apresentam concentração elevada em faixas entre 8 a 15%;
- O threshold ótimo - que retorna o maior ROE - foi de 0.08. Porém, resultou em uma política **extremamente agressiva** para a base, aprovando apenas 2 clientes;
- **Um threshold empiríco de 0.10 foi adotado** (com base na observação visual do gráfico ROE vs Threshold) como tentativa de **flexibilização da política de risco**;
- A política final adotada utiliza o modelo de regressão logística com probabilidades calibradas (Platt Scaling) e cut único (faixa A), aprovando crédito para clientes com perfil de PD <= 10%.
## ⚠️ Limitações e Considerações

- O modelo WOE-Based criado não prosseguiu para as análises econômicas por haver a necessidade de descartar duas variáveis que são impressindíveis para o contexto do negócio por não apresentarem uma relação monotônica entre as bins e o WOE. Esse modelo foi mantido apenas por caráter de aprendizado e exploratório, não sendo considerado nas decisões futuras deste projeto;
- A alocação de capital foi definida a partir de uma aproximação do valor médio de empréstimo multiplicado pelo fator de capital;
- Sobre a política de decisão final, é necessário ponderar a flexibilidade e alinhar essa decisão com a área de negócios. Segundo o modelo, muitos clientes bons estão sendo descartados (456), isso faz com que haja um aumento na fricção interna e gere um possível churn de clientes bons. Uma boa solução para isso é adicionar novas features para tentar capturar melhor esses bons pagadores;

## 🛠️ Tecnologias Utilizadas

- Python
- Pandas, Numpy
- Scikit-Learn
- XGBoost / LightGBM
- Matplotlib e Seaborn
- Jupyter Notebook
- SHAP
  
## 👤 Autores
**João Arantes**

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/joao-arantes-ds/)

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://jooarantes.github.io/)

## 🔗 Conteúdos Relacionados

- Artigo no Medium: 


## Licença

[MIT](https://github.com/jooarantes/credit-default-prediction/blob/main/LICENSE)

