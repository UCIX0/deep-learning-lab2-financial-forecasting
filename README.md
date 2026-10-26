# Pronóstico de Series de Tiempo Financieras con Arquitecturas Recurrentes (RNN, LSTM y GRU)

Este repositorio contiene la implementación y el informe experimental de un pipeline cuantitativo en **PyTorch** diseñado para predecir el precio de cierre ajustado de **Apple Inc. (`AAPL`)**, incorporando covariables de mercado como el **S&P 500 (`SPY`)** y el **Índice de Volatilidad CBOE (`^VIX`)**.

El proyecto resuelve el problema de la no-estacionariedad de los precios operando sobre el espacio de **retornos logarítmicos continuos** $r_t = \ln(P_t / P_{t-1})$ e indicadores técnicos adimensionales, garantizando una evaluación completamente libre de fugas de información (*zero data leakage*).

---

## 🛠️ Tecnologías y Requisitos

- **Python:** `3.11+`
- **Framework Principal:** PyTorch (`2.6.0+` con aceleración CUDA)
- **Análisis de Datos:** `pandas`, `numpy`, `scikit-learn`, `scipy`, `statsmodels`
- **Ingesta Financiera:** `yfinance`
- **Visualización:** `matplotlib`, `seaborn`, `mplfinance`

---

## 📁 Estructura del Proyecto

```text
.
├── checkpoints/          # [Ignorado en Git] Checkpoints binarios (.pt) guardados por EarlyStopping
├── data/
├── figures/
├── proyecto3Teorico.ipynb# Jupyter Notebook principal con la suite experimental ejecutable
├── proyecto3Teorico.md   # Informe técnico y teórico detallado
├── requirements.txt      # Dependencias exactas del entorno de desarrollo
├── .gitignore            # Reglas de exclusión de archivos pesados y cachés
└── README.md             # Documentación del proyecto

```

---

## 📊 Metodología y Salvaguardas Anti-Leakage

1. **Invarianza de Escala y Estacionariedad:** Modelado sobre retornos logarítmicos e indicadores técnicos en porcentaje/ratios (`MA_Spread`, `BB_Bandwidth`, `RSI_Scaled`, etc.). Reconstrucción exactiva en USD vía la identidad exponencial $\hat{P}_{t+k} = P_t \exp\left(\sum_{j=1}^k \hat{r}_{t+j}\right)$.
2. **Escalamiento Protegido:** Normalización (`StandardScaler`) ajustada exclusivamente (`.fit()`) sobre el conjunto de Entrenamiento.
3. **Buffer de Contexto:** Concatenación del historial previo en Validación y Prueba para evaluar el 100% de las fechas objetivo sin pérdida de muestras por ventana retrospectiva ($L$).
4. **Partición Cronológica Estricta (70% Train / 15% Val / 15% Test):**
* **Train (2019 - 2022):** 846 sesiones.
* **Val (2022 - 2023):** 181 sesiones (usado estrictamente para *EarlyStopping* y selección del modelo campeón).
* **Test (2023):** 182 sesiones (evaluación fuera de muestra).



---

## 📈 Resumen de Resultados Principales

### 🏆 Selección de Modelos ($H=1$ día)

* **Campeón Global en Validación:** `LSTM_Multiv_L60_H1` ($F=10$ canales, $L=60$ días) alcanzó el menor error absoluto medio en validación (**Val MAE = $2.2301 USD**, **Val MAPE = 1.5259%**), superando al *Naïve Persistence Baseline* ($2.2495 USD$) y a baselines clásicos como `ARIMA(1,1,1)` ($2.3042 USD$).
* **Efecto de Cambio de Régimen en Prueba:** Durante la tendencia alcista con baja volatilidad de 2023, la arquitectura univariada `GRU_Univ_L120_H1` obtuvo el menor error fuera de muestra (**Test MAE = $1.5654 USD**).

### 🚀 Pronóstico Multi-Paso Directo ($H=5$ días)

En la proyección vectorial a 5 días vista ($t+1$ a $t+5$), el modelo `GRU_Univ_L60_H5` demostró la mayor ventaja predictiva acumulada, reduciendo el error en un **-2.16% frente al baseline Naïve en el paso $t+5$** ($3.9268 USD$ vs. $4.0135 USD$).

### 🔄 Validación por Origen Móvil (Expanding Walk-Forward)

Evaluado a lo largo de 4 pliegues cronológicos (360 sesiones operativas), el modelo `LSTM_Multiv` demostró la mayor estabilidad inter-régimen, logrando un **MAE consolidado de $1.9126 ± $0.5372 USD** y el menor RMSE de todo el estudio ($2.5008 USD$).

---

## ⚡ Instalación y Uso

1. **Clonar el repositorio:**
```bash
git clone [https://github.com/tu-usuario/nombre-del-repositorio.git](https://github.com/tu-usuario/nombre-del-repositorio.git)
cd nombre-del-repositorio

```


2. **Crear y activar un entorno virtual:**
```bash
python -m venv .venv
source .venv/bin/activate  # En Linux/macOS
# .venv\Scripts\activate   # En Windows

```


3. **Instalar dependencias:**
```bash
pip install -r requirements.txt

```


4. **Ejecutar el pipeline:**
Abre y ejecuta las celdas de `proyecto3Teorico.ipynb` en tu entorno de Jupyter. El script descargará automáticamente los datos si no existen localmente, creará las carpetas necesarias, entrenará la matriz de modelos y exportará los artefactos/gráficas.

---

## 🔍 Inferencia en Producción

El repositorio incluye la función utilitaria `predict_next_days_from_checkpoint()`, la cual permite cargar el modelo en disco (`.pt`) y predecir los próximos días bursátiles utilizando la historia reciente del activo:

```python
from pathlib import Path
from proyecto3Teorico import predict_next_days_from_checkpoint

# Inferencia desde el último checkpoint
prediccion = predict_next_days_from_checkpoint(
    checkpoint_path=Path("checkpoints/LSTM_Multiv_L60_H1.pt"),
    df_history=df_feat,
    scaler_X=data_multiv_L60_H1["scaler_X"],
    scaler_y=data_multiv_L60_H1["scaler_y"]
)

```

---

## 📄 Licencia

Este proyecto está disponible bajo la licencia MIT.

