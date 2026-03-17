# 📊 Estrategia y Funcionamiento del Trading Bot

**Última actualización:** 17 de Marzo, 2026
**Versión:** 2.1.0
**Autor:** Trading Bot Indices Team

---

## 📋 Índice

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Estrategia de Trading](#estrategia-de-trading)
4. [Modelos de Inteligencia Artificial](#modelos-de-inteligencia-artificial)
5. [Indicadores Técnicos](#indicadores-técnicos)
6. [Filtros de Calidad de Señales](#filtros-de-calidad-de-señales)
7. [Gestión de Riesgo](#gestión-de-riesgo)
8. [Reglas Especiales para Índices Sintéticos](#reglas-especiales-para-índices-sintéticos)
9. [Flujo de Operación](#flujo-de-operación)
10. [Parámetros Configurables](#parámetros-configurables)
11. [Optimizaciones y Mejoras Continuas](#optimizaciones-y-mejoras-continuas)

---

## 🎯 Resumen Ejecutivo

Este bot de trading automatizado utiliza **inteligencia artificial avanzada** y **análisis técnico multi-timeframe** para generar señales de trading de alta probabilidad en índices sintéticos (GainX/PainX) en MetaTrader 5.

### Características Principales

- **Sistema de IA Multinivel**: Ensemble de 4 modelos (Random Forest, Gradient Boosting, LSTM, Pattern-Based)
- **Meta-Learning**: Filtra señales primarias usando probabilidad de éxito histórico
- **116 Características Técnicas**: Indicadores, patrones, volatilidad, momentum, soporte/resistencia
- **Análisis Multi-Timeframe**: Confirma señales en 6 timeframes (1m, 5m, 15m, 1h, 4h, 1d)
- **Gestión de Riesgo Dinámica**: SL, TP, Break Even y Trailing Stop basados en ATR
- **Filtros Avanzados Configurables**: 8 filtros de calidad para maximizar win rate
- **Trading Automático**: Ejecución automática en MT5 con gestión de posiciones
- **Notificaciones Telegram**: Señales en tiempo real con gráficos

---

## 🏗️ Arquitectura del Sistema

### Componentes Principales

```
┌─────────────────────────────────────────────────────────────┐
│                    TRADING BOT SYSTEM                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │  Data Collector  │─────▶│  Feature Engine  │           │
│  │   (MT5 Data)     │      │  (116 Features)  │           │
│  └──────────────────┘      └──────────────────┘           │
│            │                        │                      │
│            ▼                        ▼                      │
│  ┌──────────────────────────────────────────┐             │
│  │         AI Engine (4 Models)             │             │
│  │  ┌─────────┐ ┌─────────┐ ┌──────┐       │             │
│  │  │   RF    │ │   GB    │ │ LSTM │       │             │
│  │  └─────────┘ └─────────┘ └──────┘       │             │
│  │  ┌──────────────────────────────┐       │             │
│  │  │   Pattern Model (Primary)    │       │             │
│  │  └──────────────────────────────┘       │             │
│  │              │                           │             │
│  │              ▼                           │             │
│  │  ┌──────────────────────────────┐       │             │
│  │  │   Ensemble Stacking          │       │             │
│  │  │   (Logistic Regression)      │       │             │
│  │  └──────────────────────────────┘       │             │
│  └──────────────────────────────────────────┘             │
│                     │                                      │
│                     ▼                                      │
│  ┌─────────────────────────────────────────┐              │
│  │     Signal Generator                    │              │
│  │  - Confidence Threshold (75%)           │              │
│  │  - Multi-TF Validation                  │              │
│  │  - Direction Validation (GainX/PainX)   │              │
│  └─────────────────────────────────────────┘              │
│                     │                                      │
│                     ▼                                      │
│  ┌─────────────────────────────────────────┐              │
│  │     Signal Filters (8 Filters)          │              │
│  │  1. Timeframe Confluence (≥50%)         │              │
│  │  2. Trend Alignment (ADX, EMAs)         │              │
│  │  3. Volatility Check (ATR < 5%)         │              │
│  │  4. Divergence Filter (RSI/MACD)        │              │
│  │  5. Consecutive Losses (Max 2)          │              │
│  │  6. Momentum Filter (±2%)               │              │
│  │  7. S/R Proximity (0.5% from level)     │              │
│  │  8. Conflicting Signals (1h lookback)   │              │
│  └─────────────────────────────────────────┘              │
│                     │                                      │
│            ┌────────┴────────┐                             │
│            ▼                 ▼                             │
│  ┌──────────────┐  ┌──────────────┐                       │
│  │   Telegram   │  │  MT5 Trader  │                       │
│  │    Alert     │  │ (Auto-Trade) │                       │
│  └──────────────┘  └──────────────┘                       │
│                            │                               │
│                            ▼                               │
│                  ┌──────────────────┐                      │
│                  │  Risk Manager    │                      │
│                  │  - Dynamic SL/TP │                      │
│                  │  - Break Even    │                      │
│                  │  - Trailing Stop │                      │
│                  └──────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

### Módulos del Sistema

| Módulo | Archivo | Función |
|--------|---------|---------|
| **Data Collector** | `mt5_connector.py` | Conexión con MT5, descarga de datos OHLCV |
| **Feature Engine** | `feature_engineering.py` | Extracción de 116 características técnicas |
| **Technical Indicators** | `technical_indicators.py` | Cálculo de 50+ indicadores técnicos |
| **AI Models** | `ai_models.py` | 4 modelos ML + Ensemble Stacking |
| **Market Analyzer** | `market_analyzer.py` | Análisis multi-timeframe y patrones |
| **Signal Generator** | `signal_generator.py` | Generación de señales de trading |
| **Signal Filter** | `signal_filter.py` | 8 filtros de calidad configurables |
| **Risk Manager** | `risk_manager.py` | Gestión dinámica de SL/TP/Lotaje |
| **Telegram Bot** | `telegram_bot.py` | Notificaciones y gráficos |
| **Main Loop** | `main_mt5.py` | Loop principal del bot |

---

## 📈 Estrategia de Trading

### Filosofía de Trading

El bot implementa una **estrategia híbrida** que combina:

1. **Análisis Técnico Clásico**: Indicadores probados (RSI, MACD, EMAs, Bollinger Bands)
2. **Machine Learning Avanzado**: Detección de patrones complejos con LSTM y Ensemble
3. **Gestión de Riesgo Profesional**: Basada en ATR con ajustes dinámicos
4. **Filtrado Multinivel**: Solo opera señales de máxima calidad

### Tipos de Señales

#### 1. Señales BUY (Compra)

**Condiciones para generar señal BUY:**

- **Indicadores alcistas alineados:**
  - RSI < 40 (sobreventa o acercándose)
  - MACD > MACD Signal (cruce alcista)
  - Precio > SMA50 (tendencia alcista)
  - EMA9 > EMA21 > EMA50 (alineación alcista)
  - Precio cerca del Bollinger Band inferior (rebote esperado)
  - Stochastic K < 20 (sobreventa)

- **Confluencia Multi-Timeframe:**
  - ≥50% de timeframes confirman señal BUY
  - Timeframe superior (4h/1h) en tendencia alcista

- **Confirmaciones de IA:**
  - Pattern Model: Score ≥ 0.3 (alcista)
  - Random Forest: Probabilidad de BUY > 50%
  - Gradient Boosting: Probabilidad de BUY > 50%
  - LSTM: Predicción de tendencia alcista
  - Ensemble: Confianza final ≥ 75%

- **Filtros aprobados:**
  - No hay divergencia bajista (RSI bajo + precio cayendo)
  - Momentum no es excesivamente negativo (> -2%)
  - Cerca de nivel de soporte (dentro de 0.5%)
  - Sin pérdidas consecutivas recientes (< 2 SLs)
  - ADX > 25 (tendencia fuerte) si filtro mejorado está activo

**Ejemplo de señal BUY:**
```
🟢 BUY SIGNAL

Symbol: GainX 400
Timeframe: 15m
Confidence: 87.5%

📊 ENTRY
💵 Price: 100000.50

🛑 STOP LOSS
💵 Price: 99850.25 (1.5 × ATR)

🎯 TAKE PROFIT LEVELS
   TP1: 100200.75 (+0.20%, 2.0 × ATR)
   TP2: 100400.50 (+0.40%, 4.0 × ATR)

Risk/Reward: 1:2.67

💡 Reason: 5/6 timeframes aligned, RSI oversold (32), MACD bullish crossover
```

#### 2. Señales SELL (Venta)

**Condiciones para generar señal SELL:**

- **Indicadores bajistas alineados:**
  - RSI > 60 (sobrecompra o acercándose)
  - MACD < MACD Signal (cruce bajista)
  - Precio < SMA50 (tendencia bajista)
  - EMA9 < EMA21 < EMA50 (alineación bajista)
  - Precio cerca del Bollinger Band superior (rechazo esperado)
  - Stochastic K > 80 (sobrecompra)

- **Confluencia Multi-Timeframe:**
  - ≥50% de timeframes confirman señal SELL
  - Timeframe superior (4h/1h) en tendencia bajista

- **Confirmaciones de IA:**
  - Pattern Model: Score ≤ -0.3 (bajista)
  - Random Forest: Probabilidad de SELL > 50%
  - Gradient Boosting: Probabilidad de SELL > 50%
  - LSTM: Predicción de tendencia bajista
  - Ensemble: Confianza final ≥ 75%

- **Filtros aprobados:**
  - No hay divergencia alcista (RSI alto + precio subiendo)
  - Momentum no es excesivamente positivo (< +2%)
  - Cerca de nivel de resistencia (dentro de 0.5%)
  - Sin pérdidas consecutivas recientes (< 2 SLs)
  - ADX > 25 (tendencia fuerte) si filtro mejorado está activo

**Ejemplo de señal SELL:**
```
🔴 SELL SIGNAL

Symbol: PainX 400
Timeframe: 5m
Confidence: 82.3%

📊 ENTRY
💵 Price: 50000.00

🛑 STOP LOSS
💵 Price: 50150.00 (1.5 × ATR)

🎯 TAKE PROFIT LEVELS
   TP1: 49800.00 (-0.40%, 2.0 × ATR)
   TP2: 49600.00 (-0.80%, 4.0 × ATR)

Risk/Reward: 1:2.67

💡 Reason: 4/5 timeframes aligned, RSI overbought (68), Near resistance (0.3%)
```

#### 3. Señales HOLD (Sin acción)

**El bot genera HOLD cuando:**
- No hay confluencia clara entre indicadores
- Confianza < 75%
- Filtros de calidad no pasan
- Mercado lateral (ADX < 25 con filtro mejorado)
- Señal conflictiva en la última hora
- Límite diario de señales alcanzado

---

## 🤖 Modelos de Inteligencia Artificial

### 1. SimplePatternModel (Modelo Primario)

**Tipo:** Rule-Based Pattern Recognition
**Función:** Genera señales iniciales BUY/SELL/HOLD

#### Sistema de Scoring

El modelo asigna puntajes ponderados a diferentes indicadores:

| Indicador | Peso | Condición BUY | Condición SELL |
|-----------|------|---------------|----------------|
| **RSI** | 2.0 | RSI < 30 → +2.0<br>RSI < 40 → +1.0 | RSI > 70 → -2.0<br>RSI > 60 → -1.0 |
| **MACD Diff** | 1.5 | MACD > Signal → +1.5 | MACD < Signal → -1.5 |
| **SMA Trend** | 1.0 | Price > SMA50 → +1.0 | Price < SMA50 → -1.0 |
| **EMA Cross** | 1.0 | EMA9 > EMA21 → +1.0 | EMA9 < EMA21 → -1.0 |
| **Stochastic** | 0.8 | Stoch < 20 → +0.8 | Stoch > 80 → -0.8 |
| **Bollinger** | 0.7 | Price ≤ BB Low → +0.7 | Price ≥ BB High → -0.7 |
| **Momentum** | 0.5 | Momentum > 0 → +0.5 | Momentum < 0 → -0.5 |
| **ADX** | 0.5 (modifier) | ADX < 25 → Score × 0.7 | ADX < 25 → Score × 0.7 |

**Umbrales de decisión:**
- Score ≥ +0.3 → **BUY**
- Score ≤ -0.3 → **SELL**
- -0.3 < Score < +0.3 → **HOLD**

**Ventajas:**
- ✅ Genera muchas señales candidatas
- ✅ Transparente y explicable
- ✅ No requiere entrenamiento
- ✅ Rápido en tiempo real

**Limitaciones:**
- ⚠️ No captura patrones complejos
- ⚠️ Requiere filtrado posterior por meta-modelos

### 2. Random Forest Classifier

**Tipo:** Ensemble de árboles de decisión
**Configuración:** 100 estimadores, max_depth=10, calibrado con isotonic regression

#### Características

- **Entrenamiento:** 116 features técnicas → Predicción 3 clases (SELL/HOLD/BUY)
- **Calibración:** CalibratedClassifierCV para probabilidades confiables
- **Validación:** 3-fold cross-validation
- **Output:** Probabilidades calibradas para cada clase

**Ventajas:**
- ✅ Robusto a overfitting
- ✅ Maneja bien datos no lineales
- ✅ Probabilidades bien calibradas
- ✅ Importancia de features interpretable

### 3. Gradient Boosting Classifier

**Tipo:** Ensemble secuencial de árboles débiles
**Configuración:** 100 estimadores, learning_rate=0.1, max_depth=5, calibrado

#### Características

- **Entrenamiento:** 116 features → Predicción 3 clases
- **Calibración:** CalibratedClassifierCV isotonic
- **Validación:** 3-fold cross-validation
- **Output:** Probabilidades calibradas

**Ventajas:**
- ✅ Mayor precisión que RF
- ✅ Captura interacciones complejas
- ✅ Probabilidades calibradas
- ✅ Convergencia más rápida

### 4. LSTM (Long Short-Term Memory)

**Tipo:** Red neuronal recurrente
**Arquitectura:**
```
Input (50 timesteps × 116 features)
    ↓
LSTM Layer 1 (50 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (50 units)
    ↓
Dropout (0.2)
    ↓
Dense (25 units, ReLU)
    ↓
Dense (1 unit, Sigmoid)
    ↓
Output (Probability)
```

#### Configuración de Entrenamiento

- **Sequence Length:** 50 períodos
- **Epochs:** 50 (con Early Stopping patience=10)
- **Batch Size:** 32
- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Class Weights:** Automático (balanceo de clases)
- **Métricas:** Accuracy, AUC, Precision, Recall
- **Validation Split:** 10%

**Ventajas:**
- ✅ Captura dependencias temporales complejas
- ✅ Memoria de patrones históricos (50 velas)
- ✅ Class weights para datos desbalanceados
- ✅ Early stopping evita overfitting

**Desventajas:**
- ⚠️ Requiere más datos para entrenar
- ⚠️ Computacionalmente intensivo
- ⚠️ Menos interpretable que modelos tree-based

### 5. Ensemble Stacking (Meta-Modelo)

**Tipo:** Logistic Regression sobre predicciones de modelos base
**Función:** Combina predicciones de los 4 modelos para decisión final

#### Pipeline de Ensemble

```
Step 1: Base Models Predictions
├─ Random Forest    → [P(SELL), P(HOLD), P(BUY)]
├─ Gradient Boost   → [P(SELL), P(HOLD), P(BUY)]
├─ Pattern Model    → [P(SELL), P(HOLD), P(BUY)]
└─ LSTM             → [P(SELL), P(BUY)]  # Binary

Step 2: Feature Vector for Meta-Model
Meta-Features = [
    RF_proba_0, RF_proba_1, RF_proba_2,
    GB_proba_0, GB_proba_1, GB_proba_2,
    PM_proba_0, PM_proba_1, PM_proba_2,
    LSTM_proba_0, LSTM_proba_1
]  # Total: 11 features

Step 3: Meta-Model (Logistic Regression)
Meta-Features → Logistic Regression → Final [P(SELL), P(HOLD), P(BUY)]

Step 4: Decision
Max(P(SELL), P(HOLD), P(BUY)) → Final Signal + Confidence
```

**Ventajas del Stacking:**
- ✅ Combina fortalezas de cada modelo
- ✅ Reduce varianza (averaging)
- ✅ Captura consenso entre modelos
- ✅ Meta-modelo aprende cuándo confiar en cada modelo base

---

## 📊 Indicadores Técnicos

### Listado de 116 Features Extraídas

El sistema extrae 116 características técnicas de los datos OHLCV:

#### Indicadores de Tendencia (20 features)

| Indicador | Períodos | Descripción |
|-----------|----------|-------------|
| **SMA** | 10, 20, 50, 100, 200 | Simple Moving Average |
| **EMA** | 9, 12, 21, 26, 50, 100, 200 | Exponential Moving Average |
| **TEMA** | 20 | Triple Exponential Moving Average |
| **DEMA** | 20 | Double Exponential Moving Average |
| **KAMA** | 10 | Kaufman Adaptive Moving Average |
| **WMA** | 20 | Weighted Moving Average |

#### Indicadores de Momentum (25 features)

| Indicador | Configuración | Descripción |
|-----------|---------------|-------------|
| **RSI** | 14, 21 | Relative Strength Index |
| **Stochastic** | K=14, D=3 | Stochastic Oscillator |
| **Stochastic RSI** | 14 | Stochastic of RSI |
| **Williams %R** | 14 | Williams Percent Range |
| **ROC** | 12 | Rate of Change |
| **Momentum** | 5, 10 | Price Momentum |
| **CCI** | 20 | Commodity Channel Index |
| **Ultimate Oscillator** | 7/14/28 | Multi-period momentum |
| **MACD** | 12/26/9 | Moving Average Convergence Divergence |
| **PPO** | 12/26/9 | Percentage Price Oscillator |

#### Indicadores de Volatilidad (15 features)

| Indicador | Configuración | Descripción |
|-----------|---------------|-------------|
| **ATR** | 14 | Average True Range |
| **Bollinger Bands** | 20, stddev=2 | BB Upper, Middle, Lower, Width, %B |
| **Keltner Channels** | 20 | KC Upper, Middle, Lower |
| **Donchian Channels** | 20 | DC High, Low, Mid |
| **Standard Deviation** | 20 | Price volatility |

#### Indicadores de Volumen (8 features)

| Indicador | Configuración | Descripción |
|-----------|---------------|-------------|
| **OBV** | - | On-Balance Volume |
| **Volume SMA** | 20 | Volume moving average |
| **Volume Ratio** | Current / SMA | Volume strength |
| **MFI** | 14 | Money Flow Index |
| **AD** | - | Accumulation/Distribution |
| **ADL** | - | Accumulation/Distribution Line |
| **CMF** | 20 | Chaikin Money Flow |
| **Force Index** | 13 | Force Index |

#### Indicadores de Tendencia Avanzados (12 features)

| Indicador | Configuración | Descripción |
|-----------|---------------|-------------|
| **ADX** | 14 | Average Directional Index (fuerza de tendencia) |
| **+DI / -DI** | 14 | Directional Indicators |
| **Aroon** | 25 | Aroon Up, Aroon Down, Aroon Oscillator |
| **TRIX** | 15 | Triple Exponential Average |
| **Vortex** | 14 | Vortex Indicator (VI+, VI-) |
| **Mass Index** | 9/25 | Mass Index |
| **Ichimoku** | 9/26/52 | Tenkan, Kijun, Senkou A/B, Chikou |

#### Patrones de Precios (10 features)

| Patrón | Descripción |
|--------|-------------|
| **Bullish Engulfing** | Patrón alcista de reversión |
| **Bearish Engulfing** | Patrón bajista de reversión |
| **Morning Star** | Patrón alcista de reversión (3 velas) |
| **Evening Star** | Patrón bajista de reversión (3 velas) |
| **Hammer** | Patrón alcista de reversión |
| **Shooting Star** | Patrón bajista de reversión |
| **Doji** | Indecisión del mercado |
| **Price above/below MA** | Posición relativa del precio |

#### Features Derivadas (26 features)

- **Diferencias de EMAs**: EMA9-EMA21, EMA21-EMA50, etc.
- **Ratios**: Precio/SMA50, Precio/EMA50, ATR/Precio
- **Lag Features**: RSI_lag1, MACD_lag1, etc. (valores previos)
- **Cambios porcentuales**: Close_pct_change, Volume_pct_change
- **Cruces**: EMA_cross_9_21, MACD_cross_signal

### Cálculo en Tiempo Real

Durante la operación del bot:

1. **Recolección de datos**: MT5 proporciona últimas 500 velas de cada timeframe
2. **Extracción de features**: `FeatureEngineer` calcula todas las 116 features
3. **Normalización**: StandardScaler normaliza features (media 0, std 1)
4. **Validación**: Verifica que no haya NaN/Inf, rellena con valores válidos
5. **Predicción**: Features normalizadas → Modelos de IA → Señal

---

## 🛡️ Filtros de Calidad de Señales

El bot aplica **8 filtros secuenciales** antes de enviar señal a Telegram:

### 1. Filtro de Umbral de Confianza

**Condición:** `Confidence ≥ CONFIDENCE_THRESHOLD` (default: 75%)

- La confianza es el máximo de las probabilidades del ensemble: `max(P(SELL), P(HOLD), P(BUY))`
- Si confianza < 75%, la señal es rechazada inmediatamente
- Configurable en `.env`: `CONFIDENCE_THRESHOLD=0.75`

**Log:**
```
✅ Passed confidence threshold (Confidence: 87% >= 75%)
❌ Confidence 62% below threshold 75%
```

### 2. Filtro de Confluencia Multi-Timeframe

**Condición:** `≥ 50% de timeframes confirman la misma señal`

- Ejemplo: Si 5 de 6 timeframes predicen BUY, confluencia = 83% ✅
- Si solo 2 de 6 predicen BUY, confluencia = 33% ❌
- Requiere consenso mayoritario entre timeframes

**Log:**
```
✅ Timeframe confluence passed (83.3% >= 50%)
❌ Insufficient timeframe confluence (33.3% < 50%)
```

### 3. Filtro de Alineación de Tendencia

**Condición (BUY):** Precio no muy debajo de SMA50 en timeframe superior (1h/4h)

**Condición (SELL):** Precio no muy arriba de SMA50 en timeframe superior

**Con Filtro Mejorado Habilitado (`ENABLE_ENHANCED_TREND_FILTER=true`):**

- **ADX Mínimo:** ADX ≥ 25 (default) → Evita mercados laterales
- **Alineación de EMAs (BUY):** EMA9 > EMA21 > EMA50 (tendencia alcista clara)
- **Alineación de EMAs (SELL):** EMA9 < EMA21 < EMA50 (tendencia bajista clara)
- **Distancia de SMA50:** Precio dentro de ±3% de SMA50 (default)

**Log:**
```
✅ Trend alignment passed
✅ Tendencia alcista en 4h: ADX=32.5, EMAs alineadas, Precio +1.2% de SMA50
❌ Tendencia débil en 4h: ADX=18.3 < 25 (mercado lateral, evitar)
❌ Precio muy debajo de SMA50 en 4h: -4.5% (contra-tendencia bajista)
```

**Parámetros configurables:**
```bash
ENABLE_ENHANCED_TREND_FILTER=true
MIN_ADX_FOR_TREND=25.0
EMA_ALIGNMENT_REQUIRED=true
MAX_PERCENT_FROM_SMA50=3.0
```

### 4. Filtro de Volatilidad

**Condición:** `ATR < 5% del precio actual`

- Evita operar en condiciones de volatilidad extrema
- ATR es Average True Range (14 períodos)
- Si ATR > 5% del precio → mercado muy volátil, riesgo alto

**Log:**
```
✅ Volatility check passed
❌ Volatility too high - ATR 8.2% of price (need < 5%)
```

### 5. Filtro de Divergencias RSI/MACD

**Configurable:** `ENABLE_DIVERGENCE_FILTER=true` (default)

**Detecta:**

**Divergencia Alcista (Peligrosa para SELL):**
- RSI > 70 (sobrecompra) PERO precio sigue subiendo fuertemente (momentum > +1.5%)
- Indica que el rally alcista puede continuar → NO VENDER aún

**Divergencia Bajista (Peligrosa para BUY):**
- RSI < 30 (sobreventa) PERO precio sigue cayendo fuertemente (momentum < -1.5%)
- Indica que la caída bajista puede continuar → NO COMPRAR aún

**Log:**
```
✅ No hay divergencias peligrosas (RSI=55.2)
❌ Divergencia alcista detectada: RSI=72.3 (>70) pero precio sube +2.1% - Probable continuación alcista
❌ Divergencia bajista detectada: RSI=28.1 (<30) pero precio cae -2.4% - Probable continuación bajista
```

**Parámetros:**
```bash
ENABLE_DIVERGENCE_FILTER=true
DIVERGENCE_RSI_OVERBOUGHT=70.0
DIVERGENCE_RSI_OVERSOLD=30.0
DIVERGENCE_MOMENTUM_THRESHOLD=1.5
```

### 6. Filtro de Pérdidas Consecutivas

**Configurable:** `ENABLE_CONSECUTIVE_LOSSES_FILTER=true` (default)

**Condición:** Máximo 2 SLs consecutivos por símbolo (default)

- Rastrea operaciones cerradas con SL vs TP
- Si un símbolo acumula 2+ SLs consecutivos → **Período de enfriamiento de 2 horas**
- Resetea contador cuando hay un TP (Take Profit)

**Log:**
```
✅ Pérdidas consecutivas: 0/2
✅ Período de enfriamiento completado (2 SLs previos)
🔒 Período de enfriamiento activo: 2 SLs consecutivos. Esperar 87 min más
```

**Parámetros:**
```bash
ENABLE_CONSECUTIVE_LOSSES_FILTER=true
MAX_CONSECUTIVE_LOSSES=2
COOLDOWN_HOURS=2.0
```

### 7. Filtro de Momentum Excesivo

**Configurable:** `ENABLE_MOMENTUM_FILTER=true` (default)

**Condición:** Momentum en últimas 5 velas no debe ser extremo (±2% default)

**Para SELL:**
- Si momentum > +2% (precio subiendo muy rápido) → NO VENDER aún, esperar agotamiento

**Para BUY:**
- Si momentum < -2% (precio cayendo muy rápido) → NO COMPRAR aún, esperar agotamiento

**Razón:** Operar contra momentum fuerte aumenta probabilidad de SL

**Log:**
```
✅ Momentum aceptable: +0.8%
❌ Momentum alcista muy fuerte: +3.2% en últimas 5 velas - Esperar agotamiento
❌ Momentum bajista muy fuerte: -2.9% en últimas 5 velas - Esperar agotamiento
```

**Parámetros:**
```bash
ENABLE_MOMENTUM_FILTER=true
MAX_MOMENTUM_PERCENT=2.0
```

### 8. Filtro de Proximidad a Soporte/Resistencia

**Configurable:** `ENABLE_SR_PROXIMITY_FILTER=true` (default)

**Condición:**

**Para SELL:** Debe estar cerca de resistencia (≤0.5% de distancia)
- Si está > 1.5% lejos de resistencia → NO VENDER (operar solo en niveles clave)

**Para BUY:** Debe estar cerca de soporte (≤0.5% de distancia)
- Si está > 1.5% lejos de soporte → NO COMPRAR (operar solo en niveles clave)

**Razón:** Operar en niveles de S/R aumenta probabilidad de rebote/rechazo

**Log:**
```
✅ Cerca de resistencia: 0.3% (R=100,250.50)
✅ Cerca de soporte: 0.4% (S=99,750.25)
⚠️ Muy lejos de resistencia: 2.1% > 1.5% (operar solo cerca de niveles clave)
⚠️ Muy lejos de soporte: 1.8% > 1.5% (operar solo cerca de niveles clave)
```

**Parámetros:**
```bash
ENABLE_SR_PROXIMITY_FILTER=true
SR_PROXIMITY_PERCENT=0.5
SR_MAX_DISTANCE_PERCENT=1.5
```

### 9. Filtro de Señales Conflictivas

**Condición:** No señales opuestas en la última hora

- Si hubo SELL hace 30 min y ahora quiere BUY → Rechazar (señales conflictivas)
- Evita whipsaws y señales contradictorias

**Log:**
```
✅ No conflicting signals
❌ Conflicting recent signal
```

---

## ⚖️ Gestión de Riesgo

### 1. Stop Loss Dinámico (SL)

**Basado en ATR (Average True Range de 14 períodos)**

```
SL = Entry ± (ATR × STOP_LOSS_ATR_MULTIPLIER)
```

**Default:** `STOP_LOSS_ATR_MULTIPLIER = 1.5`

- **BUY:** `SL = Entry - (1.5 × ATR)`
- **SELL:** `SL = Entry + (1.5 × ATR)`

**Ventajas:**
- ✅ Se adapta a volatilidad del mercado
- ✅ En mercados tranquilos (ATR bajo) → SL más ajustado
- ✅ En mercados volátiles (ATR alto) → SL más amplio

**Ejemplo:**
```
GainX 400, Entry: 100000, ATR: 100
SL = 100000 - (1.5 × 100) = 99850
Distancia: 150 puntos (0.15%)
```

### 2. Take Profit Dinámico (TP)

**Dos niveles de Take Profit:**

```
TP1 = Entry ± (ATR × TAKE_PROFIT_1_ATR_MULTIPLIER)
TP2 = Entry ± (ATR × TAKE_PROFIT_2_ATR_MULTIPLIER)
```

**Defaults:**
- `TAKE_PROFIT_1_ATR_MULTIPLIER = 2.0`
- `TAKE_PROFIT_2_ATR_MULTIPLIER = 4.0`

**Estrategia de salida:**
- Cerrar 50% de posición en TP1 (asegurar ganancias)
- Dejar correr 50% restante hasta TP2 (maximizar profit)

**Ejemplo:**
```
PainX 400, Entry: 50000, ATR: 75
TP1 = 50000 - (2.0 × 75) = 49850  (Risk/Reward: 1:2)
TP2 = 50000 - (4.0 × 75) = 49700  (Risk/Reward: 1:4)
```

### 3. Risk/Reward Ratio

**Cálculo:**
```
Risk = |Entry - SL|
Reward = |TP1 - Entry|
RR = Reward / Risk
```

**Objetivos:**
- Mínimo: RR ≥ 1:1.5
- Ideal: RR ≥ 1:2
- Excelente: RR ≥ 1:3

**Ejemplo:**
```
Entry: 100000
SL: 99850 → Risk = 150
TP1: 100200 → Reward = 200
RR = 200/150 = 1:1.33 ✅
```

### 4. Break Even (BE)

**Configurable:** `ENABLE_BREAK_EVEN=true` (default)

**Condición:** Cuando `Profit ≥ (1.0 × ATR)`, mover SL a break even + pequeña ganancia

```
Trigger: Current Price - Entry ≥ (ATR × BREAK_EVEN_TRIGGER_ATR_MULTIPLIER)
New SL: Entry + (ATR × BREAK_EVEN_PROFIT_LOCK_ATR_MULTIPLIER)
```

**Defaults:**
- `BREAK_EVEN_TRIGGER_ATR_MULTIPLIER = 1.0`
- `BREAK_EVEN_PROFIT_LOCK_ATR_MULTIPLIER = 0.2`

**Ventaja:** Asegura pequeña ganancia y elimina riesgo de perder

**Ejemplo (BUY):**
```
Entry: 100000, ATR: 100, SL inicial: 99850
Precio sube a 100100 (profit = 100 = 1.0 × ATR)
→ Activar Break Even
→ Nuevo SL: 100000 + (0.2 × 100) = 100020
→ Asegurado +20 puntos de ganancia, riesgo = 0
```

### 5. Trailing Stop

**Configurable:** `ENABLE_TRAILING_STOP=true` (default)

**Condición:** Cuando `Profit ≥ (2.0 × ATR)`, activar trailing stop que sigue el precio

```
Trigger: Current Price - Entry ≥ (ATR × TRAILING_STOP_TRIGGER_ATR_MULTIPLIER)
SL Distance: Current Price - (ATR × TRAILING_STOP_DISTANCE_ATR_MULTIPLIER)
```

**Defaults:**
- `TRAILING_STOP_TRIGGER_ATR_MULTIPLIER = 2.0`
- `TRAILING_STOP_DISTANCE_ATR_MULTIPLIER = 1.5`

**Ventaja:** Maximiza ganancias en tendencias fuertes mientras protege profit acumulado

**Ejemplo (BUY):**
```
Entry: 100000, ATR: 100
Precio sube a 100200 (profit = 200 = 2.0 × ATR)
→ Activar Trailing Stop
→ SL: 100200 - (1.5 × 100) = 100050 (+50 asegurado)

Precio sigue subiendo a 100400
→ SL se mueve a: 100400 - 150 = 100250 (+250 asegurado)

Precio baja a 100250 → Cerrar con +250 de ganancia
```

### 6. Lotaje Dinámico

**Configurable:** `ENABLE_DYNAMIC_LOT_SIZE=true` (default)

**Fórmula:**
```
Lot Size = MIN_LOT + (Scaled Confidence × (MAX_LOT - MIN_LOT))

Scaled Confidence = (Confidence - MIN_CONFIDENCE) / (1.0 - MIN_CONFIDENCE)
```

**Defaults:**
- `MIN_LOT_SIZE = 0.10`
- `MAX_LOT_SIZE = 1.00`
- `CONFIDENCE_THRESHOLD = 0.75`

**Escalado:**
- Confianza 75% → 0.10 lotes (mínimo)
- Confianza 85% → 0.50 lotes (medio)
- Confianza 95%+ → 1.00 lotes (máximo)

**Ventaja:** Mayor riesgo solo en señales de máxima calidad

**Ejemplo:**
```
Confidence = 82%
Scaled = (0.82 - 0.75) / (1.0 - 0.75) = 0.07 / 0.25 = 0.28
Lot = 0.10 + (0.28 × (1.00 - 0.10)) = 0.10 + 0.252 = 0.35 lotes
```

### 7. Límites de Exposición

**Límites diarios configurables:**

```bash
MAX_SIGNALS_PER_DAY=10       # Máximo 10 señales ejecutadas por día
MAX_SIGNALS_PER_PAIR=3       # Máximo 3 señales por símbolo por día
MT5_MAX_OPEN_POSITIONS=3     # Máximo 3 posiciones abiertas simultáneas
```

**Ventaja:** Previene overtrading y sobreexposición

---

## 🎲 Reglas Especiales para Índices Sintéticos

### ¿Qué son los Índices Sintéticos?

Los índices sintéticos (GainX, PainX) son instrumentos de Deriv/Weltrade que simulan movimientos de mercado con características especiales:

- **GainX:** Precio con **spikes alcistas** frecuentes (súbitos aumentos de precio)
- **PainX:** Precio con **spikes bajistas** frecuentes (súbitas caídas de precio)

### Reglas del Bot para Índices Sintéticos

#### 1. GainX → Solo Señales BUY

**Configuración:** `ENFORCE_GAINX_BUY_ONLY=true` (default)

**Razón:** Los spikes de GainX son ALCISTAS → Solo tiene sentido comprar (aprovechar spikes al alza)

**Comportamiento:**
- ✅ Señales BUY generadas normalmente
- ❌ Señales SELL bloqueadas automáticamente

**Log:**
```
GainX 400: ✅ GainX index - BUY signal is valid (spike direction)
GainX 600: ❌ GainX index - Only BUY signals allowed (spikes are upward). Blocking SELL signal.
```

#### 2. PainX → Solo Señales SELL

**Configuración:** `ENFORCE_PAINX_SELL_ONLY=true` (default)

**Razón:** Los spikes de PainX son BAJISTAS → Solo tiene sentido vender (aprovechar spikes a la baja)

**Comportamiento:**
- ✅ Señales SELL generadas normalmente
- ❌ Señales BUY bloqueadas automáticamente

**Log:**
```
PainX 400: ✅ PainX index - SELL signal is valid (spike direction)
PainX 600: ❌ PainX index - Only SELL signals allowed (spikes are downward). Blocking BUY signal.
```

#### 3. Estrategia de Scalping para Sintéticos

**Timeframes recomendados:** 1m, 5m, 15m (scalping rápido)

**Configuración óptima:**
```bash
STOP_LOSS_ATR_MULTIPLIER=1.5       # SL ajustado para spikes
TAKE_PROFIT_1_ATR_MULTIPLIER=2.0   # TP1 rápido
TAKE_PROFIT_2_ATR_MULTIPLIER=4.0   # TP2 para spikes grandes

ENABLE_BREAK_EVEN=true              # Asegurar ganancias rápido
BREAK_EVEN_TRIGGER_ATR_MULTIPLIER=1.0

ENABLE_TRAILING_STOP=true           # Seguir spikes fuertes
TRAILING_STOP_TRIGGER_ATR_MULTIPLIER=2.0
```

**Ventajas para sintéticos:**
- ✅ Captura spikes frecuentes
- ✅ Break Even rápido protege de reversiones súbitas
- ✅ Trailing Stop maximiza ganancia en spikes fuertes

---

## 🔄 Flujo de Operación

### Flujo Completo (Cada 60 segundos)

```
┌──────────────────────────────────────────────────────────┐
│ 1. INICIO DEL CICLO                                      │
├──────────────────────────────────────────────────────────┤
│ • Main loop ejecuta cada 60 segundos                     │
│ • Verifica conexión MT5 activa                           │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ 2. RECOLECCIÓN DE DATOS (MT5Connector)                  │
├──────────────────────────────────────────────────────────┤
│ • Para cada símbolo configurado (GainX 400, PainX 400)  │
│ • Para cada timeframe (1m, 5m, 15m, 1h, 4h, 1d)         │
│ • Descarga últimas 500 velas OHLCV                       │
│ • Valida datos completos y sin gaps                      │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ 3. EXTRACCIÓN DE FEATURES (FeatureEngineer)             │
├──────────────────────────────────────────────────────────┤
│ • Calcula 116 características técnicas:                  │
│   - Indicadores de tendencia (SMA, EMA, etc.)           │
│   - Indicadores de momentum (RSI, MACD, etc.)           │
│   - Indicadores de volatilidad (ATR, BB, etc.)          │
│   - Patrones de precios (Engulfing, Doji, etc.)         │
│   - Features derivadas (cruces, ratios, lags)           │
│ • Normaliza con StandardScaler                           │
│ • Valida NaN/Inf, rellena valores inválidos             │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ 4. ANÁLISIS MULTI-TIMEFRAME (MarketAnalyzer)            │
├──────────────────────────────────────────────────────────┤
│ Para cada timeframe:                                     │
│ • Carga modelos entrenados del símbolo                   │
│ • Ejecuta predicción con Ensemble:                       │
│   ├─ SimplePatternModel → Señal inicial                 │
│   ├─ Random Forest → Probabilidades                     │
│   ├─ Gradient Boosting → Probabilidades                 │
│   ├─ LSTM → Probabilidades                              │
│   └─ Meta-model → Señal final + Confianza               │
│ • Detecta patrones de velas                              │
│ • Calcula niveles de soporte/resistencia                 │
│ • Genera MarketAnalysis object                           │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ 5. GENERACIÓN DE SEÑAL (SignalGenerator)                │
├──────────────────────────────────────────────────────────┤
│ • Obtiene análisis del timeframe primario (5m/15m/1h)   │
│ • Extrae señal y confianza del análisis                  │
│ • Valida dirección (GainX→BUY, PainX→SELL)              │
│ • Verifica umbral de confianza (≥75%)                    │
│ • Verifica no duplicados (SignalTracker)                 │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ 6. FILTRADO DE CALIDAD (SignalFilter)                   │
├──────────────────────────────────────────────────────────┤
│ Aplica 8 filtros secuenciales:                           │
│ ✓ 1. Confluencia multi-TF (≥50%)                        │
│ ✓ 2. Alineación de tendencia (SMA50, ADX, EMAs)         │
│ ✓ 3. Volatilidad aceptable (ATR < 5%)                   │
│ ✓ 4. Sin divergencias peligrosas (RSI/momentum)         │
│ ✓ 5. Sin pérdidas consecutivas (≤2 SLs + cooldown)      │
│ ✓ 6. Momentum no excesivo (±2%)                          │
│ ✓ 7. Cerca de S/R (≤1.5% distancia)                     │
│ ✓ 8. Sin señales conflictivas (1h lookback)             │
│                                                          │
│ Si algún filtro falla → Señal rechazada                  │
│ Si todos pasan → Continuar                               │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ 7. CÁLCULO DE RIESGO (RiskManager)                      │
├──────────────────────────────────────────────────────────┤
│ • Obtiene ATR del análisis (14 períodos)                 │
│ • Calcula Stop Loss:                                     │
│   SL = Entry ± (1.5 × ATR)                               │
│ • Calcula Take Profits:                                  │
│   TP1 = Entry ± (2.0 × ATR)                              │
│   TP2 = Entry ± (4.0 × ATR)                              │
│ • Calcula Risk/Reward ratio                              │
│ • Calcula lotaje dinámico (0.10 - 1.00)                 │
│ • Valida que SL y TP sean válidos                        │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ 8. CREACIÓN DE SEÑAL (TradingSignal)                    │
├──────────────────────────────────────────────────────────┤
│ • Genera signal_id único                                 │
│ • Almacena todos los parámetros:                         │
│   - Symbol, Type (BUY/SELL), Timeframe                   │
│   - Entry, SL, TP1, TP2                                  │
│   - Confidence, Risk/Reward, Lot Size                    │
│   - Timestamp, Reason, Analysis data                     │
│ • Registra en historial de señales                       │
│ • Registra en SignalTracker (evitar duplicados)          │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ 9. NOTIFICACIÓN TELEGRAM (TelegramBot)                  │
├──────────────────────────────────────────────────────────┤
│ • Formatea mensaje con todos los detalles               │
│ • Genera gráfico de precio con indicadores              │
│ • Envía mensaje a canal de Telegram                      │
│ • Marca señal como notificada                            │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ 10. EJECUCIÓN EN MT5 (si MT5_AUTO_TRADING=true)         │
├──────────────────────────────────────────────────────────┤
│ • Verifica límites diarios no alcanzados:                │
│   - MAX_SIGNALS_PER_DAY (10)                             │
│   - MAX_SIGNALS_PER_PAIR (3)                             │
│   - MT5_MAX_OPEN_POSITIONS (3)                           │
│ • Prepara orden MT5:                                     │
│   - Type: ORDER_TYPE_BUY / ORDER_TYPE_SELL               │
│   - Volume: lot_size calculado                           │
│   - Price: current market price                          │
│   - SL: stop_loss                                        │
│   - TP: take_profit_1                                    │
│   - Magic Number: 234000                                 │
│ • Ejecuta orden en MT5                                   │
│ • Registra ticket en logs                                │
│ • Incrementa contadores de límites                       │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ 11. GESTIÓN DE POSICIONES ABIERTAS                      │
├──────────────────────────────────────────────────────────┤
│ Para cada posición abierta:                              │
│ • Verifica precio actual vs entry                        │
│ • Si profit ≥ 1.0 × ATR → Activar Break Even            │
│   - Mover SL a entry + 0.2 × ATR                         │
│ • Si profit ≥ 2.0 × ATR → Activar Trailing Stop         │
│   - Mover SL a (current price - 1.5 × ATR)              │
│ • Si SL o TP alcanzado → Registrar resultado            │
│   - Actualizar closed_trades (para filtro pérdidas)     │
│   - Actualizar estadísticas de performance              │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ 12. LOGGING Y MONITOREO                                 │
├──────────────────────────────────────────────────────────┤
│ • Log de cada etapa en logs/trading_bot.log             │
│ • Tracking de performance (win rate, profit)            │
│ • Estadísticas de señales generadas                      │
│ • Alertas de errores o problemas                         │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ 13. ESPERAR PRÓXIMO CICLO                               │
├──────────────────────────────────────────────────────────┤
│ • Sleep 60 segundos                                      │
│ • Repetir desde paso 1                                   │
└──────────────────────────────────────────────────────────┘
```

---

## ⚙️ Parámetros Configurables

Todos los parámetros se configuran en el archivo `.env`:

### Configuración de Telegram

```bash
TELEGRAM_BOT_TOKEN=tu_token_de_botfather
TELEGRAM_CHANNEL_ID=tu_channel_id
TELEGRAM_INCLUDE_CHARTS=true
```

### Configuración de MT5

```bash
MT5_LOGIN=12345678
MT5_PASSWORD=tu_contraseña
MT5_SERVER=Weltrade-Demo
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
MT5_MAGIC_NUMBER=234000
MT5_AUTO_TRADING=false  # false = solo notificaciones, true = trading automático
```

### Símbolos y Timeframes

```bash
TRADING_SYMBOLS=GainX 400,GainX 600,PainX 400,PainX 600
TIMEFRAMES=1m,5m,15m,1h,4h,1d
```

### Parámetros de IA

```bash
CONFIDENCE_THRESHOLD=0.75  # Umbral mínimo de confianza (75%)
```

### Gestión de Riesgo

```bash
# Stop Loss y Take Profit
STOP_LOSS_ATR_MULTIPLIER=1.5
TAKE_PROFIT_1_ATR_MULTIPLIER=2.0
TAKE_PROFIT_2_ATR_MULTIPLIER=4.0

# Break Even
ENABLE_BREAK_EVEN=true
BREAK_EVEN_TRIGGER_ATR_MULTIPLIER=1.0
BREAK_EVEN_PROFIT_LOCK_ATR_MULTIPLIER=0.2

# Trailing Stop
ENABLE_TRAILING_STOP=true
TRAILING_STOP_TRIGGER_ATR_MULTIPLIER=2.0
TRAILING_STOP_DISTANCE_ATR_MULTIPLIER=1.5

# Lotaje Dinámico
ENABLE_DYNAMIC_LOT_SIZE=true
MIN_LOT_SIZE=0.10
MAX_LOT_SIZE=1.00
```

### Filtros Avanzados

```bash
# Activación de filtros (true/false)
ENABLE_DIVERGENCE_FILTER=true
ENABLE_CONSECUTIVE_LOSSES_FILTER=true
ENABLE_ENHANCED_TREND_FILTER=true
ENABLE_MOMENTUM_FILTER=true
ENABLE_SR_PROXIMITY_FILTER=true

# Parámetros de filtro de pérdidas consecutivas
MAX_CONSECUTIVE_LOSSES=2
COOLDOWN_HOURS=2.0

# Parámetros de filtro de tendencia mejorado
MIN_ADX_FOR_TREND=25.0
EMA_ALIGNMENT_REQUIRED=true
MAX_PERCENT_FROM_SMA50=3.0

# Parámetros de filtro de momentum
MAX_MOMENTUM_PERCENT=2.0

# Parámetros de filtro de soporte/resistencia
SR_PROXIMITY_PERCENT=0.5
SR_MAX_DISTANCE_PERCENT=1.5

# Parámetros de filtro de divergencias
DIVERGENCE_RSI_OVERBOUGHT=70.0
DIVERGENCE_RSI_OVERSOLD=30.0
DIVERGENCE_MOMENTUM_THRESHOLD=1.5
```

### Límites de Trading

```bash
MAX_SIGNALS_PER_DAY=10
MAX_SIGNALS_PER_PAIR=3
MT5_MAX_OPEN_POSITIONS=3
```

### Reglas de Índices Sintéticos

```bash
ENFORCE_GAINX_BUY_ONLY=true   # GainX solo BUY
ENFORCE_PAINX_SELL_ONLY=true  # PainX solo SELL
```

---

## 🚀 Optimizaciones y Mejoras Continuas

### Optimizaciones Implementadas

1. **✅ Corrección de volumen:** Usa TICKVOL en lugar de VOL (que está en 0 para sintéticos)
2. **✅ Class weights en LSTM:** Balanceo automático de clases desbalanceadas
3. **✅ Métricas adicionales:** AUC, Precision, Recall en entrenamiento LSTM
4. **✅ Validación de features:** Verifica compatibilidad al cargar modelos
5. **✅ Filtros avanzados configurables:** 5 filtros adicionales desde .env
6. **✅ Modo solo notificaciones:** Telegram sin ejecutar en MT5
7. **✅ Lotaje dinámico:** Ajuste de riesgo según confianza
8. **✅ Break Even y Trailing Stop:** Protección y maximización automática

### Áreas de Mejora Futura

1. **Backtesting Automatizado:**
   - Probar estrategia en datos históricos
   - Calcular métricas (win rate, profit factor, max drawdown)
   - Optimizar parámetros (grid search)

2. **Walk-Forward Analysis:**
   - Reentrenamiento periódico automático
   - Validación out-of-sample continua

3. **Optimización de Hiperparámetros:**
   - Bayesian optimization para parámetros de modelos
   - Grid search para thresholds de filtros

4. **Análisis de Sentimiento:**
   - Integrar noticias/sentiment de Telegram/Twitter
   - Añadir como feature adicional

5. **Visualización Avanzada:**
   - Dashboard web en tiempo real (Streamlit/Dash)
   - Gráficos de equity curve, drawdown, etc.

6. **Alertas Inteligentes:**
   - Notificaciones solo en señales excepcionales
   - Resumen diario/semanal automático

---

## 📞 Soporte

Para consultas, reportar bugs o sugerir mejoras:

- **GitHub Issues:** https://github.com/Willer1285/trading-bot-indices/issues
- **Documentación:** Ver README.md y archivos en `/docs`
- **Logs:** `logs/trading_bot.log` para debugging

---

**⚠️ ADVERTENCIA DE RIESGO:**

El trading involucra riesgo significativo de pérdida de capital. Este bot utiliza inteligencia artificial y análisis técnico, pero **NO garantiza ganancias**.

- Siempre prueba en cuenta demo primero
- No operes con dinero que no puedas permitirte perder
- Monitorea el bot constantemente
- Ajusta parámetros según tus resultados

**Este software es para fines educativos. Usa bajo tu propio riesgo.**

---

**Última actualización:** 17 de Marzo, 2026
**Versión del documento:** 1.0
**Autor:** Trading Bot Indices Team
