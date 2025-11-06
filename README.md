# Trading Bot para Índices Sintéticos (MT5)

Bot de trading automatizado con inteligencia artificial para operar índices sintéticos (GainX/PainX) en MetaTrader 5.

## Características Principales

- **Sistema de IA Avanzado**: Ensemble de modelos ML (Random Forest, Gradient Boosting, LSTM)
- **Meta-Learning**: Utiliza meta-labeling para filtrar señales de alta probabilidad
- **50+ Indicadores Técnicos**: RSI, MACD, Bollinger Bands, ATR, Ichimoku, etc.
- **Análisis Multi-Timeframe**: Confirma señales en múltiples marcos temporales
- **Gestión de Riesgo Profesional**: Stop Loss, Take Profit, Break Even y Trailing Stop dinámicos basados en ATR
- **Notificaciones en Telegram**: Alertas en tiempo real con gráficos
- **Trading Automático**: Ejecución automática de operaciones en MT5

## Arquitectura del Sistema

### Modelos de IA

1. **SimplePatternModel** (Modelo Primario)
   - Modelo basado en reglas técnicas
   - Genera señales iniciales (BUY/SELL/HOLD)
   - Utiliza RSI, MACD y tendencia

2. **Random Forest & Gradient Boosting** (Meta-modelos)
   - Filtran señales del modelo primario
   - Predicen probabilidad de éxito de cada señal
   - Calibrados con CalibratedClassifierCV

3. **LSTM** (Red Neuronal Recurrente)
   - Analiza secuencias temporales de 50 períodos
   - Captura patrones complejos en el precio
   - Entrenado con class weights para balancear datos

4. **Ensemble Stacking**
   - Combina predicciones de todos los modelos
   - Meta-modelo de Logistic Regression para decisión final
   - Genera confianza ponderada de la señal

### Proceso de Predicción

```
Datos OHLCV → Extracción de Features (116) → Modelo Primario → Señal (BUY/SELL/HOLD)
                                                                          ↓
                                                                   Meta-modelos
                                                                          ↓
                                                              Confianza (0.0 - 1.0)
                                                                          ↓
                                                                 Filtros de Calidad
                                                                          ↓
                                                             Ejecución en MT5 (si > 75%)
```

## Requisitos del Sistema

### Software
- Python 3.11+
- MetaTrader 5 (Windows o Wine en Linux)
- Cuenta de trading MT5 (demo o real)
- Bot de Telegram (opcional pero recomendado)

### Hardware
- CPU: 4+ cores (para entrenamiento de modelos)
- RAM: 8GB+ (16GB recomendado para entrenamiento)
- Almacenamiento: 5GB+ para datos y modelos

## Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/trading-bot-indices.git
cd trading-bot-indices
```

### 2. Crear Entorno Virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno

Copia el archivo `.env.example` a `.env` y completa los valores:

```bash
cp .env.example .env
```

Edita `.env` con tus credenciales:

```bash
# Telegram
TELEGRAM_BOT_TOKEN=tu_token_de_botfather
TELEGRAM_CHANNEL_ID=tu_channel_id

# MT5
MT5_LOGIN=12345678
MT5_PASSWORD=tu_contraseña
MT5_SERVER=Weltrade-Demo

# Trading
TRADING_SYMBOLS=GainX 400,GainX 600,PainX 400
TIMEFRAMES=1m,5m,15m,1h,4h,1d
CONFIDENCE_THRESHOLD=0.75

# Auto-trading (IMPORTANTE: false = solo notificaciones)
MT5_AUTO_TRADING=false
```

### 5. Crear Directorios Necesarios

```bash
mkdir -p logs models
```

## Uso

### Entrenar Modelos de IA

Antes de usar el bot, debes entrenar los modelos con datos históricos:

```bash
python train_models.py
```

Este proceso:
- Escanea `historical_data/` buscando archivos CSV
- Extrae 116 características técnicas por símbolo/timeframe
- Entrena 4 modelos por cada combinación
- Guarda los modelos en `models/{símbolo}/{timeframe}/`

**Tiempo estimado**: 30-60 minutos para 10 símbolos × 6 timeframes

### Ejecutar el Bot

```bash
python run_mt5.py
```

El bot:
1. Se conecta a MT5
2. Carga modelos entrenados
3. Analiza mercados cada 60 segundos
4. Genera señales de trading
5. Ejecuta operaciones (si `MT5_AUTO_TRADING=true`)
6. Envía notificaciones a Telegram
7. Gestiona posiciones abiertas (Break Even, Trailing Stop)

### Modo Solo Notificaciones

Para usar el bot sin ejecutar operaciones reales:

```bash
# En .env
MT5_AUTO_TRADING=false
```

Recibirás alertas en Telegram pero el bot NO abrirá operaciones.

## Estructura del Proyecto

```
trading-bot-indices/
├── src/
│   ├── ai_engine/              # Motor de IA
│   │   ├── ai_models.py        # Modelos ML (RF, GB, LSTM, Ensemble)
│   │   ├── feature_engineering.py  # Extracción de features
│   │   ├── technical_indicators.py  # 50+ indicadores
│   │   └── market_analyzer.py  # Análisis de mercado
│   ├── data_collector/         # Recolección de datos
│   │   └── mt5_connector.py    # Conexión con MT5
│   ├── signal_generator/       # Generación de señales
│   │   ├── signal_generator.py
│   │   ├── signal_filter.py    # Filtros de calidad
│   │   └── risk_manager.py     # Gestión de riesgo
│   ├── telegram_bot/           # Bot de Telegram
│   │   ├── telegram_bot.py
│   │   └── chart_generator.py
│   ├── config.py               # Configuración centralizada
│   └── main_mt5.py             # Loop principal
├── historical_data/            # Datos históricos CSV
├── models/                     # Modelos entrenados
├── logs/                       # Logs del bot
├── train_models.py             # Script de entrenamiento
├── run_mt5.py                  # Launcher del bot
├── requirements.txt            # Dependencias Python
├── .env.example                # Plantilla de configuración
└── README.md                   # Este archivo
```

## Datos Históricos

### Formato CSV Esperado

Los archivos deben estar en `historical_data/{Símbolo}/` con formato:

```
<DATE>  <TIME>  <OPEN>  <HIGH>  <LOW>  <CLOSE>  <TICKVOL>  <VOL>  <SPREAD>
2024.01.01  00:00:00  100000.00  100050.00  99950.00  100020.00  1234  0  5
```

**Importante**: El bot usa `<TICKVOL>` como volumen, no `<VOL>` (que suele estar en 0 para índices sintéticos).

### Obtener Datos Históricos

1. **Desde MT5**: Exportar datos históricos a CSV
2. **Script personalizado**: Usar `mt5_connector.py` para descargar datos
3. **Formato**: Tab-separated, columnas con `<>`, fechas `YYYY.MM.DD`

## Configuración Avanzada

### Gestión de Riesgo

```bash
# Stop Loss y Take Profit dinámicos basados en ATR
STOP_LOSS_ATR_MULTIPLIER=1.5    # SL a 1.5 × ATR
TAKE_PROFIT_1_ATR_MULTIPLIER=2.0  # TP1 a 2.0 × ATR
TAKE_PROFIT_2_ATR_MULTIPLIER=4.0  # TP2 a 4.0 × ATR

# Break Even (mover SL a punto de equilibrio)
ENABLE_BREAK_EVEN=true
BREAK_EVEN_TRIGGER_ATR_MULTIPLIER=1.0  # Activar cuando profit > 1.0 × ATR
BREAK_EVEN_PROFIT_LOCK_ATR_MULTIPLIER=0.2  # Asegurar 0.2 × ATR de ganancia

# Trailing Stop (SL dinámico que sigue el precio)
ENABLE_TRAILING_STOP=true
TRAILING_STOP_TRIGGER_ATR_MULTIPLIER=2.0  # Activar cuando profit > 2.0 × ATR
TRAILING_STOP_DISTANCE_ATR_MULTIPLIER=1.5  # Distancia del SL: 1.5 × ATR
```

### Lotaje Dinámico

```bash
ENABLE_DYNAMIC_LOT_SIZE=true
MIN_LOT_SIZE=0.10
MAX_LOT_SIZE=1.00
```

El tamaño del lote se ajusta según la confianza de la señal:
- 75% confianza → 0.10 lotes
- 85% confianza → 0.55 lotes
- 95%+ confianza → 1.00 lotes

### Reglas de Índices Sintéticos

```bash
# GainX solo señales BUY, PainX solo señales SELL
ENFORCE_GAINX_BUY_ONLY=true
ENFORCE_PAINX_SELL_ONLY=true
```

## Filtros de Calidad de Señales

El bot aplica múltiples filtros antes de ejecutar una señal:

1. **Umbral de Confianza**: ≥ 75% (configurable)
2. **Confluencia Multi-Timeframe**: ≥ 50% de timeframes de acuerdo
3. **Alineación de Tendencia**: Precio alineado con SMA50 del timeframe superior
4. **Volatilidad Aceptable**: ATR < 5% del precio
5. **Sin Señales Conflictivas**: No señales opuestas en la última hora
6. **Límites Diarios**: Max 10 señales/día, max 3 por par

## Monitoreo y Logs

### Logs del Bot

Los logs se guardan en `logs/trading_bot.log` con rotación automática:

```bash
tail -f logs/trading_bot.log
```

### Métricas de Modelos

Durante el entrenamiento, el bot registra:
- Distribución de clases (balanceo)
- Class weights aplicados
- Métricas por epoch (Loss, Accuracy, AUC, Precision, Recall)
- Métricas finales de validación

### Performance Tracking

El bot registra cada señal generada con:
- Timestamp
- Símbolo y timeframe
- Dirección (BUY/SELL)
- Confianza
- Parámetros de riesgo (SL, TP)
- Resultado (si se ejecutó)

## Solución de Problemas

### El bot no se conecta a MT5

- Verifica que MT5 esté ejecutándose
- Confirma credenciales en `.env`
- En Windows: Asegúrate de que MT5 permita conexiones de API
- En Linux: Verifica que Wine esté configurado correctamente

### Modelos no se cargan

- Verifica que existan archivos en `models/{símbolo}/{timeframe}/`
- Ejecuta `python train_models.py` para reentrenar
- Revisa logs en `logs/trading_bot.log`

### No se generan señales

- Verifica que `CONFIDENCE_THRESHOLD` no sea muy alto (recomendado: 0.75)
- Comprueba que haya suficientes datos históricos (200+ velas por timeframe)
- Revisa filtros de calidad en `signal_filter.py`

### Accuracy constante en LSTM

- Esto indica datos desbalanceados
- El bot ahora aplica class_weights automáticamente
- Verifica distribución de clases en logs de entrenamiento

## Mejores Prácticas

### 1. Comienza en Demo

Siempre prueba el bot en una cuenta demo antes de usar dinero real.

```bash
MT5_AUTO_TRADING=false
```

### 2. Reentrenamiento Periódico

Reentrena modelos cada 1-3 meses para adaptarse al mercado:

```bash
python train_models.py
```

### 3. Monitoreo Constante

- Revisa logs diariamente
- Analiza señales generadas vs ejecutadas
- Ajusta `CONFIDENCE_THRESHOLD` según resultados

### 4. Gestión de Capital

- No uses más del 1-2% del capital por operación
- Ajusta `MAX_LOT_SIZE` según tu balance
- Mantén `MAX_SIGNALS_PER_DAY` conservador (≤10)

### 5. Backtesting

Antes de trading en vivo:
- Analiza historial de señales
- Calcula win rate y risk/reward
- Ajusta parámetros según resultados

## Seguridad

- **Nunca** compartas tu archivo `.env`
- **Nunca** subas credenciales a Git (`.env` está en `.gitignore`)
- Usa cuentas demo para pruebas
- Revisa operaciones manualmente antes de activar auto-trading

## Soporte y Contribuciones

- **Issues**: https://github.com/Willer1285/trading-bot-indices/issues
- **Documentación**: Consulta este README y comentarios en el código
- **Contribuciones**: Pull requests bienvenidos

## Licencia

Este proyecto es para uso educativo y personal. Usa bajo tu propio riesgo.

**ADVERTENCIA**: Trading involucra riesgo significativo de pérdida de capital. Este bot no garantiza ganancias. Siempre prueba en demo primero.

## Changelog

### v1.1.0 (2024-11-06) - Análisis Exhaustivo y Mejoras
- ✅ Corrección de problema de volumen (usar TICKVOL en lugar de VOL)
- ✅ Mejora de entrenamiento LSTM con class weights automáticos
- ✅ Métricas adicionales en LSTM (AUC, Precision, Recall)
- ✅ Validación de features en carga de modelos
- ✅ Corrección de versión de tensorflow en requirements
- ✅ Documentación completa en README
- ✅ Archivo .env.example con todas las variables
- ✅ Análisis exhaustivo del proyecto documentado

### v1.0.0 (2024-01-01)
- Lanzamiento inicial
- Sistema de ensemble con meta-learning
- 50+ indicadores técnicos
- Gestión de riesgo dinámica
- Integración con Telegram y MT5

---

**Desarrollado con ❤️ para traders algorítmicos**
