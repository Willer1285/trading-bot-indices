# 🚀 Hybrid AI System (TFT + RL + LLM) - Guía de Uso

## 📋 Descripción General

Este sistema híbrido combina tres tecnologías de IA avanzadas para optimizar la rentabilidad del trading bot:

1. **TFT (Temporal Fusion Transformer)**: Predicción avanzada de series temporales con mecanismos de atención
2. **RL (Reinforcement Learning)**: Optimización de decisiones de trading mediante aprendizaje por refuerzo
3. **LLM (Large Language Models)**: Análisis de sentimiento del mercado a través de noticias financieras

## 🎯 Ventajas del Sistema Híbrido

### TFT (Temporal Fusion Transformer)
- ✅ Captura patrones complejos en series temporales
- ✅ Maneja múltiples horizontes de predicción
- ✅ Atención multi-cabeza para features importantes
- ✅ Estimación de incertidumbre mediante quantiles

### RL (Reinforcement Learning)
- ✅ Aprende la política óptima de trading
- ✅ Considera secuencias de decisiones (no solo señales aisladas)
- ✅ Balancea exploración y explotación
- ✅ Optimiza reward a largo plazo

### LLM Sentiment
- ✅ Incorpora información fundamental del mercado
- ✅ Detecta eventos y noticias importantes
- ✅ Análisis de sentimiento con FinBERT (especializado en finanzas)
- ✅ Complementa el análisis técnico

## 📦 Instalación de Dependencias

```bash
# Instalar dependencias adicionales para el sistema híbrido
pip install torch>=2.5.0 \
    pytorch-lightning>=2.4.0 \
    pytorch-forecasting>=1.1.0 \
    stable-baselines3>=2.3.0 \
    gymnasium>=1.0.0 \
    transformers>=4.47.0 \
    sentencepiece>=0.2.0 \
    accelerate>=1.2.0 \
    newsapi-python>=0.2.7 \
    yfinance>=0.2.50 \
    finnhub-python>=2.4.20
```

O simplemente:

```bash
pip install -r requirements.txt
```

## ⚙️ Configuración

### Variables de Entorno (.env)

Añade las siguientes variables a tu archivo `.env`:

```bash
# ========== HYBRID AI CONFIGURATION ==========

# Activar sistema híbrido
ENABLE_HYBRID_AI=true

# TFT Configuration
TFT_ENABLED=true
TFT_ENCODER_LENGTH=60
TFT_PREDICTION_LENGTH=10
TFT_HIDDEN_SIZE=64
TFT_LSTM_LAYERS=2
TFT_ATTENTION_HEADS=4
TFT_DROPOUT=0.1

# RL Configuration
RL_ENABLED=true
RL_ALGORITHM=DQN  # Options: DQN, PPO, A2C
RL_LEARNING_RATE=0.0001
RL_BUFFER_SIZE=100000
RL_BATCH_SIZE=32
RL_GAMMA=0.99
RL_TRAINING_TIMESTEPS=50000

# LLM Sentiment Configuration
LLM_ENABLED=true
LLM_MODEL_NAME=ProsusAI/finbert
NEWS_API_KEY=your_newsapi_key_here  # Get free key at https://newsapi.org
LLM_SENTIMENT_CACHE_HOURS=1
LLM_NEWS_MAX_ARTICLES=20
LLM_SENTIMENT_WEIGHT=0.3

# Model Paths
HYBRID_MODEL_PATH=models/hybrid
```

### Obtener API Key de NewsAPI

1. Visita https://newsapi.org
2. Regístrate gratis (70,000 requests/mes)
3. Copia tu API key
4. Añádela a `.env` como `NEWS_API_KEY`

## 🏋️ Entrenamiento del Sistema Híbrido

### Entrenamiento Completo

```bash
# Entrenar todos los modelos (TFT + RL + LLM)
python train_hybrid_models.py --symbol SPY --timeframe 1h
```

### Entrenamiento Selectivo

```bash
# Solo TFT y LLM (sin RL)
python train_hybrid_models.py --symbol SPY --timeframe 1h --no-rl

# Solo RL (sin TFT ni LLM)
python train_hybrid_models.py --symbol SPY --timeframe 1h --no-tft --no-llm

# Guardar en ruta específica
python train_hybrid_models.py --symbol SPY --timeframe 1h --save-path models/custom_hybrid
```

### Parámetros de Entrenamiento

| Parámetro | Descripción | Por Defecto |
|-----------|-------------|-------------|
| `--symbol` | Símbolo de trading | SPY |
| `--timeframe` | Marco temporal (1m, 5m, 15m, 30m, 1h, 4h, 1d) | 1h |
| `--no-tft` | Deshabilitar TFT | False |
| `--no-rl` | Deshabilitar RL | False |
| `--no-llm` | Deshabilitar LLM | False |
| `--save-path` | Ruta para guardar modelos | models/hybrid |

## 🚀 Uso del Sistema Híbrido

### Cargar y Usar Modelos

```python
from src.ai_engine.ai_models import EnsembleModel

# Crear ensemble híbrido
ensemble = EnsembleModel(use_hybrid=True)

# Cargar modelos entrenados
ensemble.load_all("models/hybrid")

# Hacer predicciones
predictions = ensemble.predict(X)
probabilities = ensemble.predict_proba(X)
```

### Integración con el Bot Existente

El sistema híbrido se integra automáticamente si `ENABLE_HYBRID_AI=true` en `.env`:

```python
# En train_models.py, usa:
from src.config import config

ensemble = EnsembleModel(use_hybrid=config.enable_hybrid_ai)
```

## 📊 Evaluación y Monitoreo

### Logs de Entrenamiento

Los logs se guardan en:
- `logs/hybrid_training.log`: Log detallado del entrenamiento
- `tensorboard_logs/`: Métricas de TensorBoard para RL

### Visualizar Entrenamiento RL

```bash
tensorboard --logdir tensorboard_logs/
```

Abre http://localhost:6006 en tu navegador.

### Métricas del Sistema Híbrido

```python
# Evaluar RL agent
from src.ai_engine.rl_agent import RLTradingAgent

agent = RLTradingAgent()
agent.load("models/hybrid/rl_agent.pkl")
metrics = agent.evaluate(test_data, n_episodes=10)

print(metrics)
# {
#   'mean_return': 15.2,
#   'std_return': 5.3,
#   'mean_reward': 450.2,
#   'min_return': 5.1,
#   'max_return': 25.8
# }
```

## 🔧 Troubleshooting

### Error: "pytorch-forecasting not installed"

```bash
pip install pytorch-forecasting>=1.1.0
```

### Error: "stable-baselines3 not installed"

```bash
pip install stable-baselines3>=2.3.0 gymnasium>=1.0.0
```

### Error: "transformers not installed"

```bash
pip install transformers>=4.47.0 torch>=2.5.0
```

### LLM no encuentra noticias

- Verifica que `NEWS_API_KEY` esté configurado correctamente
- El LLM funcionará con Yahoo Finance como fallback
- Revisa los logs para ver si hay errores de API

### GPU no detectado (CUDA)

```bash
# Verificar CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Instalar PyTorch con CUDA (si tienes GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 🎓 Arquitectura del Sistema

```
Hybrid AI System
│
├── TFT (Temporal Fusion Transformer)
│   ├── Variable Selection Networks
│   ├── Static Covariate Encoders
│   ├── LSTM Sequence Layer
│   ├── Multi-Head Attention
│   └── Quantile Forecasting
│
├── RL Agent (Reinforcement Learning)
│   ├── Trading Environment (Gym)
│   ├── DQN/PPO/A2C Algorithm
│   ├── Replay Buffer
│   └── Policy Network
│
├── LLM Sentiment Analyzer
│   ├── FinBERT Pre-trained Model
│   ├── News Fetching (NewsAPI/Yahoo)
│   ├── Sentiment Classification
│   └── Feature Engineering
│
└── Meta-Model (Ensemble)
    ├── Base Model Predictions
    ├── Stacking Layer
    └── Final Signal Generation
```

## 📈 Resultados Esperados

Con el sistema híbrido, se espera:

- ✅ **Mejor accuracy**: +10-15% en predicciones
- ✅ **Menor drawdown**: -20-30% reducción de pérdidas
- ✅ **Mayor Sharpe Ratio**: +0.5-1.0 puntos
- ✅ **Señales más robustas**: Confirmación multi-modelo
- ✅ **Mejor timing**: TFT captura patrones temporales complejos
- ✅ **Contexto fundamental**: LLM añade información de mercado

## 🔄 Actualización de Modelos

Se recomienda re-entrenar los modelos:

- **TFT**: Cada semana (para capturar nuevos patrones)
- **RL**: Cada 2 semanas (aprendizaje continuo)
- **LLM**: No requiere re-entrenamiento (usa modelo pre-entrenado)

```bash
# Script de actualización automatizada
./scripts/update_hybrid_models.sh
```

## 📚 Referencias

- **TFT Paper**: [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)
- **DQN Paper**: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- **FinBERT**: [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/abs/1908.10063)

## 💡 Tips y Mejores Prácticas

1. **Comienza con pocos datos**: Entrena primero con 1000 barras para verificar que todo funciona
2. **Usa GPU si es posible**: Acelera el entrenamiento 10-50x
3. **Monitorea el overfitting**: Valida en datos out-of-sample
4. **Combina señales**: El sistema híbrido es más robusto que modelos individuales
5. **Ajusta hyperparameters**: Usa grid search para optimizar parámetros

## 🤝 Soporte

Si tienes problemas o preguntas:

1. Revisa los logs en `logs/hybrid_training.log`
2. Verifica la configuración en `.env`
3. Consulta este documento
4. Abre un issue en el repositorio

---

**¡Feliz Trading con AI Híbrida! 🚀📈**
