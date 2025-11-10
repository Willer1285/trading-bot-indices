# 🔄 Sistema de Reentrenamiento Automático de Modelos

## 📋 Descripción General

El bot ahora incluye un sistema completamente automatizado para mantener los modelos de IA actualizados. El sistema descarga datos frescos desde MT5 y reentrena los modelos automáticamente cuando están desactualizados.

---

## ✨ Características

### ✅ **Reentrenamiento Automático**
- Verifica la edad de los modelos al iniciar el bot
- Descarga las últimas N velas directamente desde MT5
- Reentrena modelos si tienen más de X días de antigüedad
- Guarda metadata con timestamp de cada entrenamiento

### ✅ **Reentrenamiento Manual**
- Ejecuta reentrenamiento cuando lo necesites
- Dos modos: desde MT5 o desde archivos locales
- Total control sobre el proceso

### ✅ **Configuración Flexible**
- Símbolos y timeframes configurables en `.env`
- Número de velas configurable
- Edad máxima configurable
- Se puede activar/desactivar

---

## 🚀 Modo de Uso

### **Opción 1: Reentrenamiento Automático (Recomendado)**

El bot verifica y reentrena automáticamente al iniciar si es necesario.

**Configuración en `.env`:**
```bash
# Activar reentrenamiento automático
ENABLE_AUTO_RETRAIN=true

# Reentrenar si modelos tienen más de 7 días
AUTO_RETRAIN_DAYS=7

# Descargar últimas 5000 velas por símbolo/timeframe
RETRAIN_CANDLES=5000

# Símbolos y timeframes (ya configurados)
TRADING_SYMBOLS="PainX 400,GainX 400,PainX 600,GainX 600,PainX 800,GainX 800,PainX 999,GainX 999,PainX 1200,GainX 1200"
TIMEFRAMES="1m,1h"
```

**Flujo:**
1. Inicias el bot con `run_bot.bat`
2. El bot verifica la edad de los modelos
3. Si están desactualizados (>7 días):
   - Descarga 5000 velas de MT5 para cada símbolo/timeframe
   - Reentrena los 20 modelos automáticamente
   - Guarda los nuevos modelos con timestamp
4. Carga los modelos y continúa operando normalmente

---

### **Opción 2: Reentrenamiento Manual desde MT5**

Ejecuta el reentrenamiento manualmente cuando lo desees.

**Comando:**
```bash
# En la raíz del proyecto (con venv activado)
python train_models.py --source mt5
```

**Lo que hace:**
- Se conecta a MT5 con tus credenciales del `.env`
- Descarga las últimas `RETRAIN_CANDLES` velas para cada símbolo/timeframe
- Entrena los modelos de IA con los datos frescos
- Guarda los modelos con metadata (timestamp, número de velas, etc.)
- Elimina modelos viejos (los reemplaza con los nuevos)

**Ejemplo de salida:**
```
=============================== Training Models from MT5 Data ===============================
MT5 initialized successfully
Logged in to MT5 account 12345678

Will train models for 10 symbols × 2 timeframes = 20 models
Downloading 5000 candles per symbol/timeframe

--- Training GainX 400 [1m] ---
✅ Downloaded 5000 candles for GainX 400 [1m] from MT5
Preparing data for training with Meta-Labeling...
Training Ensemble model...
✅ Ensemble model trained successfully
✅ Finished training GainX 400 [1m]

--- Training GainX 400 [1h] ---
✅ Downloaded 5000 candles for GainX 400 [1h] from MT5
...

================================================================================
Training complete: 20 models trained, 0 failed
================================================================================
```

---

### **Opción 3: Reentrenamiento Manual desde Archivos Locales**

Usa tus archivos CSV históricos guardados localmente.

**Comando:**
```bash
# En la raíz del proyecto (con venv activado)
python train_models.py --source local
```

**Requisito:**
- Necesitas tener archivos CSV en `historical_data/` con el formato:
  ```
  historical_data/
  ├── GainX 400/
  │   ├── GainX 400_M1.csv
  │   └── GainX 400_H1.csv
  ├── PainX 400/
  │   ├── PainX 400_M1.csv
  │   └── PainX 400_H1.csv
  ...
  ```

---

## ⚙️ Variables de Configuración

### **En `.env`:**

| Variable | Valor | Descripción |
|----------|-------|-------------|
| `ENABLE_AUTO_RETRAIN` | `true` / `false` | Activar/desactivar reentrenamiento automático |
| `AUTO_RETRAIN_DAYS` | `7` (default) | Edad máxima de modelos en días antes de reentrenar |
| `RETRAIN_CANDLES` | `5000` (default) | Número de velas a descargar desde MT5 |
| `MODELS_DIRECTORY` | `models` (default) | Carpeta donde se guardan los modelos |
| `TRADING_SYMBOLS` | `"PainX 400,GainX 400,..."` | Símbolos para entrenar (separados por coma) |
| `TIMEFRAMES` | `"1m,1h"` | Timeframes para entrenar (separados por coma) |

---

## 📊 Metadata de Modelos

Cada modelo entrenado incluye un archivo `training_metadata.json`:

```json
{
  "symbol": "GainX 600",
  "timeframe": "1m",
  "trained_at": "2025-11-10T15:30:45.123456",
  "num_records": 5000,
  "source": "mt5",
  "candles_used": 5000
}
```

**Utilidad:**
- El bot lee este archivo para saber la edad del modelo
- Permite rastrear cuándo y cómo se entrenó cada modelo
- Ayuda en debugging y auditoría

---

## 🔍 Verificación de Edad de Modelos

### **Automático (al iniciar el bot):**

```
Checking model age for automatic retraining...
Model for GainX 400 [1m] is 3 days old
Model for GainX 400 [1h] is 3 days old
Model for PainX 400 [1m] is 3 days old
...
✅ Models are up to date (oldest model: 3 days old)
```

### **Si necesita reentrenar:**

```
Checking model age for automatic retraining...
Model for GainX 400 [1m] is 8 days old
⚠️  Oldest model is 8 days old (threshold: 7 days)

================================================================================
⚠️  AUTOMATIC RETRAINING TRIGGERED
Will download 5000 candles from MT5 for each symbol/timeframe
Symbols: GainX 400, PainX 400, ...
Timeframes: 1m, 1h
This may take 15-30 minutes...
================================================================================

[Proceso de reentrenamiento...]

✅ Automatic retraining completed successfully!
Models have been updated with fresh data from MT5
```

---

## 📝 Ejemplos de Uso

### **Ejemplo 1: Configuración Recomendada (Trading Real)**

```bash
# .env
ENABLE_AUTO_RETRAIN=true
AUTO_RETRAIN_DAYS=7          # Reentrenar semanalmente
RETRAIN_CANDLES=5000         # Suficiente para capturar patrones

# Inicias el bot normalmente:
run_bot.bat

# El bot verifica y reentrena automáticamente si necesario
```

---

### **Ejemplo 2: Desactivar Reentrenamiento Automático**

```bash
# .env
ENABLE_AUTO_RETRAIN=false

# El bot nunca reentrena automáticamente
# Debes reentrenar manualmente cuando quieras:
python train_models.py --source mt5
```

---

### **Ejemplo 3: Reentrenar Cada 3 Días con Más Velas**

```bash
# .env
ENABLE_AUTO_RETRAIN=true
AUTO_RETRAIN_DAYS=3          # Más frecuente
RETRAIN_CANDLES=10000        # Más datos históricos

# Modelos se actualizan cada 3 días con 10,000 velas
```

---

### **Ejemplo 4: Solo Reentrenar Manualmente los Fines de Semana**

```bash
# .env
ENABLE_AUTO_RETRAIN=false

# Los sábados o domingos:
python train_models.py --source mt5

# Luego inicias el bot con modelos frescos
run_bot.bat
```

---

## 🎯 Mejores Prácticas

### **1. Reentrenamiento Semanal (Recomendado)**
```bash
AUTO_RETRAIN_DAYS=7
RETRAIN_CANDLES=5000
```
- ✅ Balance entre actualización y estabilidad
- ✅ 5000 velas capturan patrones recientes sin sobreajuste
- ✅ Semanal es suficiente para índices sintéticos

### **2. Reentrenamiento Quincenal (Más Estable)**
```bash
AUTO_RETRAIN_DAYS=14
RETRAIN_CANDLES=7000
```
- ✅ Menos reentrenamientos = más estabilidad
- ✅ Más velas = mayor contexto histórico
- ✅ Bueno para cuentas en producción

### **3. Reentrenamiento Cada 3 Días (Agresivo)**
```bash
AUTO_RETRAIN_DAYS=3
RETRAIN_CANDLES=3000
```
- ⚠️ Modelos muy adaptados a condiciones recientes
- ⚠️ Puede perder patrones de largo plazo
- ⚠️ Usar solo en backtesting o cuentas demo

### **4. Desactivado (Control Total)**
```bash
ENABLE_AUTO_RETRAIN=false
```
- ✅ Tú decides cuándo reentrenar
- ✅ Útil durante optimización de parámetros
- ⚠️ Requiere disciplina para reentrenar manualmente

---

## 🛠️ Troubleshooting

### **Problema 1: "No se pudo conectar a MT5"**

**Causa:** Credenciales incorrectas o MT5 cerrado

**Solución:**
```bash
# Verifica tu .env:
MT5_LOGIN=12345678
MT5_PASSWORD=tu_contraseña
MT5_SERVER=Weltrade-Demo

# Asegúrate que MT5 esté abierto y en la cuenta correcta
```

---

### **Problema 2: "Reentrenamiento tomó más de 1 hora y falló"**

**Causa:** Demasiadas velas o conexión lenta

**Solución:**
```bash
# Reduce el número de velas:
RETRAIN_CANDLES=3000

# O reentrena manualmente con menos símbolos/timeframes
```

---

### **Problema 3: "Modelos no se actualizan"**

**Causa:** Metadata corrupta o `ENABLE_AUTO_RETRAIN=false`

**Solución 1: Verificar configuración**
```bash
# En .env:
ENABLE_AUTO_RETRAIN=true
```

**Solución 2: Eliminar modelos y reentrenar**
```bash
# Elimina la carpeta models (backup primero!)
# Luego reentrena manualmente:
python train_models.py --source mt5
```

---

### **Problema 4: "Failed to download data for [símbolo]"**

**Causa:** Símbolo no disponible en tu broker o nombre incorrecto

**Solución:**
```bash
# Verifica que el símbolo exista en MT5
# Algunos brokers usan nombres diferentes:
# - "Volatility 75" vs "GainX 400"
# - "Crash 500" vs "PainX 999"

# Verifica los símbolos disponibles en los logs del bot
```

---

## 📊 Estadísticas de Rendimiento

### **Tiempos Estimados de Reentrenamiento:**

| Configuración | Tiempo Estimado | Modelos | Total Velas |
|---------------|----------------|---------|-------------|
| **10 símbolos × 2 TF × 5000 velas** | 15-20 min | 20 modelos | 100,000 |
| **10 símbolos × 2 TF × 10000 velas** | 25-35 min | 20 modelos | 200,000 |
| **5 símbolos × 2 TF × 5000 velas** | 8-12 min | 10 modelos | 50,000 |
| **10 símbolos × 4 TF × 5000 velas** | 30-45 min | 40 modelos | 200,000 |

**Factores que afectan el tiempo:**
- Velocidad de CPU (entrenamiento de ML es intensivo)
- Velocidad de conexión a MT5 (descarga de datos)
- Número de velas (más velas = más tiempo de entrenamiento)

---

## 🔐 Seguridad y Backup

### **Backup Automático de Modelos Viejos:**

El sistema **NO hace backup automático**. Los modelos viejos se reemplazan directamente.

**Recomendación:**
```bash
# Antes de reentrenar, haz backup manual:
cp -r models models_backup_2025-11-10

# O crea un script de backup automático
```

---

## 📚 Referencias

- **Archivo de configuración:** `src/config.py` (líneas 72-76)
- **Script de entrenamiento:** `train_models.py`
- **Verificación de edad:** `src/main_mt5.py` (método `_check_and_retrain_models`)
- **Metadata de modelos:** `models/[símbolo]/[timeframe]/training_metadata.json`

---

## 💡 Consejos Avanzados

### **1. Reentrenamiento Programado (Cron/Task Scheduler)**

En lugar de reentrenar al iniciar el bot, programa un reentrenamiento semanal:

**Linux (cron):**
```bash
# Editar crontab
crontab -e

# Reentrenar cada domingo a las 00:00
0 0 * * 0 cd /ruta/al/bot && source venv_trading/bin/activate && python train_models.py --source mt5
```

**Windows (Task Scheduler):**
```bash
# Crear tarea programada:
# - Trigger: Semanal, domingos, 00:00
# - Action: python.exe
# - Arguments: train_models.py --source mt5
# - Start in: C:\ruta\al\bot
```

### **2. Notificaciones de Reentrenamiento**

Modifica `train_models.py` para enviar notificación Telegram al terminar:

```python
# Al final de train_from_mt5():
from src.telegram_bot.telegram_bot import TelegramBot

telegram = TelegramBot(config.telegram_bot_token, config.telegram_channel_id)
await telegram.send_message(f"✅ Reentrenamiento completado: {trained_count} modelos actualizados")
```

### **3. Métricas de Modelos**

Guarda métricas de accuracy en metadata para comparar:

```json
{
  "trained_at": "2025-11-10T15:30:45",
  "accuracy": 0.68,
  "precision": 0.72,
  "recall": 0.65
}
```

---

## 🎓 Conclusión

El sistema de reentrenamiento automático mantiene tus modelos de IA actualizados sin intervención manual. Con la configuración adecuada, tu bot siempre operará con datos frescos y patrones recientes del mercado.

**Configuración recomendada:**
```bash
ENABLE_AUTO_RETRAIN=true
AUTO_RETRAIN_DAYS=7
RETRAIN_CANDLES=5000
```

**Para mayor control:**
```bash
ENABLE_AUTO_RETRAIN=false
# Y reentrena manualmente: python train_models.py --source mt5
```

¡Mantén tus modelos frescos y tu bot rentable! 🚀📈
