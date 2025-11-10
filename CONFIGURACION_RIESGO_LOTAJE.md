# ⚙️ Configuración de Riesgo y Lotaje (Fijo/Dinámico)

## 📋 Descripción General

El bot ahora permite configurar dos modos independientes para la gestión de riesgo y lotaje:

1. **Gestión de Riesgo:** Stop Loss y Take Profit (dinámico o fijo)
2. **Gestión de Lotaje:** Tamaño del lote (dinámico o fijo)

---

## 🎯 Gestión de Riesgo (SL/TP)

### Modo Dinámico (Basado en ATR) ✅ Recomendado

El SL y TP se calculan automáticamente usando el **ATR (Average True Range)** del mercado:

```bash
# En tu archivo .env:
ENABLE_DYNAMIC_RISK=true
STOP_LOSS_ATR_MULTIPLIER=1.5
TAKE_PROFIT_1_ATR_MULTIPLIER=2.5
TAKE_PROFIT_2_ATR_MULTIPLIER=5.0
```

**Ventajas:**
- ✅ Se adapta automáticamente a la volatilidad del mercado
- ✅ Mercados volátiles = SL/TP más amplios
- ✅ Mercados tranquilos = SL/TP más ajustados
- ✅ Mejor relación Riesgo/Recompensa

**Cálculo:**
```
SL Distance = ATR × 1.5
TP1 Distance = ATR × 2.5
TP2 Distance = ATR × 5.0
```

**Ejemplo Real:**
```
GainX 600, ATR = 15.0
- Stop Loss = Entry - (15.0 × 1.5) = Entry - 22.5
- Take Profit 1 = Entry + (15.0 × 2.5) = Entry + 37.5
- Take Profit 2 = Entry + (15.0 × 5.0) = Entry + 75.0
- Risk/Reward Ratio = 2.5:1 / 5.0:1
```

---

### Modo Fijo (Puntos Manuales)

El SL y TP se configuran con **valores fijos en puntos** que nunca cambian:

```bash
# En tu archivo .env:
ENABLE_DYNAMIC_RISK=false
FIXED_STOP_LOSS_POINTS=50.0
FIXED_TAKE_PROFIT_1_POINTS=125.0
FIXED_TAKE_PROFIT_2_POINTS=250.0
```

**Ventajas:**
- ✅ Control total sobre los niveles de SL/TP
- ✅ Predictibilidad absoluta
- ✅ Útil para backtesting

**Desventajas:**
- ⚠️ No se adapta a cambios de volatilidad
- ⚠️ Puede ser demasiado ajustado en mercados volátiles
- ⚠️ Puede ser demasiado amplio en mercados tranquilos

**Conversión Puntos → Precio:**
```
Para índices sintéticos: 1 punto = 1.0 en el precio

Ejemplo:
FIXED_STOP_LOSS_POINTS=50.0
→ SL Distance = 50.0 × 1.0 = 50.0 en precio

Si el precio de entrada es 9865.00:
- BUY: SL = 9865.00 - 50.0 = 9815.00
- SELL: SL = 9865.00 + 50.0 = 9915.00
```

---

## 💰 Gestión de Lotaje

### Modo Dinámico (Basado en Confianza) ✅ Recomendado

El tamaño del lote se ajusta según la **confianza del modelo** (60% - 100%):

```bash
# En tu archivo .env:
ENABLE_DYNAMIC_LOT_SIZE=true
MIN_LOT_SIZE=0.10
MAX_LOT_SIZE=1.00
CONFIDENCE_THRESHOLD=0.75
```

**Ventajas:**
- ✅ Señales de alta confianza = Lotes mayores
- ✅ Señales de baja confianza = Lotes menores
- ✅ Gestión automática del riesgo por trade

**Cálculo:**
```python
confidence = 75% (0.75)
threshold = 75% (0.75)
max_conf = 100% (1.0)

# Escalar confianza
scaled = (0.75 - 0.75) / (1.0 - 0.75) = 0 / 0.25 = 0.0

# Calcular lote
lot = 0.10 + (0.0 × (1.00 - 0.10))
lot = 0.10 + (0.0 × 0.90)
lot = 0.10

# Si confianza fuera 87.5%:
scaled = (0.875 - 0.75) / (1.0 - 0.75) = 0.125 / 0.25 = 0.5
lot = 0.10 + (0.5 × 0.90) = 0.10 + 0.45 = 0.55

# Si confianza fuera 100%:
scaled = (1.0 - 0.75) / (1.0 - 0.75) = 0.25 / 0.25 = 1.0
lot = 0.10 + (1.0 × 0.90) = 0.10 + 0.90 = 1.00
```

**Ejemplo Real:**
```
Confianza 75% → Lote 0.10 (mínimo)
Confianza 87.5% → Lote 0.55 (medio)
Confianza 100% → Lote 1.00 (máximo)
```

---

### Modo Fijo (Lote Constante)

El tamaño del lote es **siempre el mismo**, sin importar la confianza:

```bash
# En tu archivo .env:
ENABLE_DYNAMIC_LOT_SIZE=false
MT5_LOT_SIZE=0.50
```

**Ventajas:**
- ✅ Simplicidad absoluta
- ✅ Control total del riesgo por trade
- ✅ Útil para cuentas pequeñas

**Desventajas:**
- ⚠️ No aprovecha señales de alta confianza
- ⚠️ Arriesga igual en señales débiles y fuertes

---

## 🚀 Configuraciones Recomendadas

### 1. Configuración Conservadora (Principiantes)
```bash
# Riesgo dinámico pero conservador
ENABLE_DYNAMIC_RISK=true
STOP_LOSS_ATR_MULTIPLIER=2.0          # SL más amplio
TAKE_PROFIT_1_ATR_MULTIPLIER=3.0       # TP1 moderado
TAKE_PROFIT_2_ATR_MULTIPLIER=6.0       # TP2 ambicioso

# Lotaje dinámico conservador
ENABLE_DYNAMIC_LOT_SIZE=true
MIN_LOT_SIZE=0.01                      # Muy pequeño
MAX_LOT_SIZE=0.10                      # Limitado
CONFIDENCE_THRESHOLD=0.80              # Alta selectividad
```

### 2. Configuración Balanceada (Recomendada) ⭐
```bash
# Riesgo dinámico balanceado
ENABLE_DYNAMIC_RISK=true
STOP_LOSS_ATR_MULTIPLIER=1.5
TAKE_PROFIT_1_ATR_MULTIPLIER=2.5
TAKE_PROFIT_2_ATR_MULTIPLIER=5.0

# Lotaje dinámico balanceado
ENABLE_DYNAMIC_LOT_SIZE=true
MIN_LOT_SIZE=0.10
MAX_LOT_SIZE=1.00
CONFIDENCE_THRESHOLD=0.75
```

### 3. Configuración Agresiva (Expertos)
```bash
# Riesgo dinámico agresivo
ENABLE_DYNAMIC_RISK=true
STOP_LOSS_ATR_MULTIPLIER=1.0          # SL ajustado
TAKE_PROFIT_1_ATR_MULTIPLIER=2.0       # TP1 cercano
TAKE_PROFIT_2_ATR_MULTIPLIER=4.0       # TP2 realista

# Lotaje dinámico agresivo
ENABLE_DYNAMIC_LOT_SIZE=true
MIN_LOT_SIZE=0.50                      # Mínimo alto
MAX_LOT_SIZE=2.00                      # Máximo elevado
CONFIDENCE_THRESHOLD=0.70              # Menos selectivo
```

### 4. Configuración Fija (Backtesting)
```bash
# Riesgo fijo para backtesting
ENABLE_DYNAMIC_RISK=false
FIXED_STOP_LOSS_POINTS=50.0
FIXED_TAKE_PROFIT_1_POINTS=125.0
FIXED_TAKE_PROFIT_2_POINTS=250.0

# Lotaje fijo
ENABLE_DYNAMIC_LOT_SIZE=false
MT5_LOT_SIZE=0.50
```

---

## 📊 Comparación de Modos

| Característica | Dinámico | Fijo |
|---------------|----------|------|
| **Adaptación a volatilidad** | ✅ Sí | ❌ No |
| **Predictibilidad** | ⚠️ Variable | ✅ Constante |
| **Riesgo/Recompensa óptimo** | ✅ Automático | ⚠️ Manual |
| **Complejidad** | ⚠️ Media | ✅ Simple |
| **Recomendado para** | Trading real | Backtesting |

---

## 🔍 Verificación en Logs

### Modo Dinámico
```
INFO | Risk Manager inicializado con gestión de riesgo DINÁMICA basada en ATR.
INFO | Multiplicadores: SL=1.5*ATR, TP1=2.5*ATR, TP2=5.0*ATR
INFO | Lotaje DINÁMICO activado: Min=0.1, Max=1.0

INFO | Parámetros de riesgo dinámico para GainX 600 (BUY) con ATR=15.23000:
INFO | Lotaje Dinámico Calculado: 0.55 (Confianza: 0.87)
INFO | SL=9843.155, TP1=9881.575, TP2=9957.150, RR1=2.50
```

### Modo Fijo
```
INFO | Risk Manager inicializado con gestión de riesgo FIJA.
INFO | SL Fijo=50.0 puntos, TP1 Fijo=125.0 puntos, TP2 Fijo=250.0 puntos
INFO | Lotaje FIJO activado: 0.5 lotes

INFO | Parámetros de riesgo FIJOS para GainX 600 (BUY):
INFO | Lotaje: 0.5 (Fijo)
INFO | SL=9815.00, TP1=9990.00, TP2=10115.00, RR1=2.50
```

---

## ⚠️ Advertencias Importantes

1. **ATR debe estar disponible:**
   - Si `ENABLE_DYNAMIC_RISK=true` pero no hay ATR, el bot fallará
   - Asegúrate de que `ATR_PERIOD=14` esté configurado

2. **Puntos vs Pips:**
   - El bot está configurado para **índices sintéticos** donde `1 punto = 1.0` en el precio
   - Si operas **Forex**, necesitarás ajustar `point_value` en `risk_manager.py:74` a `0.0001` (o `0.01` para pares JPY)

3. **Lotes y broker:**
   - Verifica los límites de lote de tu broker
   - Algunos brokers tienen mínimo de 0.01, otros 0.10

4. **Confianza y lotaje:**
   - `CONFIDENCE_THRESHOLD` debe ser menor a 1.0
   - Si threshold = 0.75, señales bajo 75% se rechazan
   - Solo señales ≥75% pasarán el filtro

---

## 🛠️ Cómo Cambiar de Modo

1. Abre tu archivo `.env`
2. Localiza las variables `ENABLE_DYNAMIC_RISK` y `ENABLE_DYNAMIC_LOT_SIZE`
3. Cambia a `true` o `false` según desees
4. Configura los parámetros correspondientes
5. Guarda el archivo
6. Reinicia el bot con `run_bot.bat`

**Ejemplo de cambio:**
```bash
# Antes (dinámico):
ENABLE_DYNAMIC_RISK=true

# Después (fijo):
ENABLE_DYNAMIC_RISK=false
FIXED_STOP_LOSS_POINTS=60.0
FIXED_TAKE_PROFIT_1_POINTS=150.0
FIXED_TAKE_PROFIT_2_POINTS=300.0
```

---

## 📚 Referencias

- **Commit:** `9b58856` - "feat: Agregar configuración de riesgo y lotaje fijo/dinámico"
- **Archivos modificados:** `config.py`, `risk_manager.py`
- **ATR (Average True Range):** Indicador de volatilidad que mide el rango promedio de movimiento del precio

---

## 💡 Consejos Finales

1. **Empieza con modo dinámico:** Es más robusto y se adapta mejor a diferentes condiciones
2. **Ajusta multiplicadores según tu tolerancia:** Más conservador = multiplicadores mayores
3. **Monitorea los logs:** Verifica que los SL/TP calculados sean razonables
4. **Backtesting con fijo:** Usa modo fijo para backtesting reproducible
5. **Trading real con dinámico:** Usa modo dinámico para adaptarte al mercado
