# 🔧 Solución: Features Faltantes (VOL y SPREAD)

## 📋 Problema Identificado

### Error Reportado
```
ERROR | X has 114 features, but StandardScaler is expecting 116 features as input.
ERROR | Feature names missing:
- lstm_proba_0
- lstm_proba_1
```

### Análisis Raíz

El error mencionaba `lstm_proba_0` y `lstm_proba_1` como faltantes, **pero este era un síntoma secundario**. La verdadera causa raíz era:

1. **Modelos entrenados:** Esperaban 116 features (incluyendo `VOL` y `SPREAD`)
2. **Código actual:** Solo generaba 114 features (sin `VOL` ni `SPREAD`)
3. **Resultado:** LSTM no podía hacer predicciones → Meta-modelo fallaba → Todo generaba HOLD

### Evidencia

```bash
# Features esperadas por los modelos guardados:
random_forest: 116 features
gradient_boosting: 116 features
lstm: 116 features

# Features generadas por el código:
114 features (faltaban VOL y SPREAD)
```

---

## ✅ Solución Implementada

### Archivos Modificados

#### 1. `train_models.py` (Líneas 111-127)

**Antes:**
```python
df.rename(columns={
    'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low',
    'CLOSE': 'close', 'TICKVOL': 'volume'
}, inplace=True)
# VOL y SPREAD se descartaban
```

**Después:**
```python
df.rename(columns={
    'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low',
    'CLOSE': 'close', 'TICKVOL': 'volume'
}, inplace=True)

# Asegurar que VOL y SPREAD existan (requeridas por modelos)
if 'VOL' not in df.columns:
    df['VOL'] = 0
if 'SPREAD' not in df.columns:
    df['SPREAD'] = 0
```

#### 2. `src/data_collector/mt5_market_data_manager.py` (Líneas 127-146)

**Antes:**
```python
df.rename(columns={
    'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low',
    'CLOSE': 'close', 'VOL': 'volume'  # ❌ Incorrecto
}, inplace=True)

required_cols = ['open', 'high', 'low', 'close', 'volume']
# VOL y SPREAD no se incluían
```

**Después:**
```python
df.rename(columns={
    'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low',
    'CLOSE': 'close', 'TICKVOL': 'volume'  # ✅ Correcto
}, inplace=True)

# Asegurar que VOL y SPREAD existan
if 'VOL' not in df.columns:
    df['VOL'] = 0
if 'SPREAD' not in df.columns:
    df['SPREAD'] = 0

required_cols = ['open', 'high', 'low', 'close', 'volume', 'VOL', 'SPREAD']
```

#### 3. `src/data_collector/mt5_connector.py` (Líneas 143-162, 202-217)

**Antes:**
```python
df.rename(columns={
    'time': 'timestamp',
    'tick_volume': 'volume'
}, inplace=True)

df = df[['open', 'high', 'low', 'close', 'volume']]
# VOL y SPREAD no se mapeaban desde MT5
```

**Después:**
```python
df.rename(columns={
    'time': 'timestamp',
    'tick_volume': 'volume',
    'real_volume': 'VOL',      # ✅ Agregado
    'spread': 'SPREAD'          # ✅ Agregado
}, inplace=True)

# Asegurar que VOL y SPREAD existan
if 'VOL' not in df.columns:
    df['VOL'] = 0
if 'SPREAD' not in df.columns:
    df['SPREAD'] = 0

df = df[['open', 'high', 'low', 'close', 'volume', 'VOL', 'SPREAD']]
```

---

## 🎯 Resultado Esperado

### Antes del Fix
```
❌ Features generadas: 114
❌ Features esperadas: 116
❌ LSTM: No puede hacer predicciones
❌ Ensemble: Falla
❌ Señales: Solo HOLD
```

### Después del Fix
```
✅ Features generadas: 116 (114 técnicas + VOL + SPREAD)
✅ Features esperadas: 116
✅ LSTM: Predicciones correctas
✅ Ensemble: Funciona completamente
✅ Señales: BUY/SELL/HOLD según análisis
```

---

## 📊 Mapeo de Columnas

### Archivos CSV Históricos
```
Columnas originales:
<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>

Mapeo aplicado:
OPEN      → open
HIGH      → high
LOW       → low
CLOSE     → close
TICKVOL   → volume (volumen de ticks, principal)
VOL       → VOL (volumen real, feature para modelo)
SPREAD    → SPREAD (spread, feature para modelo)
```

### Datos de MT5 Live
```
Columnas de mt5.copy_rates_from_pos():
time, open, high, low, close, tick_volume, real_volume, spread

Mapeo aplicado:
time         → timestamp
tick_volume  → volume (principal)
real_volume  → VOL (feature para modelo)
spread       → SPREAD (feature para modelo)
```

---

## 🔍 Verificación

Para verificar que el fix funcionó correctamente:

```bash
# 1. Pull de los cambios
git pull origin claude/debug-bot-execution-011CUuiScBxi1BmBobCzW3z9

# 2. Ejecutar el bot
run_bot.bat

# 3. Buscar en los logs:
# ✅ Debería mostrar: "Loaded X records..." sin errores de features
# ✅ NO debería mostrar: "X has 114 features, but StandardScaler is expecting 116"
# ✅ Debería mostrar análisis reales (BUY/SELL) en lugar de solo HOLD
```

---

## 📝 Notas Importantes

1. **Compatibilidad:** Los cambios son compatibles tanto con:
   - Datos históricos CSV (con columnas VOL y SPREAD)
   - Datos live de MT5 (con real_volume y spread)

2. **Fallback:** Si VOL o SPREAD no existen en la fuente de datos, se rellenan automáticamente con `0`

3. **No requiere reentrenamiento:** Los modelos existentes ya esperan 116 features, por lo que funcionarán inmediatamente con el fix

4. **Consistencia:** Ahora `train_models.py` y el bot usan el mismo mapeo (TICKVOL → volume)

---

## 🚀 Próximos Pasos

1. ✅ Pull de los cambios
2. ✅ Ejecutar `run_bot.bat`
3. ✅ Verificar logs (no debe haber errores de features)
4. ✅ Observar señales generadas (deberían ser variadas, no solo HOLD)
5. ✅ Monitorear ejecución durante algunos ciclos

---

## 📚 Referencias

- Commit: `361c97b` - "fix: Corregir detección de modelos entrenados"
- Archivos modificados: `train_models.py`, `mt5_market_data_manager.py`, `mt5_connector.py`
- Issue raíz: Discrepancia entre features de entrenamiento (116) y predicción (114)
