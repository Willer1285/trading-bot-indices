# 🔴 Problemas Críticos Identificados - Ejecución del Bot

## Resumen Ejecutivo

El bot no ejecuta correctamente debido a **2 problemas críticos**:

1. **Incompatibilidad de scikit-learn** (20 modelos Gradient Boosting no se cargan)
2. **Desajuste de nomenclatura de timeframes** (0 modelos encontrados para análisis)

---

## 🔍 Problema 1: Incompatibilidad de scikit-learn

### **Error:**
```
InconsistentVersionWarning: Trying to unpickle estimator from version 1.7.2 when using version 1.4.0
ERROR | Failed to load gradient_boosting: No module named '_loss'
```

### **Causa Raíz:**
- **Modelos entrenados con:** scikit-learn 1.7.2 (en venv_trading)
- **Bot ejecutándose con:** scikit-learn 1.4.0 (instalación global de Python)

### **Evidencia:**
Path del error: `C:\Users\wille\AppData\Roaming\Python\Python313\site-packages\sklearn\...`

Esto indica que Python está cargando paquetes de la instalación **GLOBAL** en lugar del **entorno virtual**.

### **Impacto:**
- ❌ **20 modelos Gradient Boosting fallaron** al cargar
- ⚠️ Solo funcionan Random Forest y LSTM
- ⚠️ El ensemble está incompleto y dará predicciones degradadas

### **Solución:**

#### **Opción A: Actualizar scikit-learn en venv (RÁPIDO)**

```cmd
# Activar entorno virtual
venv_trading\Scripts\activate

# Verificar versión actual
python -c "import sklearn; print(sklearn.__version__)"

# Si muestra 1.4.0, actualizar:
pip install --upgrade scikit-learn>=1.7.2

# Verificar nuevamente
python -c "import sklearn; print(sklearn.__version__)"
# Debe mostrar: 1.7.2 o superior
```

#### **Opción B: Re-entrenar modelos con scikit-learn 1.4.0 (LENTO)**

```cmd
# Activar entorno virtual
venv_trading\Scripts\activate

# Desinstalar scikit-learn actual
pip uninstall scikit-learn -y

# Instalar versión específica
pip install scikit-learn==1.4.0

# Re-entrenar modelos
python train_models.py
```

**Recomendación:** Opción A (actualizar a 1.7.2) es más rápido y está en el requirements.txt.

---

## 🔍 Problema 2: Desajuste de Nomenclatura de Timeframes

### **Error:**
```
WARNING | No se encontró o no está entrenado un modelo para GainX 1200 1m. Saltando análisis.
WARNING | No se encontró o no está entrenado un modelo para GainX 1200 1h. Saltando análisis.
```

### **Causa Raíz:**

**Modelos guardados con:**
- Directorios: `models/GainX_1200/GainX 1200_M1/`
- Clave en memoria: `"GainX 1200_M1"`

**Bot busca con:**
- Timeframe de config: `"1m"`, `"1h"`
- Clave buscada: `"1m"`, `"1h"`

### **Evidencia del Código:**

**src/ai_engine/market_analyzer.py:201**
```python
self.models[symbol][timeframe_dir] = model
```

Se guarda con el nombre del directorio: `"GainX 1200_M1"`

**src/ai_engine/market_analyzer.py:93**
```python
model = self.models.get(symbol, {}).get(timeframe)
```

Se busca con el timeframe de configuración: `"1m"`

**❌ No hay conversión entre "1m" → "M1" o "1h" → "H1"**

### **Impacto:**
- ❌ **0 modelos encontrados** para análisis
- ❌ Todas las señales son HOLD
- ❌ El bot no genera ninguna operación

### **Solución:**

Se requiere agregar un **mapeo de timeframes** entre los nombres de MT5 y los nombres de los modelos.

**Mapeo requerido:**
```python
TIMEFRAME_MAPPING = {
    '1m': 'M1',
    '5m': 'M5',
    '15m': 'M15',
    '1h': 'H1',
    '4h': 'H4',
    '1d': 'D1'
}
```

---

## 📊 Resumen de Modelos Cargados

### **Estado Actual:**

| Modelo Type | Cargados | Fallaron | Estado |
|-------------|----------|----------|--------|
| Random Forest | 20/20 | 0 | ✅ OK |
| LSTM | 20/20 | 0 | ✅ OK |
| Gradient Boosting | 0/20 | 20 | ❌ FALLO |
| Meta Model | 20/20 | 0 | ✅ OK |

**Total:** 60/80 modelos funcionales (75%)

### **Modelos por Símbolo/Timeframe:**

Todos los símbolos cargaron exitosamente:
- GainX 400 [M1, H1]
- GainX 600 [M1, H1]
- GainX 800 [M1, H1]
- GainX 999 [M1, H1]
- GainX 1200 [M1, H1]
- PainX 400 [M1, H1]
- PainX 600 [M1, H1]
- PainX 800 [M1, H1]
- PainX 999 [M1, H1]
- PainX 1200 [M1, H1]

**Total:** 20 combinaciones cargadas

---

## ✅ Plan de Acción

### **Paso 1: Corregir scikit-learn**

```cmd
venv_trading\Scripts\activate
pip install --upgrade scikit-learn>=1.7.2
python diagnose_environment.py
```

### **Paso 2: Corregir mapeo de timeframes**

Se implementará fix en el código (automático)

### **Paso 3: Verificar**

```cmd
python run_mt5.py
```

**Resultado esperado:**
```
✅ Modelos cargados: 80/80
✅ Gradient Boosting: Funcionando
✅ Análisis para GainX 1200 [H1]: OK
✅ Análisis para GainX 1200 [M1]: OK
```

---

## 🔧 Fix Técnico a Implementar

### **Modificación en market_analyzer.py:**

**Antes:**
```python
model = self.models.get(symbol, {}).get(timeframe)
```

**Después:**
```python
# Mapeo de timeframes MT5 → nombres de modelos
TIMEFRAME_MAPPING = {'1m': 'M1', '5m': 'M5', '15m': 'M15',
                     '1h': 'H1', '4h': 'H4', '1d': 'D1'}

# Convertir timeframe al formato del modelo
model_timeframe = TIMEFRAME_MAPPING.get(timeframe.lower(), timeframe)
model_key = f"{symbol}_{model_timeframe}"
model = self.models.get(symbol, {}).get(model_key)
```

---

## 📋 Checklist de Verificación

- [ ] Activar entorno virtual correctamente
- [ ] Actualizar scikit-learn a 1.7.2+
- [ ] Aplicar fix de mapeo de timeframes
- [ ] Ejecutar diagnóstico
- [ ] Ejecutar bot
- [ ] Verificar que los 80 modelos se cargan
- [ ] Verificar que se generan análisis (no solo HOLD)

---

## 🆘 Si Persisten Problemas

### **Verificar entorno virtual:**

```cmd
# Ver qué Python se está usando
where python

# Debe mostrar primero:
# C:\Users\wille\Downloads\trading-bot-indices\venv_trading\Scripts\python.exe

# Si muestra primero:
# C:\Users\wille\AppData\Local\Programs\Python\Python313\python.exe
# Entonces el venv NO está activado correctamente
```

### **Solución entorno virtual:**

```cmd
# Desactivar cualquier venv activo
deactivate

# Activar venv correcto
cd C:\Users\wille\Downloads\trading-bot-indices
venv_trading\Scripts\activate

# Verificar
python -c "import sys; print(sys.executable)"
# Debe mostrar el path del venv
```

---

**Fecha:** 2025-11-08
**Prioridad:** CRÍTICA
**Estado:** Pendiente de corrección

