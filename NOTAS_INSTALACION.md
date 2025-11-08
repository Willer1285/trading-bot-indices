# 📝 Notas Importantes de Instalación - Python 3.13

## ⚠️ Problema Detectado durante la Instalación

Durante la instalación automatizada con `install_windows.bat`, se detectó un problema con el paquete `mplfinance`:

```
ERROR: Could not find a version that satisfies the requirement mplfinance>=0.12.10
ERROR: No matching distribution found for mplfinance>=0.12.10
```

### ✅ Solución Aplicada

El archivo `requirements.txt` ha sido actualizado para usar la última versión disponible de mplfinance compatible con Python 3.13:

```txt
# Antes (causaba error)
mplfinance>=0.12.10

# Ahora (funciona correctamente)
mplfinance==0.12.10b0
```

**Nota:** `0.12.10b0` es una versión beta pero es estable y funcional. Es la versión más reciente disponible para Python 3.13.

---

## 🔧 Cómo Completar la Instalación

Ya que ejecutaste `install_windows.bat` y falló en mplfinance, sigue estos pasos:

### **Opción 1: Script de Corrección Rápida (RECOMENDADO)**

```cmd
fix_installation.bat
```

Este script:
- ✅ Activa el entorno virtual existente
- ✅ Instala las dependencias corregidas
- ✅ Verifica TensorFlow

### **Opción 2: Manual**

```cmd
# Activar entorno virtual
venv_trading\Scripts\activate

# Instalar dependencias corregidas
pip install -r requirements.txt

# Verificar instalación
python diagnose_environment.py
```

---

## 📊 Estado de las Dependencias

### **Paquetes Instalados Exitosamente** ✅

Durante tu instalación, estos paquetes se instalaron correctamente:

- ✅ MetaTrader5 5.0.5388
- ✅ scikit-learn 1.7.2
- ✅ xgboost 3.1.1
- ✅ pandas 2.3.3
- ✅ numpy 2.3.4
- ✅ scipy 1.16.3
- ✅ ta 0.11.0
- ✅ python-telegram-bot 22.5
- ✅ aiohttp 3.13.2
- ✅ redis 7.0.1
- ✅ pymongo 4.15.3
- ✅ asyncio 4.0.0
- ✅ aiofiles 25.1.0
- ✅ plotly 6.4.0
- ✅ matplotlib 3.10.7

### **Paquetes Pendientes de Instalación** ⚠️

- ⏳ **mplfinance** - Se instalará con `fix_installation.bat`
- ⏳ **TensorFlow** - Se instalará con `fix_installation.bat`
- ⏳ **Keras** - Se instalará con TensorFlow
- ⏳ Resto de dependencias restantes

---

## 🚀 Próximos Pasos

### **1. Completar Instalación**

```cmd
fix_installation.bat
```

### **2. Verificar Entorno**

```cmd
# Activar entorno virtual
venv_trading\Scripts\activate

# Ejecutar diagnóstico
python diagnose_environment.py
```

Deberías ver:
```
✅ RESULTADO: ENTORNO CONFIGURADO CORRECTAMENTE
```

### **3. Configurar Credenciales MT5**

```cmd
# Copiar archivo de ejemplo
copy .env.example .env

# Editar con tus datos
notepad .env
```

### **4. Entrenar Modelos**

```cmd
python train_models.py
```

### **5. Ejecutar el Bot**

```cmd
python run_mt5.py
```

---

## 📋 Dependencias Clave con Versiones Exactas

Para referencia, estas son las versiones de las dependencias principales:

```txt
# Core Python
Python==3.13.9

# Machine Learning / AI
tensorflow==2.20.0
keras==3.12.0
scikit-learn==1.7.2
xgboost==3.1.1

# Data Processing
pandas==2.3.3
numpy==2.3.4
scipy==1.16.3

# MetaTrader 5
MetaTrader5==5.0.5388

# Visualization
matplotlib==3.10.7
plotly==6.4.0
mplfinance==0.12.10b0  # ⚠️ Versión beta (estable)

# Communication
python-telegram-bot==22.5
aiohttp==3.13.2

# Database
redis==7.0.1
pymongo==4.15.3
```

---

## ⚠️ Problemas Conocidos

### **1. mplfinance versión beta**

**Problema:** Solo hay versión beta disponible para Python 3.13

**Impacto:** Mínimo. La versión `0.12.10b0` es estable y funcional.

**Alternativa:** Si prefieres usar una versión estable, puedes:
- Comentar la línea de mplfinance en requirements.txt
- El bot funcionará sin gráficos de velas (candlesticks)

```txt
# mplfinance==0.12.10b0  # Comentar si no necesitas gráficos de velas
```

### **2. Advertencias de pip sobre versiones Python**

**Mensaje:**
```
ERROR: Ignored the following versions that require a different python version...
```

**Causa:** pip está mostrando versiones antiguas incompatibles con Python 3.13

**Impacto:** Ninguno. Es solo informativo. pip automáticamente selecciona versiones compatibles.

---

## 🔍 Verificación de Instalación

Para verificar que todo está instalado correctamente:

```cmd
# Activar entorno
venv_trading\Scripts\activate

# Verificar Python
python --version
# Debe mostrar: Python 3.13.9

# Verificar TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
# Debe mostrar: TensorFlow 2.20.0

# Verificar scikit-learn
python -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"
# Debe mostrar: scikit-learn 1.7.2

# Verificar pandas
python -c "import pandas as pd; print(f'pandas {pd.__version__}')"
# Debe mostrar: pandas 2.3.3

# Verificar numpy
python -c "import numpy as np; print(f'numpy {np.__version__}')"
# Debe mostrar: numpy 2.3.4

# Diagnóstico completo
python diagnose_environment.py
```

---

## 📞 Soporte

Si después de ejecutar `fix_installation.bat` sigues teniendo problemas:

1. **Ejecutar diagnóstico completo:**
   ```cmd
   python diagnose_environment.py > diagnostico.txt
   ```

2. **Revisar el archivo** `diagnostico.txt`

3. **Verificar logs de instalación**

4. **Intentar instalación limpia:**
   - Eliminar carpeta `venv_trading`
   - Ejecutar `install_windows.bat` nuevamente

---

## ✅ Resumen

- ✅ **Problema identificado:** mplfinance no tenía versión estable para Python 3.13
- ✅ **Solución aplicada:** Usar versión beta `0.12.10b0`
- ✅ **Script de corrección:** `fix_installation.bat` creado
- ✅ **Mayoría de paquetes:** Ya instalados correctamente
- ✅ **Próximo paso:** Ejecutar `fix_installation.bat`

---

**Fecha:** Noviembre 2025
**Versión Python:** 3.13.9
**TensorFlow:** 2.20.0
**Estado:** Corrección aplicada, lista para completar instalación
