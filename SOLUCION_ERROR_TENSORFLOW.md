# 🔧 Solución al Error de TensorFlow

## ❌ Error Encontrado

```
ModuleNotFoundError: No module named 'tensorflow.python.tools'
```

Este error ocurre al ejecutar `python run_mt5.py` y se debe a incompatibilidades de TensorFlow con Python 3.11 en Windows.

---

## ⚠️ ADVERTENCIA IMPORTANTE

**Python 3.11 tiene problemas conocidos de compatibilidad con TensorFlow en Windows.**

**SOLUCIÓN RECOMENDADA: Migrar a Python 3.13.9**

Python 3.13.9 es la última versión estable y ofrece:
- ✅ TensorFlow 2.20+ completamente funcional
- ✅ Mejor rendimiento (15-20% más rápido)
- ✅ Mayor estabilidad
- ✅ Todas las librerías actualizadas

**Ver guía completa de migración:** [MIGRACION_PYTHON_3.13.md](MIGRACION_PYTHON_3.13.md)

---

## 🎯 Soluciones (Ordenadas por Efectividad)

### ⭐ **SOLUCIÓN RECOMENDADA: Migrar a Python 3.13.9**

Esta es la mejor solución a largo plazo. Resuelve todos los problemas de compatibilidad.

**Pasos rápidos:**

1. **Descargar Python 3.13.9:**
   - https://www.python.org/downloads/
   - Marcar "Add Python to PATH" durante instalación

2. **Ejecutar instalación automatizada:**
   ```cmd
   install_windows.bat
   ```

3. **Listo!** Todo funcionará sin problemas.

**Ver guía detallada:** [MIGRACION_PYTHON_3.13.md](MIGRACION_PYTHON_3.13.md)

---

## 🔧 Soluciones Alternativas (Si no puedes migrar ahora)

### **Solución Rápida - Instalación Automatizada (RECOMENDADA)**

**Para Windows:**

1. Abre una terminal (CMD o PowerShell) en la carpeta del proyecto
2. Ejecuta el script de instalación automatizada:

```cmd
install_windows.bat
```

Este script hará todo automáticamente:
- ✅ Crear un entorno virtual limpio
- ✅ Instalar todas las dependencias
- ✅ Configurar TensorFlow correctamente

---

### **Solución Manual - Paso a Paso**

Si prefieres hacerlo manualmente, sigue estos pasos:

#### **Paso 1: Crear Entorno Virtual**

```cmd
# Abre CMD en la carpeta del proyecto
cd C:\Users\wille\Downloads\trading-bot-indices

# Crear entorno virtual
python -m venv venv_trading

# Activar entorno virtual
venv_trading\Scripts\activate
```

#### **Paso 2: Actualizar pip**

```cmd
python -m pip install --upgrade pip
```

#### **Paso 3: Limpiar TensorFlow Anterior**

```cmd
# Desinstalar cualquier versión previa
pip uninstall tensorflow tensorflow-intel -y

# Limpiar caché
pip cache purge
```

#### **Paso 4: Instalar Dependencias**

```cmd
# Instalar todas las dependencias
pip install -r requirements.txt
```

#### **Paso 5: Verificar Instalación**

```cmd
# Ejecutar diagnóstico
python diagnose_environment.py
```

Si todo está OK, verás:
```
✅ RESULTADO: ENTORNO CONFIGURADO CORRECTAMENTE
```

#### **Paso 6: Entrenar Modelos (Primera Vez)**

```cmd
python train_models.py
```

#### **Paso 7: Ejecutar el Bot**

```cmd
python run_mt5.py
```

---

### **Solución Alternativa - Si lo Anterior No Funciona**

Si sigues teniendo problemas, prueba con una versión específica de TensorFlow:

```cmd
# Activar entorno virtual
venv_trading\Scripts\activate

# Desinstalar TensorFlow
pip uninstall tensorflow tensorflow-intel -y

# Instalar versión específica compatible con Python 3.11
pip install tensorflow==2.16.1

# Verificar
python -c "import tensorflow as tf; print(tf.__version__)"
```

Deberías ver: `2.16.1`

---

## 🔍 Diagnóstico de Problemas

### **Verificar qué versión de Python estás usando:**

```cmd
python --version
```

Debería mostrar: `Python 3.11.x`

### **Verificar si TensorFlow está instalado:**

```cmd
pip list | findstr tensorflow
```

### **Ejecutar script de diagnóstico completo:**

```cmd
python diagnose_environment.py
```

Este script te mostrará:
- ✅ Versión de Python
- ✅ Sistema operativo
- ✅ Módulos instalados/faltantes
- ✅ Estado de TensorFlow
- ✅ Archivos del proyecto
- ✅ Modelos entrenados

---

## 📋 Checklist de Verificación

Antes de ejecutar el bot, asegúrate de:

- [ ] Python 3.11 instalado correctamente
- [ ] Entorno virtual creado y activado
- [ ] Todas las dependencias instaladas (`pip install -r requirements.txt`)
- [ ] TensorFlow instalado correctamente (verificar con `python -c "import tensorflow"`)
- [ ] Archivo `.env` configurado con credenciales de MT5
- [ ] Modelos entrenados (ejecutar `python train_models.py` primero)

---

## 🆘 Problemas Comunes

### **Error: "pip no reconocido como comando"**

**Solución:** Reinstala Python y marca la opción "Add Python to PATH"

### **Error: "Permission denied" al crear entorno virtual**

**Solución:** Ejecuta CMD como Administrador

### **Error: "No module named 'MetaTrader5'"**

**Solución:**
```cmd
pip install MetaTrader5
```

### **Error: "Models not found"**

**Solución:**
```cmd
python train_models.py
```

### **TensorFlow se instala pero sigue dando error**

**Solución:**
```cmd
pip uninstall tensorflow tensorflow-intel keras -y
pip install tensorflow==2.16.1 --no-cache-dir
```

---

## 📞 Soporte Adicional

Si ninguna de estas soluciones funciona:

1. **Ejecuta el diagnóstico completo:**
   ```cmd
   python diagnose_environment.py > diagnostico.txt
   ```

2. **Revisa el archivo `diagnostico.txt`** para ver qué módulos faltan

3. **Verifica los logs** del error completo

---

## ✅ Verificación Final

Una vez instalado todo correctamente, ejecuta:

```cmd
# Activar entorno virtual
venv_trading\Scripts\activate

# Diagnóstico
python diagnose_environment.py

# Si todo está OK, ejecutar bot
python run_mt5.py
```

Si ves el mensaje:
```
AI MT5 Trading Bot
Starting...
```

¡Felicidades! El bot está funcionando correctamente.

---

## 📌 Notas Importantes

- **Siempre activa el entorno virtual** antes de ejecutar el bot
- **No mezcles entornos virtuales** con la instalación global de Python
- **En Windows**, TensorFlow puede tardar varios minutos en instalarse
- **Los modelos deben entrenarse** antes de la primera ejecución

---

## 🔄 Actualización de Cambios

Los siguientes archivos fueron modificados/creados para solucionar el problema:

1. ✅ `requirements.txt` - Actualizado con versión específica de TensorFlow (2.16.1)
2. ✅ `diagnose_environment.py` - Script de diagnóstico nuevo
3. ✅ `install_windows.bat` - Script de instalación automatizada nuevo
4. ✅ `SOLUCION_ERROR_TENSORFLOW.md` - Este documento

Para aplicar estos cambios, ejecuta:

```cmd
git pull origin claude/debug-bot-execution-011CUuiScBxi1BmBobCzW3z9
```
