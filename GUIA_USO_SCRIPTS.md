# 🚀 Guía de Uso - Scripts de Ejecución Automática

Esta guía explica cómo usar los scripts de ejecución automática que activan el entorno virtual por ti.

---

## 📋 Scripts Disponibles

### 1. **run_bot.bat** - Ejecutar el Trading Bot
Activa automáticamente el entorno virtual y ejecuta el bot.

**Uso:**
```cmd
# Opción 1: Desde CMD
run_bot.bat

# Opción 2: Doble clic en el archivo desde el Explorador de Windows
```

**Qué hace automáticamente:**
- ✅ Activa `venv_trading`
- ✅ Verifica que scikit-learn esté en versión 1.7.2+
- ✅ Actualiza scikit-learn si es necesario
- ✅ Verifica conexión a MT5
- ✅ Ejecuta `python run_mt5.py`
- ✅ Muestra el resultado final

---

### 2. **run_diagnostico.bat** - Ejecutar Diagnóstico
Activa automáticamente el entorno virtual y ejecuta el diagnóstico del entorno.

**Uso:**
```cmd
# Opción 1: Desde CMD
run_diagnostico.bat

# Opción 2: Doble clic en el archivo desde el Explorador de Windows
```

**Qué hace automáticamente:**
- ✅ Activa `venv_trading`
- ✅ Ejecuta `python diagnose_environment.py`
- ✅ Muestra verificación completa del entorno

---

## 🎯 Ventajas de Usar los Scripts

| Aspecto | Sin Scripts | Con Scripts |
|---------|-------------|-------------|
| **Activación venv** | Manual cada vez | ✅ Automática |
| **Verificaciones** | Manual | ✅ Automáticas |
| **Errores comunes** | Fáciles de cometer | ✅ Prevenidos |
| **Facilidad de uso** | 3-4 comandos | ✅ 1 comando |
| **Ideal para** | Desarrollo | ✅ Uso diario |

---

## ⚙️ Comparación: Ejecución Manual vs Scripts

### ❌ Ejecución Manual (Antigua Forma)
```cmd
# Paso 1: Navegar a la carpeta
cd C:\Users\wille\Downloads\trading-bot-indices

# Paso 2: Activar venv
venv_trading\Scripts\activate

# Paso 3: Verificar sklearn (opcional)
python -c "import sklearn; print(sklearn.__version__)"

# Paso 4: Ejecutar bot
python run_mt5.py
```

**Problemas:**
- 😫 Muchos pasos
- ⚠️ Fácil olvidar activar venv
- ⚠️ No verifica versiones automáticamente

---

### ✅ Ejecución con Scripts (Nueva Forma)
```cmd
# Solo esto:
run_bot.bat
```

**Ventajas:**
- 😊 Un solo comando
- ✅ Venv siempre activado
- ✅ Verificaciones automáticas
- ✅ Mensajes claros de progreso

---

## 🔧 Primera Vez: Actualizar scikit-learn

**IMPORTANTE:** La primera vez que uses `run_bot.bat`, el script puede necesitar actualizar scikit-learn automáticamente. Esto es normal.

Si prefieres hacerlo manualmente antes:

```cmd
# Ejecutar UNA VEZ para actualizar scikit-learn
venv_trading\Scripts\activate
pip install --upgrade scikit-learn>=1.7.2
deactivate
```

Después de esto, `run_bot.bat` funcionará sin necesidad de actualizaciones.

---

## 📝 Ejemplo de Uso Diario

### Escenario: Quiero ejecutar el bot cada mañana

```cmd
# 1. Asegurarse de que MT5 esté abierto
# 2. Abrir CMD en la carpeta del proyecto
# 3. Ejecutar:
run_bot.bat

# ¡Listo! El bot se ejecuta con el entorno correcto
```

---

## 🆘 Solución de Problemas

### Error: "No se encuentra el entorno virtual venv_trading"

**Causa:** Estás ejecutando el script desde una ubicación incorrecta.

**Solución:**
```cmd
# Navega primero a la carpeta del proyecto
cd C:\Users\wille\Downloads\trading-bot-indices

# Luego ejecuta el script
run_bot.bat
```

---

### Error: "El entorno virtual no se activó correctamente"

**Causa:** Problema con la instalación del venv.

**Solución:**
```cmd
# Reinstalar el entorno virtual
python -m venv venv_trading --clear
venv_trading\Scripts\activate
pip install -r requirements.txt
```

---

## 🎓 Preguntas Frecuentes

### ¿Por qué no actualizar los paquetes globales de Python?

**Respuesta:** Porque:
- ❌ Puede romper otros proyectos Python en tu sistema
- ❌ Dificulta el control de versiones
- ❌ Es una mala práctica de desarrollo
- ✅ Los scripts resuelven el problema de conveniencia manteniendo buenas prácticas

### ¿Puedo modificar los scripts?

**Sí**, los scripts son archivos `.bat` simples que puedes editar con Notepad. Por ejemplo, podrías:
- Agregar más verificaciones
- Cambiar mensajes
- Agregar logs automáticos

### ¿Funcionan los scripts si muevo el proyecto a otra carpeta?

**Sí**, los scripts usan rutas relativas. Solo asegúrate de:
1. Mover toda la carpeta del proyecto completa
2. Ejecutar los scripts desde la carpeta raíz del proyecto

---

## 📚 Resumen Rápido

```cmd
# Para ejecutar el bot:
run_bot.bat

# Para ejecutar diagnóstico:
run_diagnostico.bat

# Para instalación inicial:
install_windows.bat
```

**¡Eso es todo!** Ya no necesitas recordar activar el venv manualmente. 🎉
