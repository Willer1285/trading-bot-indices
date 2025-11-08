@echo off
REM ============================================================================
REM Script de instalación automatizada para Windows
REM Trading Bot - MT5 AI - Python 3.13+
REM ============================================================================

echo ============================================================
echo Trading Bot - Instalacion Automatizada para Windows
echo Requiere: Python 3.13.9 o superior
echo ============================================================
echo.

REM Verificar que Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no esta en el PATH
    echo.
    echo Por favor, descarga e instala Python 3.13.9 desde:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANTE: Marca la opcion "Add Python to PATH" durante la instalacion
    pause
    exit /b 1
)

echo [1/7] Verificando version de Python...
python --version
echo.

REM Verificar versión de Python (debe ser 3.13+)
python -c "import sys; exit(0 if sys.version_info >= (3, 13) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ADVERTENCIA: Se recomienda Python 3.13.9 o superior
    echo Tu version actual puede tener problemas de compatibilidad
    echo.
    echo Descarga Python 3.13.9 desde: https://www.python.org/downloads/
    echo.
    set /p CONTINUE="Continuar de todos modos? (S/N): "
    if /i not "%CONTINUE%"=="S" (
        echo Instalacion cancelada
        pause
        exit /b 1
    )
)
echo Version de Python: OK
echo.

REM Crear entorno virtual si no existe
if not exist "venv_trading" (
    echo [2/7] Creando entorno virtual Python 3.13...
    python -m venv venv_trading
    echo Entorno virtual creado: venv_trading
) else (
    echo [2/7] Entorno virtual ya existe: venv_trading
)
echo.

REM Activar entorno virtual
echo [3/7] Activando entorno virtual...
call venv_trading\Scripts\activate.bat
echo.

REM Actualizar pip y herramientas de construcción
echo [4/7] Actualizando pip, setuptools y wheel...
python -m pip install --upgrade pip setuptools wheel
echo.

REM Limpiar instalaciones previas
echo [5/7] Limpiando instalaciones previas...
pip uninstall tensorflow tensorflow-intel keras -y >nul 2>&1
pip cache purge >nul 2>&1
echo Cache limpiado
echo.

REM Instalar dependencias
echo [6/7] Instalando dependencias para Python 3.13...
echo.
echo NOTA: TensorFlow 2.20+ puede tardar varios minutos en instalarse
echo       en Windows. Por favor, se paciente...
echo.
pip install -r requirements.txt

REM Verificar instalación de TensorFlow
echo.
echo [7/7] Verificando instalacion de TensorFlow...
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} instalado correctamente')" 2>nul
if errorlevel 1 (
    echo ADVERTENCIA: TensorFlow no se instalo correctamente
    echo Intenta instalar manualmente: pip install tensorflow==2.20.0
) else (
    echo TensorFlow: OK
)

echo.
echo ============================================================
echo INSTALACION COMPLETADA EXITOSAMENTE
echo ============================================================
echo.
echo Python version requerida: 3.13.9+
echo TensorFlow version: 2.20.0+
echo.
echo PROXIMOS PASOS:
echo.
echo   1. Activa el entorno virtual:
echo      venv_trading\Scripts\activate
echo.
echo   2. Configura tus credenciales MT5:
echo      - Copia .env.example a .env
echo      - Edita .env con tus datos de MT5
echo.
echo   3. Entrena los modelos AI (primera vez):
echo      python train_models.py
echo.
echo   4. Ejecuta el bot:
echo      python run_mt5.py
echo.
echo   5. Para diagnosticar problemas:
echo      python diagnose_environment.py
echo.
echo ============================================================
pause
