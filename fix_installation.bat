@echo off
REM ============================================================================
REM Script de corrección rápida para completar la instalación
REM Trading Bot - MT5 AI - Python 3.13
REM ============================================================================

echo ============================================================
echo Trading Bot - Correccion de Instalacion
echo ============================================================
echo.

echo Este script completara la instalacion de las dependencias faltantes
echo.

REM Activar entorno virtual
echo [1/3] Activando entorno virtual...
call venv_trading\Scripts\activate.bat
echo.

REM Instalar dependencias corregidas
echo [2/3] Instalando dependencias corregidas...
echo.
pip install -r requirements.txt

REM Verificar instalación de TensorFlow
echo.
echo [3/3] Verificando instalacion de TensorFlow...
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} instalado correctamente')" 2>nul
if errorlevel 1 (
    echo ADVERTENCIA: TensorFlow no se instalo correctamente
    echo Instalando TensorFlow manualmente...
    pip install tensorflow==2.20.0
) else (
    echo TensorFlow: OK
)

echo.
echo ============================================================
echo CORRECCION COMPLETADA
echo ============================================================
echo.
echo Ahora puedes ejecutar el diagnóstico:
echo    python diagnose_environment.py
echo.
echo O entrenar los modelos:
echo    python train_models.py
echo.
pause
