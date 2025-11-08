@echo off
REM ========================================================================
REM Script de Ejecución Automática del Trading Bot
REM ========================================================================
REM Este script activa automáticamente el entorno virtual y ejecuta el bot.
REM Uso: Simplemente ejecuta "run_bot.bat" desde CMD o haz doble clic
REM ========================================================================

echo.
echo ========================================================================
echo   TRADING BOT - Iniciando con Entorno Virtual
echo ========================================================================
echo.

REM Verificar que estamos en el directorio correcto
if not exist "venv_trading\Scripts\activate.bat" (
    echo ERROR: No se encuentra el entorno virtual venv_trading
    echo Por favor, ejecuta este script desde la carpeta raiz del proyecto
    echo.
    pause
    exit /b 1
)

REM Activar el entorno virtual
echo [1/4] Activando entorno virtual...
call venv_trading\Scripts\activate.bat

REM Verificar que el entorno virtual está activo
where python | findstr "venv_trading" >nul
if errorlevel 1 (
    echo ERROR: El entorno virtual no se activó correctamente
    echo.
    pause
    exit /b 1
)
echo    OK - Entorno virtual activado
echo.

REM Verificar versión de scikit-learn
echo [2/4] Verificando scikit-learn...
python -c "import sklearn; v=sklearn.__version__; exit(0 if v>='1.7.2' else 1)" 2>nul
if errorlevel 1 (
    echo    ADVERTENCIA: scikit-learn version incorrecta o no instalada
    echo    Intentando actualizar...
    pip install --upgrade scikit-learn>=1.7.2 --quiet
)
for /f "delims=" %%i in ('python -c "import sklearn; print(sklearn.__version__)"') do set SKLEARN_VERSION=%%i
echo    OK - scikit-learn %SKLEARN_VERSION%
echo.

REM Verificar conexión a MT5 (opcional)
echo [3/4] Verificando MetaTrader 5...
python -c "import MetaTrader5 as mt5; mt5.initialize(); print('OK' if mt5.terminal_info() else 'ADVERTENCIA')" 2>nul
if errorlevel 1 (
    echo    ADVERTENCIA: MetaTrader 5 no está ejecutándose
    echo    Asegúrate de que MT5 esté abierto antes de continuar
    echo.
)
echo.

REM Ejecutar el bot
echo [4/4] Ejecutando el bot...
echo ========================================================================
echo.

python run_mt5.py

REM Capturar el código de salida
set BOT_EXIT_CODE=%errorlevel%

echo.
echo ========================================================================
if %BOT_EXIT_CODE% equ 0 (
    echo   Bot finalizado correctamente
) else (
    echo   Bot finalizado con errores (Codigo: %BOT_EXIT_CODE%^)
)
echo ========================================================================
echo.

pause
