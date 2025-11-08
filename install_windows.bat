@echo off
REM Script de instalación automatizada para Windows
REM Trading Bot - MT5 AI

echo ============================================================
echo Trading Bot - Instalacion Automatizada para Windows
echo ============================================================
echo.

REM Verificar que Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no esta en el PATH
    echo Por favor, instala Python 3.11 desde https://www.python.org/
    pause
    exit /b 1
)

echo [1/6] Python detectado correctamente
python --version
echo.

REM Crear entorno virtual si no existe
if not exist "venv_trading" (
    echo [2/6] Creando entorno virtual...
    python -m venv venv_trading
    echo Entorno virtual creado: venv_trading
) else (
    echo [2/6] Entorno virtual ya existe: venv_trading
)
echo.

REM Activar entorno virtual
echo [3/6] Activando entorno virtual...
call venv_trading\Scripts\activate.bat
echo.

REM Actualizar pip
echo [4/6] Actualizando pip...
python -m pip install --upgrade pip
echo.

REM Desinstalar TensorFlow antiguo si existe
echo [5/6] Limpiando instalaciones previas de TensorFlow...
pip uninstall tensorflow tensorflow-intel -y >nul 2>&1
echo.

REM Instalar dependencias
echo [6/6] Instalando dependencias desde requirements.txt...
echo Esto puede tomar varios minutos...
echo.
pip install -r requirements.txt

echo.
echo ============================================================
echo INSTALACION COMPLETADA
echo ============================================================
echo.
echo Para usar el bot:
echo   1. Activa el entorno virtual: venv_trading\Scripts\activate
echo   2. Configura el archivo .env con tus credenciales de MT5
echo   3. Entrena los modelos: python train_models.py
echo   4. Ejecuta el bot: python run_mt5.py
echo.
echo Para diagnosticar problemas: python diagnose_environment.py
echo.
pause
