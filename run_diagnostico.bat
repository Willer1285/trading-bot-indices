@echo off
REM ========================================================================
REM Script de Diagnóstico Automático
REM ========================================================================
REM Este script activa automáticamente el entorno virtual y ejecuta el diagnóstico.
REM Uso: Ejecuta "run_diagnostico.bat" desde CMD o haz doble clic
REM ========================================================================

echo.
echo ========================================================================
echo   DIAGNOSTICO DEL ENTORNO - Verificacion Automatica
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
echo Activando entorno virtual...
call venv_trading\Scripts\activate.bat
echo.

REM Ejecutar diagnóstico
python diagnose_environment.py

echo.
echo ========================================================================
echo   Diagnostico completado
echo ========================================================================
echo.

pause
