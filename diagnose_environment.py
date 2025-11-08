"""
Script de diagnóstico para identificar problemas con el entorno
Ejecuta este script para verificar tu configuración antes de ejecutar el bot
"""

import sys
import platform

print("=" * 60)
print("DIAGNÓSTICO DEL ENTORNO - Trading Bot")
print("=" * 60)
print()

# 1. Información del sistema
print("1. INFORMACIÓN DEL SISTEMA")
print("-" * 60)
print(f"   Sistema Operativo: {platform.system()} {platform.release()}")
print(f"   Versión de Python: {sys.version}")
print(f"   Ejecutable Python: {sys.executable}")
print()

# 2. Verificar módulos instalados
print("2. VERIFICACIÓN DE MÓDULOS REQUERIDOS")
print("-" * 60)

required_modules = {
    'MetaTrader5': 'MetaTrader5',
    'sklearn': 'scikit-learn',
    'xgboost': 'xgboost',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'ta': 'ta',
    'telegram': 'python-telegram-bot',
    'aiohttp': 'aiohttp',
    'redis': 'redis',
    'pymongo': 'pymongo',
    'asyncio': 'asyncio (built-in)',
    'aiofiles': 'aiofiles',
    'plotly': 'plotly',
    'matplotlib': 'matplotlib',
    'mplfinance': 'mplfinance',
    'dotenv': 'python-dotenv',
    'pydantic': 'pydantic',
    'yaml': 'pyyaml',
    'schedule': 'schedule',
    'ccxt': 'ccxt',
    'loguru': 'loguru',
    'prometheus_client': 'prometheus-client',
    'shap': 'shap',
    'tensorflow': 'tensorflow',
    'pytest': 'pytest',
}

missing_modules = []
installed_modules = []

for module, package_name in required_modules.items():
    try:
        if module == 'asyncio':
            # asyncio es built-in
            import asyncio
            installed_modules.append((package_name, 'built-in'))
            print(f"   ✅ {package_name:30} - OK (built-in)")
        elif module == 'sklearn':
            import sklearn
            installed_modules.append((package_name, sklearn.__version__))
            print(f"   ✅ {package_name:30} - OK (v{sklearn.__version__})")
        elif module == 'telegram':
            import telegram
            installed_modules.append((package_name, telegram.__version__))
            print(f"   ✅ {package_name:30} - OK (v{telegram.__version__})")
        elif module == 'dotenv':
            import dotenv
            # dotenv no tiene __version__ siempre
            installed_modules.append((package_name, 'installed'))
            print(f"   ✅ {package_name:30} - OK")
        elif module == 'yaml':
            import yaml
            installed_modules.append((package_name, yaml.__version__))
            print(f"   ✅ {package_name:30} - OK (v{yaml.__version__})")
        else:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            installed_modules.append((package_name, version))
            print(f"   ✅ {package_name:30} - OK (v{version})")
    except ImportError as e:
        missing_modules.append((package_name, str(e)))
        print(f"   ❌ {package_name:30} - FALTA")

print()

# 3. Verificación especial de TensorFlow
print("3. VERIFICACIÓN DETALLADA DE TENSORFLOW")
print("-" * 60)

try:
    import tensorflow as tf
    print(f"   ✅ TensorFlow instalado: v{tf.__version__}")
    print(f"   ✅ Keras disponible: {tf.keras.__version__}")

    # Verificar módulos internos problemáticos
    try:
        from tensorflow.keras.models import Sequential, load_model
        print(f"   ✅ tensorflow.keras.models - OK")
    except ImportError as e:
        print(f"   ❌ tensorflow.keras.models - ERROR: {e}")

    try:
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        print(f"   ✅ tensorflow.keras.layers - OK")
    except ImportError as e:
        print(f"   ❌ tensorflow.keras.layers - ERROR: {e}")

    try:
        from tensorflow.keras.callbacks import EarlyStopping
        print(f"   ✅ tensorflow.keras.callbacks - OK")
    except ImportError as e:
        print(f"   ❌ tensorflow.keras.callbacks - ERROR: {e}")

    try:
        from tensorflow.keras.metrics import AUC, Precision, Recall
        print(f"   ✅ tensorflow.keras.metrics - OK")
    except ImportError as e:
        print(f"   ❌ tensorflow.keras.metrics - ERROR: {e}")

    # Test básico
    try:
        model = Sequential([tf.keras.layers.Dense(1)])
        print(f"   ✅ Creación de modelo de prueba - OK")
    except Exception as e:
        print(f"   ❌ Creación de modelo de prueba - ERROR: {e}")

except ImportError as e:
    print(f"   ❌ TensorFlow NO instalado o corrupto")
    print(f"   Error: {e}")

print()

# 4. Resumen
print("4. RESUMEN")
print("-" * 60)
print(f"   Módulos instalados: {len(installed_modules)}/{len(required_modules)}")
print(f"   Módulos faltantes: {len(missing_modules)}")
print()

if missing_modules:
    print("   ⚠️  MÓDULOS FALTANTES:")
    for package, error in missing_modules:
        print(f"      - {package}")
    print()
    print("   SOLUCIÓN:")
    print("   pip install -r requirements.txt")
    print()
else:
    print("   ✅ TODOS LOS MÓDULOS ESTÁN INSTALADOS")
    print()

# 5. Verificar archivos del proyecto
print("5. VERIFICACIÓN DE ARCHIVOS DEL PROYECTO")
print("-" * 60)

import os
from pathlib import Path

project_files = [
    'run_mt5.py',
    'requirements.txt',
    'src/main_mt5.py',
    'src/ai_engine/ai_models.py',
    'src/ai_engine/market_analyzer.py',
    'src/config.py',
]

for file in project_files:
    if Path(file).exists():
        print(f"   ✅ {file}")
    else:
        print(f"   ❌ {file} - NO ENCONTRADO")

print()

# 6. Verificar modelos entrenados
print("6. VERIFICACIÓN DE MODELOS ENTRENADOS")
print("-" * 60)

models_dir = Path('models')
if models_dir.exists():
    model_files = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.keras'))
    if model_files:
        print(f"   ✅ Directorio de modelos existe")
        print(f"   ✅ Modelos encontrados: {len(model_files)}")
        for model in model_files:
            print(f"      - {model.name}")
    else:
        print(f"   ⚠️  Directorio de modelos existe pero está vacío")
        print(f"   SOLUCIÓN: Ejecutar 'python train_models.py' primero")
else:
    print(f"   ❌ Directorio de modelos no existe")
    print(f"   SOLUCIÓN: Ejecutar 'python train_models.py' primero")

print()

# Conclusión final
print("=" * 60)
if missing_modules:
    print("❌ RESULTADO: HAY PROBLEMAS CON EL ENTORNO")
    print()
    print("ACCIONES REQUERIDAS:")
    print("1. Instalar módulos faltantes: pip install -r requirements.txt")
    if 'tensorflow' in [m[0] for m in missing_modules]:
        print("2. Reinstalar TensorFlow: pip uninstall tensorflow -y && pip install tensorflow==2.16.1")
else:
    print("✅ RESULTADO: ENTORNO CONFIGURADO CORRECTAMENTE")
    print()
    print("El bot debería poder ejecutarse.")
    print("Si aún tienes problemas, verifica los modelos entrenados.")

print("=" * 60)
