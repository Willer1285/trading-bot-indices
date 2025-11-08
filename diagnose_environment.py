"""
============================================================================
Script de diagnóstico para identificar problemas con el entorno
Trading Bot - MT5 AI - Python 3.13+
============================================================================
Ejecuta este script para verificar tu configuración antes de ejecutar el bot
"""

import sys
import platform

print("=" * 70)
print("DIAGNÓSTICO DEL ENTORNO - Trading Bot MT5 AI")
print("Versión requerida: Python 3.13.9+")
print("=" * 70)
print()

# 1. Información del sistema
print("1. INFORMACIÓN DEL SISTEMA")
print("-" * 70)
print(f"   Sistema Operativo: {platform.system()} {platform.release()}")
print(f"   Arquitectura: {platform.machine()}")
print(f"   Versión de Python: {sys.version}")
print(f"   Ejecutable Python: {sys.executable}")

# Verificar versión de Python
if sys.version_info >= (3, 13):
    print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - Versión compatible")
elif sys.version_info >= (3, 12):
    print(f"   ⚠️  Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - Se recomienda actualizar a 3.13.9+")
else:
    print(f"   ❌ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - VERSIÓN NO COMPATIBLE")
    print(f"   SOLUCIÓN: Instalar Python 3.13.9 desde https://www.python.org/")

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
print("-" * 70)

tensorflow_ok = False
try:
    import tensorflow as tf
    tf_version = tuple(map(int, tf.__version__.split('.')[:2]))

    print(f"   ✅ TensorFlow instalado: v{tf.__version__}")

    # Verificar versión mínima para Python 3.13
    if tf_version >= (2, 20):
        print(f"   ✅ Versión compatible con Python 3.13 (2.20+)")
        tensorflow_ok = True
    elif tf_version >= (2, 16):
        print(f"   ⚠️  Versión antigua. Actualizar a 2.20+ para Python 3.13")
        print(f"   SOLUCIÓN: pip install --upgrade tensorflow")
    else:
        print(f"   ❌ Versión demasiado antigua ({tf.__version__})")
        print(f"   SOLUCIÓN: pip install tensorflow>=2.20.0")

    # Verificar Keras
    try:
        print(f"   ✅ Keras disponible: {tf.keras.__version__}")
    except Exception as e:
        print(f"   ⚠️  Keras: {e}")

    # Verificar módulos internos críticos
    print()
    print("   Verificando módulos internos:")

    modules_to_check = {
        'tensorflow.keras.models': ['Sequential', 'load_model'],
        'tensorflow.keras.layers': ['LSTM', 'Dense', 'Dropout'],
        'tensorflow.keras.callbacks': ['EarlyStopping'],
        'tensorflow.keras.metrics': ['AUC', 'Precision', 'Recall']
    }

    all_modules_ok = True
    for module_name, classes in modules_to_check.items():
        try:
            module = __import__(module_name, fromlist=classes)
            for cls in classes:
                getattr(module, cls)
            print(f"   ✅ {module_name:35} - OK")
        except ImportError as e:
            print(f"   ❌ {module_name:35} - ERROR: {e}")
            all_modules_ok = False

    # Test de creación de modelo
    print()
    print("   Test de funcionalidad:")
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        model = Sequential([Dense(1, input_shape=(10,))])
        print(f"   ✅ Creación de modelo de prueba - OK")
        tensorflow_ok = tensorflow_ok and all_modules_ok
    except Exception as e:
        print(f"   ❌ Creación de modelo de prueba - ERROR: {e}")
        tensorflow_ok = False

except ImportError as e:
    print(f"   ❌ TensorFlow NO instalado o corrupto")
    print(f"   Error: {e}")
    print()
    print(f"   SOLUCIÓN:")
    print(f"   pip install tensorflow>=2.20.0")

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
print("-" * 70)

models_dir = Path('models')
if models_dir.exists():
    # Buscar archivos de modelos recursivamente en todos los subdirectorios
    model_files = list(models_dir.glob('**/*.pkl')) + list(models_dir.glob('**/*.keras'))

    # Filtrar el archivo .gitkeep si existe
    model_files = [f for f in model_files if f.name != '.gitkeep']

    if model_files:
        print(f"   ✅ Directorio de modelos existe")
        print(f"   ✅ Archivos de modelos encontrados: {len(model_files)}")

        # Agrupar por símbolo/timeframe
        symbols_trained = set()
        for model in model_files:
            # Extraer símbolo y timeframe del path
            # Ejemplo: models/GainX_1200/GainX 1200_H1/random_forest.pkl
            parts = model.parts
            if len(parts) >= 3:
                symbol_timeframe = f"{parts[-3]}/{parts[-2]}"
                symbols_trained.add(symbol_timeframe)

        print(f"   ✅ Símbolos/Timeframes entrenados: {len(symbols_trained)}")

        # Mostrar algunos símbolos entrenados (primeros 5)
        for i, st in enumerate(sorted(symbols_trained)):
            if i < 5:
                print(f"      - {st}")
            elif i == 5:
                print(f"      - ... y {len(symbols_trained) - 5} más")
                break

        # Verificar tipos de modelos
        model_types = {}
        for model in model_files:
            if 'random_forest' in model.name:
                model_types['Random Forest'] = model_types.get('Random Forest', 0) + 1
            elif 'gradient_boosting' in model.name:
                model_types['Gradient Boosting'] = model_types.get('Gradient Boosting', 0) + 1
            elif 'lstm' in model.name and model.suffix == '.keras':
                model_types['LSTM (Keras)'] = model_types.get('LSTM (Keras)', 0) + 1
            elif 'meta_model' in model.name:
                model_types['Meta Model'] = model_types.get('Meta Model', 0) + 1

        print()
        print("   Tipos de modelos encontrados:")
        for model_type, count in sorted(model_types.items()):
            print(f"      - {model_type:20} {count:3} modelos")
    else:
        print(f"   ⚠️  Directorio de modelos existe pero está vacío")
        print(f"   SOLUCIÓN: Ejecutar 'python train_models.py' primero")
else:
    print(f"   ❌ Directorio de modelos no existe")
    print(f"   SOLUCIÓN: Ejecutar 'python train_models.py' primero")

print()

# Conclusión final
print("=" * 70)
has_issues = missing_modules or not tensorflow_ok or sys.version_info < (3, 13)

if has_issues:
    print("⚠️  RESULTADO: SE ENCONTRARON PROBLEMAS CON EL ENTORNO")
    print()
    print("ACCIONES REQUERIDAS:")
    print()

    if sys.version_info < (3, 13):
        print("1. ACTUALIZAR PYTHON:")
        print("   - Descargar Python 3.13.9 desde: https://www.python.org/")
        print("   - Instalar marcando 'Add Python to PATH'")
        print("   - Crear nuevo entorno virtual con Python 3.13")
        print()

    if missing_modules:
        print("2. INSTALAR MÓDULOS FALTANTES:")
        print("   pip install -r requirements.txt")
        print()
        print("   Módulos faltantes:")
        for package, _ in missing_modules:
            print(f"   - {package}")
        print()

    if not tensorflow_ok:
        print("3. REINSTALAR/ACTUALIZAR TENSORFLOW:")
        print("   pip uninstall tensorflow tensorflow-intel keras -y")
        print("   pip install tensorflow>=2.20.0")
        print()

    print("4. VERIFICAR NUEVAMENTE:")
    print("   python diagnose_environment.py")

else:
    print("✅ RESULTADO: ENTORNO CONFIGURADO CORRECTAMENTE")
    print()
    print("Configuración verificada:")
    print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"   ✅ Todos los módulos instalados")
    print(f"   ✅ TensorFlow funcionando correctamente")
    print()
    print("El bot está listo para ejecutarse.")
    print()
    print("PRÓXIMOS PASOS:")
    print("   1. Configurar .env con credenciales de MT5")
    print("   2. Entrenar modelos: python train_models.py")
    print("   3. Ejecutar bot: python run_mt5.py")

print("=" * 70)
print()
print(f"Diagnóstico completado - {platform.system()} - Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print("=" * 70)
