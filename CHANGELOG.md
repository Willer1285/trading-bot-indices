# 📝 Registro de Cambios (CHANGELOG)

> **IMPORTANTE:** Este archivo debe actualizarse **SIN EXCEPCIONES** cada vez que se realice un cambio en el bot.
>
> **Formato de versionado:** Seguimos [Semantic Versioning](https://semver.org/):
> - **MAJOR** (X.0.0): Cambios incompatibles con versiones anteriores
> - **MINOR** (0.X.0): Nuevas funcionalidades compatibles con versiones anteriores
> - **PATCH** (0.0.X): Correcciones de bugs y pequeñas mejoras

---

## 📌 Instrucciones de Uso

### Cómo registrar un cambio:

1. **Crear una nueva sección de versión** en la parte superior (justo debajo de esta sección)
2. **Formato de la sección:**
   ```markdown
   ## v[MAJOR].[MINOR].[PATCH] - [Título descriptivo] (YYYY-MM-DD)

   ### 🎯 Resumen
   Descripción breve del cambio principal

   ### ✅ Cambios Implementados
   - ✅ Cambio 1: Descripción detallada
   - ✅ Cambio 2: Descripción detallada

   ### 🐛 Bugs Corregidos (opcional)
   - 🐛 Bug 1: Descripción

   ### ⚠️ Breaking Changes (opcional, solo para MAJOR)
   - ⚠️ Cambio incompatible: Descripción

   ### 📝 Notas Adicionales (opcional)
   Información relevante para el usuario
   ```

3. **Usar emojis consistentes:**
   - ✅ Cambio completado
   - 🐛 Bug corregido
   - ⚠️ Cambio incompatible (breaking change)
   - 🔧 Configuración
   - 📊 Mejora de rendimiento
   - 🎨 Mejora de UI/UX
   - 📝 Documentación
   - 🚀 Nueva funcionalidad
   - 🔒 Seguridad

4. **Actualizar "Versión Actual" al final del archivo**

---

## 🚀 Historial de Versiones

## v2.2.0 - Documentación Completa de Estrategia y Funcionamiento (2026-03-17)

### 🎯 Resumen
Creación de documentación exhaustiva del funcionamiento interno del bot, incluyendo estrategia de trading, modelos de IA, indicadores técnicos y flujos de operación.

### ✅ Cambios Implementados

**Nuevos Archivos:**
- ✅ **ESTRATEGIA_BOT.md**: Documento completo (10,000+ palabras) que detalla:
  - Arquitectura completa del sistema con diagramas
  - Estrategia de trading detallada (señales BUY/SELL/HOLD)
  - Descripción de los 4 modelos de IA (Random Forest, Gradient Boosting, LSTM, Pattern Model)
  - Funcionamiento del Ensemble Stacking
  - Listado completo de 116 features/indicadores técnicos
  - Explicación de los 8 filtros de calidad de señales
  - Gestión de riesgo dinámica (SL, TP, Break Even, Trailing Stop, Lotaje)
  - Reglas especiales para índices sintéticos (GainX/PainX)
  - Flujo completo de operación (paso a paso)
  - Todos los parámetros configurables con ejemplos

**Mejoras en CHANGELOG.md:**
- ✅ Instrucciones claras de cómo mantener el changelog actualizado
- ✅ Formato estandarizado con Semantic Versioning
- ✅ Plantilla para registrar futuros cambios
- ✅ Uso consistente de emojis

### 📝 Notas Adicionales

Este cambio facilita:
1. **Onboarding de nuevos desarrolladores**: Pueden entender rápidamente cómo funciona el bot
2. **Debugging**: Documentación de referencia para entender el flujo
3. **Optimización**: Identificar áreas de mejora con claridad
4. **Transparencia**: Usuarios pueden entender completamente la estrategia
5. **Mantenimiento**: Registro histórico de todos los cambios realizados

**Archivo creado:** `ESTRATEGIA_BOT.md` (ubicación raíz del proyecto)

---

## v2.1.0 - Modo Canal de Señales (2025-10-26)

### ✅ Cambios Implementados

**Telegram - Solo Señales**
- ✅ El canal de Telegram **solo recibe señales** de trading
- ✅ **NO se envían confirmaciones** de ejecución de órdenes
- ✅ **NO se envían actualizaciones** horarias de balance/equity
- ✅ Canal limpio y profesional solo con señales

**Ejecución Interna**
- ✅ El bot **sigue ejecutando automáticamente** en MT5
- ✅ Todas las confirmaciones van a **logs internos**
- ✅ Detalles de tickets, volumen, precios en logs
- ✅ Sistema de monitoreo interno completo

### 📱 Qué Se Envía a Telegram

**✅ SE ENVÍA:**
```
🟢 BUY SIGNAL

Symbol: EURUSD
Timeframe: 4h

📈 ENTRY
💵 Price: 1.09500

🛑 STOP LOSS
💵 Price: 1.09200

🎯 TAKE PROFIT LEVELS
   TP1: 1.10100 (+0.55%)
   TP2: 1.10400 (+0.82%)
   TP3: 1.10900 (+1.28%)

Confidence: 85%
Signal Strength: 87/100
Risk/Reward: 1:2.5

💡 Reason: 5/6 timeframes aligned
```

**❌ NO SE ENVÍA:**
- Confirmaciones de ejecución
- Números de ticket
- Volumen ejecutado
- Actualizaciones de balance
- Actualizaciones de equity
- Estados de posiciones

### 📊 Qué Queda en Logs Internos

**Archivo: logs/trading_bot.log**

```
2025-01-26 14:30:15 | INFO | ✅ ORDER EXECUTED
  Ticket: 123456789
  Symbol: EURUSD
  Type: BUY
  Volume: 0.10 lots
  Entry: 1.09500
  SL: 1.09200
  TP: 1.10100
  Time: 2025-01-26 14:30:15

2025-01-26 15:00:00 | INFO | Hourly status
  Balance: 10150.50
  Equity: 10175.30
  Profit: 25.50
  Open Positions: 2
  Signals: 5
```

### 🎯 Beneficios

1. **Canal Profesional**
   - Solo señales limpias
   - Fácil de seguir
   - Sin ruido de confirmaciones

2. **Privacidad**
   - No se muestra cuánto se ejecuta
   - No se muestran tickets reales
   - Ideal para canales públicos/compartidos

3. **Seguimiento Interno**
   - Todo registrado en logs
   - Auditoría completa
   - Monitoreo detallado

### 🔧 Configuración

El comportamiento es automático. Si quieres cambiar:

**Habilitar confirmaciones en Telegram:**

Editar `src/main_mt5.py`, línea ~280:

```python
# Descomentar estas líneas para enviar confirmaciones
await self.telegram_bot.send_message(
    f"✅ ORDER EXECUTED\n"
    f"Ticket: {result['ticket']}\n"
    ...
)
```

**Habilitar actualizaciones horarias:**

Editar `src/main_mt5.py`, línea ~330:

```python
# Descomentar para actualizaciones horarias
await self.telegram_bot.send_message(
    f"📊 Hourly Status Update\n"
    ...
)
```

### 📁 Ver Logs

**Windows:**
```bash
type logs\trading_bot.log
```

**Linux/Mac:**
```bash
tail -f logs/trading_bot.log
```

**Ver solo ejecuciones:**
```bash
# Windows
findstr "ORDER EXECUTED" logs\trading_bot.log

# Linux/Mac
grep "ORDER EXECUTED" logs/trading_bot.log
```

---

## Versiones Anteriores

### v2.0.0 - Sistema MT5 Completo
- ✅ Integración con MetaTrader 5
- ✅ Ejecución automática de órdenes
- ✅ Compatible con Weltrade
- ✅ Gestión de riesgo automática

### v1.0.0 - Sistema Crypto Original
- ✅ Integración con exchanges crypto
- ✅ Análisis con IA
- ✅ Señales a Telegram
- ✅ Solo señales (sin ejecución)

---

## 📊 Información del Proyecto

**Última actualización:** 17 de Marzo, 2026
**Versión Actual:** v2.2.0
**Repositorio:** https://github.com/Willer1285/trading-bot-indices

---

## ⚠️ Recordatorio para Desarrolladores

**ESTE ARCHIVO DEBE ACTUALIZARSE EN CADA CAMBIO SIN EXCEPCIONES.**

Antes de hacer commit:
1. ✅ Actualizar CHANGELOG.md con los cambios realizados
2. ✅ Seguir el formato estandarizado arriba
3. ✅ Usar emojis consistentes
4. ✅ Actualizar "Versión Actual" al final
5. ✅ Incluir fecha en formato YYYY-MM-DD

**Si los cambios afectan la estrategia o funcionamiento del bot, también actualizar `ESTRATEGIA_BOT.md`**
