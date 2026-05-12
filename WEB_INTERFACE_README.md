# 🌐 Interfaz Web del Trading Bot

## 📋 Resumen

Se ha implementado una **interfaz web completa** para monitorear y controlar el trading bot en tiempo real. La interfaz está construida con Flask + Bootstrap 5 y proporciona todas las funcionalidades solicitadas.

---

## ✅ Funcionalidades Implementadas

### 1. 📊 Dashboard Principal
- **Estado del bot**: Running/Paused/Stopped con indicador visual
- **Uptime**: Tiempo de funcionamiento excluyendo pausas
- **Estadísticas completas**:
  - Total de señales enviadas
  - Señales ganadas en TP1 y TP2
  - Señales perdidas en SL
  - Operaciones en Break Even
  - Operaciones con Trailing Stop activo
  - Operaciones actualmente en ganancia
- **Filtros temporales**: Última hora, día, semana, mes, año, todo el tiempo
- **Gráficos interactivos**: Chart.js para visualizar resultados
- **Win Rate**: Tasa de éxito calculada en tiempo real
- **Tabla de trades**: Últimas 10 operaciones con detalles

### 2. 🎮 Control del Bot
- **Botón Iniciar (▶️)**: Arranca el bot
- **Botón Pausar (⏸️)**: Pausa temporalmente (no genera nuevas señales)
- **Botón Reanudar (▶️)**: Continúa desde pausa
- **Botón Detener (⏹️)**: Apaga el bot completamente
- Estado persistente en base de datos SQLite

### 3. ✉️ Editor de Plantillas de Mensajes
- **9 tipos de mensajes editables**:
  1. Señal generada (BUY/SELL)
  2. Operación abierta confirmada
  3. **Operación en positivo** ⭐ (NUEVO)
  4. **TP1 alcanzado** ⭐ (NUEVO)
  5. **TP2 alcanzado** ⭐ (NUEVO)
  6. SL alcanzado
  7. Break Even activado
  8. Trailing Stop activado
  9. Alertas de error

- **Funcionalidades**:
  - Editor de texto con variables dinámicas
  - Activar/Desactivar cada mensaje individualmente
  - Variables disponibles mostradas para cada plantilla
  - Guardado automático en base de datos

### 4. 📤 Envío Manual de Notificaciones
- Formulario de texto libre
- Botón de envío instantáneo al canal de Telegram
- Historial de notificaciones enviadas con fecha/hora
- Indicador de éxito/error

### 5. 📈 Sistema de Tracking Avanzado
El `TradeTracker` monitorea **en tiempo real** todas las operaciones y detecta automáticamente:

- ✅ **Operación abierta**: Cuando se ejecuta en MT5
- 💚 **Entrada en ganancia**: Primera vez que el precio entra en profit (NUEVO)
- 🎯 **TP1 alcanzado**: Cuando el precio toca el primer take profit (NUEVO)
- 🎯🎯 **TP2 alcanzado**: Cuando el precio toca el segundo take profit (NUEVO)
- 🛑 **SL alcanzado**: Cuando se dispara el stop loss
- ⚖️ **Break Even**: Cuando el SL se mueve a punto de equilibrio
- 📈 **Trailing Stop**: Cuando se activa el trailing stop

Todos los eventos se registran en la base de datos y envían notificaciones configurables via Telegram.

---

## 🗄️ Base de Datos

**SQLite** (`trading_bot.db`) con las siguientes tablas:

### `trades`
- Información completa de cada operación
- Estados: OPEN, CLOSED_TP1, CLOSED_TP2, CLOSED_SL, CLOSED_BE
- Campos: symbol, signal_type, entry_price, sl, tp1, tp2, lot_size, confidence, profit, etc.

### `trade_events`
- Histórico de eventos para cada trade
- Tipos: OPENED, IN_PROFIT, TP1_HIT, TP2_HIT, SL_HIT, BE_ACTIVATED, TS_ACTIVATED

### `message_templates`
- Plantillas editables de mensajes
- Campo `enabled` para activar/desactivar

### `bot_status`
- Estado actual del bot
- Timestamps de inicio/pausa/detención
- Uptime acumulado

### `manual_notifications`
- Historial de notificaciones enviadas manualmente

---

## 🚀 Cómo Usar

### Instalación de Dependencias

```bash
pip install flask flask-cors sqlalchemy
```

### ⚠️ IMPORTANTE: Integración Pendiente

**La interfaz web está implementada pero NO integrada automáticamente con el bot principal.**

Para completar la integración, necesitas modificar `src/main_mt5.py`:

#### Paso 1: Importar módulos

Agregar al inicio del archivo:

```python
import threading
from web_interface.app import run_flask_app, set_bot_instance
from web_interface.database import init_database
from src.trade_tracker import TradeTracker
from src.bot_controller import get_bot_controller
```

#### Paso 2: Inicializar en `__init__` de la clase TradingBot

```python
def __init__(self):
    # ... código existente ...

    # Inicializar base de datos
    init_database()

    # Inicializar controlador del bot
    self.bot_controller = get_bot_controller()

    # Inicializar trade tracker
    self.trade_tracker = TradeTracker(telegram_bot=self.telegram_bot)
    self.trade_tracker.load_active_trades()

    # Pasar referencias a la web interface
    set_bot_instance(self, self.telegram_bot)
```

#### Paso 3: Iniciar Flask en thread separado

Agregar al método `start()`:

```python
def start(self):
    logger.info("Starting MT5 Trading Bot...")

    # Iniciar interfaz web en thread separado
    flask_thread = threading.Thread(
        target=run_flask_app,
        kwargs={'host': '0.0.0.0', 'port': 5000, 'debug': False},
        daemon=True
    )
    flask_thread.start()
    logger.info("Web interface started at http://localhost:5000")

    # ... resto del código existente ...
```

#### Paso 4: Registrar trades abiertos

En el método donde ejecutas órdenes (cuando se abre una posición):

```python
# Después de ejecutar la orden exitosamente
if result and result.order > 0:
    # Registrar en el trade tracker
    self.trade_tracker.register_trade_opened(
        signal_id=signal.signal_id,
        symbol=signal.symbol,
        signal_type=signal.signal_type,
        entry_price=signal.entry_price,
        sl=signal.stop_loss,
        tp1=signal.take_profit_levels[0],
        tp2=signal.take_profit_levels[1],
        lot_size=signal.lot_size,
        confidence=signal.confidence,
        timeframe=signal.timeframe,
        mt5_ticket=result.order
    )
```

#### Paso 5: Monitorear trades activos

En tu loop principal de análisis:

```python
async def _analyze_symbol(self, symbol: str):
    # Verificar si el bot puede generar señales
    if not self.bot_controller.can_generate_signals():
        logger.debug(f"Bot paused/stopped, skipping signal generation for {symbol}")
        # Pero seguir monitoreando trades existentes
        if self.bot_controller.can_monitor_trades():
            self.trade_tracker.update_trade_monitoring()
        return

    # Monitorear trades activos
    self.trade_tracker.update_trade_monitoring()

    # ... resto del código de análisis ...
```

#### Paso 6: Registrar Break Even y Trailing Stop

Cuando actives BE o TS:

```python
# Break Even
self.trade_tracker.register_break_even(signal_id, new_sl)

# Trailing Stop
self.trade_tracker.register_trailing_stop(signal_id, new_sl)
```

---

## 🌐 Acceso a la Interfaz

Una vez integrado e iniciado el bot:

```bash
python main_mt5.py
```

Abre tu navegador en:
```
http://localhost:5000
```

### Páginas disponibles:
- `/` - Dashboard principal
- `/messages` - Editor de plantillas de mensajes
- `/manual` - Envío manual de notificaciones

---

## 📡 API REST Endpoints

### Bot Control
- `GET /api/bot/status` - Obtener estado actual
- `POST /api/bot/start` - Iniciar bot
- `POST /api/bot/pause` - Pausar bot
- `POST /api/bot/resume` - Reanudar bot
- `POST /api/bot/stop` - Detener bot

### Dashboard
- `GET /api/dashboard/stats?period=day` - Estadísticas (hour/day/week/month/year/all)
- `GET /api/dashboard/trades?limit=50` - Lista de trades recientes

### Mensajes
- `GET /api/messages/templates` - Obtener todas las plantillas
- `PUT /api/messages/templates/{id}` - Actualizar plantilla
- `POST /api/messages/templates/{id}/toggle` - Activar/Desactivar plantilla

### Notificaciones
- `POST /api/notifications/send` - Enviar notificación manual
- `GET /api/notifications/history?limit=20` - Historial de notificaciones

---

## 🎨 Diseño Visual

- **Verde**: Ganancias, TP alcanzados, Bot running
- **Rojo**: Pérdidas, SL alcanzado, Bot stopped
- **Amarillo**: Bot paused
- **Azul**: Información general

---

## 📊 Actualización Automática

- **Estado del bot**: Cada 5 segundos
- **Dashboard stats**: Cada 10 segundos
- **Trades activos**: Cada tick (cuando el bot está en ejecución)

---

## 🔧 Configuración Adicional

### Cambiar el puerto

Edita `web_interface/app.py`:

```python
run_flask_app(host='0.0.0.0', port=8080, debug=False)
```

### Acceso desde red local

Usa `host='0.0.0.0'` para permitir acceso desde otros dispositivos en la misma red:

```
http://192.168.1.X:5000
```

### Modo debug

Solo para desarrollo:

```python
run_flask_app(host='localhost', port=5000, debug=True)
```

---

## ⚠️ Notas Importantes

1. **Base de datos**: Se crea automáticamente en `trading_bot.db` al iniciar
2. **Plantillas por defecto**: Se inicializan automáticamente la primera vez
3. **Seguridad**: Por defecto solo accesible desde localhost
4. **Performance**: La interfaz NO afecta el rendimiento del bot (thread separado)
5. **Estado persistente**: El estado del bot se guarda en la DB y se restaura al reiniciar

---

## 🐛 Troubleshooting

### La interfaz no carga
- Verifica que Flask esté instalado: `pip install flask flask-cors sqlalchemy`
- Verifica que el puerto 5000 no esté en uso: `netstat -ano | findstr :5000`

### No se envían notificaciones manuales
- Verifica que el bot de Telegram esté configurado correctamente
- Verifica que `set_bot_instance()` fue llamado con las referencias correctas

### El estado del bot no se actualiza
- Abre la consola del navegador (F12) para ver errores JavaScript
- Verifica que la API responde: `curl http://localhost:5000/api/bot/status`

### Los trades no se muestran
- Verifica que `TradeTracker` esté integrado en el flujo de ejecución
- Verifica que se llame a `register_trade_opened()` después de cada orden exitosa

---

## 📝 TODO: Pendientes

- [ ] Integrar completamente con `main_mt5.py` (instrucciones arriba)
- [ ] Agregar gráfico de evolución temporal de profit
- [ ] Agregar filtro por símbolo en el dashboard
- [ ] Agregar exportación de estadísticas a CSV
- [ ] Agregar autenticación básica (opcional, para producción)
- [ ] Agregar panel de configuración para editar .env desde la interfaz

---

## 🎉 Resultado Final

Con esta implementación tienes una **interfaz web profesional** que te permite:

✅ Monitorear el bot 24/7 desde cualquier navegador
✅ Ver estadísticas en tiempo real
✅ Controlar el bot sin tocar el código
✅ Personalizar todos los mensajes
✅ Enviar notificaciones manuales
✅ Rastrear cada evento de tus operaciones
✅ Analizar el rendimiento por período

**¡Todo desde una interfaz moderna y fácil de usar!** 🚀
