# Sistema de Watchdog - Prevención de Bloqueos en Producción

## 🎯 Problema Resuelto

El bot puede quedarse **paralizado** durante horas esperando respuestas de MT5, modelos de IA, o conexiones de red. Esto es crítico en producción VPS donde no hay supervisión humana.

**Ejemplo de problema detectado:**
```
2025-11-19 15:37:13 | Starting New Analysis Cycle
[SILENCIO DE 3 HORAS - BOT CONGELADO]
2025-11-19 18:10:45 | Analyzing PainX 400...
```

## 🛡️ Solución Implementada

### 1. **Sistema de Watchdog**
Monitoreo independiente en thread separado que:
- ✅ Detecta cuando el main loop deja de responder
- ✅ Envía alertas críticas por logs y Telegram
- ✅ Registra estadísticas del sistema (CPU, memoria)
- ✅ Intenta recuperación automática

### 2. **Heartbeat Logging**
Logs periódicos que demuestran que el bot está vivo:
```
💓 Heartbeat #15 - Bot is alive (last: 60s ago) - Health: HEALTHY, Checks: 30
```

### 3. **Timeouts Robustos**
Todas las operaciones críticas tienen timeouts:
- **MT5 fetch_ohlcv**: 30 segundos
- **Análisis por símbolo**: 30 segundos (configurado en main loop)
- **Operaciones de red**: timeouts estándar

### 4. **Alertas Automáticas**
Cuando se detecta un bloqueo (>3 minutos sin heartbeat):
```
🚨 CRITICAL: Bot Freeze Detected!

⏱️ Time since last heartbeat: 185.2s
⚠️ Status: Main loop is not responding

Possible causes:
• MT5 connection blocked
• Network issues
• Resource constraints (CPU/Memory)
• Model inference timeout

Action: Monitoring for auto-recovery...
```

## 📊 Configuración del Watchdog

En `src/main_mt5.py`:

```python
self.watchdog = BotWatchdog(
    timeout_seconds=180,  # 3 minutos sin heartbeat = alerta
    check_interval=30,    # Verificar cada 30 segundos
    alert_callback=self._on_freeze_detected,
    recovery_callback=self._on_freeze_recovery
)
```

### Parámetros Ajustables:

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `timeout_seconds` | 180 | Tiempo máximo sin heartbeat antes de alertar |
| `check_interval` | 30 | Intervalo de verificación en segundos |
| `log_interval` | 60 | Intervalo de heartbeat logs |

## 🔍 Monitoreo en Logs

### Logs Normales (Bot Saludable):
```
2025-11-20 10:30:15 | INFO | Watchdog check OK (last heartbeat: 2.1s ago)
2025-11-20 10:31:00 | INFO | 💓 Heartbeat #5 - Bot is alive - Health: HEALTHY, Checks: 10
```

### Logs de Alerta (Bot Congelado):
```
2025-11-20 10:35:00 | CRITICAL | 🚨 BOT FREEZE DETECTED! No heartbeat for 185.3s (timeout: 180s)
2025-11-20 10:35:00 | WARNING | 🔄 Attempting automatic recovery from freeze...
2025-11-20 10:35:00 | INFO | System status - CPU: 45.2%, Memory: 512.3 MB
2025-11-20 10:35:00 | ERROR | MT5 connection is down - this may be causing the freeze
```

### Logs de Recuperación:
```
2025-11-20 10:38:45 | SUCCESS | 🔄 Bot recovered from freeze! Duration: 223.7s
```

## 🚨 Qué Hacer Cuando se Detecta un Bloqueo

### 1. **Revisar los Logs**
Buscar el mensaje de alerta:
```bash
grep "BOT FREEZE" logs/trading_bot.log
```

### 2. **Verificar Estado del Sistema**
El watchdog registra automáticamente:
- Uso de CPU
- Uso de memoria
- Estado de conexión MT5

### 3. **Causas Comunes y Soluciones**

| Causa | Solución |
|-------|----------|
| **MT5 desconectado** | Verificar credenciales y conexión de red |
| **Red bloqueada** | Revisar firewall, VPN, o problemas de ISP |
| **Memoria alta** | Reiniciar bot, revisar memory leaks |
| **CPU alta** | Reducir símbolos monitoreados o intervalo de análisis |
| **Modelo IA lento** | Optimizar modelos o usar GPU |

### 4. **Recuperación Manual**

Si el bot no se recupera automáticamente:

```bash
# 1. Detener el bot
pkill -f run_mt5.py

# 2. Revisar logs para identificar causa
tail -n 100 logs/trading_bot.log

# 3. Reiniciar el bot
python run_mt5.py
```

## 📈 Estadísticas del Watchdog

El watchdog registra estadísticas disponibles en:
- **Logs**: Cada minuto en heartbeat
- **Web Interface**: Dashboard de monitoreo
- **API**: `watchdog.get_statistics()`

Ejemplo de estadísticas:
```python
{
    'is_running': True,
    'uptime_hours': 24.5,
    'total_checks': 2940,
    'alerts_sent': 0,
    'recoveries_attempted': 0,
    'is_frozen': False,
    'last_heartbeat': '2025-11-20T10:45:32',
    'time_since_heartbeat': 1.2
}
```

## 🔧 Personalización

### Ajustar Sensibilidad del Watchdog

Para entornos con análisis más lentos:
```python
self.watchdog = BotWatchdog(
    timeout_seconds=300,  # 5 minutos (menos sensible)
    check_interval=60,    # Verificar cada minuto
)
```

Para entornos que requieren respuesta rápida:
```python
self.watchdog = BotWatchdog(
    timeout_seconds=120,  # 2 minutos (más sensible)
    check_interval=20,    # Verificar cada 20 segundos
)
```

### Deshabilitar Watchdog (No Recomendado)

Si necesitas deshabilitar temporalmente:
```python
# En src/main_mt5.py, comentar estas líneas:
# self.watchdog.start()
# self.watchdog.heartbeat()
```

## ✅ Verificación del Sistema

Para verificar que el watchdog está funcionando:

1. **Verificar inicio del watchdog**:
   ```bash
   grep "Watchdog started" logs/trading_bot.log
   ```

2. **Verificar heartbeats periódicos**:
   ```bash
   grep "Heartbeat #" logs/trading_bot.log | tail -n 5
   ```

3. **Simular un bloqueo** (testing):
   ```python
   # En código de prueba, agregar:
   import time
   time.sleep(200)  # Simular bloqueo de 3+ minutos
   ```

## 🎯 Mejoras Futuras

- [ ] Integración con servicios de monitoreo externos (Datadog, New Relic)
- [ ] Auto-restart del bot cuando se detectan bloqueos persistentes
- [ ] Métricas de performance en dashboard web
- [ ] Alertas por email/SMS además de Telegram
- [ ] Análisis de patrones de bloqueo para predicción

## 📞 Soporte

Si experimentas bloqueos frecuentes:
1. Revisa los logs con las herramientas de este documento
2. Ajusta la configuración del watchdog según tu entorno
3. Considera optimizar los modelos de IA o reducir la carga de análisis
4. Verifica la estabilidad de la conexión MT5 y red

---

**Última actualización:** 2025-11-20
**Versión del sistema:** v2.0 con Watchdog Protection
