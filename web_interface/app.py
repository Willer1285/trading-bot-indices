"""
Flask Web Application
Interfaz web para monitorear y controlar el trading bot
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
from loguru import logger
import sys
from pathlib import Path

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web_interface.database import get_database, Trade, TradeEvent, MessageTemplate, ManualNotification
from src.bot_controller import get_bot_controller
from sqlalchemy import func, and_

app = Flask(__name__)
CORS(app)

# Configuración
app.config['SECRET_KEY'] = 'trading-bot-secret-key-2024'
app.config['JSON_SORT_KEYS'] = False

# Obtener instancias globales
db = get_database()
bot_controller = get_bot_controller()

# Variable global para almacenar referencia al bot y telegram
_bot_instance = None
_telegram_instance = None


def set_bot_instance(bot, telegram):
    """Establece las instancias del bot y telegram"""
    global _bot_instance, _telegram_instance
    _bot_instance = bot
    _telegram_instance = telegram


# ==================== RUTAS PRINCIPALES ====================

@app.route('/')
def index():
    """Página principal - Dashboard"""
    return render_template('dashboard.html')


@app.route('/messages')
def messages():
    """Página de edición de mensajes"""
    return render_template('messages.html')


@app.route('/manual')
def manual_send():
    """Página de envío manual de notificaciones"""
    return render_template('manual_send.html')


# ==================== API - BOT CONTROL ====================

@app.route('/api/bot/status', methods=['GET'])
def get_bot_status():
    """Obtiene el estado actual del bot"""
    try:
        status_info = bot_controller.get_status_info()
        return jsonify({'success': True, 'data': status_info})
    except Exception as e:
        logger.error(f"Error getting bot status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    """Inicia el bot"""
    try:
        success = bot_controller.start()
        return jsonify({
            'success': success,
            'message': 'Bot iniciado' if success else 'El bot ya está en ejecución',
            'status': bot_controller.get_status()
        })
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/bot/pause', methods=['POST'])
def pause_bot():
    """Pausa el bot"""
    try:
        success = bot_controller.pause()
        return jsonify({
            'success': success,
            'message': 'Bot pausado' if success else 'El bot no está en ejecución',
            'status': bot_controller.get_status()
        })
    except Exception as e:
        logger.error(f"Error pausing bot: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/bot/resume', methods=['POST'])
def resume_bot():
    """Reanuda el bot"""
    try:
        success = bot_controller.resume()
        return jsonify({
            'success': success,
            'message': 'Bot reanudado' if success else 'El bot no está pausado',
            'status': bot_controller.get_status()
        })
    except Exception as e:
        logger.error(f"Error resuming bot: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    """Detiene el bot"""
    try:
        success = bot_controller.stop()
        return jsonify({
            'success': success,
            'message': 'Bot detenido' if success else 'El bot ya está detenido',
            'status': bot_controller.get_status()
        })
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== API - DASHBOARD STATISTICS ====================

@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """Obtiene estadísticas para el dashboard"""
    try:
        period = request.args.get('period', 'all')  # hour, day, week, month, year, all
        session = db.get_session()

        # Calcular fecha límite según el período
        now = datetime.utcnow()
        date_filter = None

        if period == 'hour':
            date_filter = now - timedelta(hours=1)
        elif period == 'day':
            date_filter = now - timedelta(days=1)
        elif period == 'week':
            date_filter = now - timedelta(weeks=1)
        elif period == 'month':
            date_filter = now - timedelta(days=30)
        elif period == 'year':
            date_filter = now - timedelta(days=365)

        # Query base
        query = session.query(Trade)
        if date_filter:
            query = query.filter(Trade.opened_at >= date_filter)

        all_trades = query.all()

        # Estadísticas
        total_signals = len(all_trades)
        closed_trades = [t for t in all_trades if t.status.startswith('CLOSED')]

        stats = {
            'total_signals': total_signals,
            'open_trades': len([t for t in all_trades if t.status == 'OPEN']),
            'closed_tp1': len([t for t in closed_trades if t.status == 'CLOSED_TP1']),
            'closed_tp2': len([t for t in closed_trades if t.status == 'CLOSED_TP2']),
            'closed_sl': len([t for t in closed_trades if t.status == 'CLOSED_SL']),
            'closed_be': len([t for t in closed_trades if t.status == 'CLOSED_BE']),
            'total_profit': sum(t.profit for t in closed_trades if t.profit),
            'win_rate': (len([t for t in closed_trades if t.profit and t.profit > 0]) / len(closed_trades) * 100) if closed_trades else 0,
            'period': period
        }

        # Eventos especiales
        events_query = session.query(TradeEvent)
        if date_filter:
            events_query = events_query.filter(TradeEvent.timestamp >= date_filter)

        events = events_query.all()
        stats['events_in_profit'] = len([e for e in events if e.event_type == 'IN_PROFIT'])
        stats['events_be'] = len([e for e in events if e.event_type == 'BE_ACTIVATED'])
        stats['events_ts'] = len([e for e in events if e.event_type == 'TS_ACTIVATED'])

        session.close()

        return jsonify({'success': True, 'data': stats})

    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dashboard/trades', methods=['GET'])
def get_recent_trades():
    """Obtiene la lista de trades recientes"""
    try:
        limit = request.args.get('limit', 50, type=int)
        session = db.get_session()

        trades = session.query(Trade).order_by(Trade.opened_at.desc()).limit(limit).all()

        trades_data = []
        for trade in trades:
            trades_data.append({
                'id': trade.id,
                'signal_id': trade.signal_id,
                'symbol': trade.symbol,
                'signal_type': trade.signal_type,
                'entry_price': trade.entry_price,
                'sl': trade.sl,
                'tp1': trade.tp1,
                'tp2': trade.tp2,
                'lot_size': trade.lot_size,
                'confidence': trade.confidence,
                'timeframe': trade.timeframe,
                'status': trade.status,
                'opened_at': trade.opened_at.isoformat() if trade.opened_at else None,
                'closed_at': trade.closed_at.isoformat() if trade.closed_at else None,
                'profit': trade.profit,
                'mt5_ticket': trade.mt5_ticket
            })

        session.close()

        return jsonify({'success': True, 'data': trades_data})

    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== API - MESSAGE TEMPLATES ====================

@app.route('/api/messages/templates', methods=['GET'])
def get_message_templates():
    """Obtiene todas las plantillas de mensajes"""
    try:
        session = db.get_session()

        templates = session.query(MessageTemplate).all()

        templates_data = []
        for template in templates:
            templates_data.append({
                'id': template.id,
                'message_type': template.message_type,
                'name': template.name,
                'template': template.template,
                'enabled': template.enabled,
                'variables': template.variables,
                'updated_at': template.updated_at.isoformat() if template.updated_at else None
            })

        session.close()

        return jsonify({'success': True, 'data': templates_data})

    except Exception as e:
        logger.error(f"Error getting templates: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/messages/templates/<int:template_id>', methods=['PUT'])
def update_message_template(template_id):
    """Actualiza una plantilla de mensaje"""
    try:
        data = request.json
        session = db.get_session()

        template = session.query(MessageTemplate).filter_by(id=template_id).first()

        if not template:
            return jsonify({'success': False, 'error': 'Template not found'}), 404

        # Actualizar campos
        if 'template' in data:
            template.template = data['template']
        if 'enabled' in data:
            template.enabled = data['enabled']

        template.updated_at = datetime.utcnow()

        session.commit()
        session.close()

        return jsonify({'success': True, 'message': 'Template updated successfully'})

    except Exception as e:
        logger.error(f"Error updating template: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/messages/templates/<int:template_id>/toggle', methods=['POST'])
def toggle_message_template(template_id):
    """Activa/desactiva una plantilla"""
    try:
        session = db.get_session()

        template = session.query(MessageTemplate).filter_by(id=template_id).first()

        if not template:
            session.close()
            return jsonify({'success': False, 'error': 'Template not found'}), 404

        # Toggle enabled state
        template.enabled = not template.enabled
        new_enabled_state = template.enabled  # Capture before closing session

        session.commit()
        session.close()

        return jsonify({'success': True, 'enabled': new_enabled_state})

    except Exception as e:
        logger.error(f"Error toggling template: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== API - MANUAL NOTIFICATIONS ====================

@app.route('/api/notifications/send', methods=['POST'])
def send_manual_notification():
    """Envía una notificación manual via Telegram (con soporte para imágenes)"""
    try:
        # Obtener datos del formulario
        message = request.form.get('message', '').strip()
        image_file = request.files.get('image')

        if not message and not image_file:
            return jsonify({'success': False, 'error': 'Message or image is required'}), 400

        session = db.get_session()

        # Intentar enviar via Telegram
        success = False
        error_msg = None

        if _telegram_instance:
            try:
                import asyncio
                import tempfile
                import os
                import threading
                from concurrent.futures import ThreadPoolExecutor

                # Función para ejecutar en thread separado con su propio loop
                def send_telegram_message():
                    try:
                        # Crear un nuevo loop para este thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)

                        try:
                            # Si hay imagen, enviar con send_photo
                            if image_file:
                                # Guardar temporalmente la imagen
                                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.filename)[1]) as tmp_file:
                                    image_file.save(tmp_file.name)
                                    tmp_path = tmp_file.name

                                try:
                                    # Enviar imagen con caption
                                    new_loop.run_until_complete(_telegram_instance.send_photo(tmp_path, caption=message if message else None))
                                finally:
                                    # Eliminar archivo temporal
                                    if os.path.exists(tmp_path):
                                        os.remove(tmp_path)
                            else:
                                # Solo mensaje de texto
                                new_loop.run_until_complete(_telegram_instance.send_message(message))

                            return True, None
                        finally:
                            new_loop.close()
                    except Exception as e:
                        return False, str(e)

                # Ejecutar en thread separado
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(send_telegram_message)
                success, error_msg = future.result(timeout=10)  # 10 segundos timeout
                executor.shutdown(wait=False)

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error sending Telegram notification: {e}")
        else:
            error_msg = "Telegram bot not initialized"

        # Guardar en base de datos
        notification = ManualNotification(
            message=message if message else '[Imagen enviada]',
            success=success,
            error_message=error_msg
        )
        session.add(notification)
        session.commit()
        session.close()

        return jsonify({
            'success': success,
            'message': 'Notification sent successfully' if success else 'Failed to send notification',
            'error': error_msg
        })

    except Exception as e:
        logger.error(f"Error sending manual notification: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/notifications/history', methods=['GET'])
def get_notification_history():
    """Obtiene el historial de notificaciones manuales"""
    try:
        limit = request.args.get('limit', 20, type=int)
        session = db.get_session()

        notifications = session.query(ManualNotification).order_by(ManualNotification.sent_at.desc()).limit(limit).all()

        notifications_data = []
        for notif in notifications:
            notifications_data.append({
                'id': notif.id,
                'message': notif.message,
                'sent_at': notif.sent_at.isoformat() if notif.sent_at else None,
                'success': notif.success,
                'error_message': notif.error_message
            })

        session.close()

        return jsonify({'success': True, 'data': notifications_data})

    except Exception as e:
        logger.error(f"Error getting notification history: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def run_flask_app(host='0.0.0.0', port=5000, debug=False):
    """
    Inicia la aplicación Flask

    Args:
        host: Host para el servidor
        port: Puerto para el servidor
        debug: Modo debug
    """
    logger.info(f"Starting Flask web interface on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == '__main__':
    run_flask_app(debug=True)
