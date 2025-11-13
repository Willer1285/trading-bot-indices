"""
Cleanup Manager
Manages automatic cleanup of logs, cache, and memory optimization for 24/7 VPS operation
"""

import os
import gc
import psutil
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
from loguru import logger


class CleanupManager:
    """Manages automatic cleanup and memory optimization"""

    def __init__(
        self,
        log_retention_days: int = 7,
        max_log_size_mb: int = 100,
        memory_threshold_percent: float = 80.0,
        cleanup_interval_hours: int = 6
    ):
        """
        Initialize Cleanup Manager

        Args:
            log_retention_days: Days to keep old log files
            max_log_size_mb: Maximum size of individual log files (MB)
            memory_threshold_percent: Memory usage threshold for cleanup (%)
            cleanup_interval_hours: Hours between automatic cleanups
        """
        self.log_retention_days = log_retention_days
        self.max_log_size_mb = max_log_size_mb * 1024 * 1024  # Convert to bytes
        self.memory_threshold = memory_threshold_percent
        self.cleanup_interval_hours = cleanup_interval_hours
        self.last_cleanup = datetime.utcnow()

        logger.info(
            f"Cleanup Manager initialized - "
            f"Log retention: {log_retention_days}d, "
            f"Max log size: {max_log_size_mb}MB, "
            f"Memory threshold: {memory_threshold_percent}%, "
            f"Cleanup interval: {cleanup_interval_hours}h"
        )

    def should_run_cleanup(self) -> bool:
        """Check if cleanup should run"""
        time_since_cleanup = (datetime.utcnow() - self.last_cleanup).total_seconds() / 3600
        return time_since_cleanup >= self.cleanup_interval_hours

    def run_cleanup(self) -> Dict:
        """
        Run complete cleanup routine

        Returns:
            Dictionary with cleanup statistics
        """
        logger.info("🧹 Starting automatic cleanup...")

        stats = {
            'logs_deleted': 0,
            'logs_size_freed_mb': 0,
            'memory_freed_mb': 0,
            'cache_cleared': False
        }

        try:
            # 1. Clean old log files
            log_stats = self._cleanup_old_logs()
            stats['logs_deleted'] = log_stats['files_deleted']
            stats['logs_size_freed_mb'] = log_stats['size_freed_mb']

            # 2. Rotate large log files
            rotation_stats = self._rotate_large_logs()
            stats['logs_deleted'] += rotation_stats['files_rotated']

            # 3. Check memory usage and force garbage collection if needed
            memory_before = self._get_memory_usage()
            if memory_before > self.memory_threshold:
                logger.warning(f"Memory usage high ({memory_before:.1f}%), forcing garbage collection")
                gc.collect()
                memory_after = self._get_memory_usage()
                stats['memory_freed_mb'] = (memory_before - memory_after) * psutil.virtual_memory().total / 100 / (1024 ** 2)
                logger.info(f"Memory freed: {stats['memory_freed_mb']:.2f} MB")

            # 4. Clear Python cache
            stats['cache_cleared'] = self._clear_python_cache()

            self.last_cleanup = datetime.utcnow()

            logger.success(
                f"✅ Cleanup completed - "
                f"Deleted {stats['logs_deleted']} logs, "
                f"Freed {stats['logs_size_freed_mb']:.2f}MB disk, "
                f"Freed {stats['memory_freed_mb']:.2f}MB RAM"
            )

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        return stats

    def _cleanup_old_logs(self) -> Dict:
        """Delete log files older than retention period"""
        stats = {'files_deleted': 0, 'size_freed_mb': 0}

        logs_dir = Path("logs")
        if not logs_dir.exists():
            return stats

        cutoff_date = datetime.utcnow() - timedelta(days=self.log_retention_days)

        for log_file in logs_dir.glob("*.log*"):
            try:
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)

                if file_mtime < cutoff_date:
                    file_size = log_file.stat().st_size
                    log_file.unlink()
                    stats['files_deleted'] += 1
                    stats['size_freed_mb'] += file_size / (1024 ** 2)
                    logger.debug(f"Deleted old log: {log_file.name}")

            except Exception as e:
                logger.warning(f"Could not delete {log_file.name}: {e}")

        if stats['files_deleted'] > 0:
            logger.info(f"Deleted {stats['files_deleted']} old log files ({stats['size_freed_mb']:.2f}MB)")

        return stats

    def _rotate_large_logs(self) -> Dict:
        """Rotate log files that exceed size limit"""
        stats = {'files_rotated': 0}

        logs_dir = Path("logs")
        if not logs_dir.exists():
            return stats

        for log_file in logs_dir.glob("*.log"):
            try:
                if log_file.stat().st_size > self.max_log_size_mb:
                    # Rename to .old and keep only last rotation
                    old_file = log_file.with_suffix('.log.old')
                    if old_file.exists():
                        old_file.unlink()

                    log_file.rename(old_file)
                    stats['files_rotated'] += 1
                    logger.info(f"Rotated large log: {log_file.name} -> {old_file.name}")

            except Exception as e:
                logger.warning(f"Could not rotate {log_file.name}: {e}")

        return stats

    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent

    def _clear_python_cache(self) -> bool:
        """Clear Python __pycache__ directories"""
        try:
            project_root = Path(__file__).parent.parent.parent
            pycache_dirs = list(project_root.rglob("__pycache__"))

            for pycache_dir in pycache_dirs:
                try:
                    shutil.rmtree(pycache_dir)
                except Exception as e:
                    logger.debug(f"Could not delete {pycache_dir}: {e}")

            if pycache_dirs:
                logger.info(f"Cleared {len(pycache_dirs)} __pycache__ directories")

            return True

        except Exception as e:
            logger.warning(f"Error clearing Python cache: {e}")
            return False

    def get_disk_usage(self) -> Dict:
        """Get disk usage statistics"""
        try:
            project_root = Path(__file__).parent.parent.parent

            # Logs directory
            logs_size = 0
            logs_dir = project_root / "logs"
            if logs_dir.exists():
                for f in logs_dir.rglob("*"):
                    if f.is_file():
                        logs_size += f.stat().st_size

            # Models directory
            models_size = 0
            models_dir = project_root / "models"
            if models_dir.exists():
                for f in models_dir.rglob("*"):
                    if f.is_file():
                        models_size += f.stat().st_size

            # Historical data directory
            historical_size = 0
            historical_dir = project_root / "historical_data"
            if historical_dir.exists():
                for f in historical_dir.rglob("*"):
                    if f.is_file():
                        historical_size += f.stat().st_size

            return {
                'logs_mb': logs_size / (1024 ** 2),
                'models_mb': models_size / (1024 ** 2),
                'historical_mb': historical_size / (1024 ** 2),
                'total_mb': (logs_size + models_size + historical_size) / (1024 ** 2)
            }

        except Exception as e:
            logger.error(f"Error calculating disk usage: {e}")
            return {}

    def get_system_stats(self) -> Dict:
        """Get system resource statistics"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_mb': psutil.virtual_memory().available / (1024 ** 2),
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
