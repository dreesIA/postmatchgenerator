"""Opta Scout Report - Individual Player Scouting Report Generator."""

from .opta_client import OptaClient
from .data_loader import DataLoader, PlayerData
from .normalize import EventNormalizer
from .cohort import CohortBuilder
from .metrics import MetricsEngine

__version__ = "1.0.0"
__all__ = [
    'OptaClient',
    'DataLoader',
    'PlayerData',
    'EventNormalizer',
    'CohortBuilder',
    'MetricsEngine',
]
