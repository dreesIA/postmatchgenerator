"""Report sections for the Opta Scout Report."""

from .base import BaseSection
from .snapshot import SnapshotSection
from .role_usage import RoleUsageSection
from .physical import PhysicalSection
from .involvement import InvolvementSection
from .passing import PassingSection
from .carrying import CarryingSection
from .chance_creation import ChanceCreationSection
from .defending import DefendingSection
from .discipline import DisciplineSection
from .benchmarking import BenchmarkingSection
from .trends import TrendsSection
from .recommendation import RecommendationSection

__all__ = [
    'BaseSection',
    'SnapshotSection',
    'RoleUsageSection',
    'PhysicalSection',
    'InvolvementSection',
    'PassingSection',
    'CarryingSection',
    'ChanceCreationSection',
    'DefendingSection',
    'DisciplineSection',
    'BenchmarkingSection',
    'TrendsSection',
    'RecommendationSection',
]
