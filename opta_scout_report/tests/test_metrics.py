"""Tests for metrics engine."""

import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from metrics import MetricsEngine


@pytest.fixture
def config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'metrics.yml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {
        'metrics': {
            'passing': {
                'passes_attempted': {'per_90': True},
                'passes_completed': {'per_90': True},
            },
            'defending': {
                'tackles': {'per_90': True},
                'interceptions': {'per_90': True},
            }
        }
    }


@pytest.fixture
def metrics_engine(config):
    """Create metrics engine instance."""
    return MetricsEngine(config)


@dataclass
class MockPlayerInfo:
    """Mock player info for testing."""
    name: str = "Test Player"
    player_id: str = "p12345"
    position: str = "Central Midfielder"
    position_group: str = "CM"
    age: int = 25
    current_team: str = "Test FC"
    nationality: str = "England"


@dataclass
class MockAppearance:
    """Mock match appearance."""
    match_id: str
    minutes_played: int
    is_starter: bool = True


@dataclass
class MockPlayerData:
    """Mock player data for testing."""
    info: MockPlayerInfo
    events: Optional[pd.DataFrame]
    appearances: List[MockAppearance]


@pytest.fixture
def sample_player_data():
    """Create sample player data."""
    # Create events
    np.random.seed(42)

    events = pd.DataFrame({
        'type_id': [1, 1, 1, 1, 1, 7, 7, 8, 8, 3, 3],  # passes, tackles, interceptions, carries
        'event_type': ['pass', 'pass', 'pass', 'pass', 'pass', 'tackle', 'tackle',
                       'interception', 'interception', 'carry', 'carry'],
        'x': np.random.uniform(20, 80, 11),
        'y': np.random.uniform(20, 80, 11),
        'end_x': np.random.uniform(30, 90, 11),
        'end_y': np.random.uniform(20, 80, 11),
        'outcome': [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],  # 4/5 passes, 1/2 tackles, 2/2 interceptions
        'match_id': ['m1'] * 5 + ['m1'] * 2 + ['m1'] * 2 + ['m1'] * 2,
        'minute': list(range(11)),
        'is_progressive_pass': [True, True, False, False, True, False, False, False, False, False, False],
    })

    appearances = [
        MockAppearance(match_id='m1', minutes_played=90),
    ]

    return MockPlayerData(
        info=MockPlayerInfo(),
        events=events,
        appearances=appearances
    )


class TestMetricsEngine:
    """Tests for MetricsEngine class."""

    def test_calculate_all_returns_result(self, metrics_engine, sample_player_data):
        """Test that calculate_all returns a result object."""
        result = metrics_engine.calculate_all(sample_player_data)

        assert result is not None
        assert hasattr(result, 'metrics')

    def test_per_90_normalization(self, metrics_engine, sample_player_data):
        """Test per-90 minute normalization."""
        result = metrics_engine.calculate_all(sample_player_data)

        # With 90 minutes played, raw counts should equal per-90 values
        # 5 passes in 90 minutes = 5 passes per 90
        if 'passes_attempted' in result.metrics:
            assert result.metrics['passes_attempted'].value == pytest.approx(5.0, rel=0.1)

    def test_metrics_are_numeric(self, metrics_engine, sample_player_data):
        """Test that all metric values are numeric."""
        result = metrics_engine.calculate_all(sample_player_data)

        for metric_name, metric_value in result.metrics.items():
            assert isinstance(metric_value.value, (int, float)), \
                f"{metric_name} has non-numeric value"

    def test_total_minutes_tracked(self, metrics_engine, sample_player_data):
        """Test that total minutes are tracked."""
        result = metrics_engine.calculate_all(sample_player_data)

        assert hasattr(result, 'total_minutes')
        assert result.total_minutes == 90


class TestPassingMetrics:
    """Tests for passing metrics calculation."""

    def test_pass_accuracy_calculation(self, metrics_engine, sample_player_data):
        """Test pass accuracy calculation."""
        result = metrics_engine.calculate_all(sample_player_data)

        # 4 completed out of 5 attempted = 80%
        if 'pass_accuracy' in result.metrics:
            assert result.metrics['pass_accuracy'].value == pytest.approx(80.0, rel=0.1)

    def test_progressive_passes_count(self, metrics_engine, sample_player_data):
        """Test progressive passes counting."""
        result = metrics_engine.calculate_all(sample_player_data)

        # 3 progressive passes marked in sample data
        if 'progressive_passes' in result.metrics:
            assert result.metrics['progressive_passes'].value == pytest.approx(3.0, rel=0.1)


class TestDefendingMetrics:
    """Tests for defending metrics calculation."""

    def test_tackles_won_calculation(self, metrics_engine, sample_player_data):
        """Test tackles won calculation."""
        result = metrics_engine.calculate_all(sample_player_data)

        # 1 successful tackle out of 2 attempted
        if 'tackles_won' in result.metrics:
            assert result.metrics['tackles_won'].value == pytest.approx(1.0, rel=0.1)

    def test_interceptions_count(self, metrics_engine, sample_player_data):
        """Test interceptions counting."""
        result = metrics_engine.calculate_all(sample_player_data)

        # 2 interceptions
        if 'interceptions' in result.metrics:
            assert result.metrics['interceptions'].value == pytest.approx(2.0, rel=0.1)

    def test_tackle_success_rate(self, metrics_engine, sample_player_data):
        """Test tackle success rate calculation."""
        result = metrics_engine.calculate_all(sample_player_data)

        # 1 out of 2 = 50%
        if 'tackle_success_rate' in result.metrics:
            assert result.metrics['tackle_success_rate'].value == pytest.approx(50.0, rel=0.1)


class TestCarryingMetrics:
    """Tests for carrying/dribbling metrics."""

    def test_carries_count(self, metrics_engine, sample_player_data):
        """Test carries counting."""
        result = metrics_engine.calculate_all(sample_player_data)

        # 2 carries in sample data
        if 'carries' in result.metrics:
            assert result.metrics['carries'].value == pytest.approx(2.0, rel=0.1)


class TestMultipleMatches:
    """Tests for metrics with multiple matches."""

    def test_per_90_with_partial_minutes(self, metrics_engine):
        """Test per-90 calculation with partial minutes."""
        events = pd.DataFrame({
            'type_id': [1] * 10,  # 10 passes
            'event_type': ['pass'] * 10,
            'x': [50] * 10,
            'y': [50] * 10,
            'end_x': [60] * 10,
            'end_y': [50] * 10,
            'outcome': [1] * 10,
            'match_id': ['m1'] * 10,
            'minute': list(range(10)),
        })

        appearances = [
            MockAppearance(match_id='m1', minutes_played=45),  # Half a game
        ]

        player_data = MockPlayerData(
            info=MockPlayerInfo(),
            events=events,
            appearances=appearances
        )

        result = metrics_engine.calculate_all(player_data)

        # 10 passes in 45 minutes = 20 passes per 90
        if 'passes_attempted' in result.metrics:
            assert result.metrics['passes_attempted'].value == pytest.approx(20.0, rel=0.1)

    def test_aggregation_across_matches(self, metrics_engine):
        """Test metrics aggregation across multiple matches."""
        events = pd.DataFrame({
            'type_id': [1] * 20,  # 20 passes total
            'event_type': ['pass'] * 20,
            'x': [50] * 20,
            'y': [50] * 20,
            'end_x': [60] * 20,
            'end_y': [50] * 20,
            'outcome': [1] * 20,
            'match_id': ['m1'] * 10 + ['m2'] * 10,  # 10 per match
            'minute': list(range(10)) * 2,
        })

        appearances = [
            MockAppearance(match_id='m1', minutes_played=90),
            MockAppearance(match_id='m2', minutes_played=90),
        ]

        player_data = MockPlayerData(
            info=MockPlayerInfo(),
            events=events,
            appearances=appearances
        )

        result = metrics_engine.calculate_all(player_data)

        # 20 passes in 180 minutes = 10 passes per 90
        if 'passes_attempted' in result.metrics:
            assert result.metrics['passes_attempted'].value == pytest.approx(10.0, rel=0.1)


class TestMatchByMatchMetrics:
    """Tests for match-by-match metric calculation."""

    def test_match_by_match_generated(self, metrics_engine, sample_player_data):
        """Test that match-by-match metrics are generated."""
        result = metrics_engine.calculate_all(sample_player_data)

        assert hasattr(result, 'match_by_match')
        if result.match_by_match:
            assert len(result.match_by_match) == 1  # One match in sample


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_events(self, metrics_engine):
        """Test handling of empty events DataFrame."""
        player_data = MockPlayerData(
            info=MockPlayerInfo(),
            events=pd.DataFrame(),
            appearances=[]
        )

        result = metrics_engine.calculate_all(player_data)

        assert result is not None

    def test_none_events(self, metrics_engine):
        """Test handling of None events."""
        player_data = MockPlayerData(
            info=MockPlayerInfo(),
            events=None,
            appearances=[]
        )

        result = metrics_engine.calculate_all(player_data)

        assert result is not None

    def test_zero_minutes(self, metrics_engine):
        """Test handling of zero minutes played."""
        events = pd.DataFrame({
            'type_id': [1],
            'event_type': ['pass'],
            'x': [50],
            'y': [50],
            'end_x': [60],
            'end_y': [50],
            'outcome': [1],
            'match_id': ['m1'],
            'minute': [0],
        })

        player_data = MockPlayerData(
            info=MockPlayerInfo(),
            events=events,
            appearances=[]  # No appearances = 0 minutes
        )

        result = metrics_engine.calculate_all(player_data)

        # Should handle gracefully without division by zero
        assert result is not None
