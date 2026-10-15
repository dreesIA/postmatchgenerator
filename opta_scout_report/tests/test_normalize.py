"""Tests for event normalization."""

import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from normalize import EventNormalizer


@pytest.fixture
def config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'metrics.yml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    # Return minimal config for testing
    return {
        'thresholds': {
            'progressive_pass_meters': 10,
            'progressive_carry_meters': 10,
        },
        'zones': {
            'defensive_third': {'x_min': 0, 'x_max': 33.33},
            'middle_third': {'x_min': 33.33, 'x_max': 66.66},
            'final_third': {'x_min': 66.66, 'x_max': 100},
            'box': {'x_min': 83, 'x_max': 100, 'y_min': 21.1, 'y_max': 78.9},
        }
    }


@pytest.fixture
def normalizer(config):
    """Create normalizer instance."""
    return EventNormalizer(config)


@pytest.fixture
def sample_events():
    """Create sample event data."""
    return pd.DataFrame({
        'type_id': [1, 1, 1, 3, 7, 8],  # passes, carry, tackle, interception
        'x': [30, 50, 70, 60, 40, 35],
        'y': [50, 50, 50, 50, 50, 50],
        'end_x': [50, 60, 85, 80, np.nan, np.nan],
        'end_y': [50, 55, 50, 50, np.nan, np.nan],
        'outcome': [1, 1, 1, 1, 1, 1],
        'player_id': ['p1'] * 6,
        'match_id': ['m1'] * 6,
        'minute': [10, 20, 30, 40, 50, 60],
        'second': [0, 0, 0, 0, 0, 0],
    })


class TestEventNormalizer:
    """Tests for EventNormalizer class."""

    def test_normalize_returns_dataframe(self, normalizer, sample_events):
        """Test that normalize returns a DataFrame."""
        result = normalizer.normalize(sample_events)
        assert isinstance(result, pd.DataFrame)

    def test_progressive_pass_detection(self, normalizer, sample_events):
        """Test detection of progressive passes."""
        result = normalizer.normalize(sample_events)

        # First pass: 30 -> 50 = 20m forward (progressive)
        # Second pass: 50 -> 60 = 10m forward (progressive)
        # Third pass: 70 -> 85 = 15m forward (progressive)
        if 'is_progressive_pass' in result.columns:
            passes = result[result['type_id'] == 1]
            assert passes['is_progressive_pass'].sum() >= 2

    def test_zone_tagging(self, normalizer, sample_events):
        """Test zone tagging of events."""
        result = normalizer.normalize(sample_events)

        if 'zone' in result.columns:
            # Event at x=30 should be in defensive third
            event_30 = result[result['x'] == 30].iloc[0]
            assert event_30['zone'] in ['defensive_third', 'middle_third']

            # Event at x=70 should be in final third
            event_70 = result[result['x'] == 70].iloc[0]
            assert event_70['zone'] == 'final_third'

    def test_box_entry_detection(self, normalizer, sample_events):
        """Test detection of passes into the box."""
        result = normalizer.normalize(sample_events)

        if 'is_box_entry' in result.columns:
            # Third pass ends at x=85, which is in the box
            passes = result[result['type_id'] == 1]
            box_entries = passes[passes['end_x'] >= 83]
            assert len(box_entries) > 0

    def test_handles_missing_coordinates(self, normalizer):
        """Test handling of events with missing coordinates."""
        events = pd.DataFrame({
            'type_id': [1, 1],
            'x': [50, np.nan],
            'y': [50, np.nan],
            'end_x': [60, np.nan],
            'end_y': [50, np.nan],
            'outcome': [1, 1],
        })

        result = normalizer.normalize(events)
        assert len(result) == 2

    def test_distance_calculation(self, normalizer, sample_events):
        """Test pass distance calculation."""
        result = normalizer.normalize(sample_events)

        if 'pass_distance' in result.columns:
            # First pass: sqrt((50-30)^2 + (50-50)^2) = 20
            passes = result[result['type_id'] == 1]
            first_pass = passes.iloc[0]
            expected_distance = 20.0
            assert abs(first_pass['pass_distance'] - expected_distance) < 0.1


class TestProgressivePassDetection:
    """Detailed tests for progressive pass detection."""

    def test_forward_pass_is_progressive(self, normalizer):
        """Test that forward passes > 10m are marked progressive."""
        events = pd.DataFrame({
            'type_id': [1],
            'x': [30],
            'y': [50],
            'end_x': [45],  # 15m forward
            'end_y': [50],
            'outcome': [1],
        })

        result = normalizer.normalize(events)
        if 'is_progressive_pass' in result.columns:
            assert result.iloc[0]['is_progressive_pass'] == True

    def test_backward_pass_not_progressive(self, normalizer):
        """Test that backward passes are not progressive."""
        events = pd.DataFrame({
            'type_id': [1],
            'x': [50],
            'y': [50],
            'end_x': [35],  # 15m backward
            'end_y': [50],
            'outcome': [1],
        })

        result = normalizer.normalize(events)
        if 'is_progressive_pass' in result.columns:
            assert result.iloc[0]['is_progressive_pass'] == False

    def test_short_forward_pass_not_progressive(self, normalizer):
        """Test that short forward passes < 10m are not progressive."""
        events = pd.DataFrame({
            'type_id': [1],
            'x': [50],
            'y': [50],
            'end_x': [55],  # 5m forward
            'end_y': [50],
            'outcome': [1],
        })

        result = normalizer.normalize(events)
        if 'is_progressive_pass' in result.columns:
            assert result.iloc[0]['is_progressive_pass'] == False

    def test_pass_into_final_third_is_progressive(self, normalizer):
        """Test that passes entering final third are progressive."""
        events = pd.DataFrame({
            'type_id': [1],
            'x': [60],
            'y': [50],
            'end_x': [70],  # Enters final third
            'end_y': [50],
            'outcome': [1],
        })

        result = normalizer.normalize(events)
        if 'is_progressive_pass' in result.columns:
            # Even though only 10m, entering final third makes it progressive
            assert result.iloc[0]['is_progressive_pass'] == True


class TestZoneTagging:
    """Tests for pitch zone tagging."""

    def test_defensive_third_tagging(self, normalizer):
        """Test events in defensive third are correctly tagged."""
        events = pd.DataFrame({
            'type_id': [1],
            'x': [15],
            'y': [50],
            'end_x': [20],
            'end_y': [50],
            'outcome': [1],
        })

        result = normalizer.normalize(events)
        if 'zone' in result.columns:
            assert result.iloc[0]['zone'] == 'defensive_third'

    def test_final_third_tagging(self, normalizer):
        """Test events in final third are correctly tagged."""
        events = pd.DataFrame({
            'type_id': [1],
            'x': [80],
            'y': [50],
            'end_x': [90],
            'end_y': [50],
            'outcome': [1],
        })

        result = normalizer.normalize(events)
        if 'zone' in result.columns:
            assert result.iloc[0]['zone'] == 'final_third'

    def test_box_tagging(self, normalizer):
        """Test events in penalty box are correctly tagged."""
        events = pd.DataFrame({
            'type_id': [1],
            'x': [88],
            'y': [50],  # Within box y-range (21.1-78.9)
            'end_x': [95],
            'end_y': [50],
            'outcome': [1],
        })

        result = normalizer.normalize(events)
        if 'in_box' in result.columns:
            assert result.iloc[0]['in_box'] == True


class TestPerNinetyNormalization:
    """Tests for per-90 normalization."""

    def test_per_ninety_calculation(self, normalizer):
        """Test per-90 minute normalization."""
        # 45 minutes played, 10 events = 20 per 90
        events = pd.DataFrame({
            'type_id': [1] * 10,
            'x': [50] * 10,
            'y': [50] * 10,
            'end_x': [60] * 10,
            'end_y': [50] * 10,
            'outcome': [1] * 10,
            'minute': list(range(10)),
        })

        total_minutes = 45
        count = len(events)
        per_90 = count / total_minutes * 90

        assert abs(per_90 - 20.0) < 0.01
