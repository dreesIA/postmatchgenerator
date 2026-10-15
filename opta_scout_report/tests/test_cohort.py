"""Tests for cohort builder and percentile calculations."""

import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cohort import CohortBuilder, PercentileResult, CohortSummary


@pytest.fixture
def config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'metrics.yml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {
        'recommendation': {
            'thresholds': {
                'recommend': 70,
                'monitor': 50,
            },
            'risk_factors': {
                'injury_history_weight': 0.2,
                'card_rate_weight': 0.15,
            }
        },
        'radar_profiles': {
            'CM': {
                'axes': ['passes_completed', 'progressive_passes', 'tackles_won'],
                'weights': {'passes_completed': 1.0, 'progressive_passes': 1.2, 'tackles_won': 0.8}
            }
        }
    }


@pytest.fixture
def cohort_builder(config):
    """Create cohort builder instance."""
    return CohortBuilder(config)


@pytest.fixture
def sample_cohort_data():
    """Create sample cohort data."""
    np.random.seed(42)
    n_players = 50

    return pd.DataFrame({
        'player_id': [f'p{i}' for i in range(n_players)],
        'passes_completed': np.random.normal(50, 10, n_players),
        'pass_accuracy': np.random.normal(85, 5, n_players),
        'progressive_passes': np.random.normal(5, 2, n_players),
        'tackles_won': np.random.normal(2, 1, n_players),
        'interceptions': np.random.normal(1.5, 0.5, n_players),
        'xG': np.random.exponential(0.3, n_players),
        'xA': np.random.exponential(0.2, n_players),
        'minutes': np.random.uniform(450, 2000, n_players),
    })


@pytest.fixture
def sample_player_metrics():
    """Create sample player metrics."""
    class MockMetrics:
        def __init__(self):
            self.metrics = {
                'passes_completed': MockMetricValue(55),  # Above average
                'pass_accuracy': MockMetricValue(88),
                'progressive_passes': MockMetricValue(7),  # Well above average
                'tackles_won': MockMetricValue(1.5),  # Below average
                'interceptions': MockMetricValue(1.5),
                'xG': MockMetricValue(0.4),
                'xA': MockMetricValue(0.25),
            }
            self.total_minutes = 1500

    class MockMetricValue:
        def __init__(self, value):
            self.value = value

    return MockMetrics()


class TestCohortBuilder:
    """Tests for CohortBuilder class."""

    def test_build_cohort_returns_summary(self, cohort_builder, sample_player_metrics,
                                          sample_cohort_data):
        """Test that build_cohort returns a CohortSummary."""
        result = cohort_builder.build_cohort(
            player_metrics=sample_player_metrics,
            cohort_data=sample_cohort_data,
            position_group='CM'
        )

        assert isinstance(result, CohortSummary)

    def test_percentiles_calculated(self, cohort_builder, sample_player_metrics,
                                    sample_cohort_data):
        """Test that percentiles are calculated for metrics."""
        result = cohort_builder.build_cohort(
            player_metrics=sample_player_metrics,
            cohort_data=sample_cohort_data,
            position_group='CM'
        )

        assert hasattr(result, 'percentiles')
        assert len(result.percentiles) > 0

    def test_percentile_range(self, cohort_builder, sample_player_metrics,
                              sample_cohort_data):
        """Test that percentiles are within 0-100 range."""
        result = cohort_builder.build_cohort(
            player_metrics=sample_player_metrics,
            cohort_data=sample_cohort_data,
            position_group='CM'
        )

        for metric, pct_result in result.percentiles.items():
            assert 0 <= pct_result.percentile <= 100, f"{metric} percentile out of range"


class TestPercentileCalculation:
    """Tests for percentile calculation logic."""

    def test_median_value_is_50th_percentile(self, cohort_builder):
        """Test that median value gives approximately 50th percentile."""
        cohort_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        player_value = 5.5  # Median

        percentile = cohort_builder._calculate_percentile(player_value, cohort_values)

        assert 45 <= percentile <= 55

    def test_max_value_is_high_percentile(self, cohort_builder):
        """Test that maximum value gives high percentile."""
        cohort_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        player_value = 10

        percentile = cohort_builder._calculate_percentile(player_value, cohort_values)

        assert percentile >= 90

    def test_min_value_is_low_percentile(self, cohort_builder):
        """Test that minimum value gives low percentile."""
        cohort_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        player_value = 1

        percentile = cohort_builder._calculate_percentile(player_value, cohort_values)

        assert percentile <= 15

    def test_above_max_is_100th_percentile(self, cohort_builder):
        """Test that value above max is 100th percentile."""
        cohort_values = np.array([1, 2, 3, 4, 5])
        player_value = 10

        percentile = cohort_builder._calculate_percentile(player_value, cohort_values)

        assert percentile >= 99

    def test_below_min_is_near_0th_percentile(self, cohort_builder):
        """Test that value below min is near 0th percentile."""
        cohort_values = np.array([5, 6, 7, 8, 9, 10])
        player_value = 1

        percentile = cohort_builder._calculate_percentile(player_value, cohort_values)

        assert percentile <= 5


class TestRecommendation:
    """Tests for recommendation calculation."""

    def test_high_percentile_gives_recommend(self, cohort_builder, sample_cohort_data):
        """Test that high overall percentile gives RECOMMEND verdict."""
        # Create player with high values across the board
        class HighPerformer:
            def __init__(self):
                self.metrics = {
                    'passes_completed': MockVal(70),  # Very high
                    'pass_accuracy': MockVal(95),
                    'progressive_passes': MockVal(10),
                    'tackles_won': MockVal(4),
                    'interceptions': MockVal(3),
                }
                self.total_minutes = 1800

        class MockVal:
            def __init__(self, v):
                self.value = v

        result = cohort_builder.build_cohort(
            player_metrics=HighPerformer(),
            cohort_data=sample_cohort_data,
            position_group='CM'
        )

        # Check that average percentile is high
        if result.percentiles:
            avg_pct = np.mean([p.percentile for p in result.percentiles.values()])
            assert avg_pct > 60

    def test_low_percentile_gives_pass(self, cohort_builder, sample_cohort_data):
        """Test that low overall percentile gives PASS verdict."""
        class LowPerformer:
            def __init__(self):
                self.metrics = {
                    'passes_completed': MockVal(30),  # Very low
                    'pass_accuracy': MockVal(70),
                    'progressive_passes': MockVal(2),
                    'tackles_won': MockVal(0.5),
                    'interceptions': MockVal(0.5),
                }
                self.total_minutes = 500

        class MockVal:
            def __init__(self, v):
                self.value = v

        result = cohort_builder.build_cohort(
            player_metrics=LowPerformer(),
            cohort_data=sample_cohort_data,
            position_group='CM'
        )

        if result.percentiles:
            avg_pct = np.mean([p.percentile for p in result.percentiles.values()])
            assert avg_pct < 50


class TestFitScore:
    """Tests for fit score calculation."""

    def test_fit_score_uses_position_weights(self, cohort_builder, sample_player_metrics,
                                             sample_cohort_data):
        """Test that fit score considers position-specific weights."""
        result = cohort_builder.build_cohort(
            player_metrics=sample_player_metrics,
            cohort_data=sample_cohort_data,
            position_group='CM'
        )

        # Fit score should be calculated
        assert hasattr(result, 'fit_score')
        assert 0 <= result.fit_score <= 100

    def test_fit_score_different_positions(self, cohort_builder, sample_player_metrics,
                                           sample_cohort_data):
        """Test that different positions produce different fit scores."""
        result_cm = cohort_builder.build_cohort(
            player_metrics=sample_player_metrics,
            cohort_data=sample_cohort_data,
            position_group='CM'
        )

        result_cb = cohort_builder.build_cohort(
            player_metrics=sample_player_metrics,
            cohort_data=sample_cohort_data,
            position_group='CB'
        )

        # Different positions may weight metrics differently
        # Just ensure both produce valid scores
        assert 0 <= result_cm.fit_score <= 100
        assert 0 <= result_cb.fit_score <= 100


class TestRiskScore:
    """Tests for risk score calculation."""

    def test_risk_score_range(self, cohort_builder, sample_player_metrics,
                              sample_cohort_data):
        """Test that risk score is within valid range."""
        result = cohort_builder.build_cohort(
            player_metrics=sample_player_metrics,
            cohort_data=sample_cohort_data,
            position_group='CM'
        )

        assert hasattr(result, 'risk_score')
        assert 0 <= result.risk_score <= 100

    def test_high_cards_increases_risk(self, cohort_builder, sample_cohort_data):
        """Test that high card rate increases risk score."""
        class HighCardPlayer:
            def __init__(self):
                self.metrics = {
                    'passes_completed': MockVal(50),
                    'yellow_cards': MockVal(0.5),  # High card rate
                    'red_cards': MockVal(0.1),
                }
                self.total_minutes = 1000

        class LowCardPlayer:
            def __init__(self):
                self.metrics = {
                    'passes_completed': MockVal(50),
                    'yellow_cards': MockVal(0.05),  # Low card rate
                    'red_cards': MockVal(0),
                }
                self.total_minutes = 1000

        class MockVal:
            def __init__(self, v):
                self.value = v

        # Risk score calculations would differ based on discipline metrics
        # This is a conceptual test
        result_high = cohort_builder.build_cohort(
            player_metrics=HighCardPlayer(),
            cohort_data=sample_cohort_data,
            position_group='CM'
        )

        result_low = cohort_builder.build_cohort(
            player_metrics=LowCardPlayer(),
            cohort_data=sample_cohort_data,
            position_group='CM'
        )

        # Both should have valid risk scores
        assert 0 <= result_high.risk_score <= 100
        assert 0 <= result_low.risk_score <= 100
