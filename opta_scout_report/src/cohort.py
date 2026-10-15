"""
Cohort building and percentile calculations for benchmarking.
"""

import logging
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PercentileResult:
    """Result of percentile calculation for a metric."""
    metric: str
    value: float
    per_90_value: float
    percentile: float
    rank: int
    total_players: int
    cohort_mean: float
    cohort_median: float
    cohort_std: float


@dataclass
class CohortSummary:
    """Summary of cohort for a player."""
    player_id: str
    position_group: str
    competition: str
    season: str
    cohort_size: int
    min_minutes: int
    percentiles: Dict[str, PercentileResult] = field(default_factory=dict)
    radar_values: Dict[str, float] = field(default_factory=dict)
    fit_score: float = 0.0
    risk_score: float = 0.0


class CohortBuilder:
    """
    Builds comparison cohorts and calculates percentile ranks.
    """

    def __init__(self, config: Optional[Union[str, Path, Mapping[str, Any]]] = None):
        """
        Initialize cohort builder.

        Args:
            config: Either a loaded config dict, or a path to `metrics.yml`.
        """
        if config is None:
            config_path = Path(__file__).parent.parent / "config" / "metrics.yml"
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, (str, Path)):
            with open(config, "r") as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, Mapping):
            self.config = dict(config)
        else:
            raise TypeError(
                "config must be a path (str/Path) or a mapping (dict-like), "
                f"got {type(config).__name__}"
            )

        self.position_groups = self.config.get('position_groups', {})
        self.radar_profiles = self.config.get('radar_profiles', {})

    def get_min_minutes(self, position_group: str) -> int:
        """Get minimum minutes threshold for a position group."""
        pg_config = self.position_groups.get(position_group, {})
        return pg_config.get('min_minutes', 450)

    def build_cohort(
        self,
        cohort_df: Optional[pd.DataFrame] = None,
        player_stats: Optional[Dict[str, Any]] = None,
        player_minutes: Optional[int] = None,
        position_group: Optional[str] = None,
        *,
        player_metrics: Any = None,
        cohort_data: Optional[pd.DataFrame] = None,
    ) -> CohortSummary:
        """
        Build cohort comparison and calculate percentiles.

        Args:
            cohort_df: (Legacy) DataFrame with cohort player stats
            player_stats: (Legacy) Subject player's stats dictionary
            player_minutes: (Legacy) Subject player's total minutes
            position_group: Position group
            player_metrics: MetricsResult-like object or dict of metric -> value
            cohort_data: Alias for cohort_df used by report generator/tests

        Returns:
            CohortSummary with all percentile calculations
        """
        if cohort_data is not None:
            cohort_df = cohort_data
        if cohort_df is None:
            cohort_df = pd.DataFrame()

        if player_stats is None:
            player_stats = {}

        # Prefer the newer API when provided.
        if player_metrics is not None:
            extracted = self._extract_metric_values(player_metrics)
            player_stats = {**player_stats, **extracted["values"]}
            if extracted.get("player_id"):
                player_stats.setdefault("player_id", extracted["player_id"])
            if player_minutes is None:
                player_minutes = extracted.get("minutes")

        if position_group is None:
            raise TypeError("position_group is required")

        if cohort_df.empty:
            logger.warning("Empty cohort DataFrame provided")
            return CohortSummary(
                player_id=player_stats.get('player_id', ''),
                position_group=position_group,
                competition='',
                season='',
                cohort_size=0,
                min_minutes=self.get_min_minutes(position_group)
            )

        # Get radar profile for position
        radar_config = self.radar_profiles.get(position_group, {})
        radar_axes = radar_config.get('axes', [])
        radar_weights = radar_config.get('weights', {})

        # Calculate percentiles for each metric
        percentiles = {}
        radar_values = {}

        minutes_series: Optional[pd.Series] = None
        if 'minutes' in cohort_df.columns:
            minutes_series = pd.to_numeric(cohort_df['minutes'], errors='coerce')

        # All numeric columns as potential metrics (including numeric-looking strings)
        numeric_cols = [
            c for c in cohort_df.columns
            if c not in ['player_id', 'team_id', 'minutes']
            and pd.api.types.is_numeric_dtype(pd.to_numeric(cohort_df[c], errors='coerce'))
        ]

        for metric in numeric_cols:
            # Get cohort values
            metric_series = pd.to_numeric(cohort_df[metric], errors='coerce')
            cohort_values = metric_series.replace([np.inf, -np.inf], np.nan).dropna()

            if len(cohort_values) < 3:
                continue

            # Get player value
            player_value = self._coerce_float(player_stats.get(metric, 0))

            # Calculate per-90 values if minutes are available; otherwise assume values are already comparable.
            if minutes_series is not None:
                cohort_per90 = self._calculate_per90(metric_series.fillna(0), minutes_series.fillna(0))
                player_per90 = (
                    self._calculate_per90_single(player_value, int(player_minutes or 0))
                    if player_minutes is not None
                    else player_value
                )
            else:
                cohort_per90 = cohort_values
                player_per90 = player_value

            # Calculate percentile
            percentile = self._calculate_percentile(player_per90, cohort_per90)

            # Calculate rank
            rank = int((cohort_per90 < player_per90).sum() + 1)

            percentiles[metric] = PercentileResult(
                metric=metric,
                value=player_value,
                per_90_value=player_per90,
                percentile=percentile,
                rank=rank,
                total_players=len(cohort_values) + 1,  # Include subject
                cohort_mean=cohort_per90.mean(),
                cohort_median=cohort_per90.median(),
                cohort_std=cohort_per90.std()
            )

            # Store radar value if in radar axes
            if metric in radar_axes:
                radar_values[metric] = percentile

        # Calculate fit score
        fit_score = self._calculate_fit_score(radar_values, radar_weights)

        # Calculate risk score
        risk_score = self._calculate_risk_score(player_stats, percentiles)

        return CohortSummary(
            player_id=player_stats.get('player_id', ''),
            position_group=position_group,
            competition=player_stats.get('competition', ''),
            season=player_stats.get('season', ''),
            cohort_size=len(cohort_df) + 1,
            min_minutes=self.get_min_minutes(position_group),
            percentiles=percentiles,
            radar_values=radar_values,
            fit_score=fit_score,
            risk_score=risk_score
        )

    def _extract_metric_values(self, player_metrics: Any) -> Dict[str, Any]:
        """
        Extract a {metric: value} mapping and basic metadata from a MetricsResult-like object.
        """
        player_id = getattr(player_metrics, "player_id", "") or getattr(player_metrics, "id", "")
        minutes = getattr(player_metrics, "minutes", None)
        if minutes is None:
            minutes = getattr(player_metrics, "total_minutes", None)

        if hasattr(player_metrics, "to_per90_dict") and callable(player_metrics.to_per90_dict):
            values = player_metrics.to_per90_dict()
        elif isinstance(player_metrics, Mapping):
            values = dict(player_metrics)
        else:
            metrics = getattr(player_metrics, "metrics", None)
            if isinstance(metrics, Mapping):
                values = {}
                for name, metric_value in metrics.items():
                    if hasattr(metric_value, "per_90") and getattr(metric_value, "per_90") is not None:
                        values[name] = getattr(metric_value, "per_90")
                    elif hasattr(metric_value, "value"):
                        values[name] = getattr(metric_value, "value")
                    else:
                        values[name] = metric_value
            else:
                raise TypeError(
                    "player_metrics must be a MetricsResult-like object (with .metrics or .to_per90_dict) "
                    f"or a mapping, got {type(player_metrics).__name__}"
                )

        return {"values": values, "player_id": player_id, "minutes": minutes}

    def _coerce_float(self, value: Any) -> float:
        """Best-effort conversion to float for metric values."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float, np.number)):
            return float(value)
        if hasattr(value, "per_90") and getattr(value, "per_90") is not None:
            return float(getattr(value, "per_90"))
        if hasattr(value, "value"):
            return float(getattr(value, "value"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _calculate_per90(self, values: pd.Series, minutes: pd.Series) -> pd.Series:
        """Calculate per-90 values for a series."""
        per90 = values * 90 / minutes.replace(0, np.nan)
        return per90.fillna(0)

    def _calculate_per90_single(self, value: float, minutes: int) -> float:
        """Calculate per-90 value for a single player."""
        if minutes <= 0:
            return 0.0
        return value * 90 / minutes

    def _calculate_percentile(self, value: float, cohort_values: pd.Series) -> float:
        """
        Calculate percentile rank for a value within a distribution.

        Args:
            value: Player's value
            cohort_values: Cohort distribution

        Returns:
            Percentile (0-100)
        """
        if len(cohort_values) == 0:
            return 50.0

        # Handle NaN and infinite values
        cohort_clean = cohort_values.replace([np.inf, -np.inf], np.nan).dropna()

        if len(cohort_clean) == 0:
            return 50.0

        # Calculate percentile using scipy-style method
        below = (cohort_clean < value).sum()
        equal = (cohort_clean == value).sum()

        percentile = (below + 0.5 * equal) / len(cohort_clean) * 100

        return min(99, max(1, percentile))

    def _calculate_fit_score(
        self,
        radar_values: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate fit score based on weighted percentiles.

        Args:
            radar_values: Dictionary of metric -> percentile
            weights: Dictionary of metric -> weight

        Returns:
            Fit score (0-100)
        """
        if not radar_values:
            return 50.0

        weighted_sum = 0.0
        weight_total = 0.0

        for metric, percentile in radar_values.items():
            weight = abs(weights.get(metric, 1.0))
            weighted_sum += percentile * weight
            weight_total += weight

        if weight_total == 0:
            return 50.0

        return weighted_sum / weight_total

    def _calculate_risk_score(
        self,
        player_stats: Dict[str, float],
        percentiles: Dict[str, PercentileResult]
    ) -> float:
        """
        Calculate risk score based on negative metrics.

        Lower is better (less risky).

        Args:
            player_stats: Player statistics
            percentiles: Calculated percentiles

        Returns:
            Risk score (0-100)
        """
        risk_factors = []

        # Card rate
        if 'yellow_cards' in percentiles:
            risk_factors.append(percentiles['yellow_cards'].percentile)
        if 'red_cards' in percentiles:
            risk_factors.append(percentiles['red_cards'].percentile * 2)

        # Turnover rate
        if 'dispossessions' in percentiles:
            risk_factors.append(percentiles['dispossessions'].percentile)
        if 'turnovers' in percentiles:
            risk_factors.append(percentiles['turnovers'].percentile)

        # Errors
        if 'errors_leading_to_shot' in percentiles:
            risk_factors.append(percentiles['errors_leading_to_shot'].percentile * 1.5)

        # Foul rate
        if 'fouls_committed' in percentiles:
            risk_factors.append(percentiles['fouls_committed'].percentile * 0.5)

        if not risk_factors:
            return 50.0

        return sum(risk_factors) / len(risk_factors)

    def get_radar_config(self, position_group: str) -> Tuple[List[str], Dict[str, float]]:
        """
        Get radar configuration for a position group.

        Args:
            position_group: Position group

        Returns:
            (axes, weights) tuple
        """
        config = self.radar_profiles.get(position_group, {})
        return config.get('axes', []), config.get('weights', {})

    def calculate_recommendation(
        self,
        fit_score: float,
        risk_score: float,
        minutes_stability: float = 0.5,
        age_factor: float = 0.0
    ) -> Tuple[str, str]:
        """
        Calculate recommendation based on scores.

        Args:
            fit_score: Fit score (0-100)
            risk_score: Risk score (0-100)
            minutes_stability: Minutes stability factor (0-1)
            age_factor: Age adjustment (-10 to +10)

        Returns:
            (recommendation, reason) tuple
        """
        thresholds = self.config.get('recommendation', {}).get('thresholds', {})
        recommend_threshold = thresholds.get('recommend', 70)
        monitor_threshold = thresholds.get('monitor', 50)

        # Adjust score based on factors
        adjusted_score = fit_score * 0.6 + (100 - risk_score) * 0.25 + minutes_stability * 15
        adjusted_score += age_factor

        # Generate recommendation
        if adjusted_score >= recommend_threshold:
            recommendation = "RECOMMEND"
            if fit_score >= 75:
                reason = f"Strong fit ({fit_score:.0f}%) with acceptable risk ({risk_score:.0f}%)"
            else:
                reason = f"Good overall profile with balanced strengths"
        elif adjusted_score >= monitor_threshold:
            recommendation = "MONITOR"
            if risk_score > 65:
                reason = f"Promising profile ({fit_score:.0f}%) but elevated risk factors ({risk_score:.0f}%)"
            else:
                reason = f"Moderate fit ({fit_score:.0f}%), needs further evaluation"
        else:
            recommendation = "PASS"
            if fit_score < 40:
                reason = f"Profile doesn't match role requirements ({fit_score:.0f}%)"
            else:
                reason = f"Risk factors outweigh potential ({risk_score:.0f}% risk)"

        return recommendation, reason


def calculate_percentile_ranks(
    player_values: Dict[str, float],
    cohort_df: pd.DataFrame,
    minutes_col: str = 'minutes'
) -> Dict[str, float]:
    """
    Convenience function to calculate percentile ranks.

    Args:
        player_values: Player's values (already per-90 adjusted)
        cohort_df: Cohort data (raw values)
        minutes_col: Column name for minutes

    Returns:
        Dictionary of metric -> percentile
    """
    builder = CohortBuilder()
    result = {}

    for metric, value in player_values.items():
        if metric in cohort_df.columns:
            cohort_per90 = builder._calculate_per90(
                cohort_df[metric], cohort_df[minutes_col]
            )
            result[metric] = builder._calculate_percentile(value, cohort_per90)

    return result
