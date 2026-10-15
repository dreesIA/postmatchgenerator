"""
Metrics calculation engine for player scouting reports.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """A calculated metric value."""
    name: str
    value: float
    per_90: Optional[float] = None
    possession_adjusted: Optional[float] = None
    description: str = ""
    format_type: str = "number"  # number, percentage, decimal


@dataclass
class MetricsResult:
    """Complete metrics calculation result."""
    player_id: str
    minutes: int
    matches: int
    metrics: Dict[str, MetricValue] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert to simple dictionary of values."""
        return {name: m.value for name, m in self.metrics.items()}

    def to_per90_dict(self) -> Dict[str, float]:
        """Convert to dictionary of per-90 values."""
        return {
            name: m.per_90 if m.per_90 is not None else m.value
            for name, m in self.metrics.items()
        }


class MetricsEngine:
    """
    Calculates all metrics for a player from event data.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize metrics engine.

        Args:
            config_path: Path to metrics.yml config
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "metrics.yml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.thresholds = self.config.get('thresholds', {})
        self.zones = self.config.get('zones', {})

    def calculate_all_metrics(
        self,
        events_df: pd.DataFrame,
        minutes: int,
        team_possession: float = 50.0
    ) -> MetricsResult:
        """
        Calculate all metrics for a player.

        Args:
            events_df: Player's event data
            minutes: Total minutes played
            team_possession: Team's average possession percentage

        Returns:
            MetricsResult with all calculated metrics
        """
        if events_df.empty:
            logger.warning("Empty events DataFrame")
            return MetricsResult(
                player_id="",
                minutes=minutes,
                matches=0
            )

        player_id = events_df['player_id'].iloc[0] if 'player_id' in events_df.columns else ""
        matches = events_df['match_id'].nunique() if 'match_id' in events_df.columns else 0

        result = MetricsResult(
            player_id=player_id,
            minutes=minutes,
            matches=matches
        )

        # Calculate all metric categories
        result.metrics.update(self._calculate_involvement(events_df, minutes, team_possession))
        result.metrics.update(self._calculate_passing(events_df, minutes, team_possession))
        result.metrics.update(self._calculate_carrying(events_df, minutes, team_possession))
        result.metrics.update(self._calculate_chance_creation(events_df, minutes, team_possession))
        result.metrics.update(self._calculate_defending(events_df, minutes, team_possession))
        result.metrics.update(self._calculate_discipline(events_df, minutes, team_possession))

        return result

    def _per_90(self, value: float, minutes: int) -> float:
        """Calculate per-90 value."""
        if minutes <= 0:
            return 0.0
        return value * 90 / minutes

    def _possession_adjust(self, value: float, possession: float, is_attacking: bool = True) -> float:
        """
        Adjust value for possession.

        For attacking metrics: adjust up if low possession
        For defensive metrics: adjust down if high possession
        """
        if possession <= 0:
            return value

        if is_attacking:
            # Adjust up for lower possession (fewer opportunities)
            factor = 50 / possession
        else:
            # Adjust down for higher possession (fewer defensive situations)
            factor = possession / 50

        return value * factor

    def _calculate_involvement(
        self,
        df: pd.DataFrame,
        minutes: int,
        possession: float
    ) -> Dict[str, MetricValue]:
        """Calculate involvement metrics."""
        metrics = {}

        # Touches
        touches = df['is_touch'].sum() if 'is_touch' in df.columns else len(df)
        metrics['touches'] = MetricValue(
            name='touches',
            value=touches,
            per_90=self._per_90(touches, minutes),
            description="Total touches of the ball"
        )

        # Touches by zone
        if 'in_defensive_third' in df.columns:
            def_touches = df[df['in_defensive_third']]['is_touch'].sum()
            metrics['defensive_third_touches'] = MetricValue(
                name='defensive_third_touches',
                value=def_touches,
                per_90=self._per_90(def_touches, minutes),
                description="Touches in defensive third"
            )

        if 'in_middle_third' in df.columns:
            mid_touches = df[df['in_middle_third']]['is_touch'].sum()
            metrics['middle_third_touches'] = MetricValue(
                name='middle_third_touches',
                value=mid_touches,
                per_90=self._per_90(mid_touches, minutes),
                description="Touches in middle third"
            )

        if 'in_final_third' in df.columns:
            att_touches = df[df['in_final_third']]['is_touch'].sum()
            metrics['final_third_touches'] = MetricValue(
                name='final_third_touches',
                value=att_touches,
                per_90=self._per_90(att_touches, minutes),
                description="Touches in attacking third"
            )

        if 'in_box' in df.columns:
            box_touches = df[df['in_box']]['is_touch'].sum()
            metrics['box_touches'] = MetricValue(
                name='box_touches',
                value=box_touches,
                per_90=self._per_90(box_touches, minutes),
                description="Touches in penalty area"
            )

        return metrics

    def _calculate_passing(
        self,
        df: pd.DataFrame,
        minutes: int,
        possession: float
    ) -> Dict[str, MetricValue]:
        """Calculate passing metrics."""
        metrics = {}

        passes = df[df['is_pass']] if 'is_pass' in df.columns else pd.DataFrame()

        if passes.empty:
            return metrics

        # Total passes
        total_passes = len(passes)
        successful_passes = passes['outcome_success'].sum() if 'outcome_success' in passes.columns else 0

        metrics['passes_attempted'] = MetricValue(
            name='passes_attempted',
            value=total_passes,
            per_90=self._per_90(total_passes, minutes),
            description="Total passes attempted"
        )

        metrics['passes_completed'] = MetricValue(
            name='passes_completed',
            value=successful_passes,
            per_90=self._per_90(successful_passes, minutes),
            description="Successful passes"
        )

        pass_accuracy = (successful_passes / total_passes * 100) if total_passes > 0 else 0
        metrics['pass_accuracy'] = MetricValue(
            name='pass_accuracy',
            value=pass_accuracy,
            description="Pass completion percentage",
            format_type="percentage"
        )

        # Progressive passes
        if 'is_progressive_pass' in passes.columns:
            prog_passes = passes['is_progressive_pass'].sum()
            metrics['progressive_passes'] = MetricValue(
                name='progressive_passes',
                value=prog_passes,
                per_90=self._per_90(prog_passes, minutes),
                description="Passes that advance ball significantly"
            )

        # Line breaking passes
        if 'is_line_break' in passes.columns:
            line_breaks = passes['is_line_break'].sum()
            metrics['line_breaking_passes'] = MetricValue(
                name='line_breaking_passes',
                value=line_breaks,
                per_90=self._per_90(line_breaks, minutes),
                description="Passes that break defensive lines"
            )

        # Final third entries
        if 'end_in_final_third' in passes.columns:
            ft_entries = passes[
                (passes['outcome_success']) &
                (passes['x'] < 66.66) &
                (passes['end_in_final_third'])
            ]
            metrics['final_third_entries'] = MetricValue(
                name='final_third_entries',
                value=len(ft_entries),
                per_90=self._per_90(len(ft_entries), minutes),
                description="Passes into the final third"
            )

        # Box entries
        if 'end_in_box' in passes.columns:
            box_entries = passes[
                (passes['outcome_success']) &
                (~passes['in_box']) &
                (passes['end_in_box'])
            ]
            metrics['penalty_area_entries'] = MetricValue(
                name='penalty_area_entries',
                value=len(box_entries),
                per_90=self._per_90(len(box_entries), minutes),
                description="Passes into the penalty area"
            )

        # Long passes
        if 'distance_meters' in passes.columns:
            long_threshold = self.thresholds.get('long_pass_meters', 25)
            long_passes = passes[passes['distance_meters'] >= long_threshold]
            long_successful = long_passes['outcome_success'].sum() if 'outcome_success' in long_passes.columns else 0

            metrics['long_passes_attempted'] = MetricValue(
                name='long_passes_attempted',
                value=len(long_passes),
                per_90=self._per_90(len(long_passes), minutes),
                description=f"Passes >= {long_threshold}m"
            )

            long_accuracy = (long_successful / len(long_passes) * 100) if len(long_passes) > 0 else 0
            metrics['long_pass_accuracy'] = MetricValue(
                name='long_pass_accuracy',
                value=long_accuracy,
                description="Long pass completion percentage",
                format_type="percentage"
            )

        # Switches
        if 'is_switch' in passes.columns:
            switches = passes['is_switch'].sum()
            metrics['switches'] = MetricValue(
                name='switches',
                value=switches,
                per_90=self._per_90(switches, minutes),
                description="Switches of play"
            )

        # Through balls
        if 'is_through_ball' in passes.columns:
            through_balls = passes['is_through_ball'].sum()
            metrics['through_balls'] = MetricValue(
                name='through_balls',
                value=through_balls,
                per_90=self._per_90(through_balls, minutes),
                description="Through balls played"
            )

        # Crosses
        if 'is_cross' in passes.columns:
            crosses = passes[passes['is_cross']]
            successful_crosses = crosses['outcome_success'].sum() if 'outcome_success' in crosses.columns else 0

            metrics['crosses'] = MetricValue(
                name='crosses',
                value=len(crosses),
                per_90=self._per_90(len(crosses), minutes),
                description="Crosses attempted"
            )

            cross_accuracy = (successful_crosses / len(crosses) * 100) if len(crosses) > 0 else 0
            metrics['cross_accuracy'] = MetricValue(
                name='cross_accuracy',
                value=cross_accuracy,
                description="Cross completion percentage",
                format_type="percentage"
            )

        # Key passes
        if 'is_key_pass' in passes.columns:
            key_passes = passes['is_key_pass'].sum()
            metrics['key_passes'] = MetricValue(
                name='key_passes',
                value=key_passes,
                per_90=self._per_90(key_passes, minutes),
                description="Passes leading to a shot"
            )

        return metrics

    def _calculate_carrying(
        self,
        df: pd.DataFrame,
        minutes: int,
        possession: float
    ) -> Dict[str, MetricValue]:
        """Calculate carrying/dribbling metrics."""
        metrics = {}

        dribbles = df[df['is_dribble']] if 'is_dribble' in df.columns else pd.DataFrame()

        if dribbles.empty:
            return metrics

        # Total carries/dribbles
        total_dribbles = len(dribbles)
        successful_dribbles = dribbles['outcome_success'].sum() if 'outcome_success' in dribbles.columns else 0

        metrics['dribbles_attempted'] = MetricValue(
            name='dribbles_attempted',
            value=total_dribbles,
            per_90=self._per_90(total_dribbles, minutes),
            description="Take-on attempts"
        )

        dribble_success = (successful_dribbles / total_dribbles * 100) if total_dribbles > 0 else 0
        metrics['dribble_success_rate'] = MetricValue(
            name='dribble_success_rate',
            value=dribble_success,
            description="Take-on success percentage",
            format_type="percentage"
        )

        # Progressive carries
        if 'is_progressive_carry' in dribbles.columns:
            prog_carries = dribbles['is_progressive_carry'].sum()
            metrics['progressive_carries'] = MetricValue(
                name='progressive_carries',
                value=prog_carries,
                per_90=self._per_90(prog_carries, minutes),
                description="Carries that advance ball significantly"
            )

        # Carry distance
        if 'distance_meters' in dribbles.columns:
            total_distance = dribbles['distance_meters'].sum()
            metrics['carry_distance'] = MetricValue(
                name='carry_distance',
                value=total_distance,
                per_90=self._per_90(total_distance, minutes),
                description="Total carry distance (meters)"
            )

        # Carries into final third
        if 'end_in_final_third' in dribbles.columns:
            ft_carries = dribbles[
                (dribbles['outcome_success']) &
                (dribbles['x'] < 66.66) &
                (dribbles['end_in_final_third'])
            ]
            metrics['carries_into_final_third'] = MetricValue(
                name='carries_into_final_third',
                value=len(ft_carries),
                per_90=self._per_90(len(ft_carries), minutes),
                description="Carries ending in final third"
            )

        # Carries into box
        if 'end_in_box' in dribbles.columns:
            box_carries = dribbles[
                (dribbles['outcome_success']) &
                (~dribbles['in_box']) &
                (dribbles['end_in_box'])
            ]
            metrics['carries_into_box'] = MetricValue(
                name='carries_into_box',
                value=len(box_carries),
                per_90=self._per_90(len(box_carries), minutes),
                description="Carries ending in penalty area"
            )

        # Dispossessions
        dispossessions = df[df['is_dispossessed']].shape[0] if 'is_dispossessed' in df.columns else 0
        metrics['dispossessions'] = MetricValue(
            name='dispossessions',
            value=dispossessions,
            per_90=self._per_90(dispossessions, minutes),
            description="Times dispossessed"
        )

        return metrics

    def _calculate_chance_creation(
        self,
        df: pd.DataFrame,
        minutes: int,
        possession: float
    ) -> Dict[str, MetricValue]:
        """Calculate chance creation and shooting metrics."""
        metrics = {}

        shots = df[df['is_shot']] if 'is_shot' in df.columns else pd.DataFrame()

        # Shots
        total_shots = len(shots)
        metrics['shots'] = MetricValue(
            name='shots',
            value=total_shots,
            per_90=self._per_90(total_shots, minutes),
            description="Shots taken"
        )

        if not shots.empty:
            # Shots on target
            on_target = shots['is_on_target'].sum() if 'is_on_target' in shots.columns else 0
            metrics['shots_on_target'] = MetricValue(
                name='shots_on_target',
                value=on_target,
                per_90=self._per_90(on_target, minutes),
                description="Shots on target"
            )

            # Goals
            goals = shots['is_goal'].sum() if 'is_goal' in shots.columns else 0
            metrics['goals'] = MetricValue(
                name='goals',
                value=goals,
                per_90=self._per_90(goals, minutes),
                description="Goals scored"
            )

            # Shot on target %
            shot_accuracy = (on_target / total_shots * 100) if total_shots > 0 else 0
            metrics['shot_accuracy'] = MetricValue(
                name='shot_accuracy',
                value=shot_accuracy,
                description="Shot accuracy percentage",
                format_type="percentage"
            )

            # Conversion rate
            conversion = (goals / total_shots * 100) if total_shots > 0 else 0
            metrics['conversion_rate'] = MetricValue(
                name='conversion_rate',
                value=conversion,
                description="Goal conversion rate",
                format_type="percentage"
            )

        # xG (would need to be calculated or provided)
        # Placeholder for xG metrics
        metrics['xG'] = MetricValue(
            name='xG',
            value=0.0,
            per_90=0.0,
            description="Expected goals"
        )

        # xA (would need shot data from teammates)
        metrics['xA'] = MetricValue(
            name='xA',
            value=0.0,
            per_90=0.0,
            description="Expected assists"
        )

        # Assists (from is_assist qualifier)
        assists = df['is_assist'].sum() if 'is_assist' in df.columns else 0
        metrics['assists'] = MetricValue(
            name='assists',
            value=assists,
            per_90=self._per_90(assists, minutes),
            description="Assists"
        )

        # Big chances created
        big_chances = df['is_big_chance'].sum() if 'is_big_chance' in df.columns else 0
        metrics['big_chances_created'] = MetricValue(
            name='big_chances_created',
            value=big_chances,
            per_90=self._per_90(big_chances, minutes),
            description="Big chances created"
        )

        return metrics

    def _calculate_defending(
        self,
        df: pd.DataFrame,
        minutes: int,
        possession: float
    ) -> Dict[str, MetricValue]:
        """Calculate defensive metrics."""
        metrics = {}

        # Tackles
        tackles = df[df['is_tackle']] if 'is_tackle' in df.columns else pd.DataFrame()
        total_tackles = len(tackles)
        successful_tackles = tackles['outcome_success'].sum() if not tackles.empty and 'outcome_success' in tackles.columns else 0

        metrics['tackles'] = MetricValue(
            name='tackles',
            value=total_tackles,
            per_90=self._per_90(total_tackles, minutes),
            possession_adjusted=self._possession_adjust(
                self._per_90(total_tackles, minutes), possession, is_attacking=False
            ),
            description="Tackles attempted"
        )

        metrics['tackles_won'] = MetricValue(
            name='tackles_won',
            value=successful_tackles,
            per_90=self._per_90(successful_tackles, minutes),
            description="Successful tackles"
        )

        tackle_success = (successful_tackles / total_tackles * 100) if total_tackles > 0 else 0
        metrics['tackle_success_rate'] = MetricValue(
            name='tackle_success_rate',
            value=tackle_success,
            description="Tackle success percentage",
            format_type="percentage"
        )

        # Interceptions
        interceptions = df['is_interception'].sum() if 'is_interception' in df.columns else 0
        metrics['interceptions'] = MetricValue(
            name='interceptions',
            value=interceptions,
            per_90=self._per_90(interceptions, minutes),
            possession_adjusted=self._possession_adjust(
                self._per_90(interceptions, minutes), possession, is_attacking=False
            ),
            description="Interceptions"
        )

        # Blocks
        blocks = df['is_blocked'].sum() if 'is_blocked' in df.columns else 0
        metrics['blocks'] = MetricValue(
            name='blocks',
            value=blocks,
            per_90=self._per_90(blocks, minutes),
            description="Shots and passes blocked"
        )

        # Clearances
        clearances = df['is_clearance'].sum() if 'is_clearance' in df.columns else 0
        metrics['clearances'] = MetricValue(
            name='clearances',
            value=clearances,
            per_90=self._per_90(clearances, minutes),
            description="Clearances"
        )

        # Aerial duels
        aerials = df[df['is_aerial']] if 'is_aerial' in df.columns else pd.DataFrame()
        total_aerials = len(aerials)
        successful_aerials = aerials['outcome_success'].sum() if not aerials.empty and 'outcome_success' in aerials.columns else 0

        metrics['aerial_duels'] = MetricValue(
            name='aerial_duels',
            value=total_aerials,
            per_90=self._per_90(total_aerials, minutes),
            description="Aerial duels contested"
        )

        aerial_win_rate = (successful_aerials / total_aerials * 100) if total_aerials > 0 else 0
        metrics['aerial_duel_win_rate'] = MetricValue(
            name='aerial_duel_win_rate',
            value=aerial_win_rate,
            description="Aerial duel success percentage",
            format_type="percentage"
        )

        # Ball recoveries
        recoveries = df['is_recovery'].sum() if 'is_recovery' in df.columns else 0
        metrics['ball_recoveries'] = MetricValue(
            name='ball_recoveries',
            value=recoveries,
            per_90=self._per_90(recoveries, minutes),
            description="Ball recoveries"
        )

        # Total defensive actions
        def_actions = interceptions + blocks + clearances + successful_tackles
        metrics['defensive_actions'] = MetricValue(
            name='defensive_actions',
            value=def_actions,
            per_90=self._per_90(def_actions, minutes),
            description="Total defensive actions"
        )

        return metrics

    def _calculate_discipline(
        self,
        df: pd.DataFrame,
        minutes: int,
        possession: float
    ) -> Dict[str, MetricValue]:
        """Calculate discipline and risk metrics."""
        metrics = {}

        # Fouls
        fouls = df[df['is_foul']] if 'is_foul' in df.columns else pd.DataFrame()
        fouls_committed = len(fouls)

        metrics['fouls_committed'] = MetricValue(
            name='fouls_committed',
            value=fouls_committed,
            per_90=self._per_90(fouls_committed, minutes),
            description="Fouls committed"
        )

        # Cards
        cards = df[df['is_card']] if 'is_card' in df.columns else pd.DataFrame()
        yellow_cards = len(cards)  # Simplified - would need qualifier check for card type

        metrics['yellow_cards'] = MetricValue(
            name='yellow_cards',
            value=yellow_cards,
            per_90=self._per_90(yellow_cards, minutes),
            description="Yellow cards"
        )

        metrics['red_cards'] = MetricValue(
            name='red_cards',
            value=0,  # Would need specific qualifier
            per_90=0.0,
            description="Red cards"
        )

        # Turnovers
        dispossessions = df['is_dispossessed'].sum() if 'is_dispossessed' in df.columns else 0
        failed_passes = len(df[(df['is_pass']) & (~df['outcome_success'])]) if 'is_pass' in df.columns else 0

        turnovers = dispossessions + failed_passes
        metrics['turnovers'] = MetricValue(
            name='turnovers',
            value=turnovers,
            per_90=self._per_90(turnovers, minutes),
            description="Total turnovers"
        )

        return metrics

    def calculate_match_by_match(
        self,
        events_df: pd.DataFrame,
        minutes_by_match: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Calculate metrics match-by-match for trend analysis.

        Args:
            events_df: All player events
            minutes_by_match: Dictionary of match_id -> minutes

        Returns:
            DataFrame with metrics per match
        """
        match_metrics = []

        for match_id in events_df['match_id'].unique():
            match_events = events_df[events_df['match_id'] == match_id]
            match_minutes = minutes_by_match.get(match_id, 90)

            if match_minutes <= 0:
                continue

            result = self.calculate_all_metrics(match_events, match_minutes)

            match_row = {
                'match_id': match_id,
                'minutes': match_minutes
            }
            match_row.update(result.to_per90_dict())
            match_metrics.append(match_row)

        return pd.DataFrame(match_metrics)


def calculate_rolling_averages(
    df: pd.DataFrame,
    window: int = 5,
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Calculate rolling averages for metrics.

    Args:
        df: DataFrame with match-by-match metrics
        window: Rolling window size
        metrics: List of metrics to calculate (defaults to all numeric)

    Returns:
        DataFrame with rolling averages added
    """
    if metrics is None:
        metrics = df.select_dtypes(include=[np.number]).columns.tolist()
        metrics = [m for m in metrics if m not in ['match_id', 'minutes']]

    result = df.copy()

    for metric in metrics:
        if metric in df.columns:
            result[f'{metric}_rolling_{window}'] = df[metric].rolling(
                window=window, min_periods=1
            ).mean()

    return result
