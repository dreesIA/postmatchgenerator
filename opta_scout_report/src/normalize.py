"""
Event data normalization and standardization.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# Opta event type mappings
EVENT_TYPE_MAP = {
    1: 'pass',
    2: 'offside_pass',
    3: 'take_on',
    4: 'foul',
    5: 'out',
    6: 'corner_awarded',
    7: 'tackle',
    8: 'interception',
    9: 'turnover',
    10: 'save',
    11: 'claim',
    12: 'clearance',
    13: 'miss',
    14: 'post',
    15: 'attempt_saved',
    16: 'goal',
    17: 'card',
    18: 'player_off',
    19: 'player_on',
    20: 'player_retired',
    21: 'player_returns',
    22: 'player_becomes_goalkeeper',
    23: 'goalkeeper_becomes_player',
    24: 'condition_change',
    25: 'official_change',
    27: 'start_delay',
    28: 'end_delay',
    30: 'end',
    32: 'start',
    34: 'team_set_up',
    35: 'player_changed_position',
    36: 'player_changed_jersey',
    37: 'collection_end',
    38: 'temp_goal',
    39: 'temp_attempt',
    40: 'formation_change',
    41: 'punch',
    42: 'good_skill',
    43: 'deleted_event',
    44: 'aerial',
    45: 'challenge',
    49: 'ball_recovery',
    50: 'blocked_pass',
    51: 'delay_of_play',
    52: 'keeper_pick_up',
    53: 'chance_missed',
    54: 'ball_touch',
    55: 'temp_save',
    56: 'resume',
    57: 'contentious_referee_decision',
    58: 'possession_data',
    59: 'fifty_fifty',
    60: 'referee_drop_ball',
    61: 'failed_to_block',
    63: 'injury_clearance',
    64: 'tackle',
    65: 'attendance',
    66: 'referee_stop',
    67: 'keeper_sweeper',
    68: 'chance_missed',
    69: 'ball_recovery',
    70: 'blocked',
    71: 'delay',
    72: 'offside',
    73: 'attempt_blocked',
    74: 'dispossessed',
    75: 'error',
    76: 'shield_ball_opp',
    77: 'ball_recovery',
}

# Qualifier type mappings
QUALIFIER_MAP = {
    1: 'long_ball',
    2: 'cross',
    3: 'head_pass',
    4: 'through_ball',
    5: 'free_kick_taken',
    6: 'corner_taken',
    7: 'players_caught_offside',
    13: 'keeper_throw',
    15: 'head',
    16: 'direct_free_kick',
    17: 'right_foot',
    18: 'second_yellow',
    19: 'involved_foul',
    20: 'foul_def_half',
    21: 'foul_att_half',
    22: 'contested',
    24: 'out_right',
    25: 'won',
    26: 'overrun',
    29: 'not_caught',
    30: 'accurate',
    31: 'goal_mouth_blocked',
    33: 'bigChance',
    34: 'dive_fall',
    41: 'no_touch',
    44: 'goal_left_penalty_area',
    55: 'smothered',
    56: 'keeperposition_standing',
    59: 'red_card',
    70: 'deflection',
    72: 'six_yard_blocked',
    82: 'out_bottom',
    83: 'hand',
    87: 'assisting_pass',
    91: 'chance',
    107: 'involved_card',
    113: 'intentional_assist',
    117: 'assisted',
    123: 'leading_to_attempt',
    124: 'leading_to_goal',
    127: 'pass_end_x',
    128: 'pass_end_y',
    140: 'end_x',
    141: 'end_y',
    145: 'end_zones',
    146: 'shot_end_zone',
    148: 'penalty',
    152: 'goal_zone_left',
    153: 'goal_zone_right',
    154: 'goal_zone_centre',
    155: 'goal_zone_low_left',
    156: 'goal_zone_low_right',
    157: 'goal_zone_low_centre',
    158: 'goal_zone_high_left',
    159: 'goal_zone_high_right',
    160: 'goal_zone_high_centre',
    194: 'progressive_pass',
    196: 'saved_off_line',
    197: 'shot_centre_goal',
    199: 'shot_left_goal',
    200: 'shot_right_goal',
    233: 'switch_of_play',
    322: 'big_chance',
    321: 'key_pass',
}


@dataclass
class ZoneDefinitions:
    """Pitch zone definitions (as percentage of pitch, 0-100)."""
    # Thirds (x-axis)
    defensive_third: Tuple[float, float] = (0, 33.33)
    middle_third: Tuple[float, float] = (33.33, 66.66)
    final_third: Tuple[float, float] = (66.66, 100)

    # Channels (y-axis)
    left_channel: Tuple[float, float] = (0, 21.1)
    left_half_space: Tuple[float, float] = (21.1, 36.8)
    central: Tuple[float, float] = (36.8, 63.2)
    right_half_space: Tuple[float, float] = (63.2, 78.9)
    right_channel: Tuple[float, float] = (78.9, 100)

    # Key areas
    box_x: Tuple[float, float] = (83, 100)
    box_y: Tuple[float, float] = (21.1, 78.9)

    zone_14_x: Tuple[float, float] = (66.66, 83)
    zone_14_y: Tuple[float, float] = (36.8, 63.2)


class EventNormalizer:
    """
    Normalizes Opta event data into a consistent DataFrame format.
    """

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        """
        Initialize normalizer.

        Args:
            pitch_length: Pitch length in meters
            pitch_width: Pitch width in meters
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.zones = ZoneDefinitions()

    def normalize_match(
        self,
        match_data: Dict,
        team_id: str = None,
        standardize_direction: bool = True
    ) -> pd.DataFrame:
        """
        Normalize match event data to DataFrame.

        Args:
            match_data: Raw Opta match event data
            team_id: Team ID for direction standardization
            standardize_direction: Whether to standardize attack direction

        Returns:
            Normalized DataFrame
        """
        events = []

        # Get match info
        match_info = match_data.get('matchInfo', {})
        match_id = match_info.get('id', '')

        # Get contestants
        contestants = {
            c.get('id'): c for c in match_info.get('contestant', [])
        }

        # Get live data events
        live_data = match_data.get('liveData', {})
        event_list = live_data.get('event', [])

        for event in event_list:
            normalized = self._normalize_event(
                event, match_id, contestants, team_id, standardize_direction
            )
            if normalized:
                events.append(normalized)

        if not events:
            return pd.DataFrame()

        df = pd.DataFrame(events)

        # Add derived columns
        df = self._add_derived_columns(df)

        return df

    def _normalize_event(
        self,
        event: Dict,
        match_id: str,
        contestants: Dict,
        home_team_id: str,
        standardize_direction: bool
    ) -> Optional[Dict]:
        """Normalize a single event."""
        try:
            type_id = int(event.get('typeId', 0))
            player_id = event.get('playerId', '')
            team_id = event.get('contestantId', '')

            # Get coordinates
            x = float(event.get('x', 0) or 0)
            y = float(event.get('y', 0) or 0)

            # Get end coordinates from qualifiers
            end_x, end_y = self._extract_end_coordinates(event)

            # Extract qualifiers
            qualifiers = self._extract_qualifiers(event)

            # Get outcome
            outcome = int(event.get('outcome', 0))

            # Standardize direction (home team attacks left to right)
            if standardize_direction and team_id != home_team_id:
                x = 100 - x
                y = 100 - y
                if end_x is not None:
                    end_x = 100 - end_x
                if end_y is not None:
                    end_y = 100 - end_y

            return {
                'match_id': match_id,
                'event_id': event.get('id', ''),
                'period': int(event.get('periodId', 0)),
                'minute': int(event.get('timeMin', 0)),
                'second': int(event.get('timeSec', 0)),
                'type_id': type_id,
                'type_name': EVENT_TYPE_MAP.get(type_id, 'unknown'),
                'player_id': str(player_id),
                'team_id': str(team_id),
                'x': x,
                'y': y,
                'end_x': end_x,
                'end_y': end_y,
                'outcome': outcome,
                'outcome_success': outcome == 1,
                'qualifiers': qualifiers,
                'is_long_ball': 1 in qualifiers,
                'is_cross': 2 in qualifiers,
                'is_through_ball': 4 in qualifiers,
                'is_head': 15 in qualifiers,
                'is_key_pass': 321 in qualifiers,
                'is_big_chance': 322 in qualifiers or 33 in qualifiers,
                'is_assist': 87 in qualifiers or 117 in qualifiers,
            }

        except Exception as e:
            logger.warning(f"Error normalizing event: {e}")
            return None

    def _extract_end_coordinates(self, event: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Extract end coordinates from qualifiers."""
        end_x = None
        end_y = None

        qualifiers = event.get('qualifier', [])
        if not isinstance(qualifiers, list):
            qualifiers = [qualifiers]

        for qual in qualifiers:
            qual_id = int(qual.get('qualifierId', 0))
            value = qual.get('value')

            if qual_id == 140:  # end_x
                try:
                    end_x = float(value)
                except (ValueError, TypeError):
                    pass
            elif qual_id == 141:  # end_y
                try:
                    end_y = float(value)
                except (ValueError, TypeError):
                    pass
            elif qual_id == 127:  # pass_end_x (alternative)
                if end_x is None:
                    try:
                        end_x = float(value)
                    except (ValueError, TypeError):
                        pass
            elif qual_id == 128:  # pass_end_y (alternative)
                if end_y is None:
                    try:
                        end_y = float(value)
                    except (ValueError, TypeError):
                        pass

        return end_x, end_y

    def _extract_qualifiers(self, event: Dict) -> List[int]:
        """Extract qualifier IDs from event."""
        qualifier_ids = []

        qualifiers = event.get('qualifier', [])
        if not isinstance(qualifiers, list):
            qualifiers = [qualifiers]

        for qual in qualifiers:
            try:
                qual_id = int(qual.get('qualifierId', 0))
                qualifier_ids.append(qual_id)
            except (ValueError, TypeError):
                pass

        return qualifier_ids

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns to the DataFrame."""
        # Event type flags
        df['is_pass'] = df['type_id'] == 1
        df['is_shot'] = df['type_id'].isin([13, 14, 15, 16])
        df['is_goal'] = df['type_id'] == 16
        df['is_on_target'] = df['type_id'].isin([15, 16])
        df['is_tackle'] = df['type_id'] == 7
        df['is_interception'] = df['type_id'] == 8
        df['is_clearance'] = df['type_id'] == 12
        df['is_aerial'] = df['type_id'] == 44
        df['is_dribble'] = df['type_id'] == 3
        df['is_foul'] = df['type_id'] == 4
        df['is_card'] = df['type_id'] == 17
        df['is_recovery'] = df['type_id'].isin([49, 69, 77])
        df['is_dispossessed'] = df['type_id'] == 74
        df['is_blocked'] = df['type_id'].isin([10, 70])
        df['is_touch'] = df['type_id'].isin([1, 3, 13, 14, 15, 16, 44, 49, 54])

        # Zone tagging
        df['zone_x'] = df['x'].apply(self._get_third)
        df['zone_y'] = df['y'].apply(self._get_channel)
        df['in_box'] = self._in_box(df['x'], df['y'])
        df['in_final_third'] = df['x'] >= 66.66
        df['in_middle_third'] = (df['x'] >= 33.33) & (df['x'] < 66.66)
        df['in_defensive_third'] = df['x'] < 33.33
        df['in_zone_14'] = self._in_zone_14(df['x'], df['y'])

        # End zone tagging
        df['end_zone_x'] = df['end_x'].apply(
            lambda x: self._get_third(x) if pd.notna(x) else None
        )
        df['end_in_box'] = self._in_box(df['end_x'], df['end_y'])
        df['end_in_final_third'] = df['end_x'] >= 66.66

        # Distance calculations
        df['distance'] = self._calculate_distance(
            df['x'], df['y'], df['end_x'], df['end_y']
        )
        df['distance_meters'] = df['distance'] * self.pitch_length / 100

        # Progressive actions
        df['x_progress'] = df['end_x'] - df['x']
        df['is_progressive_pass'] = (
            (df['is_pass']) &
            (df['outcome_success']) &
            (
                (df['x_progress'] >= 10) |  # 10% of pitch = ~10.5m
                ((df['x'] < 66.66) & (df['end_x'] >= 66.66))  # Into final third
            )
        )
        df['is_progressive_carry'] = (
            (df['is_dribble']) &
            (df['outcome_success']) &
            (
                (df['x_progress'] >= 10) |
                ((df['x'] < 66.66) & (df['end_x'] >= 66.66))
            )
        )

        # Line breaking (simplified - passes that go through vertical space)
        df['is_line_break'] = (
            (df['is_pass']) &
            (df['outcome_success']) &
            (df['x_progress'] >= 5) &
            (abs(df['y'] - df['end_y'].fillna(df['y'])) < 20)  # Relatively direct
        )

        # Switch of play
        df['lateral_distance'] = abs(df['end_y'].fillna(df['y']) - df['y'])
        df['is_switch'] = (
            (df['is_pass']) &
            (df['outcome_success']) &
            (df['lateral_distance'] >= 40)
        )

        return df

    def _get_third(self, x: float) -> str:
        """Get the third of the pitch for x coordinate."""
        if pd.isna(x):
            return 'unknown'
        if x < 33.33:
            return 'defensive'
        elif x < 66.66:
            return 'middle'
        else:
            return 'attacking'

    def _get_channel(self, y: float) -> str:
        """Get the channel for y coordinate."""
        if pd.isna(y):
            return 'unknown'
        if y < 21.1:
            return 'left_wide'
        elif y < 36.8:
            return 'left_half_space'
        elif y < 63.2:
            return 'central'
        elif y < 78.9:
            return 'right_half_space'
        else:
            return 'right_wide'

    def _in_box(self, x: pd.Series, y: pd.Series) -> pd.Series:
        """Check if coordinates are in the box."""
        return (x >= 83) & (y >= 21.1) & (y <= 78.9)

    def _in_zone_14(self, x: pd.Series, y: pd.Series) -> pd.Series:
        """Check if coordinates are in zone 14."""
        return (x >= 66.66) & (x < 83) & (y >= 36.8) & (y <= 63.2)

    def _calculate_distance(
        self,
        x1: pd.Series,
        y1: pd.Series,
        x2: pd.Series,
        y2: pd.Series
    ) -> pd.Series:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def normalize_events_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize an existing events DataFrame (convenience function).

    Args:
        df: Events DataFrame

    Returns:
        Normalized DataFrame
    """
    normalizer = EventNormalizer()
    return normalizer._add_derived_columns(df.copy())


def convert_to_meters(x: float, y: float, pitch_length: float = 105, pitch_width: float = 68) -> Tuple[float, float]:
    """
    Convert percentage coordinates to meters.

    Args:
        x: X coordinate (0-100)
        y: Y coordinate (0-100)
        pitch_length: Pitch length in meters
        pitch_width: Pitch width in meters

    Returns:
        (x_meters, y_meters)
    """
    return (x * pitch_length / 100, y * pitch_width / 100)


def convert_to_percentage(x_m: float, y_m: float, pitch_length: float = 105, pitch_width: float = 68) -> Tuple[float, float]:
    """
    Convert meter coordinates to percentage.

    Args:
        x_m: X coordinate in meters
        y_m: Y coordinate in meters
        pitch_length: Pitch length in meters
        pitch_width: Pitch width in meters

    Returns:
        (x_percent, y_percent)
    """
    return (x_m * 100 / pitch_length, y_m * 100 / pitch_width)
