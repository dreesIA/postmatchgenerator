"""
Data loader and coordinator for gathering player scouting data.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .opta_client import OptaClient
from .normalize import EventNormalizer, normalize_events_dataframe

logger = logging.getLogger(__name__)


@dataclass
class PlayerInfo:
    """Player information."""
    player_id: str
    name: str
    first_name: str
    last_name: str
    position: str
    position_group: str
    team_id: str
    team_name: str
    nationality: str = ""
    age: int = 0
    height: int = 0
    weight: int = 0
    foot: str = ""
    jersey_number: int = 0


@dataclass
class MatchAppearance:
    """Player appearance in a match."""
    match_id: str
    date: str
    opponent: str
    home_away: str
    result: str
    minutes: int
    started: bool
    goals: int = 0
    assists: int = 0
    position: str = ""


@dataclass
class PlayerData:
    """Complete player data for scouting report."""
    player_info: PlayerInfo
    appearances: List[MatchAppearance]
    events_df: pd.DataFrame
    total_minutes: int
    matches_played: int
    matches_started: int
    competition_name: str
    season: str
    date_range: Tuple[str, str]
    team_possession: Dict[str, float] = field(default_factory=dict)
    physical_data: Optional[pd.DataFrame] = None


class DataLoader:
    """
    Coordinates data loading for player scouting reports.

    Handles:
    - Loading match list and filtering by date range
    - Loading player appearances and minutes
    - Loading and normalizing event data
    - Loading team possession for adjustment
    - Loading physical data (if available)
    """

    def __init__(self, client: OptaClient = None):
        """
        Initialize the data loader.

        Args:
            client: OptaClient instance (creates new one if not provided)
        """
        self.client = client or OptaClient()
        self.normalizer = EventNormalizer()

    def load_player_data(
        self,
        player_id: str,
        competition_id: str,
        season_id: str,
        team_id: str = None,
        start_date: str = None,
        end_date: str = None,
        position_group: str = None
    ) -> PlayerData:
        """
        Load all data needed for a player scouting report.

        Args:
            player_id: Player ID
            competition_id: Competition ID
            season_id: Season ID
            team_id: Team ID (optional, will be detected)
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            position_group: Position group override

        Returns:
            PlayerData with all necessary information
        """
        logger.info(f"Loading data for player {player_id}")

        # Get player info and team
        player_info = self._get_player_info(
            player_id, competition_id, season_id, team_id
        )

        if position_group:
            player_info.position_group = position_group

        # Get matches for the team
        matches = self.client.get_matches(
            competition_id=competition_id,
            season_id=season_id,
            team_id=player_info.team_id,
            start_date=start_date,
            end_date=end_date
        )

        logger.info(f"Found {len(matches)} matches for team {player_info.team_name}")

        # Load appearances and events
        appearances = []
        all_events = []
        team_possession = {}

        for match in matches:
            match_id = self._extract_match_id(match)
            if not match_id:
                continue

            try:
                # Get match events
                events_data = self.client.get_events(match_id)

                # Check if player participated
                player_events = self._filter_player_events(events_data, player_id)

                if player_events:
                    # Get match details
                    appearance = self._extract_appearance(
                        match, events_data, player_id, player_info.team_id
                    )
                    appearances.append(appearance)

                    # Normalize events
                    normalized = self.normalizer.normalize_match(
                        events_data, player_info.team_id
                    )
                    all_events.append(normalized)

                    # Get possession data
                    try:
                        poss = self.client.get_team_possession(match_id)
                        team_possession[match_id] = poss.get(player_info.team_id, 50)
                    except Exception:
                        team_possession[match_id] = 50.0

            except Exception as e:
                logger.warning(f"Error loading match {match_id}: {e}")

        # Combine all events
        if all_events:
            events_df = pd.concat(all_events, ignore_index=True)
            # Filter to player events
            events_df = events_df[events_df['player_id'] == player_id].copy()
        else:
            events_df = pd.DataFrame()

        # Calculate totals
        total_minutes = sum(a.minutes for a in appearances)
        matches_started = sum(1 for a in appearances if a.started)

        # Determine date range
        if appearances:
            dates = [a.date for a in appearances if a.date]
            date_range = (min(dates), max(dates)) if dates else (start_date, end_date)
        else:
            date_range = (start_date, end_date)

        # Get competition name
        competition_name = self._get_competition_name(competition_id)

        logger.info(
            f"Loaded {len(appearances)} appearances, "
            f"{total_minutes} minutes, {len(events_df)} events"
        )

        return PlayerData(
            player_info=player_info,
            appearances=appearances,
            events_df=events_df,
            total_minutes=total_minutes,
            matches_played=len(appearances),
            matches_started=matches_started,
            competition_name=competition_name,
            season=season_id,
            date_range=date_range,
            team_possession=team_possession,
            physical_data=None  # Physical data loaded separately if available
        )

    def _get_player_info(
        self,
        player_id: str,
        competition_id: str,
        season_id: str,
        team_id: str = None
    ) -> PlayerInfo:
        """Get player information from squad data."""
        # If team_id not provided, we need to find it
        if not team_id:
            # Get player stats to find team
            player_stats = self.client.get_player_stats(competition_id, season_id)
            team_id = self._find_player_team(player_stats, player_id)

        if not team_id:
            raise ValueError(f"Could not find team for player {player_id}")

        # Get squad data
        try:
            squad_data = self.client.get_squads(competition_id, season_id, team_id)
            player_data = self._find_player_in_squad(squad_data, player_id)
        except Exception:
            player_data = None

        if player_data:
            return self._parse_player_info(player_data, team_id, squad_data)
        else:
            # Return basic info
            return PlayerInfo(
                player_id=player_id,
                name=f"Player {player_id}",
                first_name="",
                last_name=f"Player {player_id}",
                position="Unknown",
                position_group="CM",
                team_id=team_id,
                team_name=f"Team {team_id}"
            )

    def _find_player_team(self, player_stats: Dict, player_id: str) -> Optional[str]:
        """Find which team a player belongs to."""
        # Navigate stats structure to find player
        for team in player_stats.get('team', []):
            for player in team.get('player', []):
                if str(player.get('playerId', '')) == str(player_id):
                    return team.get('contestantId')
        return None

    def _find_player_in_squad(self, squad_data: Dict, player_id: str) -> Optional[Dict]:
        """Find player in squad data."""
        for player in squad_data.get('player', []):
            if str(player.get('id', '')) == str(player_id):
                return player
            if str(player.get('playerId', '')) == str(player_id):
                return player
        return None

    def _parse_player_info(
        self,
        player_data: Dict,
        team_id: str,
        squad_data: Dict
    ) -> PlayerInfo:
        """Parse player info from API data."""
        position = player_data.get('position', 'Unknown')
        position_group = self._map_position_group(position)

        # Get team name
        team_name = squad_data.get('contestantName', f"Team {team_id}")

        # Calculate age if DOB available
        age = 0
        dob = player_data.get('dateOfBirth', player_data.get('dob'))
        if dob:
            try:
                birth_date = datetime.strptime(dob[:10], '%Y-%m-%d')
                age = (datetime.now() - birth_date).days // 365
            except ValueError:
                pass

        return PlayerInfo(
            player_id=str(player_data.get('id', player_data.get('playerId'))),
            name=player_data.get('name', player_data.get('matchName', '')),
            first_name=player_data.get('firstName', ''),
            last_name=player_data.get('lastName', ''),
            position=position,
            position_group=position_group,
            team_id=team_id,
            team_name=team_name,
            nationality=player_data.get('nationality', ''),
            age=age,
            height=int(player_data.get('height', 0) or 0),
            weight=int(player_data.get('weight', 0) or 0),
            foot=player_data.get('foot', ''),
            jersey_number=int(player_data.get('shirtNumber', 0) or 0)
        )

    def _map_position_group(self, position: str) -> str:
        """Map position to position group."""
        position_lower = position.lower()

        if 'goalkeeper' in position_lower:
            return 'GK'
        elif 'centre-back' in position_lower or 'central defender' in position_lower:
            return 'CB'
        elif 'back' in position_lower or 'wing-back' in position_lower:
            return 'FB'
        elif 'defensive mid' in position_lower or 'holding' in position_lower:
            return 'DM'
        elif 'attacking mid' in position_lower:
            return 'AM'
        elif 'midfielder' in position_lower or 'central mid' in position_lower:
            return 'CM'
        elif 'winger' in position_lower or 'wide' in position_lower:
            return 'W'
        elif 'forward' in position_lower or 'striker' in position_lower:
            return 'ST'
        else:
            return 'CM'

    def _extract_match_id(self, match: Dict) -> Optional[str]:
        """Extract match ID from match data."""
        if '@attributes' in match:
            match_id = match['@attributes'].get('uId', '')
        else:
            match_id = match.get('uId', match.get('id', match.get('matchId', '')))

        if match_id:
            # Remove 'g' prefix if present
            return str(match_id).replace('g', '')
        return None

    def _filter_player_events(self, events_data: Dict, player_id: str) -> List[Dict]:
        """Filter events for a specific player."""
        events = []

        live_data = events_data.get('liveData', {})
        event_list = live_data.get('event', [])

        for event in event_list:
            if str(event.get('playerId', '')) == str(player_id):
                events.append(event)

        return events

    def _extract_appearance(
        self,
        match: Dict,
        events_data: Dict,
        player_id: str,
        team_id: str
    ) -> MatchAppearance:
        """Extract appearance details from match data."""
        match_info = events_data.get('matchInfo', {})

        # Get match ID
        match_id = self._extract_match_id(match) or match_info.get('id', '')

        # Get date
        date = match.get('date', match.get('matchDate', ''))
        if not date:
            date = match_info.get('date', '')

        # Get teams and score
        contestants = match_info.get('contestant', [])
        home_team = away_team = None
        home_score = away_score = 0

        for team in contestants:
            if team.get('position') == 'home':
                home_team = team
                home_score = int(team.get('score', 0))
            else:
                away_team = team
                away_score = int(team.get('score', 0))

        # Determine opponent and home/away
        if home_team and str(home_team.get('id', '')) == str(team_id):
            opponent = away_team.get('name', 'Unknown') if away_team else 'Unknown'
            home_away = 'H'
            result = 'W' if home_score > away_score else 'L' if home_score < away_score else 'D'
        else:
            opponent = home_team.get('name', 'Unknown') if home_team else 'Unknown'
            home_away = 'A'
            result = 'W' if away_score > home_score else 'L' if away_score < home_score else 'D'

        # Get player stats from match
        minutes = 0
        started = False
        goals = 0
        assists = 0
        position = ""

        for team in contestants:
            for player in team.get('player', []):
                if str(player.get('playerId', '')) == str(player_id):
                    stats = player.get('stats', {})
                    minutes = int(stats.get('minsPlayed', 0))
                    started = player.get('position', '') != 'Substitute'
                    goals = int(stats.get('goals', 0))
                    assists = int(stats.get('assists', 0))
                    position = player.get('position', '')
                    break

        return MatchAppearance(
            match_id=match_id,
            date=date,
            opponent=opponent,
            home_away=home_away,
            result=result,
            minutes=minutes,
            started=started,
            goals=goals,
            assists=assists,
            position=position
        )

    def _get_competition_name(self, competition_id: str) -> str:
        """Get competition name from ID."""
        competition_names = {
            '8': 'Premier League',
            '23': 'La Liga',
            '9': 'Bundesliga',
            '21': 'Serie A',
            '24': 'Ligue 1',
            '5': 'Champions League',
            '6': 'Europa League',
        }
        return competition_names.get(competition_id, f"Competition {competition_id}")

    def load_cohort_data(
        self,
        competition_id: str,
        season_id: str,
        position_group: str,
        min_minutes: int = 450,
        exclude_player_id: str = None
    ) -> pd.DataFrame:
        """
        Load data for cohort players (same position, same competition).

        Args:
            competition_id: Competition ID
            season_id: Season ID
            position_group: Position group to filter
            min_minutes: Minimum minutes played
            exclude_player_id: Player to exclude (the subject)

        Returns:
            DataFrame with cohort player stats
        """
        logger.info(f"Loading cohort data for {position_group} in competition {competition_id}")

        # Get player stats for the competition
        try:
            player_stats = self.client.get_player_stats(competition_id, season_id)
        except Exception as e:
            logger.error(f"Failed to load player stats: {e}")
            return pd.DataFrame()

        cohort_players = []

        for team in player_stats.get('team', []):
            team_id = team.get('contestantId')
            team_name = team.get('contestantName', '')

            for player in team.get('player', []):
                player_id = str(player.get('playerId', ''))

                # Skip the subject player
                if player_id == str(exclude_player_id):
                    continue

                # Get player position
                position = player.get('position', '')
                player_position_group = self._map_position_group(position)

                if player_position_group != position_group:
                    continue

                # Get minutes
                stats = player.get('stat', [])
                minutes = 0
                for stat in stats:
                    if stat.get('type') == 'minsPlayed':
                        minutes = int(stat.get('value', 0))
                        break

                if minutes < min_minutes:
                    continue

                # Extract all stats
                player_stats_dict = {
                    'player_id': player_id,
                    'player_name': player.get('name', player.get('matchName', '')),
                    'team_id': team_id,
                    'team_name': team_name,
                    'position': position,
                    'position_group': player_position_group,
                    'minutes': minutes
                }

                for stat in stats:
                    stat_type = stat.get('type', '')
                    stat_value = stat.get('value', 0)
                    try:
                        player_stats_dict[stat_type] = float(stat_value)
                    except (ValueError, TypeError):
                        player_stats_dict[stat_type] = stat_value

                cohort_players.append(player_stats_dict)

        if cohort_players:
            df = pd.DataFrame(cohort_players)
            logger.info(f"Loaded {len(df)} cohort players")
            return df
        else:
            logger.warning("No cohort players found")
            return pd.DataFrame()
