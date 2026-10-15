"""
Opta F24 Event Data API Module
Retrieves and processes match event data from Perform API via scoresway.com
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


class RateLimitError(Exception):
    """Raised when the Opta feed keeps returning HTTP 429 after retries"""


@dataclass
class LeagueConfig:
    """Configuration for a specific league/tournament"""
    key: str
    name: str
    country: str
    tournament_id: str
    outlet_key: str
    season: str


class ConfigLoader:
    """Load and manage league configurations from JSON file"""

    DEFAULT_CONFIG_PATH = Path(__file__).parent / "leagues.json"

    def __init__(self, config_path: str = None):
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self._config = None
        self._leagues: Dict[str, LeagueConfig] = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from JSON file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self._config = json.load(f)

        for key, data in self._config.get('leagues', {}).items():
            self._leagues[key] = LeagueConfig(
                key=key,
                name=data.get('name', key),
                country=data.get('country', ''),
                tournament_id=data['tournament_id'],
                outlet_key=data['outlet_key'],
                season=data.get('season', '')
            )

    def get_league(self, league_key: str) -> LeagueConfig:
        """Get configuration for a specific league"""
        if league_key not in self._leagues:
            available = ', '.join(self._leagues.keys())
            raise ValueError(f"Unknown league: {league_key}. Available: {available}")
        return self._leagues[league_key]

    def get_default_league(self) -> LeagueConfig:
        """Get the default league configuration"""
        default_key = self._config.get('default_league')
        if default_key and default_key in self._leagues:
            return self._leagues[default_key]
        return next(iter(self._leagues.values()))

    def list_leagues(self) -> List[str]:
        """List all available league keys"""
        return list(self._leagues.keys())

    def get_all_leagues(self) -> Dict[str, LeagueConfig]:
        """Get all league configurations"""
        return self._leagues.copy()


@dataclass
class MatchInfo:
    """Container for match metadata"""
    match_id: str
    home_team_id: str
    away_team_id: str
    home_team_name: str
    away_team_name: str
    competition: str
    date: str
    venue: str
    home_score: int
    away_score: int
    status: str = ''
    coverage_level: str = ''


class OptaAPI:
    """
    Client for retrieving Opta F24 event data from Perform API

    Can be configured for specific leagues via LeagueConfig or used with defaults.

    Usage:
        # Config-driven (recommended)
        api = OptaAPI.for_league("usl_championship")
        matches = api.get_matches_by_tournament()  # Uses config

        # Manual configuration
        api = OptaAPI()
        matches = api.get_matches_by_tournament("tournament_id")
    """

    BASE_URL = "https://api.performfeeds.com/soccerdata"
    DEFAULT_OUTLET_KEY = "qxcx5jmswgto1qeqcjzghtddt"  # Public outlet key from scoresway
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/116.0',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Prefer': 'safe',
        'Referer': 'https://www.scoresway.com/',
        'Origin': 'https://www.scoresway.com',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'Sec-GPC': '1',
        'Cache-Control': 'max-age=0',
    }
    
    # Event Type IDs
    EVENT_TYPES = {
        1: 'Pass',
        2: 'Offside Pass',
        3: 'Take On',
        4: 'Foul',
        5: 'Out',
        6: 'Corner Awarded',
        7: 'Tackle',
        8: 'Interception',
        10: 'Save',
        11: 'Claim',
        12: 'Clearance',
        13: 'Miss',
        14: 'Post',
        15: 'Attempt Saved',
        16: 'Goal',
        17: 'Card',
        18: 'Player Off',
        19: 'Player On',
        27: 'Start Delay',
        28: 'End Delay',
        30: 'End',
        32: 'Start',
        34: 'Team Set Up',
        37: 'End Match',
        40: 'Formation Change',
        41: 'Punch',
        43: 'Deleted Event',
        44: 'Aerial',
        45: 'Challenge',
        49: 'Ball Recovery',
        50: 'Blocked Pass',
        51: 'Dispossessed',
        52: 'Keeper Pick-up',
        53: 'Chance Missed',
        54: 'Ball Touch',
        55: 'Temp_Goal',
        56: 'Resume',
        57: 'Contentious Referee Decision',
        58: 'Possession Data',
        59: 'Fifty Fifty',
        60: 'Referee Drop Ball',
        61: 'Failed To Block',
        64: 'Offside',
        65: 'Attempt Blocked',
        66: 'Tackle Foul',
        67: 'Temp_Retake Penalty',
        68: 'Offside Provoked',
        69: 'Keeper Sweeper',
        70: 'Injury Clearance',
        71: 'Tackle',
        72: 'Temp_Save',
        73: 'Temp_Resume',
        74: 'Shield Ball OOP',
        75: 'Ball Recovery',
        76: 'Blocked Pass',
        77: 'Second Yellow',
        78: 'Delayed Start',
        79: 'Tackle',
        80: 'Cross not claimed',
        81: 'Smother',
        82: 'Error',
        83: 'Temp_Shot',
        84: 'Condition Change',
    }
    
    # Qualifier IDs for passes
    PASS_QUALIFIERS = {
        1: 'long_ball',
        2: 'cross',
        3: 'head_pass',
        4: 'through_ball',
        5: 'free_kick',
        6: 'corner',
        7: 'offside_player',
        106: 'attacking_pass',
        107: 'throw_in',
        140: 'pass_end_x',
        141: 'pass_end_y',
        155: 'chipped',
        156: 'lay_off',
        157: 'launch',
        168: 'flick_on',
        195: 'pull_back',
        196: 'switch_of_play',
        210: 'assist',
        212: 'length',
        213: 'angle',
        218: 'second_assist',
    }
    
    # Qualifier IDs for shots
    SHOT_QUALIFIERS = {
        15: 'head',
        20: 'right_foot',
        22: 'regular_play',
        24: 'set_piece',
        25: 'from_corner',
        26: 'free_kick',
        29: 'assisted',
        72: 'left_foot',
        82: 'blocked',
        102: 'goalmouth_y',
        103: 'goalmouth_z',
        146: 'blocked_x',
        147: 'blocked_y',
        154: 'intentional_assist',
        214: 'big_chance',
        215: 'individual_play',
    }

    def __init__(self, league: LeagueConfig = None, outlet_key: str = None):
        """
        Initialize API client

        Args:
            league: LeagueConfig object for the target league
            outlet_key: Override outlet key (optional, uses league config or default)
        """
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

        self.league = league
        if outlet_key:
            self.outlet_key = outlet_key
        elif league:
            self.outlet_key = league.outlet_key
        else:
            self.outlet_key = self.DEFAULT_OUTLET_KEY

        # Optional relay in front of the feed (see cloudflare_worker/), for hosts the feed
        # rate limits. Set as env vars or as root-level Streamlit secrets.
        relay_url = os.environ.get('OPTA_BASE_URL', '').strip().rstrip('/')
        if relay_url and not relay_url.endswith('/soccerdata'):
            relay_url += '/soccerdata'
        self.base_url = relay_url or self.BASE_URL
        relay_key = os.environ.get('OPTA_RELAY_KEY', '').strip()
        if relay_key:
            self.session.headers['X-Relay-Key'] = relay_key

    @classmethod
    def for_league(cls, league_key: str, config_path: str = None) -> 'OptaAPI':
        """
        Factory method to create API client for a specific league

        Args:
            league_key: League identifier (e.g., "usl_championship")
            config_path: Path to config file (optional)

        Returns:
            OptaAPI instance configured for the league
        """
        config_loader = ConfigLoader(config_path)
        league = config_loader.get_league(league_key)
        return cls(league=league)
    
    def _parse_jsonp(self, response_text: str) -> dict:
        """Parse JSONP response to JSON"""
        soup = BeautifulSoup(response_text, 'lxml')
        text = soup.get_text()
        # Remove JSONP callback wrapper
        start = text.find('(') + 1
        end = text.rfind(')')
        json_str = text[start:end]
        return json.loads(json_str)

    def _fetch(self, url: str, params: dict, max_retries: int = 3) -> dict:
        """GET a feed URL and parse it, retrying with backoff when rate limited (429)"""
        for attempt in range(max_retries + 1):
            response = self.session.get(url, params=params, timeout=30)
            # The feed reports errors either as a real HTTP status or inside a 200 body
            rate_limited = response.status_code == 429
            data = None
            if not rate_limited:
                response.raise_for_status()
                data = self._parse_jsonp(response.text)
                rate_limited = str(data.get('httpStatus')) == '429'
            if not rate_limited:
                return data
            if attempt < max_retries:
                retry_after = response.headers.get('Retry-After', '')
                delay = int(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1)
                time.sleep(min(delay, 30))

        raise RateLimitError(
            "The Opta feed is rate limiting requests from this server (HTTP 429). "
            "Wait a few minutes and try again."
        )

    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request and parse response"""
        if params is None:
            params = {}
        
        # Add default params
        params.update({
            '_rt': 'c',
            '_lcl': 'en',
            '_fmt': 'jsonp',
            '_clbk': 'callback'
        })
        
        url = f"{self.base_url}/{endpoint}/{self.outlet_key}"
        return self._fetch(url, params)
    
    def get_match_events(self, match_id: str) -> Tuple[pd.DataFrame, MatchInfo]:
        """
        Retrieve all events for a match

        Args:
            match_id: Opta match ID

        Returns:
            Tuple of (events DataFrame, MatchInfo object)
        """
        return self.parse_match_events(self.get_match_events_raw(match_id), match_id)

    def get_match_events_raw(self, match_id: str) -> dict:
        """
        Retrieve the raw match event feed (as saved by download_match.py)

        Args:
            match_id: Opta match ID

        Returns:
            Feed response as a dictionary
        """
        url = f"{self.base_url}/matchevent/{self.outlet_key}/{match_id}"
        params = {
            '_rt': 'c',
            '_lcl': 'en',
            '_fmt': 'jsonp',
            '_clbk': 'callback'
        }
        
        data = self._fetch(url, params)

        # The feed returns HTTP 200 with an error body when the match can't be served
        # (rate limiting is retried and raised separately in _fetch)
        if 'errorCode' in data:
            raise ValueError(
                f"Opta feed returned error {data.get('errorCode')} (HTTP {data.get('httpStatus')}) "
                f"for match {match_id}. The match may not have been played yet, the ID may be "
                f"wrong, or this competition isn't available with the current outlet key."
            )

        return data

    def parse_match_events(self, data: dict, match_id: str) -> Tuple[pd.DataFrame, MatchInfo]:
        """
        Turn a raw match event feed into an events DataFrame and MatchInfo

        Args:
            data: Raw feed from get_match_events_raw (or an uploaded match file)
            match_id: Opta match ID

        Returns:
            Tuple of (events DataFrame, MatchInfo object)
        """
        # Extract match info
        match_info_raw = data.get('matchInfo', {})
        contestants = match_info_raw.get('contestant', [])
        
        home_team = next((c for c in contestants if c.get('position') == 'home'), {})
        away_team = next((c for c in contestants if c.get('position') == 'away'), {})
        
        scores = data.get('liveData', {}).get('matchDetails', {}).get('scores', {}).get('total', {})
        
        match_info = MatchInfo(
            match_id=match_id,
            home_team_id=home_team.get('id', ''),
            away_team_id=away_team.get('id', ''),
            home_team_name=home_team.get('name', ''),
            away_team_name=away_team.get('name', ''),
            competition=match_info_raw.get('competition', {}).get('name', ''),
            date=match_info_raw.get('date', ''),
            venue=match_info_raw.get('venue', {}).get('longName', ''),
            home_score=scores.get('home', 0),
            away_score=scores.get('away', 0),
            status=data.get('liveData', {}).get('matchDetails', {}).get('matchStatus', ''),
            coverage_level=match_info_raw.get('coverageLevel', '')
        )

        # Extract events
        events = data.get('liveData', {}).get('event', [])
        events_df = pd.json_normalize(events)
        
        # Add event type names
        if 'typeId' in events_df.columns:
            events_df['event_type'] = events_df['typeId'].map(self.EVENT_TYPES)
        
        return events_df, match_info
    
    def get_match_stats(self, match_id: str) -> dict:
        """
        Retrieve match statistics
        
        Args:
            match_id: Opta match ID
            
        Returns:
            Dictionary with team and player stats
        """
        url = f"{self.base_url}/matchstats/{self.outlet_key}/{match_id}"
        params = {
            '_rt': 'c',
            'detailed': 'yes',
            '_lcl': 'en',
            '_fmt': 'jsonp',
            '_clbk': 'callback'
        }
        
        return self._fetch(url, params)
    
    def get_matches_by_tournament(self, tournament_id: str = None, page_size: int = 400) -> pd.DataFrame:
        """
        Get all matches for a tournament/season

        Args:
            tournament_id: Tournament calendar ID (uses league config if not provided)
            page_size: Number of matches per page

        Returns:
            DataFrame with match list
        """
        if tournament_id is None and self.league:
            tournament_id = self.league.tournament_id

        if tournament_id is None:
            raise ValueError("tournament_id must be provided or set via league config")

        url = f"{self.base_url}/match/{self.outlet_key}"
        params = {
            '_rt': 'c',
            'tmcl': tournament_id,
            'live': 'yes',
            '_pgSz': str(page_size),
            '_lcl': 'en',
            '_fmt': 'jsonp',
            '_clbk': 'callback'
        }
        
        data = self._fetch(url, params)
        
        matches = data.get('match', [])
        return pd.json_normalize(matches)
    
    def get_squad(self, team_id: str, tournament_id: str) -> pd.DataFrame:
        """
        Get squad for a team in a tournament
        
        Args:
            team_id: Contestant/team ID
            tournament_id: Tournament calendar ID
            
        Returns:
            DataFrame with player info
        """
        url = f"{self.base_url}/squads/{self.outlet_key}"
        params = {
            '_rt': 'c',
            'ctst': team_id,
            'tmcl': tournament_id,
            '_pgSz': '200',
            '_lcl': 'en',
            '_fmt': 'jsonp',
            '_clbk': 'callback'
        }
        
        data = self._fetch(url, params)
        
        squad = data.get('squad', [{}])[0]
        players = squad.get('person', [])
        
        return pd.json_normalize(players)
    
    def get_standings(self, tournament_id: str) -> pd.DataFrame:
        """
        Get league standings
        
        Args:
            tournament_id: Tournament calendar ID
            
        Returns:
            DataFrame with standings
        """
        url = f"{self.base_url}/standings/{self.outlet_key}"
        params = {
            '_rt': 'c',
            'tmcl': tournament_id,
            'live': 'yes',
            '_lcl': 'en',
            '_fmt': 'jsonp',
            '_clbk': 'callback'
        }
        
        data = self._fetch(url, params)
        
        # Navigate to rankings
        stages = data.get('stage', [])
        if stages:
            divisions = stages[0].get('division', [])
            if divisions:
                rankings = divisions[0].get('ranking', [])
                return pd.json_normalize(rankings)
        
        return pd.DataFrame()

    def get_player_info(self, player_id: str, team_id: str = None, tournament_id: str = None) -> Optional[Dict]:
        """
        Get player information from squad data

        Args:
            player_id: Player ID
            team_id: Team ID (optional if league is configured)
            tournament_id: Tournament ID (optional if league is configured)

        Returns:
            Dictionary with player info or None if not found
        """
        if tournament_id is None and self.league:
            tournament_id = self.league.tournament_id

        if tournament_id is None:
            raise ValueError("tournament_id must be provided or set via league config")

        # If team_id not provided, we need to search all teams
        if team_id:
            squad_df = self.get_squad(team_id, tournament_id)
            if 'id' in squad_df.columns:
                player_row = squad_df[squad_df['id'] == player_id]
                if len(player_row) > 0:
                    return player_row.iloc[0].to_dict()
        return None

    def get_team_matches(self, team_id: str, tournament_id: str = None) -> pd.DataFrame:
        """
        Get all matches for a specific team in a tournament

        Args:
            team_id: Team/contestant ID
            tournament_id: Tournament ID (optional if league is configured)

        Returns:
            DataFrame with match info
        """
        if tournament_id is None and self.league:
            tournament_id = self.league.tournament_id

        if tournament_id is None:
            raise ValueError("tournament_id must be provided or set via league config")

        all_matches = self.get_matches_by_tournament(tournament_id)

        # Filter for team's matches
        team_matches = []
        for _, match in all_matches.iterrows():
            contestants = match.get('matchInfo.contestant', [])
            if isinstance(contestants, list):
                for c in contestants:
                    if isinstance(c, dict) and c.get('id') == team_id:
                        team_matches.append({
                            'match_id': match.get('matchInfo.id'),
                            'date': match.get('matchInfo.date'),
                            'competition': match.get('matchInfo.competition.name'),
                            'home_team': contestants[0].get('name') if contestants else '',
                            'away_team': contestants[1].get('name') if len(contestants) > 1 else '',
                            'home_id': contestants[0].get('id') if contestants else '',
                            'away_id': contestants[1].get('id') if len(contestants) > 1 else '',
                        })
                        break

        return pd.DataFrame(team_matches)

    def parse_scoresway_url(self, url: str) -> Dict[str, str]:
        """
        Parse a Scoresway URL to extract IDs

        Args:
            url: Full Scoresway URL

        Returns:
            Dictionary with extracted IDs (tournament_id, team_id, player_id, match_id)
        """
        import re

        result = {}

        # Tournament ID pattern: /soccer/league-name/TOURNAMENT_ID/
        tournament_match = re.search(r'/soccer/[^/]+/([a-z0-9]+)/', url)
        if tournament_match:
            result['tournament_id'] = tournament_match.group(1)

        # Team ID pattern: /teams/view/TEAM_ID/
        team_match = re.search(r'/teams/view/([a-z0-9]+)', url)
        if team_match:
            result['team_id'] = team_match.group(1)

        # Player ID pattern: /player/view/PLAYER_ID
        player_match = re.search(r'/player/view/([a-z0-9]+)', url)
        if player_match:
            result['player_id'] = player_match.group(1)

        # Match ID pattern: /match/view/MATCH_ID/
        match_match = re.search(r'/match/view/([a-z0-9]+)', url)
        if match_match:
            result['match_id'] = match_match.group(1)

        return result


class EventProcessor:
    """
    Process raw Opta events into analysis-ready DataFrames
    """
    
    # Shot type IDs
    SHOT_TYPES = [13, 14, 15, 16]  # Miss, Post, Saved, Goal
    
    def __init__(self, events_df: pd.DataFrame, match_info: MatchInfo):
        self.events_df = events_df.copy()
        self.match_info = match_info
        self._preprocess_events()
    
    def _preprocess_events(self):
        """Add common derived columns"""
        # Ensure qualifier column exists and handle NaN
        if 'qualifier' not in self.events_df.columns:
            self.events_df['qualifier'] = [[] for _ in range(len(self.events_df))]
        else:
            self.events_df['qualifier'] = self.events_df['qualifier'].apply(
                lambda x: x if isinstance(x, list) else []
            )
        
        # Add team names
        team_map = {
            self.match_info.home_team_id: self.match_info.home_team_name,
            self.match_info.away_team_id: self.match_info.away_team_name
        }
        self.events_df['team_name'] = self.events_df['contestantId'].map(team_map)
        
        # Add is_home flag
        self.events_df['is_home'] = self.events_df['contestantId'] == self.match_info.home_team_id
        
        # Add open play flag
        self.events_df['is_open_play'] = self.events_df['qualifier'].apply(
            lambda q: not any(item.get('qualifierId') in [5, 6, 107] for item in q)
        )
    
    def _extract_qualifier_value(self, qualifiers: list, qualifier_id: int) -> Optional[float]:
        """Extract value from qualifier list"""
        for q in qualifiers:
            if q.get('qualifierId') == qualifier_id:
                val = q.get('value')
                if val is not None:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return None
        return None
    
    def _has_qualifier(self, qualifiers: list, qualifier_id: int) -> bool:
        """Check if qualifier exists"""
        return any(q.get('qualifierId') == qualifier_id for q in qualifiers)
    
    def get_passes(self) -> pd.DataFrame:
        """
        Extract and enrich pass events
        
        Returns:
            DataFrame with pass data including end coordinates
        """
        passes = self.events_df[self.events_df['typeId'] == 1].copy()
        
        # Extract end coordinates from qualifiers
        passes['end_x'] = passes['qualifier'].apply(
            lambda q: self._extract_qualifier_value(q, 140)
        )
        passes['end_y'] = passes['qualifier'].apply(
            lambda q: self._extract_qualifier_value(q, 141)
        )
        
        # Extract pass attributes
        passes['length'] = passes['qualifier'].apply(
            lambda q: self._extract_qualifier_value(q, 212)
        )
        passes['angle'] = passes['qualifier'].apply(
            lambda q: self._extract_qualifier_value(q, 213)
        )
        
        # Pass type flags
        passes['is_long_ball'] = passes['qualifier'].apply(
            lambda q: self._has_qualifier(q, 1)
        )
        passes['is_cross'] = passes['qualifier'].apply(
            lambda q: self._has_qualifier(q, 2)
        )
        passes['is_through_ball'] = passes['qualifier'].apply(
            lambda q: self._has_qualifier(q, 4)
        )
        passes['is_corner'] = passes['qualifier'].apply(
            lambda q: self._has_qualifier(q, 6)
        )
        passes['is_free_kick'] = passes['qualifier'].apply(
            lambda q: self._has_qualifier(q, 5)
        )
        passes['is_throw_in'] = passes['qualifier'].apply(
            lambda q: self._has_qualifier(q, 107)
        )
        passes['is_assist'] = passes['qualifier'].apply(
            lambda q: self._has_qualifier(q, 210)
        )
        passes['is_key_pass'] = passes['qualifier'].apply(
            lambda q: self._has_qualifier(q, 210) or self._has_qualifier(q, 218)
        )
        
        # Calculate pass success
        passes['is_successful'] = passes['outcome'] == 1
        
        # Select and rename columns
        cols = [
            'id', 'eventId', 'periodId', 'timeMin', 'timeSec',
            'contestantId', 'team_name', 'is_home',
            'playerId', 'playerName',
            'x', 'y', 'end_x', 'end_y',
            'is_successful', 'is_open_play',
            'length', 'angle',
            'is_long_ball', 'is_cross', 'is_through_ball',
            'is_corner', 'is_free_kick', 'is_throw_in',
            'is_assist', 'is_key_pass'
        ]
        
        return passes[[c for c in cols if c in passes.columns]]
    
    def get_shots(self) -> pd.DataFrame:
        """
        Extract and enrich shot events with xG estimation
        
        Returns:
            DataFrame with shot data including xG
        """
        shots = self.events_df[self.events_df['typeId'].isin(self.SHOT_TYPES)].copy()
        
        # Shot outcome
        shots['is_goal'] = shots['typeId'] == 16
        shots['is_on_target'] = shots['typeId'].isin([15, 16])
        shots['hit_post'] = shots['typeId'] == 14
        
        # Body part
        shots['body_part'] = 'right_foot'  # default
        shots.loc[shots['qualifier'].apply(lambda q: self._has_qualifier(q, 72)), 'body_part'] = 'left_foot'
        shots.loc[shots['qualifier'].apply(lambda q: self._has_qualifier(q, 15)), 'body_part'] = 'head'
        
        # Situation
        shots['is_big_chance'] = shots['qualifier'].apply(
            lambda q: self._has_qualifier(q, 214)
        )
        shots['is_from_set_piece'] = shots['qualifier'].apply(
            lambda q: self._has_qualifier(q, 24) or self._has_qualifier(q, 25) or self._has_qualifier(q, 26)
        )
        shots['is_penalty'] = shots['qualifier'].apply(
            lambda q: self._has_qualifier(q, 9)
        )
        
        # Goalmouth coordinates
        shots['goalmouth_y'] = shots['qualifier'].apply(
            lambda q: self._extract_qualifier_value(q, 102)
        )
        shots['goalmouth_z'] = shots['qualifier'].apply(
            lambda q: self._extract_qualifier_value(q, 103)
        )
        
        # Calculate xG
        shots['xG'] = shots.apply(self._calculate_xg, axis=1)
        
        # Select columns
        cols = [
            'id', 'eventId', 'periodId', 'timeMin', 'timeSec',
            'contestantId', 'team_name', 'is_home',
            'playerId', 'playerName',
            'x', 'y',
            'is_goal', 'is_on_target', 'hit_post',
            'body_part', 'is_big_chance', 'is_from_set_piece', 'is_penalty',
            'is_open_play', 'xG',
            'goalmouth_y', 'goalmouth_z'
        ]
        
        return shots[[c for c in cols if c in shots.columns]]
    
    def _calculate_xg(self, shot: pd.Series) -> float:
        """
        Calculate expected goals for a shot using an improved model.

        Based on research from StatsBomb, FBRef, and academic xG models.
        Uses logistic regression with calibrated coefficients.

        Args:
            shot: Series with shot data

        Returns:
            xG value between 0 and 1
        """
        # Handle penalty - historical conversion rate
        if shot.get('is_penalty', False):
            return 0.76

        # Get coordinates (Opta uses 0-100 scale, goal at x=100, y=0-100)
        x = shot.get('x', 50)
        y = shot.get('y', 50)

        # Convert Opta coordinates to meters (pitch is ~105m x 68m)
        x_meters = (100 - x) * 1.05  # Distance from goal line in meters
        y_meters = (y - 50) * 0.68   # Distance from center in meters

        # Distance from center of goal
        distance = np.sqrt(x_meters**2 + y_meters**2)

        # Calculate visible goal angle (radians)
        # Goal posts are at y = 50 ± (7.32/2) / 0.68 ≈ 50 ± 5.38 in Opta coords
        goal_width = 7.32  # meters
        half_goal = goal_width / 2

        # Angle from shot location to both posts
        left_post_y = -half_goal
        right_post_y = half_goal

        # Angle to goal using law of cosines
        dist_left = np.sqrt(x_meters**2 + (y_meters - left_post_y)**2)
        dist_right = np.sqrt(x_meters**2 + (y_meters - right_post_y)**2)

        # Visible angle to goal
        cos_angle = (dist_left**2 + dist_right**2 - goal_width**2) / (2 * dist_left * dist_right)
        cos_angle = np.clip(cos_angle, -1, 1)
        goal_angle = np.arccos(cos_angle)

        # Logistic regression coefficients (calibrated to approximate professional xG models)
        # Note: Opta's proprietary xG uses additional features (defender/GK position, shot speed)
        # that are not available in the standard event feed. This model provides a reasonable
        # approximation based on shot location and basic contextual features.
        intercept = -1.00
        coef_distance = -0.095
        coef_angle = 1.45
        coef_distance_sq = -0.0015

        # Linear predictor
        log_odds = (
            intercept +
            coef_distance * distance +
            coef_angle * goal_angle +
            coef_distance_sq * (distance ** 2)
        )

        # Body part adjustment
        body_part = shot.get('body_part', 'right_foot')
        if body_part == 'head':
            log_odds -= 0.45  # Headers are harder to convert

        # Shot location adjustments
        # Shots from very central positions are easier
        if abs(y_meters) < 3:
            log_odds += 0.15

        # Shots from inside the 6-yard box
        if x_meters < 5.5 and abs(y_meters) < 9.16:
            log_odds += 0.40

        # Big chance indicator (when available from Opta)
        if shot.get('is_big_chance', False):
            log_odds += 0.60

        # Set piece shots typically have lower xG (more defenders)
        if shot.get('is_from_set_piece', False) and not shot.get('is_penalty', False):
            log_odds -= 0.20

        # Convert log odds to probability
        xg = 1 / (1 + np.exp(-log_odds))

        # Clamp to valid range
        return max(0.01, min(0.95, xg))
    
    def get_defensive_actions(self) -> pd.DataFrame:
        """
        Extract defensive actions (tackles, interceptions, clearances, blocks)
        
        Returns:
            DataFrame with defensive action data
        """
        defensive_types = [7, 8, 12, 10]  # Tackle, Interception, Clearance, Save
        
        defensive = self.events_df[self.events_df['typeId'].isin(defensive_types)].copy()
        
        # Action type name
        action_map = {7: 'Tackle', 8: 'Interception', 12: 'Clearance', 10: 'Save'}
        defensive['action_type'] = defensive['typeId'].map(action_map)
        
        # Success
        defensive['is_successful'] = defensive['outcome'] == 1
        
        cols = [
            'id', 'eventId', 'periodId', 'timeMin', 'timeSec',
            'contestantId', 'team_name', 'is_home',
            'playerId', 'playerName',
            'x', 'y',
            'action_type', 'is_successful'
        ]
        
        return defensive[[c for c in cols if c in defensive.columns]]
    
    def get_ball_recoveries(self) -> pd.DataFrame:
        """
        Extract ball recovery events
        
        Returns:
            DataFrame with ball recovery locations
        """
        recoveries = self.events_df[self.events_df['typeId'] == 49].copy()
        
        # Add pitch third
        recoveries['pitch_third'] = pd.cut(
            recoveries['x'],
            bins=[0, 33.33, 66.66, 100],
            labels=['Defensive', 'Middle', 'Attacking']
        )
        
        cols = [
            'id', 'eventId', 'periodId', 'timeMin', 'timeSec',
            'contestantId', 'team_name', 'is_home',
            'playerId', 'playerName',
            'x', 'y', 'pitch_third'
        ]
        
        return recoveries[[c for c in cols if c in recoveries.columns]]
    
    def get_take_ons(self) -> pd.DataFrame:
        """
        Extract dribble/take-on events
        
        Returns:
            DataFrame with take-on data
        """
        take_ons = self.events_df[self.events_df['typeId'] == 3].copy()
        take_ons['is_successful'] = take_ons['outcome'] == 1
        
        cols = [
            'id', 'eventId', 'periodId', 'timeMin', 'timeSec',
            'contestantId', 'team_name', 'is_home',
            'playerId', 'playerName',
            'x', 'y', 'is_successful'
        ]
        
        return take_ons[[c for c in cols if c in take_ons.columns]]
    
    def get_aerials(self) -> pd.DataFrame:
        """
        Extract aerial duel events
        
        Returns:
            DataFrame with aerial duel data
        """
        aerials = self.events_df[self.events_df['typeId'] == 44].copy()
        aerials['is_won'] = aerials['outcome'] == 1
        
        cols = [
            'id', 'eventId', 'periodId', 'timeMin', 'timeSec',
            'contestantId', 'team_name', 'is_home',
            'playerId', 'playerName',
            'x', 'y', 'is_won'
        ]
        
        return aerials[[c for c in cols if c in aerials.columns]]
    
    def get_fouls(self) -> pd.DataFrame:
        """
        Extract foul events
        
        Returns:
            DataFrame with foul data
        """
        fouls = self.events_df[self.events_df['typeId'] == 4].copy()
        
        # Foul committed (outcome=1) vs foul won (outcome=0)
        fouls['foul_committed'] = fouls['outcome'] == 1
        
        cols = [
            'id', 'eventId', 'periodId', 'timeMin', 'timeSec',
            'contestantId', 'team_name', 'is_home',
            'playerId', 'playerName',
            'x', 'y', 'foul_committed'
        ]
        
        return fouls[[c for c in cols if c in fouls.columns]]


class PassingNetworkBuilder:
    """
    Build passing network data for visualization
    """

    def __init__(self, passes_df: pd.DataFrame, match_info: MatchInfo, events_df: pd.DataFrame = None):
        self.passes = passes_df.copy()
        self.match_info = match_info
        self.events_df = events_df  # Full events for identifying substitutes

    def build_network(self, team_id: str = None, period: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build passing network nodes and edges

        Args:
            team_id: Filter by team (optional)
            period: Filter by period (optional)

        Returns:
            Tuple of (nodes_df, edges_df)
        """
        passes = self.passes.copy()

        # Filter by team
        if team_id:
            passes = passes[passes['contestantId'] == team_id]

        # Filter by period
        if period:
            passes = passes[passes['periodId'] == period]

        # Only successful passes
        passes = passes[passes['is_successful'] == True]

        # Get receiver from next event (simplified - you may want to enhance this)
        # For now, we'll use the pass end location to estimate position

        # Build nodes: average position per player
        nodes = passes.groupby(['playerId', 'playerName']).agg({
            'x': 'mean',
            'y': 'mean',
            'id': 'count'  # pass count
        }).reset_index()
        nodes.columns = ['player_id', 'player_name', 'avg_x', 'avg_y', 'pass_count']

        # Identify substitutes from "Player On" events (typeId == 19)
        sub_player_ids = set()
        if self.events_df is not None:
            team_events = self.events_df
            if team_id:
                team_events = team_events[team_events['contestantId'] == team_id]
            # Player On events (typeId 19) indicate substitutes coming on
            sub_events = team_events[team_events['typeId'] == 19]
            sub_player_ids = set(sub_events['playerId'].dropna().unique())

        # Mark starters (players NOT in sub_player_ids)
        nodes['is_starter'] = ~nodes['player_id'].isin(sub_player_ids)
        
        # Build edges: pass combinations
        # We need to identify the receiver - this requires matching with the next event
        # For now, we'll create a simplified version based on sequential passes
        
        edges_list = []
        team_passes = passes.sort_values(['periodId', 'timeMin', 'timeSec'])
        
        for i in range(len(team_passes) - 1):
            current = team_passes.iloc[i]
            next_pass = team_passes.iloc[i + 1]
            
            # Check if same team and close in time (likely same possession)
            if (current['contestantId'] == next_pass['contestantId'] and
                abs(current['timeMin'] - next_pass['timeMin']) <= 1):
                
                edges_list.append({
                    'passer_id': current['playerId'],
                    'passer_name': current['playerName'],
                    'receiver_id': next_pass['playerId'],
                    'receiver_name': next_pass['playerName']
                })
        
        edges = pd.DataFrame(edges_list)
        
        if len(edges) > 0:
            # Count pass combinations
            edges = edges.groupby(
                ['passer_id', 'passer_name', 'receiver_id', 'receiver_name']
            ).size().reset_index(name='pass_count')
        
        return nodes, edges


if __name__ == "__main__":
    # Example usage
    api = OptaAPI()
    
    # Example match ID (Swiss Super League match from the notebook)
    match_id = "31tcbue0pnhmssglpr9ezvkt0"
    
    print("Fetching match events...")
    events_df, match_info = api.get_match_events(match_id)
    
    print(f"\nMatch: {match_info.home_team_name} vs {match_info.away_team_name}")
    print(f"Score: {match_info.home_score} - {match_info.away_score}")
    print(f"Events: {len(events_df)}")
    
    # Process events
    processor = EventProcessor(events_df, match_info)
    
    passes = processor.get_passes()
    shots = processor.get_shots()
    recoveries = processor.get_ball_recoveries()
    
    print(f"\nPasses: {len(passes)}")
    print(f"Shots: {len(shots)}")
    print(f"Ball Recoveries: {len(recoveries)}")
    
    # Build passing network
    network_builder = PassingNetworkBuilder(passes, match_info)
    nodes, edges = network_builder.build_network(team_id=match_info.home_team_id)
    
    print(f"\nPassing Network - {match_info.home_team_name}")
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
