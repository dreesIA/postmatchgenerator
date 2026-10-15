"""
Opta API Client with caching, retries, and rate limiting.
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached API response."""
    data: Any
    timestamp: datetime
    endpoint: str
    params: Dict


class DiskCache:
    """Simple disk-based cache for API responses."""

    def __init__(self, cache_dir: str = ".opta_cache", expiry_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expiry_hours = expiry_hours

    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate a unique cache key from endpoint and params."""
        param_str = json.dumps(params, sort_keys=True)
        key_str = f"{endpoint}:{param_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.json"

    def get(self, endpoint: str, params: Dict) -> Optional[Any]:
        """Retrieve cached data if valid."""
        cache_key = self._get_cache_key(endpoint, params)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)

            cached_time = datetime.fromisoformat(cached['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=self.expiry_hours):
                cache_path.unlink()
                return None

            logger.debug(f"Cache hit for {endpoint}")
            return cached['data']
        except (json.JSONDecodeError, KeyError):
            cache_path.unlink()
            return None

    def set(self, endpoint: str, params: Dict, data: Any) -> None:
        """Store data in cache."""
        cache_key = self._get_cache_key(endpoint, params)
        cache_path = self._get_cache_path(cache_key)

        cached = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'params': params,
            'data': data
        }

        with open(cache_path, 'w') as f:
            json.dump(cached, f)

        logger.debug(f"Cached response for {endpoint}")

    def clear(self) -> None:
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info("Cache cleared")

    def clear_expired(self) -> int:
        """Remove expired cache entries. Returns count of removed entries."""
        removed = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                cached_time = datetime.fromisoformat(cached['timestamp'])
                if datetime.now() - cached_time > timedelta(hours=self.expiry_hours):
                    cache_file.unlink()
                    removed += 1
            except (json.JSONDecodeError, KeyError):
                cache_file.unlink()
                removed += 1
        return removed


class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0

    def wait(self) -> None:
        """Wait if necessary to respect rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()


class OptaClient:
    """
    Opta API client with caching, retries, and rate limiting.

    Usage:
        client = OptaClient()
        matches = client.get_matches(competition_id="8", season_id="2023")
        events = client.get_events(match_id="12345")
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the Opta client.

        Args:
            config_path: Path to opta_endpoints.json config file
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "opta_endpoints.json"

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Set up authentication
        self.base_url = self.config['base_url']
        self._setup_auth()

        # Set up session with retries
        self.session = self._create_session()

        # Set up caching
        cache_config = self.config.get('cache', {})
        if cache_config.get('enabled', True):
            self.cache = DiskCache(
                cache_dir=cache_config.get('directory', '.opta_cache'),
                expiry_hours=cache_config.get('expiry_hours', 24)
            )
        else:
            self.cache = None

        # Set up rate limiting
        rate_config = self.config.get('rate_limits', {})
        self.rate_limiter = RateLimiter(
            requests_per_minute=rate_config.get('requests_per_minute', 60)
        )

        # Request tracking
        self.request_count = 0
        self.cache_hits = 0

    def _setup_auth(self) -> None:
        """Set up authentication credentials from environment."""
        auth_config = self.config.get('auth', {})

        if auth_config.get('type') == 'basic':
            username_env = auth_config.get('username_env', 'OPTA_USERNAME')
            password_env = auth_config.get('password_env', 'OPTA_PASSWORD')

            self.username = os.environ.get(username_env)
            self.password = os.environ.get(password_env)

            if not self.username or not self.password:
                logger.warning(
                    f"Opta credentials not found in environment variables "
                    f"({username_env}, {password_env}). API calls may fail."
                )

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()

        rate_config = self.config.get('rate_limits', {})
        retry_strategy = Retry(
            total=rate_config.get('retry_attempts', 3),
            backoff_factor=rate_config.get('backoff_factor', 2),
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # Set auth if available
        if hasattr(self, 'username') and self.username:
            session.auth = (self.username, self.password)

        return session

    def _build_url(self, endpoint_name: str, path_params: Dict = None) -> str:
        """Build the full URL for an endpoint."""
        endpoint_config = self.config['endpoints'].get(endpoint_name)
        if not endpoint_config:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")

        path = endpoint_config['path']
        if path_params:
            for key, value in path_params.items():
                path = path.replace(f"{{{key}}}", str(value))

        return f"{self.base_url}{path}"

    def _build_params(self, endpoint_name: str, custom_params: Dict = None) -> Dict:
        """Build query parameters for an endpoint."""
        endpoint_config = self.config['endpoints'].get(endpoint_name)
        params = dict(endpoint_config.get('params', {}))

        # Replace placeholders in params
        if custom_params:
            for key, value in list(params.items()):
                if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                    param_name = value[1:-1]
                    if param_name in custom_params:
                        params[key] = custom_params[param_name]
                    else:
                        del params[key]

            # Add any additional custom params
            for key, value in custom_params.items():
                if key not in params and not key.startswith('_'):
                    params[key] = value

        return params

    def _request(
        self,
        endpoint_name: str,
        path_params: Dict = None,
        query_params: Dict = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Make an API request with caching and rate limiting.

        Args:
            endpoint_name: Name of the endpoint from config
            path_params: Parameters for URL path
            query_params: Additional query parameters
            use_cache: Whether to use cache

        Returns:
            Parsed JSON response
        """
        url = self._build_url(endpoint_name, path_params)
        params = self._build_params(endpoint_name, {**(path_params or {}), **(query_params or {})})

        # Check cache first
        if use_cache and self.cache:
            cached = self.cache.get(url, params)
            if cached is not None:
                self.cache_hits += 1
                return cached

        # Rate limit
        self.rate_limiter.wait()

        # Make request
        start_time = time.time()
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            duration = time.time() - start_time
            logger.info(
                f"API Request: {endpoint_name} | Status: {response.status_code} | "
                f"Duration: {duration:.2f}s"
            )

            self.request_count += 1

            # Cache response
            if use_cache and self.cache:
                self.cache.set(url, params, data)

            return data

        except requests.exceptions.RequestException as e:
            duration = time.time() - start_time
            logger.error(
                f"API Error: {endpoint_name} | Error: {str(e)} | Duration: {duration:.2f}s"
            )
            raise

    # =========================================================================
    # High-level API methods
    # =========================================================================

    def get_matches(
        self,
        competition_id: str,
        season_id: str,
        team_id: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> List[Dict]:
        """
        Get matches for a competition/season.

        Args:
            competition_id: Competition ID
            season_id: Season ID
            team_id: Optional team filter
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            List of match dictionaries
        """
        params = {
            'competition_id': competition_id,
            'season_id': season_id
        }
        if team_id:
            params['teamId'] = team_id
        if start_date:
            params['fromDate'] = start_date
        if end_date:
            params['toDate'] = end_date

        data = self._request('matches', path_params=params, query_params=params)

        # Extract matches from response
        matches = []
        if 'match' in data:
            matches = data['match'] if isinstance(data['match'], list) else [data['match']]
        elif 'matchDate' in data:
            for match_date in data['matchDate']:
                if 'match' in match_date:
                    match_list = match_date['match']
                    if isinstance(match_list, list):
                        matches.extend(match_list)
                    else:
                        matches.append(match_list)

        return matches

    def get_events(self, match_id: str) -> Dict:
        """
        Get event data for a match.

        Args:
            match_id: Match ID

        Returns:
            Match event data
        """
        return self._request(
            'match_events',
            path_params={'match_id': match_id}
        )

    def get_match_stats(self, match_id: str) -> Dict:
        """
        Get match statistics.

        Args:
            match_id: Match ID

        Returns:
            Match statistics data
        """
        return self._request(
            'match_stats',
            path_params={'match_id': match_id}
        )

    def get_player_stats(
        self,
        competition_id: str,
        season_id: str,
        team_id: str = None
    ) -> Dict:
        """
        Get player statistics for a competition/season.

        Args:
            competition_id: Competition ID
            season_id: Season ID
            team_id: Optional team filter

        Returns:
            Player statistics data
        """
        params = {
            'competition_id': competition_id,
            'season_id': season_id
        }
        if team_id:
            params['team_id'] = team_id

        return self._request('player_stats', path_params=params, query_params=params)

    def get_team_stats(self, competition_id: str, season_id: str) -> Dict:
        """
        Get team statistics for a competition/season.

        Args:
            competition_id: Competition ID
            season_id: Season ID

        Returns:
            Team statistics data
        """
        return self._request(
            'team_stats',
            path_params={
                'competition_id': competition_id,
                'season_id': season_id
            }
        )

    def get_squads(
        self,
        competition_id: str,
        season_id: str,
        team_id: str
    ) -> Dict:
        """
        Get squad information for a team.

        Args:
            competition_id: Competition ID
            season_id: Season ID
            team_id: Team ID

        Returns:
            Squad data
        """
        return self._request(
            'squads',
            path_params={
                'competition_id': competition_id,
                'season_id': season_id,
                'team_id': team_id
            }
        )

    def get_player_minutes(
        self,
        player_id: str,
        competition_id: str,
        season_id: str
    ) -> List[Dict]:
        """
        Get minutes played by a player in each match.

        Args:
            player_id: Player ID
            competition_id: Competition ID
            season_id: Season ID

        Returns:
            List of match appearances with minutes
        """
        # Get all matches
        matches = self.get_matches(competition_id, season_id)

        appearances = []
        for match in matches:
            match_id = match.get('@attributes', {}).get('uId', match.get('uId', match.get('id')))
            if not match_id:
                continue

            try:
                stats = self.get_match_stats(match_id.replace('g', ''))

                # Find player in match data
                for team_data in stats.get('matchInfo', {}).get('contestant', []):
                    for player in team_data.get('player', []):
                        if str(player.get('playerId', '')) == str(player_id):
                            appearances.append({
                                'match_id': match_id,
                                'minutes': player.get('stats', {}).get('minsPlayed', 0),
                                'started': player.get('position', '') != 'Substitute',
                                'date': match.get('date', match.get('matchDate'))
                            })
            except Exception as e:
                logger.warning(f"Could not get stats for match {match_id}: {e}")

        return appearances

    def get_team_possession(self, match_id: str) -> Dict[str, float]:
        """
        Get team possession percentages for a match.

        Args:
            match_id: Match ID

        Returns:
            Dictionary with team_id -> possession percentage
        """
        stats = self.get_match_stats(match_id)

        possession = {}
        for team_data in stats.get('liveData', {}).get('lineUp', []):
            team_id = team_data.get('contestantId')
            team_stats = team_data.get('stat', [])

            for stat in team_stats:
                if stat.get('type') == 'possessionPercentage':
                    possession[team_id] = float(stat.get('value', 50))

        return possession

    def get_stats(self) -> Dict:
        """Get client usage statistics."""
        return {
            'requests': self.request_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(1, self.request_count + self.cache_hits)
        }

    def clear_cache(self) -> None:
        """Clear the API cache."""
        if self.cache:
            self.cache.clear()
