"""
Bundle Opta feed responses into snapshots/ so the app works without reaching the feed.

The app serves bundled responses before calling the feed, so everything recorded here
loads even when the feed rate limits the host. Run this on a computer that can reach the
feed, then commit snapshots/. Re-run it to add newly played matches; matches that are
already bundled are skipped, while match lists and squads are always refreshed.

Usage:
    python build_snapshots.py --season <tournament ID>     # every played match in a season
    python build_snapshots.py <Scoresway URL> [...]        # a match, or a team/player's season
"""

import argparse
import sys
import time

from opta_api import OptaAPI, RateLimitError

# Only finished matches: an in-progress match would be bundled with partial data
FINISHED_STATUSES = {'Played', 'Awarded'}
PAUSE_SECONDS = 0.3     # between live requests, to stay under the feed's rate limit
RATE_LIMIT_WAIT = 60    # seconds to wait when the feed rate limits us anyway


def with_rate_limit_retries(func, *args, attempts: int = 5):
    """Call func, waiting and retrying if the feed rate limits us"""
    for attempt in range(attempts):
        try:
            return func(*args)
        except RateLimitError:
            if attempt == attempts - 1:
                raise
            print(f"  rate limited, waiting {RATE_LIMIT_WAIT}s...", flush=True)
            time.sleep(RATE_LIMIT_WAIT)


def record_match(api: OptaAPI, match_id: str) -> bool:
    """Bundle a match's events and stats; returns False if the feed has no events for it"""
    try:
        with_rate_limit_retries(api.get_match_events_raw, match_id)
    except ValueError:
        return False
    stats = with_rate_limit_retries(api.get_match_stats, match_id)
    if 'errorCode' in stats:
        print(f"  {match_id}: events bundled, but match stats unavailable "
              f"(error {stats['errorCode']}), so its reports need the live feed", flush=True)
    return True


def contestant_ids(match) -> set:
    contestants = match.get('matchInfo.contestant')
    return {c.get('id') for c in contestants} if isinstance(contestants, list) else set()


def record_season(api: OptaAPI, tournament_id: str, team_id: str = None):
    """Bundle a season's match list, squads, and played matches (optionally one team's)"""
    # Match lists and squads change during a season, so always refresh them
    api.read_snapshots = False
    matches = with_rate_limit_retries(api.get_matches_by_tournament, tournament_id)
    season_name = matches['matchInfo.tournamentCalendar.name'].iloc[0] if len(matches) else ''
    competition = matches['matchInfo.competition.name'].iloc[0] if len(matches) else ''

    all_team_ids = set().union(*(contestant_ids(m) for _, m in matches.iterrows()))
    team_ids = sorted(t for t in all_team_ids if t and team_id in (None, t))
    for tid in team_ids:
        with_rate_limit_retries(api.get_squad, tid, tournament_id)
    api.read_snapshots = True

    played = []
    for _, match in matches.iterrows():
        if match.get('liveData.matchDetails.matchStatus') not in FINISHED_STATUSES:
            continue
        if team_id and team_id not in contestant_ids(match):
            continue
        played.append(match['matchInfo.id'])

    scope = f"team {team_id}" if team_id else "all teams"
    print(f"{competition} {season_name} ({scope}): {len(played)} matches, {len(team_ids)} squads",
          flush=True)
    record_matches(api, played)


def record_matches(api: OptaAPI, match_ids: list):
    """Bundle each match, pausing between live requests"""
    skipped = []
    for i, match_id in enumerate(match_ids, 1):
        before = api.live_requests
        if not record_match(api, match_id):
            skipped.append(match_id)
        if api.live_requests > before:
            time.sleep(PAUSE_SECONDS)
        if i % 25 == 0 or i == len(match_ids):
            print(f"  {i}/{len(match_ids)} matches", flush=True)
    if skipped:
        print(f"  no event data yet for {len(skipped)}: {', '.join(skipped)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Bundle Opta feed data into snapshots/")
    parser.add_argument('urls', nargs='*', help="Scoresway match, team, or player URLs")
    parser.add_argument('--season', action='append', default=[], metavar='TOURNAMENT_ID',
                        help="Bundle every played match in this season (repeatable)")
    args = parser.parse_args()
    if not args.urls and not args.season:
        parser.error("give at least one Scoresway URL or --season")

    api = OptaAPI(record_snapshots=True)

    for tournament_id in args.season:
        record_season(api, tournament_id)

    for url in args.urls:
        ids = api.parse_scoresway_url(url)
        if 'team_id' in ids and 'tournament_id' in ids:
            record_season(api, ids['tournament_id'], ids['team_id'])
            if 'player_id' in ids:
                # The player page looks the player up in the team's squad, recorded above
                print(f"  player {ids['player_id']} covered by the team's season", flush=True)
        elif 'match_id' in ids:
            print(f"Match {ids['match_id']}", flush=True)
            record_matches(api, [ids['match_id']])
        else:
            print(f"Skipped {url}: not a Scoresway match, team, or player URL", file=sys.stderr)

    print(f"Done ({api.live_requests} feed requests)")


if __name__ == '__main__':
    main()
