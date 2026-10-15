"""
Download Opta match data to a file for the app's "Or upload a match file" option.

Use this when the hosted app is rate limited by the Opta feed: run it on a computer
that can reach the feed, then upload the saved .json file in the app's sidebar.

Usage:
    python download_match.py <Scoresway match URL or match ID> [more ...] [-o OUTPUT_DIR]
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from opta_api import OptaAPI


def download_match(api: OptaAPI, match_id: str, output_dir: Path) -> Path:
    """Fetch a match's events and stats and save them as one JSON file"""
    events = api.get_match_events_raw(match_id)
    if not events.get('liveData', {}).get('event'):
        raise ValueError("the feed returned no events for this match")

    stats = api.get_match_stats(match_id)
    if 'errorCode' in stats:
        raise ValueError(f"match stats unavailable (error {stats['errorCode']})")

    info = events.get('matchInfo', {})
    label = f"{info.get('description', '')}_{info.get('localDate') or info.get('date', '')}"
    label = re.sub(r'[^A-Za-z0-9-]+', '_', label).strip('_')
    path = output_dir / (f"{label}_{match_id}.json" if label else f"{match_id}.json")

    payload = {
        'source': 'download_match.py',
        'match_id': match_id,
        'downloaded_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'matchevent': events,
        'matchstats': stats,
    }
    path.write_text(json.dumps(payload, separators=(',', ':')))
    return path


def main():
    parser = argparse.ArgumentParser(description="Download Opta match data for upload to the app")
    parser.add_argument('matches', nargs='+', help="Scoresway match URLs or match IDs")
    parser.add_argument('-o', '--output-dir', default='match_files',
                        help="Folder to save files in (default: match_files)")
    args = parser.parse_args()

    api = OptaAPI()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for value in args.matches:
        value = value.strip()
        match_id = api.parse_scoresway_url(value).get('match_id')
        if not match_id and re.fullmatch(r'[a-z0-9]+', value):
            match_id = value
        if not match_id:
            print(f"Skipped {value}: not a Scoresway match URL or match ID", file=sys.stderr)
            failures += 1
            continue

        try:
            path = download_match(api, match_id, output_dir)
            print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"Failed {match_id}: {e}", file=sys.stderr)
            failures += 1

    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
