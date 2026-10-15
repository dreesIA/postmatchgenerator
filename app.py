"""
Streamlit App for Football Match & Player Analysis
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import gzip
import json
import os
import sys
from pathlib import Path

APP_DIR = os.path.realpath(Path(__file__).parent)


def _source_signature(path):
    try:
        stat = os.stat(path)
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def _project_modules():
    """This app's own imported modules (not this script itself or installed packages)"""
    for name, module in list(sys.modules.items()):
        path = os.path.realpath(getattr(module, '__file__', None) or '/')
        if name not in ('__main__', __name__) and path.startswith(APP_DIR + os.sep) \
                and 'site-packages' not in path:
            yield name, module


# Streamlit Community Cloud re-runs this script after a git push but keeps the app's other
# modules loaded from before, so new app code would run against old modules until a reboot.
# When any of their files changed, unload them all so they're imported fresh below.
if any(getattr(m, '_source_signature', None) != _source_signature(m.__file__)
       for _, m in _project_modules()):
    for _name, _ in list(_project_modules()):
        del sys.modules[_name]

# Add scout report to path
sys.path.insert(0, str(Path(__file__).parent / 'opta_scout_report' / 'src'))

from opta_api import OptaAPI, EventProcessor, ConfigLoader, PassingNetworkBuilder, RateLimitError, SNAPSHOT_DIR
from player_viz import PlayerDataAggregator, PlayerShotMap, PlayerPassMap, PlayerHeatMap, PlayerProfile
from pitch_viz import ShotMap, PassingNetwork
from match_overview import MatchOverview
from phase_report import FullPhaseReport

# Scout report imports (conditional to avoid errors if not fully set up)
try:
    from opta_scout_report.src.report_generator import ReportGenerator
    from opta_scout_report.src.data_loader import DataLoader, PlayerData, PlayerInfo, MatchAppearance
    from opta_scout_report.src.normalize import EventNormalizer
    from opta_scout_report.src.metrics import MetricsEngine
    from opta_scout_report.src.cohort import CohortBuilder
    from opta_scout_report.src.report_sections import (
        SnapshotSection, RoleUsageSection, PassingSection, CarryingSection,
        ChanceCreationSection, DefendingSection, DisciplineSection,
        BenchmarkingSection, TrendsSection, RecommendationSection
    )
    from opta_scout_report.src.pdf import PDFBuilder
    SCOUT_REPORT_AVAILABLE = True
except ImportError as e:
    SCOUT_REPORT_AVAILABLE = False
    SCOUT_IMPORT_ERROR = str(e)

# Record what each project module was loaded from (see the unloading above)
for _, _module in _project_modules():
    if not hasattr(_module, '_source_signature'):
        _module._source_signature = _source_signature(_module.__file__)

# Page config
st.set_page_config(
    page_title="Football Analytics",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .stat-card {
        background-color: #16213e;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stat-value {
        font-size: 2em;
        font-weight: bold;
        color: #f39c12;
    }
    .stat-label {
        color: #888;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_config_loader():
    """Cache the config loader"""
    return ConfigLoader()


@st.cache_data(ttl=3600)
def fetch_match_events(_api, match_id):
    """Cache match event data for 1 hour"""
    return _api.get_match_events(match_id)


@st.cache_data(ttl=3600)
def fetch_match_stats(_api, match_id):
    """Cache match stats for 1 hour"""
    return _api.get_match_stats(match_id)


@st.cache_data(ttl=600)
def bundled_seasons():
    """Seasons with matches in snapshots/ as (name, matches bundled, last match date)"""
    bundled = {p.name.split('.')[0] for p in (SNAPSHOT_DIR / 'matchevent').glob('*.json.gz')}
    seasons = []
    for path in sorted((SNAPSHOT_DIR / 'match').glob('*.json.gz')):
        matches = json.loads(gzip.decompress(path.read_bytes())).get('match', [])
        dates = [m['matchInfo'].get('date', '')[:10] for m in matches if m['matchInfo']['id'] in bundled]
        if dates:
            info = matches[0]['matchInfo']
            seasons.append((f"{info['competition']['name']} {info['tournamentCalendar']['name']}",
                            len(dates), max(dates)))
    return seasons


def bundled_seasons_text():
    seasons = bundled_seasons()
    if not seasons:
        return "none found on this server"
    return "; ".join(f"{name}: {count} matches through {last}" for name, count, last in seasons)


@st.cache_data(ttl=3600)
def load_uploaded_match(_api, file_bytes):
    """Parse a match file from download_match.py into (match_id, (events_df, match_info, match_stats))"""
    not_match_file = ValueError("This isn't a match file from download_match.py")
    try:
        data = json.loads(file_bytes)
    except ValueError:
        raise not_match_file
    if not isinstance(data, dict) or not isinstance(data.get('matchevent'), dict) \
            or not isinstance(data.get('matchstats'), dict):
        raise not_match_file
    match_id = data.get('match_id') or data['matchevent'].get('matchInfo', {}).get('id', '')
    events_df, match_info = _api.parse_match_events(data['matchevent'], match_id)
    return match_id, (events_df, match_info, data['matchstats'])


@st.cache_data(ttl=3600)
def fetch_tournament_matches(_api, tournament_id):
    """Cache tournament matches for 1 hour"""
    return _api.get_matches_by_tournament(tournament_id)


@st.cache_data(ttl=3600)
def fetch_team_matches(_api, team_id, tournament_id):
    """Cache team matches for 1 hour"""
    return _api.get_team_matches(team_id, tournament_id)


@st.cache_data(ttl=3600)
def fetch_squad(_api, team_id, tournament_id):
    """Cache squad data for 1 hour"""
    return _api.get_squad(team_id, tournament_id)


def main():
    st.title("⚽ Football Analytics Dashboard")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # URL input - primary method
        st.subheader("Paste Scoresway URL")
        url_input = st.text_input(
            "Scoresway URL",
            placeholder="https://www.scoresway.com/...",
            help="Paste a match or player URL from Scoresway"
        )

        # Parse URL if provided
        parsed_ids = {}
        if url_input:
            api_temp = OptaAPI()
            parsed_ids = api_temp.parse_scoresway_url(url_input)
            if parsed_ids:
                st.success("URL parsed successfully!")
                with st.expander("Extracted IDs"):
                    for key, value in parsed_ids.items():
                        st.code(f"{key}: {value}")

        # Match file upload - for when this server can't reach the Opta feed
        uploaded_match = st.file_uploader(
            "Or upload a match file",
            type="json",
            help="Create one on a computer that can reach the feed with: "
                 "python download_match.py <match URL>"
        )

        st.caption(f"Bundled data (loads without the live feed): {bundled_seasons_text()}")

        st.divider()

        # League selection
        config = get_config_loader()
        leagues = config.list_leagues()
        league_names = {key: config.get_league(key).name for key in leagues}

        selected_league_key = st.selectbox(
            "Select League",
            options=leagues,
            format_func=lambda x: league_names[x]
        )

        league_config = config.get_league(selected_league_key)
        st.caption(f"Season: {league_config.season}")

        # Initialize API
        api = OptaAPI.for_league(selected_league_key)

        st.divider()

        # Manual ID inputs (as fallback)
        with st.expander("Manual ID Entry"):
            manual_match_id = st.text_input("Match ID", placeholder="e.g., 19a81k01783lyow7nexg2c844")
            manual_team_id = st.text_input("Team ID", placeholder="e.g., e3k8yw2rbmphyfu5htn4yi5jk")
            manual_player_id = st.text_input("Player ID", placeholder="e.g., caoq1dpq6lz1btrd75s9xuz56")

        # Determine what IDs we have
        match_id = parsed_ids.get('match_id') or manual_match_id
        team_id = parsed_ids.get('team_id') or manual_team_id
        player_id = parsed_ids.get('player_id') or manual_player_id
        tournament_id = parsed_ids.get('tournament_id') or league_config.tournament_id

    # Main content - an uploaded match file takes priority, then available IDs
    if uploaded_match is not None:
        try:
            uploaded_match_id, preloaded = load_uploaded_match(api, uploaded_match.getvalue())
        except ValueError as e:
            st.error(f"Couldn't read the match file: {e}")
            return
        show_match_page(api, uploaded_match_id, preloaded=preloaded)
    elif player_id and team_id:
        # Player season analysis
        show_player_season_analysis(api, player_id, team_id, tournament_id)
    elif match_id:
        # Match analysis with optional player selection
        show_match_page(api, match_id)
    elif team_id:
        # Team overview
        show_team_overview(api, team_id, tournament_id)
    else:
        # Landing page
        show_landing_page()


def show_landing_page():
    """Show instructions when no IDs are provided"""
    st.info("👈 Paste a Scoresway URL in the sidebar to get started")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Match Analysis")
        st.markdown("""
        Paste a match URL to see:
        - Match statistics and xG
        - Shot maps for both teams
        - Passing networks
        - Player-level analysis
        """)

        with st.expander("Example Match URL"):
            st.code("https://www.scoresway.com/en_GB/soccer/usl-championship-2025/8hdq5pmnb4m3p9t24ytwgjitw/match/view/19a81k01783lyow7nexg2c844/match-summary")

    with col2:
        st.markdown("### Player Analysis")
        st.markdown("""
        Paste a player URL to see:
        - Season statistics across all matches
        - Shot map with xG
        - Pass map with key passes
        - Touch heat map
        """)

        with st.expander("Example Player URL"):
            st.code("https://www.scoresway.com/en_GB/soccer/usl-championship-2025/8hdq5pmnb4m3p9t24ytwgjitw/teams/view/e3k8yw2rbmphyfu5htn4yi5jk/player/view/caoq1dpq6lz1btrd75s9xuz56")


def show_team_overview(api, team_id, tournament_id):
    """Show team overview with squad and matches"""
    st.subheader("Team Overview")

    # Get squad
    try:
        squad_df = fetch_squad(api, team_id, tournament_id)
        if len(squad_df) > 0:
            team_name = squad_df.iloc[0].get('contestantName', 'Unknown Team') if 'contestantName' in squad_df.columns else 'Team'
            st.markdown(f"### {team_name}")

            # Squad table
            if 'firstName' in squad_df.columns:
                display_cols = ['firstName', 'lastName', 'position', 'shirtNumber', 'nationality']
                display_cols = [c for c in display_cols if c in squad_df.columns]
                st.dataframe(squad_df[display_cols], use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error loading team data: {e}")


def show_match_page(api, match_id, preloaded=None):
    """Show match analysis page

    preloaded: (events_df, match_info, match_stats) from an uploaded match file;
    when given, nothing is fetched from the Opta feed
    """
    match_stats = None
    try:
        if preloaded is not None:
            events_df, match_info, match_stats = preloaded
        else:
            with st.spinner("Fetching match data..."):
                events_df, match_info = fetch_match_events(api, match_id)

        if len(events_df) == 0:
            status = getattr(match_info, 'status', '')
            coverage = getattr(match_info, 'coverage_level', '')
            if status and status != 'Played':
                st.warning(f"This match has no event data yet (status: {status}). "
                           "Event data is available once the match has been played.")
            else:
                teams = [t for t in (match_info.home_team_name, match_info.away_team_name) if t]
                desc = f" ({' vs '.join(teams)})" if teams else ''
                st.error(f"No event data found for this match{desc}. "
                         f"Coverage level: {coverage or 'unknown'}. "
                         "This competition may not include event-level data.")
            return

    except RateLimitError:
        if (SNAPSHOT_DIR / 'matchevent' / f'{match_id}.json.gz').exists():
            st.error("This match is bundled with the app, but the server didn't use the bundled "
                     "data. Reboot the app so it loads the current code.")
        else:
            st.error("This match isn't in the app's bundled data, and the Opta feed is rate "
                     "limiting this server, so it can't be loaded live right now. You can upload "
                     f"a match file from download_match.py in the sidebar. Bundled: {bundled_seasons_text()}.")
        return
    except Exception as e:
        st.error(f"Error fetching match data: {str(e)}")
        return

    # Display match header
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(f"### {match_info.home_team_name}")
    with col2:
        st.markdown(f"### {match_info.home_score} - {match_info.away_score}")
    with col3:
        st.markdown(f"### {match_info.away_team_name}")

    st.caption(f"{match_info.competition} | {match_info.date} | {match_info.venue}")
    st.divider()

    # Analysis type tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Match Analysis", "Player Analysis", "Full Report", "Phase Report"])

    with tab1:
        show_match_analysis(events_df, match_info)

    with tab2:
        show_player_analysis(api, events_df, match_info, match_id)

    with tab3:
        show_match_overview(api, events_df, match_info, match_id, match_stats)

    with tab4:
        show_phase_report(api, events_df, match_info, match_id, match_stats)


def show_player_season_analysis(api, player_id, team_id, tournament_id):
    """Show player analysis across the entire season"""

    # Get player info
    try:
        player_info = api.get_player_info(player_id, team_id, tournament_id)
        with st.spinner("Loading season matches..."):
            team_matches = fetch_team_matches(api, team_id, tournament_id)
    except RateLimitError:
        st.error("This team's season isn't in the app's bundled data, and the Opta feed is rate "
                 f"limiting this server, so it can't be loaded live right now. Bundled: {bundled_seasons_text()}.")
        return
    except Exception as e:
        st.error(f"Error loading player data: {e}")
        return

    if player_info:
        player_name = f"{player_info.get('firstName', '')} {player_info.get('lastName', '')}".strip()
        st.title(f"📊 {player_name}")
        st.caption(f"#{player_info.get('shirtNumber', '')} | {player_info.get('position', '')} | {player_info.get('nationality', '')}")
    else:
        st.title("Player Analysis")

    st.divider()

    # Only finished matches have complete event data
    if 'status' in team_matches.columns:
        team_matches = team_matches[team_matches['status'].isin(['Played', 'Awarded'])]

    if len(team_matches) == 0:
        st.warning("No matches found for this team")
        return

    st.info(f"Found {len(team_matches)} matches for the season")

    # Match selection
    col1, col2 = st.columns([1, 3])
    with col1:
        num_matches = st.slider(
            "Matches to analyze",
            min_value=1,
            max_value=len(team_matches),
            value=min(5, len(team_matches)),
            help="More matches = more complete stats but slower loading"
        )

    # Load player data
    with st.spinner(f"Analyzing {num_matches} matches..."):
        aggregator = PlayerDataAggregator(api)

        progress_bar = st.progress(0)
        match_ids = team_matches['match_id'].tolist()[:num_matches]

        for i, match_id in enumerate(match_ids):
            try:
                aggregator.load_match(match_id, player_id, events=fetch_match_events(api, match_id))
            except Exception as e:
                st.warning(f"Could not load match {match_id}: {e}")
            progress_bar.progress((i + 1) / num_matches)

        progress_bar.empty()

    if aggregator.match_count == 0:
        st.warning("No data found for this player in the selected matches")
        return

    # Summary stats
    stats = aggregator.get_summary_stats()

    st.subheader(f"Season Statistics ({aggregator.match_count} matches)")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Goals", stats['goals'])
    with col2:
        st.metric("xG", f"{stats['total_xg']:.2f}")
    with col3:
        st.metric("Assists", stats['assists'])
    with col4:
        st.metric("Key Passes", stats['key_passes'])
    with col5:
        st.metric("Pass %", f"{stats['pass_completion']:.0f}%")
    with col6:
        st.metric("Shots", stats['total_shots'])

    st.divider()

    # Visualizations
    if SCOUT_REPORT_AVAILABLE:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Shot Map", "Pass Map", "Heat Map", "Full Profile", "Scout Report"])
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["Shot Map", "Pass Map", "Heat Map", "Full Profile"])

    with tab1:
        if len(aggregator.shots) > 0:
            shot_map = PlayerShotMap(aggregator.shots, player_name if player_info else "Player")
            fig, ax = shot_map.plot(title=f"Shot Map - {aggregator.match_count} matches")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No shots recorded")

    with tab2:
        if len(aggregator.passes) > 0:
            pass_map = PlayerPassMap(aggregator.passes, player_name if player_info else "Player")
            fig, ax = pass_map.plot(title=f"Pass Map - {aggregator.match_count} matches")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No passes recorded")

    with tab3:
        if len(aggregator.touches) > 0:
            heat_map = PlayerHeatMap(aggregator.touches, player_name if player_info else "Player")
            fig, ax = heat_map.plot(title=f"Touch Heat Map - {aggregator.match_count} matches")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No touch data recorded")

    with tab4:
        profile = PlayerProfile(aggregator)
        fig = profile.generate_profile()
        st.pyplot(fig)
        plt.close(fig)

    if SCOUT_REPORT_AVAILABLE:
        with tab5:
            show_scout_report(api, aggregator, player_info, tournament_id, stats)

    # Detailed stats
    st.divider()
    st.subheader("Detailed Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Attacking**")
        st.write(f"- Goals: {stats['goals']}")
        st.write(f"- xG: {stats['total_xg']:.2f}")
        xg_diff = stats['goals'] - stats['total_xg']
        st.write(f"- G-xG: {'+' if xg_diff >= 0 else ''}{xg_diff:.2f}")
        st.write(f"- Shots: {stats['total_shots']}")
        st.write(f"- Assists: {stats['assists']}")
        st.write(f"- Key Passes: {stats['key_passes']}")

    with col2:
        st.markdown("**Passing**")
        st.write(f"- Total: {stats['total_passes']}")
        st.write(f"- Completion: {stats['pass_completion']:.1f}%")

        st.markdown("**Dribbling**")
        take_on_pct = (stats['take_ons_successful'] / stats['take_ons_attempted'] * 100) if stats['take_ons_attempted'] > 0 else 0
        st.write(f"- Take-ons: {stats['take_ons_successful']}/{stats['take_ons_attempted']} ({take_on_pct:.0f}%)")

    with col3:
        st.markdown("**Defending**")
        st.write(f"- Tackles: {stats['tackles']}")
        st.write(f"- Interceptions: {stats['interceptions']}")
        st.write(f"- Recoveries: {stats['ball_recoveries']}")

        st.markdown("**Aerial**")
        aerial_pct = (stats['aerials_won'] / stats['aerials_total'] * 100) if stats['aerials_total'] > 0 else 0
        st.write(f"- Won: {stats['aerials_won']}/{stats['aerials_total']} ({aerial_pct:.0f}%)")


def show_match_analysis(events_df, match_info):
    """Display match-level analysis"""

    processor = EventProcessor(events_df, match_info)
    passes = processor.get_passes()
    shots = processor.get_shots()
    recoveries = processor.get_ball_recoveries()
    defensive = processor.get_defensive_actions()

    # Summary stats
    home_shots = shots[shots['team_name'] == match_info.home_team_name]
    away_shots = shots[shots['team_name'] == match_info.away_team_name]
    home_xg = home_shots['xG'].sum() if len(home_shots) > 0 else 0
    away_xg = away_shots['xG'].sum() if len(away_shots) > 0 else 0

    home_passes = passes[passes['team_name'] == match_info.home_team_name]
    away_passes = passes[passes['team_name'] == match_info.away_team_name]

    # Stats row
    st.subheader("Match Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("xG", f"{home_xg:.2f} - {away_xg:.2f}")
    with col2:
        st.metric("Shots", f"{len(home_shots)} - {len(away_shots)}")
    with col3:
        home_pct = home_passes['is_successful'].mean() * 100 if len(home_passes) > 0 else 0
        away_pct = away_passes['is_successful'].mean() * 100 if len(away_passes) > 0 else 0
        st.metric("Pass Accuracy", f"{home_pct:.0f}% - {away_pct:.0f}%")
    with col4:
        st.metric("Total Events", len(events_df))

    st.divider()

    # Visualizations
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Shot Maps", "Passing Networks", "Defensive Actions"])

    with viz_tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{match_info.home_team_name}**")
            if len(home_shots) > 0:
                shot_map = ShotMap(home_shots, match_info.home_team_name)
                fig, ax = shot_map.plot()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No shots")

        with col2:
            st.markdown(f"**{match_info.away_team_name}**")
            if len(away_shots) > 0:
                shot_map = ShotMap(away_shots, match_info.away_team_name)
                fig, ax = shot_map.plot()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No shots")

    with viz_tab2:
        network_builder = PassingNetworkBuilder(passes, match_info)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{match_info.home_team_name}**")
            home_nodes, home_edges = network_builder.build_network(team_id=match_info.home_team_id)
            if len(home_nodes) > 0:
                pass_net = PassingNetwork(home_nodes, home_edges, match_info.home_team_name)
                fig, ax = pass_net.plot()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Not enough data for passing network")

        with col2:
            st.markdown(f"**{match_info.away_team_name}**")
            away_nodes, away_edges = network_builder.build_network(team_id=match_info.away_team_id)
            if len(away_nodes) > 0:
                pass_net = PassingNetwork(away_nodes, away_edges, match_info.away_team_name)
                fig, ax = pass_net.plot()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Not enough data for passing network")

    with viz_tab3:
        if len(defensive) > 0:
            from pitch_viz import DefensiveActionsMap
            def_map = DefensiveActionsMap(defensive, match_info)
            fig, ax = def_map.plot()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No defensive action data available")

    # Shot details table
    st.divider()
    st.subheader("Shot Details")

    if len(shots) > 0:
        shot_display = shots[['timeMin', 'team_name', 'playerName', 'xG', 'is_goal', 'is_on_target']].copy()
        shot_display.columns = ['Minute', 'Team', 'Player', 'xG', 'Goal', 'On Target']
        shot_display = shot_display.sort_values('Minute')
        st.dataframe(shot_display, use_container_width=True, hide_index=True)


def show_match_overview(api, events_df, match_info, match_id, match_stats=None):
    """Display the full match overview one-pager"""
    import tempfile
    import os

    st.subheader("Full Match Report")
    st.caption("Comprehensive one-page match overview with all visualizations")

    # Generate button
    if st.button("Generate Full Report", type="primary"):
        with st.spinner("Generating match overview... This may take a moment."):
            try:
                # Fetch match stats for accurate possession data (already loaded for uploaded files)
                if match_stats is None:
                    match_stats = fetch_match_stats(api, match_id)

                # Create the match overview with real stats
                overview = MatchOverview(events_df, match_info, match_stats)

                # Save to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    tmp_path = tmp_file.name

                fig = overview.generate(save_path=tmp_path)

                # Display the figure
                st.pyplot(fig)
                plt.close(fig)

                # Create download button
                with open(tmp_path, 'rb') as f:
                    png_data = f.read()

                # Clean filename
                filename = f"{match_info.home_team_name}_vs_{match_info.away_team_name}_{match_info.date}.png"
                filename = filename.replace(' ', '_').replace('/', '-')

                st.download_button(
                    label="Download PNG",
                    data=png_data,
                    file_name=filename,
                    mime="image/png"
                )

                # Clean up temp file
                os.unlink(tmp_path)

                st.success("Report generated successfully!")

            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("Click 'Generate Full Report' to create the comprehensive match overview")

        # Preview what will be included
        with st.expander("What's included in the report?"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"**{match_info.home_team_name}**")
                st.markdown("""
                - Passing Network
                - Defensive Actions
                - Final Third Entries
                - Zone 14 & Half-Space Entries
                - Chance Origins
                - Pass Flow Map
                """)

            with col2:
                st.markdown("**Match Stats**")
                st.markdown("""
                - xG Comparison
                - Possession & Field Tilt
                - PPDA Analysis
                - Combined Shot Map
                - Game Control Timeline
                - Key Statistics Table
                """)

            with col3:
                st.markdown(f"**{match_info.away_team_name}**")
                st.markdown("""
                - Passing Network
                - Defensive Actions
                - Final Third Entries
                - Zone 14 & Half-Space Entries
                - Chance Origins
                - Pass Flow Map
                """)


def show_phase_report(api, events_df, match_info, match_id, match_stats=None):
    """Display the phase-by-phase match report (7 pages)"""
    import re
    from io import BytesIO

    st.subheader("Phase-by-Phase Analysis")
    st.caption("Detailed analysis of each phase of play: In Possession, Out of Possession, Transitions")

    # Phase selection
    phase_options = {
        "1. Build-Up Phase": "buildup",
        "2. Middle Progression": "middle",
        "3. Final Third": "final_third",
        "4. High Press": "high_press",
        "5. Mid/Low Block": "mid_low_block",
        "6. Positive Transitions": "positive_transition",
        "7. Negative Transitions": "negative_transition",
        "All Phases (Full Report)": "all"
    }

    selected_phase = st.selectbox(
        "Select Phase to Analyze",
        list(phase_options.keys()),
        index=7  # Default to "All Phases"
    )

    phase_key = phase_options[selected_phase]

    def _safe_filename(value: str) -> str:
        value = re.sub(r"\s+", "_", str(value or "").strip())
        value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
        return value.strip("_.-") or "phase_report"

    base_name = _safe_filename(
        f"{match_info.home_team_name}_vs_{match_info.away_team_name}_{match_info.date}_{match_id}"
    )
    download_state_key = f"phase_report_download::{match_id}::{phase_key}"

    # Show last-generated download (persists across reruns)
    if download_state_key in st.session_state:
        dl = st.session_state[download_state_key]
        st.download_button(
            label=dl["label"],
            data=dl["data"],
            file_name=dl["file_name"],
            mime=dl["mime"],
            type="secondary",
            use_container_width=False,
            key=f"download_button::{download_state_key}",
        )

    # Preview what's included
    with st.expander("What's included in each phase?"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**In Possession**")
            st.markdown("""
            - **Build-Up**: Passes from own third, progressive carries, corridor preference
            - **Middle Progression**: Final third entries, line-breaking passes, take-ons
            - **Final Third**: Crosses, shots, xG analysis, chance creation
            """)

            st.markdown("**Transitions**")
            st.markdown("""
            - **Positive**: Ball recoveries, counter-attack progressions, direct play
            - **Negative**: Ball losses, counter-press actions, tactical fouls
            """)

        with col2:
            st.markdown("**Out of Possession**")
            st.markdown("""
            - **High Press**: Pressing triggers, high recoveries, PPDA
            - **Mid/Low Block**: Defensive shape, clearances, blocks
            """)

    if st.button("Generate Phase Report", type="primary"):
        with st.spinner("Generating phase analysis... This may take a moment."):
            try:
                # Fetch match stats (already loaded for uploaded files)
                if match_stats is None:
                    match_stats = fetch_match_stats(api, match_id)

                # Create the phase report
                report = FullPhaseReport(events_df, match_info, match_stats)

                if phase_key == "all":
                    # Generate all pages
                    figures = report.generate_all_pages()

                    # Build a single downloadable PDF from the figures
                    from matplotlib.backends.backend_pdf import PdfPages
                    pdf_buffer = BytesIO()
                    with PdfPages(pdf_buffer) as pdf:
                        for fig in figures:
                            pdf.savefig(fig, bbox_inches='tight')
                    pdf_bytes = pdf_buffer.getvalue()

                    for i, fig in enumerate(figures):
                        st.pyplot(fig)
                        plt.close(fig)

                    st.session_state[download_state_key] = {
                        "label": "Download Phase Report (PDF)",
                        "data": pdf_bytes,
                        "file_name": f"{base_name}_phase_report.pdf",
                        "mime": "application/pdf",
                    }
                    st.download_button(
                        label=st.session_state[download_state_key]["label"],
                        data=st.session_state[download_state_key]["data"],
                        file_name=st.session_state[download_state_key]["file_name"],
                        mime=st.session_state[download_state_key]["mime"],
                        type="primary",
                        key=f"download_button_now::{download_state_key}",
                    )
                    st.success("Full phase report generated!")

                else:
                    # Generate specific phase
                    from phase_report import (
                        BuildUpPhasePage, MiddleProgressionPhasePage, FinalThirdPhasePage,
                        HighPressPhasePage, MidLowBlockPhasePage,
                        PositiveTransitionPage, NegativeTransitionPage
                    )

                    common_args = {
                        'events_df': events_df,
                        'match_info': match_info,
                        'passes': report.passes,
                        'shots': report.shots,
                        'defensive': report.defensive,
                        'recoveries': report.recoveries,
                        'take_ons': report.take_ons,
                        'match_stats': match_stats
                    }

                    phase_map = {
                        "buildup": BuildUpPhasePage,
                        "middle": MiddleProgressionPhasePage,
                        "final_third": FinalThirdPhasePage,
                        "high_press": HighPressPhasePage,
                        "mid_low_block": MidLowBlockPhasePage,
                        "positive_transition": PositiveTransitionPage,
                        "negative_transition": NegativeTransitionPage,
                    }

                    page_class = phase_map[phase_key]
                    page = page_class(**common_args)
                    fig = page.generate()

                    # Build a downloadable PNG for the phase
                    png_buffer = BytesIO()
                    fig.savefig(png_buffer, format="png", dpi=150, bbox_inches="tight")
                    png_bytes = png_buffer.getvalue()

                    st.pyplot(fig)
                    plt.close(fig)

                    st.session_state[download_state_key] = {
                        "label": f"Download {selected_phase} (PNG)",
                        "data": png_bytes,
                        "file_name": f"{base_name}_{phase_key}.png",
                        "mime": "image/png",
                    }
                    st.download_button(
                        label=st.session_state[download_state_key]["label"],
                        data=st.session_state[download_state_key]["data"],
                        file_name=st.session_state[download_state_key]["file_name"],
                        mime=st.session_state[download_state_key]["mime"],
                        type="primary",
                        key=f"download_button_now::{download_state_key}",
                    )
                    st.success(f"{selected_phase} generated!")

            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("Select a phase and click 'Generate Phase Report' to create the analysis")


def show_player_analysis(api, events_df, match_info, match_id):
    """Display player-level analysis for a single match"""

    # Get unique players from the match
    players_home = events_df[events_df['contestantId'] == match_info.home_team_id][['playerId', 'playerName']].drop_duplicates()
    players_away = events_df[events_df['contestantId'] == match_info.away_team_id][['playerId', 'playerName']].drop_duplicates()

    # Player selection
    col1, col2 = st.columns(2)

    with col1:
        team_choice = st.radio(
            "Select Team",
            [match_info.home_team_name, match_info.away_team_name],
            horizontal=True,
            key="player_team_select"
        )

    players_df = players_home if team_choice == match_info.home_team_name else players_away
    players_df = players_df.dropna()

    if len(players_df) == 0:
        st.warning("No player data available for this team")
        return

    # Create player options
    player_options = {row['playerId']: row['playerName'] for _, row in players_df.iterrows()}

    with col2:
        selected_player_id = st.selectbox(
            "Select Player",
            options=list(player_options.keys()),
            format_func=lambda x: player_options.get(x, "Unknown"),
            key="player_select"
        )

    if not selected_player_id:
        return

    # Load player data
    with st.spinner("Loading player data..."):
        aggregator = PlayerDataAggregator(api)
        aggregator.load_match(match_id, selected_player_id, events=(events_df, match_info))

    if aggregator.player_info is None:
        st.warning("No data found for this player in this match")
        return

    # Player header
    st.divider()
    st.subheader(f"📊 {aggregator.player_info.player_name}")
    st.caption(f"{aggregator.player_info.team_name}")

    # Summary stats
    stats = aggregator.get_summary_stats()

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Passes", f"{stats['total_passes']}")
    with col2:
        st.metric("Pass %", f"{stats['pass_completion']:.0f}%")
    with col3:
        st.metric("Shots", stats['total_shots'])
    with col4:
        st.metric("Goals", stats['goals'])
    with col5:
        st.metric("xG", f"{stats['total_xg']:.2f}")
    with col6:
        st.metric("Key Passes", stats['key_passes'])

    st.divider()

    # Player visualizations
    player_tab1, player_tab2, player_tab3, player_tab4 = st.tabs(["Shot Map", "Pass Map", "Heat Map", "Full Profile"])

    with player_tab1:
        if len(aggregator.shots) > 0:
            shot_map = PlayerShotMap(aggregator.shots, aggregator.player_info.player_name)
            fig, ax = shot_map.plot()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No shots recorded for this player")

    with player_tab2:
        if len(aggregator.passes) > 0:
            pass_map = PlayerPassMap(aggregator.passes, aggregator.player_info.player_name)
            fig, ax = pass_map.plot()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No passes recorded for this player")

    with player_tab3:
        if len(aggregator.touches) > 0:
            heat_map = PlayerHeatMap(aggregator.touches, aggregator.player_info.player_name)
            fig, ax = heat_map.plot()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No touch data recorded for this player")

    with player_tab4:
        profile = PlayerProfile(aggregator)
        fig = profile.generate_profile()
        st.pyplot(fig)
        plt.close(fig)


def show_scout_report(api, aggregator, player_info, tournament_id, stats):
    """Display the scout report for a player"""
    import yaml
    import numpy as np
    from dataclasses import dataclass
    import tempfile
    import os

    st.subheader("Individual Scout Report")
    st.caption("Comprehensive scouting analysis with percentile rankings and recommendations")

    # Load config
    config_path = Path(__file__).parent / 'opta_scout_report' / 'config' / 'metrics.yml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        st.error(f"Could not load metrics config: {e}")
        return

    # Build player data from aggregator
    player_name = f"{player_info.get('firstName', '')} {player_info.get('lastName', '')}".strip() if player_info else "Unknown"

    # Determine position group
    position = player_info.get('position', 'Midfielder') if player_info else 'Midfielder'
    position_group = determine_position_group(position)

    # Create mock PlayerInfo and PlayerData for the scout report
    @dataclass
    class MockPlayerInfo:
        name: str
        player_id: str
        position: str
        position_group: str
        age: int
        current_team: str
        nationality: str
        height: str = None
        weight: str = None
        preferred_foot: str = None

    @dataclass
    class MockAppearance:
        match_id: str
        minutes_played: int
        is_starter: bool = True

    @dataclass
    class MockPlayerData:
        info: MockPlayerInfo
        events: pd.DataFrame
        appearances: list

    # Build player info
    mock_info = MockPlayerInfo(
        name=player_name,
        player_id=player_info.get('id', 'unknown') if player_info else 'unknown',
        position=position,
        position_group=position_group,
        age=player_info.get('age', 25) if player_info else 25,
        current_team=aggregator.player_info.team_name if aggregator.player_info else 'Unknown',
        nationality=player_info.get('nationality', '') if player_info else '',
    )

    # Build events DataFrame from aggregator data
    all_events = []

    # Add passes
    if len(aggregator.passes) > 0:
        passes_df = aggregator.passes.copy()
        passes_df['event_type'] = 'pass'
        passes_df['type_id'] = 1
        all_events.append(passes_df)

    # Add shots
    if len(aggregator.shots) > 0:
        shots_df = aggregator.shots.copy()
        shots_df['event_type'] = 'shot'
        shots_df['type_id'] = 13
        all_events.append(shots_df)

    # Add touches for other events
    if len(aggregator.touches) > 0:
        touches_df = aggregator.touches.copy()
        if 'event_type' not in touches_df.columns:
            touches_df['event_type'] = 'touch'
        all_events.append(touches_df)

    if all_events:
        events_df = pd.concat(all_events, ignore_index=True)
    else:
        events_df = pd.DataFrame()

    # Build appearances
    appearances = [
        MockAppearance(match_id=f'm{i}', minutes_played=90)
        for i in range(aggregator.match_count)
    ]

    # Create player data
    player_data = MockPlayerData(
        info=mock_info,
        events=events_df,
        appearances=appearances
    )

    # Display position group selector
    col1, col2 = st.columns([1, 3])
    with col1:
        position_override = st.selectbox(
            "Position Group",
            options=['GK', 'CB', 'FB', 'DM', 'CM', 'AM', 'W', 'ST'],
            index=['GK', 'CB', 'FB', 'DM', 'CM', 'AM', 'W', 'ST'].index(position_group) if position_group in ['GK', 'CB', 'FB', 'DM', 'CM', 'AM', 'W', 'ST'] else 4,
            help="Adjust position group for appropriate benchmarking"
        )
        player_data.info.position_group = position_override

    # Calculate metrics from aggregator stats
    @dataclass
    class MockMetricValue:
        value: float

    @dataclass
    class MockMetricsResult:
        metrics: dict
        total_minutes: float
        match_by_match: list = None

    # Build metrics from aggregator stats
    total_minutes = aggregator.match_count * 90  # Estimate

    metrics = {
        'passes_attempted': MockMetricValue(stats['total_passes'] / aggregator.match_count * 90 / 90 if aggregator.match_count > 0 else 0),
        'passes_completed': MockMetricValue(stats['total_passes'] * stats['pass_completion'] / 100 / aggregator.match_count * 90 / 90 if aggregator.match_count > 0 else 0),
        'pass_accuracy': MockMetricValue(stats['pass_completion']),
        'shots': MockMetricValue(stats['total_shots'] / aggregator.match_count * 90 / 90 if aggregator.match_count > 0 else 0),
        'goals': MockMetricValue(stats['goals'] / aggregator.match_count * 90 / 90 if aggregator.match_count > 0 else 0),
        'xG': MockMetricValue(stats['total_xg'] / aggregator.match_count * 90 / 90 if aggregator.match_count > 0 else 0),
        'assists': MockMetricValue(stats['assists'] / aggregator.match_count * 90 / 90 if aggregator.match_count > 0 else 0),
        'key_passes': MockMetricValue(stats['key_passes'] / aggregator.match_count * 90 / 90 if aggregator.match_count > 0 else 0),
        'tackles_won': MockMetricValue(stats['tackles'] / aggregator.match_count * 90 / 90 if aggregator.match_count > 0 else 0),
        'interceptions': MockMetricValue(stats['interceptions'] / aggregator.match_count * 90 / 90 if aggregator.match_count > 0 else 0),
        'ball_recoveries': MockMetricValue(stats['ball_recoveries'] / aggregator.match_count * 90 / 90 if aggregator.match_count > 0 else 0),
        'aerial_duel_win_rate': MockMetricValue((stats['aerials_won'] / stats['aerials_total'] * 100) if stats['aerials_total'] > 0 else 50),
        'dribbles_attempted': MockMetricValue(stats['take_ons_attempted'] / aggregator.match_count * 90 / 90 if aggregator.match_count > 0 else 0),
        'dribble_success_rate': MockMetricValue((stats['take_ons_successful'] / stats['take_ons_attempted'] * 100) if stats['take_ons_attempted'] > 0 else 50),
        'progressive_passes': MockMetricValue(stats.get('progressive_passes', 0) / aggregator.match_count * 90 / 90 if aggregator.match_count > 0 else 0),
        'touches': MockMetricValue(len(aggregator.touches) / aggregator.match_count * 90 / 90 if aggregator.match_count > 0 else 0),
    }

    metrics_result = MockMetricsResult(
        metrics=metrics,
        total_minutes=total_minutes
    )

    # Build cohort (using position-based benchmarks from config)
    cohort_builder = CohortBuilder(config)

    # Create mock cohort data using typical values for the position
    # In a real implementation, this would come from a database of player stats
    cohort_size = 100
    np.random.seed(42)

    cohort_data = pd.DataFrame({
        'passes_attempted': np.random.normal(45, 15, cohort_size),
        'passes_completed': np.random.normal(38, 12, cohort_size),
        'pass_accuracy': np.random.normal(82, 8, cohort_size),
        'shots': np.random.exponential(2, cohort_size),
        'goals': np.random.exponential(0.3, cohort_size),
        'xG': np.random.exponential(0.35, cohort_size),
        'assists': np.random.exponential(0.15, cohort_size),
        'key_passes': np.random.exponential(1.5, cohort_size),
        'tackles_won': np.random.normal(2.5, 1.5, cohort_size),
        'interceptions': np.random.normal(1.5, 1, cohort_size),
        'ball_recoveries': np.random.normal(5, 2, cohort_size),
        'aerial_duel_win_rate': np.random.normal(50, 15, cohort_size),
        'dribbles_attempted': np.random.exponential(2, cohort_size),
        'dribble_success_rate': np.random.normal(50, 15, cohort_size),
        'progressive_passes': np.random.exponential(4, cohort_size),
        'touches': np.random.normal(50, 15, cohort_size),
    })

    cohort_summary = cohort_builder.build_cohort(
        player_metrics=metrics_result,
        cohort_data=cohort_data,
        position_group=position_override
    )

    # Display sections
    st.divider()

    # Section selection
    section_options = {
        'Executive Summary': 'snapshot',
        'Passing Analysis': 'passing',
        'Carrying & Dribbling': 'carrying',
        'Chance Creation': 'chance_creation',
        'Defending': 'defending',
        'Benchmarking': 'benchmarking',
        'Recommendation': 'recommendation',
    }

    selected_sections = st.multiselect(
        "Select Report Sections",
        options=list(section_options.keys()),
        default=['Executive Summary', 'Benchmarking', 'Recommendation']
    )

    if st.button("Generate Scout Report Sections", type="primary"):
        with st.spinner("Generating scout report..."):
            section_map = {
                'snapshot': SnapshotSection,
                'passing': PassingSection,
                'carrying': CarryingSection,
                'chance_creation': ChanceCreationSection,
                'defending': DefendingSection,
                'benchmarking': BenchmarkingSection,
                'recommendation': RecommendationSection,
            }

            for section_name in selected_sections:
                section_key = section_options[section_name]
                section_class = section_map.get(section_key)

                if section_class:
                    try:
                        section = section_class(
                            player_data=player_data,
                            metrics_result=metrics_result,
                            cohort_summary=cohort_summary,
                            config=config
                        )
                        content = section.generate()

                        st.markdown(f"### {content.title}")

                        # Display figures
                        for fig, caption in zip(content.figures, content.figure_captions):
                            st.pyplot(fig)
                            st.caption(caption)
                            plt.close(fig)

                        # Display takeaways
                        if content.takeaways:
                            for takeaway in content.takeaways:
                                if takeaway.is_positive:
                                    st.success(f"✓ {takeaway.text}")
                                else:
                                    st.warning(f"⚠ {takeaway.text}")

                        st.divider()

                    except Exception as e:
                        st.error(f"Error generating {section_name}: {e}")
                        import traceback
                        st.code(traceback.format_exc())

    # PDF Export option
    st.divider()
    st.subheader("Export Options")

    if st.button("Generate Full PDF Report"):
        with st.spinner("Generating PDF report..."):
            try:
                # Create temp file for PDF
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    tmp_path = tmp_file.name

                # Build PDF
                pdf_builder = PDFBuilder(
                    output_path=tmp_path,
                    player_name=player_name,
                    player_team=aggregator.player_info.team_name if aggregator.player_info else 'Unknown',
                    position=position_override,
                    season="2024/25"
                )

                # Generate all sections
                all_sections = []
                for section_class in [SnapshotSection, PassingSection, CarryingSection,
                                      ChanceCreationSection, DefendingSection,
                                      BenchmarkingSection, RecommendationSection]:
                    try:
                        section = section_class(
                            player_data=player_data,
                            metrics_result=metrics_result,
                            cohort_summary=cohort_summary,
                            config=config
                        )
                        all_sections.append(section.generate())
                    except Exception:
                        pass

                pdf_path = pdf_builder.build(all_sections)

                # Create download button
                with open(pdf_path, 'rb') as f:
                    pdf_data = f.read()

                filename = f"scout_report_{player_name.replace(' ', '_').lower()}.pdf"

                st.download_button(
                    label="Download PDF Report",
                    data=pdf_data,
                    file_name=filename,
                    mime="application/pdf"
                )

                # Clean up
                os.unlink(tmp_path)

                st.success("PDF report generated!")

            except Exception as e:
                st.error(f"Error generating PDF: {e}")
                import traceback
                st.code(traceback.format_exc())


def determine_position_group(position: str) -> str:
    """Determine position group from position string"""
    position = position.lower() if position else ''

    if 'goalkeeper' in position or 'keeper' in position or position == 'gk':
        return 'GK'
    elif 'centre-back' in position or 'center-back' in position or 'central defender' in position or position == 'cb':
        return 'CB'
    elif 'full-back' in position or 'left-back' in position or 'right-back' in position or 'wing-back' in position:
        return 'FB'
    elif 'defensive mid' in position or 'holding mid' in position or position == 'dm' or position == 'cdm':
        return 'DM'
    elif 'attacking mid' in position or 'playmaker' in position or position == 'am' or position == 'cam':
        return 'AM'
    elif 'winger' in position or 'left mid' in position or 'right mid' in position or 'wide' in position:
        return 'W'
    elif 'striker' in position or 'forward' in position or 'centre-forward' in position or position == 'st' or position == 'cf':
        return 'ST'
    else:
        return 'CM'  # Default to central midfielder


if __name__ == "__main__":
    main()
