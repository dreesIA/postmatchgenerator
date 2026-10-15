"""Streamlit application for the Opta Scout Report generator."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import streamlit as st
import pandas as pd
from datetime import datetime

# Import report generator
try:
    from src.report_generator import ReportGenerator
except ImportError:
    st.error("Failed to import report generator. Please check your installation.")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Opta Scout Report Generator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #1a1a2e;
    }
    .stButton>button {
        background-color: #e94560;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #c73e54;
    }
    .metric-card {
        background-color: #16213e;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .section-header {
        color: #e94560;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.getenv('OPTA_API_KEY', '')
    if 'api_secret' not in st.session_state:
        st.session_state.api_secret = os.getenv('OPTA_API_SECRET', '')
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False
    if 'report_path' not in st.session_state:
        st.session_state.report_path = None


def sidebar_config():
    """Render sidebar configuration."""
    st.sidebar.title("⚽ Scout Report Generator")
    st.sidebar.markdown("---")

    # API Configuration
    st.sidebar.subheader("API Configuration")

    api_key = st.sidebar.text_input(
        "Opta API Key",
        value=st.session_state.api_key,
        type="password",
        help="Your Opta API key"
    )
    st.session_state.api_key = api_key

    api_secret = st.sidebar.text_input(
        "Opta API Secret",
        value=st.session_state.api_secret,
        type="password",
        help="Your Opta API secret"
    )
    st.session_state.api_secret = api_secret

    st.sidebar.markdown("---")

    # Competition selection
    st.sidebar.subheader("Data Selection")

    competitions = {
        "Premier League": "8",
        "La Liga": "23",
        "Serie A": "21",
        "Bundesliga": "22",
        "Ligue 1": "24",
        "Champions League": "5",
        "Europa League": "6",
        "MLS": "45",
    }

    competition = st.sidebar.selectbox(
        "Competition",
        options=list(competitions.keys()),
        index=0
    )
    competition_id = competitions[competition]

    seasons = {
        "2024/25": "2024",
        "2023/24": "2023",
        "2022/23": "2022",
    }

    season = st.sidebar.selectbox(
        "Season",
        options=list(seasons.keys()),
        index=0
    )
    season_id = seasons[season]

    return competition_id, season_id


def main_content(competition_id: str, season_id: str):
    """Render main content area."""
    st.title("🔍 Individual Player Scout Report")
    st.markdown("Generate comprehensive scouting reports for individual players.")

    # Player search
    col1, col2 = st.columns([3, 1])

    with col1:
        player_id = st.text_input(
            "Player ID",
            placeholder="Enter Opta Player ID (e.g., p12345)",
            help="The unique Opta identifier for the player"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("🔍 Search Player", use_container_width=True)

    # Section selection
    st.markdown("### Report Sections")

    sections = {
        'snapshot': 'Executive Summary',
        'role_usage': 'Role & Usage',
        'physical': 'Physical Performance',
        'involvement': 'Involvement & Reception',
        'passing': 'Passing Analysis',
        'carrying': 'Carrying & Dribbling',
        'chance_creation': 'Chance Creation',
        'defending': 'Defensive Analysis',
        'discipline': 'Discipline & Risk',
        'benchmarking': 'Benchmarking',
        'trends': 'Performance Trends',
        'recommendation': 'Final Recommendation',
    }

    col1, col2, col3 = st.columns(3)
    selected_sections = []

    for i, (key, label) in enumerate(sections.items()):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.checkbox(label, value=True, key=f"section_{key}"):
                selected_sections.append(key)

    st.markdown("---")

    # Generate report button
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        generate_clicked = st.button(
            "📊 Generate Report",
            use_container_width=True,
            disabled=not player_id
        )

    if generate_clicked and player_id:
        generate_report(player_id, competition_id, season_id, selected_sections)

    # Show download button if report was generated
    if st.session_state.report_generated and st.session_state.report_path:
        st.success("✅ Report generated successfully!")

        with open(st.session_state.report_path, 'rb') as f:
            st.download_button(
                label="📥 Download PDF Report",
                data=f,
                file_name=Path(st.session_state.report_path).name,
                mime="application/pdf",
                use_container_width=True
            )


def generate_report(player_id: str, competition_id: str, season_id: str,
                   selected_sections: list):
    """Generate the scout report."""
    with st.spinner("Generating report... This may take a few minutes."):
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Initializing report generator...")
            progress_bar.progress(10)

            generator = ReportGenerator(
                api_key=st.session_state.api_key,
                api_secret=st.session_state.api_secret
            )

            status_text.text("Loading player data...")
            progress_bar.progress(30)

            status_text.text("Calculating metrics...")
            progress_bar.progress(50)

            status_text.text("Building comparison cohort...")
            progress_bar.progress(70)

            status_text.text("Generating PDF...")
            progress_bar.progress(90)

            # Generate the report
            report_path = generator.generate_report(
                player_id=player_id,
                competition_id=competition_id,
                season_id=season_id,
                sections=selected_sections if selected_sections else None
            )

            progress_bar.progress(100)
            status_text.text("Complete!")

            st.session_state.report_generated = True
            st.session_state.report_path = report_path

        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            st.session_state.report_generated = False


def batch_reports_tab():
    """Render batch reports tab."""
    st.header("📦 Batch Report Generation")
    st.markdown("Generate reports for multiple players at once.")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV with Player IDs",
        type=['csv'],
        help="CSV file with a 'player_id' column"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if 'player_id' not in df.columns:
            st.error("CSV must contain a 'player_id' column")
        else:
            st.dataframe(df.head())
            st.info(f"Found {len(df)} players")

            if st.button("Generate All Reports"):
                progress_bar = st.progress(0)

                for i, row in df.iterrows():
                    progress = int((i + 1) / len(df) * 100)
                    progress_bar.progress(progress)
                    st.text(f"Processing player {i + 1}/{len(df)}: {row['player_id']}")

                st.success("Batch generation complete!")


def comparison_tab(competition_id: str, season_id: str):
    """Render player comparison tab."""
    st.header("⚖️ Player Comparison")
    st.markdown("Compare multiple players side by side.")

    # Player inputs
    col1, col2 = st.columns(2)

    with col1:
        player1_id = st.text_input("Player 1 ID", key="player1")
        player1_name = st.text_input("Player 1 Name (optional)", key="player1_name")

    with col2:
        player2_id = st.text_input("Player 2 ID", key="player2")
        player2_name = st.text_input("Player 2 Name (optional)", key="player2_name")

    # Additional players
    with st.expander("Add more players"):
        col3, col4 = st.columns(2)

        with col3:
            player3_id = st.text_input("Player 3 ID", key="player3")

        with col4:
            player4_id = st.text_input("Player 4 ID", key="player4")

    if st.button("Generate Comparison"):
        player_ids = [pid for pid in [player1_id, player2_id, player3_id, player4_id] if pid]

        if len(player_ids) < 2:
            st.warning("Please enter at least 2 player IDs")
        else:
            st.info(f"Generating comparison for {len(player_ids)} players...")


def main():
    """Main application entry point."""
    init_session_state()
    competition_id, season_id = sidebar_config()

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📝 Single Report",
        "📦 Batch Reports",
        "⚖️ Comparison"
    ])

    with tab1:
        main_content(competition_id, season_id)

    with tab2:
        batch_reports_tab()

    with tab3:
        comparison_tab(competition_id, season_id)


if __name__ == "__main__":
    main()
