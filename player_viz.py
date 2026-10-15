"""
Player Visualization Module for Opta F24 Data
Creates individual player analysis graphics: shot maps, pass maps, heat maps, and profiles
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

from pitch_viz import Pitch, PitchColors
from opta_api import OptaAPI, EventProcessor, MatchInfo


@dataclass
class PlayerInfo:
    """Container for player metadata"""
    player_id: str
    player_name: str
    team_id: str
    team_name: str
    position: Optional[str] = None


class PlayerDataAggregator:
    """
    Aggregate player event data across single or multiple matches
    """

    def __init__(self, api: OptaAPI):
        """
        Initialize aggregator with API client

        Args:
            api: OptaAPI instance (can be configured for specific league)
        """
        self.api = api
        self._passes = pd.DataFrame()
        self._shots = pd.DataFrame()
        self._defensive_actions = pd.DataFrame()
        self._ball_recoveries = pd.DataFrame()
        self._take_ons = pd.DataFrame()
        self._aerials = pd.DataFrame()
        self._touches = pd.DataFrame()
        self._match_count = 0
        self._player_info: Optional[PlayerInfo] = None

    def load_match(self, match_id: str, player_id: str) -> 'PlayerDataAggregator':
        """
        Load and aggregate data from a single match

        Args:
            match_id: Opta match ID
            player_id: Target player ID

        Returns:
            self (for method chaining)
        """
        events_df, match_info = self.api.get_match_events(match_id)
        processor = EventProcessor(events_df, match_info)

        # Filter events for target player
        passes = processor.get_passes()
        player_passes = passes[passes['playerId'] == player_id].copy()
        player_passes['match_id'] = match_id

        shots = processor.get_shots()
        player_shots = shots[shots['playerId'] == player_id].copy()
        player_shots['match_id'] = match_id

        defensive = processor.get_defensive_actions()
        player_defensive = defensive[defensive['playerId'] == player_id].copy()
        player_defensive['match_id'] = match_id

        recoveries = processor.get_ball_recoveries()
        player_recoveries = recoveries[recoveries['playerId'] == player_id].copy()
        player_recoveries['match_id'] = match_id

        take_ons = processor.get_take_ons()
        player_take_ons = take_ons[take_ons['playerId'] == player_id].copy()
        player_take_ons['match_id'] = match_id

        aerials = processor.get_aerials()
        player_aerials = aerials[aerials['playerId'] == player_id].copy()
        player_aerials['match_id'] = match_id

        # All touches (events with x, y coordinates)
        touch_cols = ['x', 'y', 'typeId', 'playerId', 'playerName', 'contestantId', 'team_name']
        available_cols = [c for c in touch_cols if c in events_df.columns]
        all_events = events_df[
            (events_df['playerId'] == player_id) &
            (events_df['x'].notna()) &
            (events_df['y'].notna())
        ][available_cols].copy()
        all_events['match_id'] = match_id

        # Aggregate
        self._passes = pd.concat([self._passes, player_passes], ignore_index=True)
        self._shots = pd.concat([self._shots, player_shots], ignore_index=True)
        self._defensive_actions = pd.concat([self._defensive_actions, player_defensive], ignore_index=True)
        self._ball_recoveries = pd.concat([self._ball_recoveries, player_recoveries], ignore_index=True)
        self._take_ons = pd.concat([self._take_ons, player_take_ons], ignore_index=True)
        self._aerials = pd.concat([self._aerials, player_aerials], ignore_index=True)
        self._touches = pd.concat([self._touches, all_events], ignore_index=True)

        self._match_count += 1

        # Set player info from first match
        if self._player_info is None and len(player_passes) > 0:
            row = player_passes.iloc[0]
            self._player_info = PlayerInfo(
                player_id=player_id,
                player_name=row.get('playerName', 'Unknown'),
                team_id=row.get('contestantId', ''),
                team_name=row.get('team_name', '')
            )

        return self

    def load_matches(self, match_ids: List[str], player_id: str) -> 'PlayerDataAggregator':
        """
        Load and aggregate data from multiple matches

        Args:
            match_ids: List of Opta match IDs
            player_id: Target player ID

        Returns:
            self (for method chaining)
        """
        for match_id in match_ids:
            self.load_match(match_id, player_id)
        return self

    def load_season(self, player_id: str, team_id: str = None,
                    max_matches: int = None) -> 'PlayerDataAggregator':
        """
        Load all matches from the configured league/season

        Args:
            player_id: Target player ID
            team_id: Filter by team (optional - speeds up processing)
            max_matches: Limit number of matches (for testing)

        Returns:
            self (for method chaining)
        """
        if self.api.league is None:
            raise ValueError("API must be configured with a league to load season data")

        matches = self.api.get_matches_by_tournament()
        match_ids = matches['matchInfo.id'].tolist() if 'matchInfo.id' in matches.columns else []

        if max_matches:
            match_ids = match_ids[:max_matches]

        return self.load_matches(match_ids, player_id)

    @property
    def passes(self) -> pd.DataFrame:
        return self._passes.copy()

    @property
    def shots(self) -> pd.DataFrame:
        return self._shots.copy()

    @property
    def defensive_actions(self) -> pd.DataFrame:
        return self._defensive_actions.copy()

    @property
    def ball_recoveries(self) -> pd.DataFrame:
        return self._ball_recoveries.copy()

    @property
    def take_ons(self) -> pd.DataFrame:
        return self._take_ons.copy()

    @property
    def aerials(self) -> pd.DataFrame:
        return self._aerials.copy()

    @property
    def touches(self) -> pd.DataFrame:
        return self._touches.copy()

    @property
    def player_info(self) -> Optional[PlayerInfo]:
        return self._player_info

    @property
    def match_count(self) -> int:
        return self._match_count

    def get_summary_stats(self) -> Dict:
        """Calculate summary statistics across all loaded matches"""
        stats = {
            'matches_played': self._match_count,
            'total_passes': len(self._passes),
            'pass_completion': self._passes['is_successful'].mean() * 100 if len(self._passes) > 0 else 0,
            'total_shots': len(self._shots),
            'goals': int(self._shots['is_goal'].sum()) if len(self._shots) > 0 and 'is_goal' in self._shots.columns else 0,
            'total_xg': self._shots['xG'].sum() if len(self._shots) > 0 and 'xG' in self._shots.columns else 0,
            'key_passes': int(self._passes['is_key_pass'].sum()) if len(self._passes) > 0 and 'is_key_pass' in self._passes.columns else 0,
            'assists': int(self._passes['is_assist'].sum()) if len(self._passes) > 0 and 'is_assist' in self._passes.columns else 0,
            'tackles': len(self._defensive_actions[self._defensive_actions['action_type'] == 'Tackle']) if len(self._defensive_actions) > 0 and 'action_type' in self._defensive_actions.columns else 0,
            'interceptions': len(self._defensive_actions[self._defensive_actions['action_type'] == 'Interception']) if len(self._defensive_actions) > 0 and 'action_type' in self._defensive_actions.columns else 0,
            'ball_recoveries': len(self._ball_recoveries),
            'take_ons_attempted': len(self._take_ons),
            'take_ons_successful': int(self._take_ons['is_successful'].sum()) if len(self._take_ons) > 0 and 'is_successful' in self._take_ons.columns else 0,
            'aerials_won': int(self._aerials['is_won'].sum()) if len(self._aerials) > 0 and 'is_won' in self._aerials.columns else 0,
            'aerials_total': len(self._aerials),
        }
        return stats


class PlayerShotMap:
    """
    Create shot map visualization for a single player
    """

    def __init__(self, shots_df: pd.DataFrame, player_name: str = None):
        """
        Initialize player shot map

        Args:
            shots_df: DataFrame with shot data (filtered for target player)
            player_name: Player name for title
        """
        self.shots = shots_df.copy()
        self.player_name = player_name
        self.colors = PitchColors()

    def plot(self,
             ax: plt.Axes = None,
             show_xg: bool = True,
             half_pitch: bool = True,
             title: str = None,
             show_stats: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create player shot map with xG values and outcomes

        Args:
            ax: Matplotlib axes
            show_xg: Size markers by xG value
            half_pitch: Show only attacking half
            title: Custom title
            show_stats: Show summary statistics box

        Returns:
            Tuple of (figure, axes)
        """
        pitch = Pitch(half=half_pitch, figsize=(10, 8))
        fig, ax = pitch.draw(ax)

        if len(self.shots) == 0:
            ax.text(75, 50, "No shots recorded", fontsize=14, color='white',
                    ha='center', va='center')
            if title:
                ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
            fig.patch.set_facecolor('#1a1a2e')
            return fig, ax

        # Color by outcome
        colors = []
        for _, shot in self.shots.iterrows():
            if shot.get('is_goal', False):
                colors.append(self.colors.goal)
            elif shot.get('is_on_target', False):
                colors.append(self.colors.save)
            elif shot.get('hit_post', False):
                colors.append(self.colors.post)
            else:
                colors.append(self.colors.miss)

        # Size by xG
        if show_xg and 'xG' in self.shots.columns:
            sizes = self.shots['xG'] * 1000 + 50
        else:
            sizes = 150

        # Plot shots
        ax.scatter(
            self.shots['x'],
            self.shots['y'],
            c=colors,
            s=sizes,
            alpha=0.85,
            edgecolors='white',
            linewidths=2,
            zorder=10
        )

        # xG labels for all shots
        for _, shot in self.shots.iterrows():
            xg_val = shot.get('xG', 0)
            ax.annotate(
                f"{xg_val:.2f}",
                (shot['x'], shot['y']),
                xytext=(0, -12),
                textcoords='offset points',
                fontsize=7,
                color='white',
                ha='center',
                alpha=0.8
            )

        # Legend
        legend_elements = [
            plt.scatter([], [], c=self.colors.goal, s=150, label='Goal', edgecolors='white'),
            plt.scatter([], [], c=self.colors.save, s=150, label='Saved', edgecolors='white'),
            plt.scatter([], [], c=self.colors.post, s=150, label='Post', edgecolors='white'),
            plt.scatter([], [], c=self.colors.miss, s=150, label='Miss', edgecolors='white'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

        # Stats box
        if show_stats:
            total_xg = self.shots['xG'].sum() if 'xG' in self.shots.columns else 0
            goals = self.shots.get('is_goal', pd.Series([False])).sum()
            on_target = self.shots.get('is_on_target', pd.Series([False])).sum()

            xg_per_shot = total_xg / len(self.shots) if len(self.shots) > 0 else 0
            xg_diff = goals - total_xg
            diff_str = f"+{xg_diff:.2f}" if xg_diff >= 0 else f"{xg_diff:.2f}"

            stats_text = (
                f"Shots: {len(self.shots)}\n"
                f"Goals: {int(goals)}\n"
                f"On Target: {int(on_target)}\n"
                f"xG: {total_xg:.2f}\n"
                f"xG/Shot: {xg_per_shot:.3f}\n"
                f"G-xG: {diff_str}"
            )

            ax.text(97, 55, stats_text, fontsize=9, color='white',
                    ha='right', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
        elif self.player_name:
            ax.set_title(f"{self.player_name} - Shot Map", fontsize=14,
                         fontweight='bold', color='white', pad=10)

        fig.patch.set_facecolor('#1a1a2e')

        return fig, ax


class PlayerPassMap:
    """
    Create pass map visualization showing passes made by a player
    """

    def __init__(self, passes_df: pd.DataFrame, player_name: str = None):
        """
        Initialize player pass map

        Args:
            passes_df: DataFrame with pass data (filtered for target player)
            player_name: Player name for title
        """
        self.passes = passes_df.copy()
        self.player_name = player_name
        self.colors = PitchColors()

    def plot(self,
             ax: plt.Axes = None,
             show_successful: bool = True,
             show_unsuccessful: bool = True,
             highlight_key_passes: bool = True,
             highlight_assists: bool = True,
             title: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create pass map showing all passes made

        Args:
            ax: Matplotlib axes
            show_successful: Show successful passes
            show_unsuccessful: Show unsuccessful passes
            highlight_key_passes: Highlight key passes with different color
            highlight_assists: Highlight assists with special marker
            title: Custom title

        Returns:
            Tuple of (figure, axes)
        """
        pitch = Pitch(figsize=(12, 8))
        fig, ax = pitch.draw(ax)

        if len(self.passes) == 0:
            ax.text(50, 50, "No passes recorded", fontsize=14, color='white',
                    ha='center', va='center')
            if title:
                ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
            fig.patch.set_facecolor('#1a1a2e')
            return fig, ax

        # Filter passes with valid end coordinates
        valid_passes = self.passes[
            self.passes['end_x'].notna() & self.passes['end_y'].notna()
        ].copy()

        # Draw unsuccessful passes first (underneath)
        if show_unsuccessful and 'is_successful' in valid_passes.columns:
            unsuccessful = valid_passes[valid_passes['is_successful'] == False]
            for _, p in unsuccessful.iterrows():
                ax.annotate(
                    '',
                    xy=(p['end_x'], p['end_y']),
                    xytext=(p['x'], p['y']),
                    arrowprops=dict(
                        arrowstyle='->',
                        color='#e74c3c',
                        alpha=0.4,
                        linewidth=1,
                        shrinkA=0,
                        shrinkB=0
                    ),
                    zorder=5
                )

        # Draw successful passes
        if show_successful and 'is_successful' in valid_passes.columns:
            is_key = valid_passes.get('is_key_pass', pd.Series([False] * len(valid_passes)))
            is_assist = valid_passes.get('is_assist', pd.Series([False] * len(valid_passes)))

            successful = valid_passes[
                (valid_passes['is_successful'] == True) &
                (is_key == False) &
                (is_assist == False)
            ]
            for _, p in successful.iterrows():
                ax.annotate(
                    '',
                    xy=(p['end_x'], p['end_y']),
                    xytext=(p['x'], p['y']),
                    arrowprops=dict(
                        arrowstyle='->',
                        color='#3498db',
                        alpha=0.5,
                        linewidth=1.2,
                        shrinkA=0,
                        shrinkB=0
                    ),
                    zorder=6
                )

        # Highlight key passes
        if highlight_key_passes and 'is_key_pass' in valid_passes.columns:
            is_assist = valid_passes.get('is_assist', pd.Series([False] * len(valid_passes)))
            key_passes = valid_passes[
                (valid_passes['is_key_pass'] == True) &
                (is_assist == False)
            ]
            for _, p in key_passes.iterrows():
                ax.annotate(
                    '',
                    xy=(p['end_x'], p['end_y']),
                    xytext=(p['x'], p['y']),
                    arrowprops=dict(
                        arrowstyle='->',
                        color='#f39c12',
                        alpha=0.9,
                        linewidth=2.5,
                        shrinkA=0,
                        shrinkB=0
                    ),
                    zorder=8
                )
                ax.scatter(p['end_x'], p['end_y'], c='#f39c12', s=50,
                           marker='o', edgecolors='white', linewidths=1, zorder=9)

        # Highlight assists
        if highlight_assists and 'is_assist' in valid_passes.columns:
            assists = valid_passes[valid_passes['is_assist'] == True]
            for _, p in assists.iterrows():
                ax.annotate(
                    '',
                    xy=(p['end_x'], p['end_y']),
                    xytext=(p['x'], p['y']),
                    arrowprops=dict(
                        arrowstyle='->',
                        color='#2ecc71',
                        alpha=1.0,
                        linewidth=3,
                        shrinkA=0,
                        shrinkB=0
                    ),
                    zorder=10
                )
                ax.scatter(p['end_x'], p['end_y'], c='#2ecc71', s=120,
                           marker='*', edgecolors='white', linewidths=1, zorder=11)

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color='#3498db', linewidth=2, label='Successful'),
            plt.Line2D([0], [0], color='#e74c3c', linewidth=2, label='Unsuccessful'),
            plt.Line2D([0], [0], color='#f39c12', linewidth=3, label='Key Pass'),
            plt.Line2D([0], [0], color='#2ecc71', linewidth=3, label='Assist'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

        # Stats box
        total = len(self.passes)
        successful_count = self.passes['is_successful'].sum() if 'is_successful' in self.passes.columns else 0
        completion = (successful_count / total * 100) if total > 0 else 0
        key_pass_count = self.passes['is_key_pass'].sum() if 'is_key_pass' in self.passes.columns else 0
        assist_count = self.passes['is_assist'].sum() if 'is_assist' in self.passes.columns else 0

        stats_text = (
            f"Total Passes: {total}\n"
            f"Completed: {int(successful_count)} ({completion:.1f}%)\n"
            f"Key Passes: {int(key_pass_count)}\n"
            f"Assists: {int(assist_count)}"
        )

        ax.text(98, 3, stats_text, fontsize=9, color='white',
                ha='right', va='bottom', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
        elif self.player_name:
            ax.set_title(f"{self.player_name} - Pass Map", fontsize=14,
                         fontweight='bold', color='white', pad=10)

        fig.patch.set_facecolor('#1a1a2e')

        return fig, ax


class PlayerHeatMap:
    """
    Create heat map showing player touch locations
    """

    def __init__(self, touches_df: pd.DataFrame, player_name: str = None):
        """
        Initialize player heat map

        Args:
            touches_df: DataFrame with all player touches (x, y coordinates)
            player_name: Player name for title
        """
        self.touches = touches_df.copy()
        self.player_name = player_name

    def plot(self,
             ax: plt.Axes = None,
             bins: int = 12,
             cmap: str = 'hot',
             title: str = None,
             show_touches: bool = False) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create heat map of player touch locations

        Args:
            ax: Matplotlib axes
            bins: Number of bins for histogram (higher = more detail)
            cmap: Matplotlib colormap name
            title: Custom title
            show_touches: Overlay individual touch points

        Returns:
            Tuple of (figure, axes)
        """
        pitch = Pitch(figsize=(12, 8))
        fig, ax = pitch.draw(ax)

        if len(self.touches) == 0:
            ax.text(50, 50, "No touches recorded", fontsize=14, color='white',
                    ha='center', va='center')
            if title:
                ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
            fig.patch.set_facecolor('#1a1a2e')
            return fig, ax

        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            self.touches['x'],
            self.touches['y'],
            bins=bins,
            range=[[0, 100], [0, 100]]
        )

        # Apply Gaussian smoothing for cleaner visualization
        try:
            from scipy.ndimage import gaussian_filter
            heatmap_smooth = gaussian_filter(heatmap, sigma=1)
        except ImportError:
            heatmap_smooth = heatmap

        # Custom colormap (transparent to hot)
        colors_list = ['#1a472a00', '#1a472a', '#2ecc71', '#f1c40f', '#e74c3c', '#ffffff']
        custom_cmap = LinearSegmentedColormap.from_list('custom_hot', colors_list)

        # Plot heatmap
        extent = [0, 100, 0, 100]
        ax.imshow(
            heatmap_smooth.T,
            origin='lower',
            extent=extent,
            cmap=custom_cmap,
            alpha=0.7,
            aspect='auto',
            zorder=2
        )

        # Overlay individual touches
        if show_touches:
            ax.scatter(
                self.touches['x'],
                self.touches['y'],
                c='white',
                s=5,
                alpha=0.3,
                zorder=3
            )

        # Calculate average position
        avg_x = self.touches['x'].mean()
        avg_y = self.touches['y'].mean()

        # Mark average position
        ax.scatter([avg_x], [avg_y], c='white', s=200, marker='x',
                   linewidths=3, zorder=10)
        ax.annotate(
            'AVG',
            (avg_x, avg_y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            color='white',
            fontweight='bold'
        )

        # Stats
        total_touches = len(self.touches)
        left_half = (self.touches['x'] < 50).sum() / total_touches * 100
        right_half = (self.touches['x'] >= 50).sum() / total_touches * 100
        final_third = (self.touches['x'] >= 66.66).sum() / total_touches * 100

        stats_text = (
            f"Total Touches: {total_touches}\n"
            f"Avg Position: ({avg_x:.1f}, {avg_y:.1f})\n"
            f"Own Half: {left_half:.0f}%\n"
            f"Opp Half: {right_half:.0f}%\n"
            f"Final Third: {final_third:.0f}%"
        )

        ax.text(2, 3, stats_text, fontsize=9, color='white',
                ha='left', va='bottom', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
        elif self.player_name:
            ax.set_title(f"{self.player_name} - Touch Heat Map", fontsize=14,
                         fontweight='bold', color='white', pad=10)

        fig.patch.set_facecolor('#1a1a2e')

        return fig, ax


class PlayerProfile:
    """
    Generate complete player profile with multiple visualizations
    """

    def __init__(self, aggregator: PlayerDataAggregator):
        """
        Initialize player profile

        Args:
            aggregator: PlayerDataAggregator with loaded match data
        """
        self.aggregator = aggregator
        self.player_info = aggregator.player_info

    def generate_profile(self,
                         save_path: str = None,
                         figsize: Tuple[int, int] = (20, 16)) -> plt.Figure:
        """
        Generate full player profile dashboard

        Layout:
        +-------------------+-------------------+
        |     Shot Map      |     Pass Map      |
        +-------------------+-------------------+
        |     Heat Map      |    Stats Panel    |
        +-------------------+-------------------+

        Args:
            save_path: Path to save figure (optional)
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor('#1a1a2e')

        # Create grid
        gs = GridSpec(2, 2, figure=fig, hspace=0.15, wspace=0.1)

        # Player name header
        player_name = self.player_info.player_name if self.player_info else "Unknown Player"
        team_name = self.player_info.team_name if self.player_info else ""
        matches = self.aggregator.match_count

        fig.suptitle(
            f"{player_name}\n{team_name} | {matches} Match{'es' if matches != 1 else ''}",
            fontsize=20, fontweight='bold', color='white', y=0.98
        )

        # Shot Map (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        shot_map = PlayerShotMap(self.aggregator.shots, player_name)
        shot_map.plot(ax=ax1, title="Shot Map")

        # Pass Map (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        pass_map = PlayerPassMap(self.aggregator.passes, player_name)
        pass_map.plot(ax=ax2, title="Pass Map")

        # Heat Map (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        heat_map = PlayerHeatMap(self.aggregator.touches, player_name)
        heat_map.plot(ax=ax3, title="Touch Heat Map")

        # Stats Panel (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        self._draw_stats_panel(ax4)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())

        return fig

    def _draw_stats_panel(self, ax: plt.Axes):
        """Draw statistics summary panel"""
        ax.set_facecolor('#1a1a2e')
        ax.axis('off')

        stats = self.aggregator.get_summary_stats()

        # Organize stats into categories
        categories = {
            'Overview': [
                ('Matches', stats['matches_played']),
            ],
            'Attacking': [
                ('Goals', stats['goals']),
                ('xG', f"{stats['total_xg']:.2f}"),
                ('Shots', stats['total_shots']),
                ('Assists', stats['assists']),
                ('Key Passes', stats['key_passes']),
            ],
            'Passing': [
                ('Total Passes', stats['total_passes']),
                ('Pass Completion', f"{stats['pass_completion']:.1f}%"),
            ],
            'Dribbling': [
                ('Take-Ons', f"{stats['take_ons_successful']}/{stats['take_ons_attempted']}"),
            ],
            'Defending': [
                ('Tackles', stats['tackles']),
                ('Interceptions', stats['interceptions']),
                ('Ball Recoveries', stats['ball_recoveries']),
            ],
            'Aerial': [
                ('Aerials Won', f"{stats['aerials_won']}/{stats['aerials_total']}"),
            ]
        }

        y_pos = 0.95
        for category, items in categories.items():
            # Category header
            ax.text(0.5, y_pos, category, fontsize=14, fontweight='bold',
                    color='#f39c12', transform=ax.transAxes, ha='center')
            y_pos -= 0.06

            # Items
            for label, value in items:
                ax.text(0.3, y_pos, label, fontsize=11, color='white',
                        transform=ax.transAxes, ha='right')
                ax.text(0.35, y_pos, str(value), fontsize=11, color='white',
                        fontweight='bold', transform=ax.transAxes, ha='left')
                y_pos -= 0.045

            y_pos -= 0.02  # Space between categories


def quick_player_profile(league_key: str, match_id: str, player_id: str,
                         save_path: str = None) -> plt.Figure:
    """
    Generate player profile from single match with one function call

    Args:
        league_key: League identifier (e.g., "usl_championship")
        match_id: Opta match ID
        player_id: Target player ID
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    api = OptaAPI.for_league(league_key)
    aggregator = PlayerDataAggregator(api).load_match(match_id, player_id)
    return PlayerProfile(aggregator).generate_profile(save_path=save_path)
