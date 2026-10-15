"""
Phase-by-Phase Match Report Generator
Creates detailed analysis pages for each phase of play:
- In Possession: Build-up, Middle Progression, Final Third
- Out of Possession: High Press, Mid/Low Block
- Transitions: Positive (to attack), Negative (to defense)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Arrow
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass
from io import BytesIO

from opta_api import OptaAPI, EventProcessor, MatchInfo, PassingNetworkBuilder


# Color scheme
BACKGROUND_COLOR = '#f5f5f0'
TEXT_COLOR = '#1a1a1a'
TEXT_SECONDARY = '#666666'
PITCH_COLOR = '#c8d6a0'
PITCH_LINES = '#2d5a27'


@dataclass
class TeamColors:
    """Color scheme for home and away teams"""
    home: str = '#e63946'
    away: str = '#457b9d'
    home_light: str = '#f4a5ab'
    away_light: str = '#a8c5d9'


class PhasePage:
    """Base class for phase analysis pages"""

    def __init__(self, events_df: pd.DataFrame, match_info: MatchInfo,
                 passes: pd.DataFrame, shots: pd.DataFrame,
                 defensive: pd.DataFrame, recoveries: pd.DataFrame,
                 take_ons: pd.DataFrame, match_stats: dict = None):
        self.events_df = events_df
        self.match_info = match_info
        self.passes = passes
        self.shots = shots
        self.defensive = defensive
        self.recoveries = recoveries
        self.take_ons = take_ons
        self.match_stats = match_stats
        self.colors = TeamColors()

        # Split by team
        self.home_passes = passes[passes['is_home'] == True]
        self.away_passes = passes[passes['is_home'] == False]
        self.home_shots = shots[shots['is_home'] == True]
        self.away_shots = shots[shots['is_home'] == False]
        self.home_defensive = defensive[defensive['is_home'] == True]
        self.away_defensive = defensive[defensive['is_home'] == False]
        self.home_recoveries = recoveries[recoveries['is_home'] == True]
        self.away_recoveries = recoveries[recoveries['is_home'] == False]
        self.home_take_ons = take_ons[take_ons['is_home'] == True]
        self.away_take_ons = take_ons[take_ons['is_home'] == False]

    def _draw_pitch(self, ax: plt.Axes, half: str = 'full'):
        """Draw pitch on axes. half can be 'full', 'left', 'right'"""
        ax.set_facecolor(PITCH_COLOR)
        lw = 1.2
        lc = PITCH_LINES

        if half == 'full':
            # Full pitch
            ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color=lc, lw=lw)
            ax.plot([50, 50], [0, 100], color=lc, lw=lw)
            center_circle = Circle((50, 50), 9.15, fill=False, color=lc, lw=lw)
            ax.add_patch(center_circle)
            # Left penalty area
            ax.plot([0, 17, 17, 0], [21.1, 21.1, 78.9, 78.9], color=lc, lw=lw)
            ax.plot([0, 5.8, 5.8, 0], [36.8, 36.8, 63.2, 63.2], color=lc, lw=lw)
            # Right penalty area
            ax.plot([100, 83, 83, 100], [21.1, 21.1, 78.9, 78.9], color=lc, lw=lw)
            ax.plot([100, 94.2, 94.2, 100], [36.8, 36.8, 63.2, 63.2], color=lc, lw=lw)
            ax.set_xlim(-3, 103)
            ax.set_ylim(-5, 105)
        elif half == 'left':
            # Left half only (defensive)
            ax.plot([0, 50, 50, 0, 0], [0, 0, 100, 100, 0], color=lc, lw=lw)
            ax.plot([0, 17, 17, 0], [21.1, 21.1, 78.9, 78.9], color=lc, lw=lw)
            ax.plot([0, 5.8, 5.8, 0], [36.8, 36.8, 63.2, 63.2], color=lc, lw=lw)
            ax.set_xlim(-3, 53)
            ax.set_ylim(-5, 105)
        elif half == 'right':
            # Right half only (attacking)
            ax.plot([50, 100, 100, 50, 50], [0, 0, 100, 100, 0], color=lc, lw=lw)
            ax.plot([100, 83, 83, 100], [21.1, 21.1, 78.9, 78.9], color=lc, lw=lw)
            ax.plot([100, 94.2, 94.2, 100], [36.8, 36.8, 63.2, 63.2], color=lc, lw=lw)
            ax.set_xlim(47, 103)
            ax.set_ylim(-5, 105)

        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

    def _draw_vertical_pitch(self, ax: plt.Axes, third: str = 'attacking', goal_at_top: bool = True):
        """
        Draw a vertical pitch showing the full half, rotated 90 degrees.
        Goal is at top (for attacking) or bottom (for defending).
        Data is filtered to the relevant third, but the full half is displayed.

        Args:
            ax: Matplotlib axes
            third: 'attacking' (final third) or 'defensive' (own third)
            goal_at_top: If True, goal faces up (attacking view). If False, goal faces down (defending view).
        """
        ax.set_facecolor(PITCH_COLOR)
        lw = 1.2
        lc = PITCH_LINES

        # Vertical pitch showing full half: width=100, height=50
        # Pitch outline
        ax.plot([0, 100, 100, 0, 0], [0, 0, 50, 50, 0], color=lc, lw=lw)

        # Halfway line (at y=0 for attacking, y=50 for defending when goal at top)
        if third == 'attacking':
            if goal_at_top:
                # Goal at top (y=50), halfway line at bottom (y=0)
                # Penalty area at top - complete box with bottom line
                ax.plot([21.1, 21.1], [33, 50], color=lc, lw=lw)  # Left vertical
                ax.plot([78.9, 78.9], [33, 50], color=lc, lw=lw)  # Right vertical
                ax.plot([21.1, 78.9], [33, 33], color=lc, lw=lw)  # Bottom horizontal
                # 6-yard box - complete box with bottom line
                ax.plot([36.8, 36.8], [44.2, 50], color=lc, lw=lw)  # Left vertical
                ax.plot([63.2, 63.2], [44.2, 50], color=lc, lw=lw)  # Right vertical
                ax.plot([36.8, 63.2], [44.2, 44.2], color=lc, lw=lw)  # Bottom horizontal
                # Goal line indicator
                ax.plot([44, 56], [50, 50], color=lc, lw=2.5)
                # Center circle arc (partial, at bottom - only show top half that's visible)
                theta = np.linspace(0, np.pi, 50)
                circle_x = 50 + 9.15 * np.cos(theta)
                circle_y = 0 + 9.15 * np.sin(theta)
                ax.plot(circle_x, circle_y, color=lc, lw=lw)
            else:
                # Goal at bottom (y=0), halfway line at top (y=50)
                # Penalty area at bottom - complete box with top line
                ax.plot([21.1, 21.1], [0, 17], color=lc, lw=lw)  # Left vertical
                ax.plot([78.9, 78.9], [0, 17], color=lc, lw=lw)  # Right vertical
                ax.plot([21.1, 78.9], [17, 17], color=lc, lw=lw)  # Top horizontal
                # 6-yard box - complete box with top line
                ax.plot([36.8, 36.8], [0, 5.8], color=lc, lw=lw)  # Left vertical
                ax.plot([63.2, 63.2], [0, 5.8], color=lc, lw=lw)  # Right vertical
                ax.plot([36.8, 63.2], [5.8, 5.8], color=lc, lw=lw)  # Top horizontal
                # Goal line indicator
                ax.plot([44, 56], [0, 0], color=lc, lw=2.5)
                # Center circle arc (partial, at top - only show bottom half that's visible)
                theta = np.linspace(np.pi, 2*np.pi, 50)
                circle_x = 50 + 9.15 * np.cos(theta)
                circle_y = 50 + 9.15 * np.sin(theta)
                ax.plot(circle_x, circle_y, color=lc, lw=lw)

        else:  # defensive third
            if goal_at_top:
                # Goal at top (y=50) - own goal being defended
                # Penalty area at top - complete box with bottom line
                ax.plot([21.1, 21.1], [33, 50], color=lc, lw=lw)  # Left vertical
                ax.plot([78.9, 78.9], [33, 50], color=lc, lw=lw)  # Right vertical
                ax.plot([21.1, 78.9], [33, 33], color=lc, lw=lw)  # Bottom horizontal
                # 6-yard box - complete box with bottom line
                ax.plot([36.8, 36.8], [44.2, 50], color=lc, lw=lw)  # Left vertical
                ax.plot([63.2, 63.2], [44.2, 50], color=lc, lw=lw)  # Right vertical
                ax.plot([36.8, 63.2], [44.2, 44.2], color=lc, lw=lw)  # Bottom horizontal
                # Goal line indicator
                ax.plot([44, 56], [50, 50], color=lc, lw=2.5)
                # Center circle arc (partial, at bottom - only show top half that's visible)
                theta = np.linspace(0, np.pi, 50)
                circle_x = 50 + 9.15 * np.cos(theta)
                circle_y = 0 + 9.15 * np.sin(theta)
                ax.plot(circle_x, circle_y, color=lc, lw=lw)
            else:
                # Goal at bottom (y=0) - own goal being defended
                # Penalty area at bottom - complete box with top line
                ax.plot([21.1, 21.1], [0, 17], color=lc, lw=lw)  # Left vertical
                ax.plot([78.9, 78.9], [0, 17], color=lc, lw=lw)  # Right vertical
                ax.plot([21.1, 78.9], [17, 17], color=lc, lw=lw)  # Top horizontal
                # 6-yard box - complete box with top line
                ax.plot([36.8, 36.8], [0, 5.8], color=lc, lw=lw)  # Left vertical
                ax.plot([63.2, 63.2], [0, 5.8], color=lc, lw=lw)  # Right vertical
                ax.plot([36.8, 63.2], [5.8, 5.8], color=lc, lw=lw)  # Top horizontal
                # Goal line indicator
                ax.plot([44, 56], [0, 0], color=lc, lw=2.5)
                # Center circle arc (partial, at top - only show bottom half that's visible)
                theta = np.linspace(np.pi, 2*np.pi, 50)
                circle_x = 50 + 9.15 * np.cos(theta)
                circle_y = 50 + 9.15 * np.sin(theta)
                ax.plot(circle_x, circle_y, color=lc, lw=lw)

        ax.set_xlim(-5, 105)
        ax.set_ylim(-3, 53)

        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

    def _transform_to_vertical(self, x: float, y: float, third: str = 'attacking',
                                goal_at_top: bool = True, is_home: bool = True) -> Tuple[float, float]:
        """
        Transform coordinates from horizontal pitch to vertical half-pitch view.

        The visualization shows the full half (0-50 on y-axis), but data is only
        from the relevant third.

        For attacking third (original x: 50-100 maps to full half):
        - Original x maps to new y (distance from halfway line)
        - Original y maps to new x (width)

        Args:
            x, y: Original coordinates (0-100 scale)
            third: 'attacking' or 'defensive'
            goal_at_top: Whether goal is at top of visualization
            is_home: Whether this is home team (affects coordinate flipping)
        """
        if third == 'attacking':
            # Map x from 50-100 to 0-50 (half-pitch scale)
            if goal_at_top:
                # x=50 (halfway) -> y=0, x=100 (goal) -> y=50
                new_y = x - 50
            else:
                # x=50 (halfway) -> y=50, x=100 (goal) -> y=0
                new_y = 50 - (x - 50)

            new_x = y  # y stays as x (width)

        else:  # defensive third
            # Map x from 0-50 to 0-50
            if goal_at_top:
                # x=0 (goal) -> y=50 (top), x=50 (halfway) -> y=0
                new_y = 50 - x
            else:
                # x=0 (goal) -> y=0 (bottom), x=50 (halfway) -> y=50
                new_y = x

            new_x = y

        return new_x, new_y

    def _draw_header(self, ax: plt.Axes, title: str, subtitle: str = None):
        """Draw page header"""
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis('off')

        # Title
        ax.text(0.5, 0.7, title, fontsize=24, fontweight='bold',
                color=TEXT_COLOR, ha='center', va='center', transform=ax.transAxes)

        # Match info
        match_text = f"{self.match_info.home_team_name} {self.match_info.home_score} - {self.match_info.away_score} {self.match_info.away_team_name}"
        ax.text(0.5, 0.35, match_text, fontsize=14, color=TEXT_SECONDARY,
                ha='center', va='center', transform=ax.transAxes)

        if subtitle:
            ax.text(0.5, 0.1, subtitle, fontsize=11, color=TEXT_SECONDARY,
                    ha='center', va='center', transform=ax.transAxes, style='italic')

    def _draw_footer(self, ax: plt.Axes):
        """Draw page footer"""
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis('off')
        ax.text(0.5, 0.5, "Generated by Idrees Ihsanullah",
                fontsize=9, color=TEXT_SECONDARY, ha='center', va='center', transform=ax.transAxes)

    def _draw_stat_box(self, ax: plt.Axes, home_val, away_val, label: str,
                       format_str: str = '{}', higher_better: bool = True):
        """Draw a stat comparison box"""
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis('off')

        # Determine which is better
        home_better = (home_val > away_val) if higher_better else (home_val < away_val)
        away_better = not home_better and home_val != away_val

        # Home value
        home_color = self.colors.home if home_better else TEXT_SECONDARY
        ax.text(0.2, 0.5, format_str.format(home_val), fontsize=16,
                color=home_color, ha='center', va='center', transform=ax.transAxes,
                fontweight='bold' if home_better else 'normal')

        # Label
        ax.text(0.5, 0.5, label, fontsize=10, color=TEXT_COLOR,
                ha='center', va='center', transform=ax.transAxes)

        # Away value
        away_color = self.colors.away if away_better else TEXT_SECONDARY
        ax.text(0.8, 0.5, format_str.format(away_val), fontsize=16,
                color=away_color, ha='center', va='center', transform=ax.transAxes,
                fontweight='bold' if away_better else 'normal')

    def generate(self, save_path: str = None) -> plt.Figure:
        """Override in subclasses"""
        raise NotImplementedError


class BuildUpPhasePage(PhasePage):
    """
    Build-Up Phase Analysis (Own Third - x: 0-33.33)

    Visualizations:
    1. Build-up passing network (GK + defenders)
    2. Pass map from own third
    3. Progressive passes from build-up
    4. Ball progression corridors (left/center/right)
    5. Key stats: GK distribution, build-up success rate, passes under pressure
    """

    def generate(self, save_path: str = None) -> plt.Figure:
        fig = plt.figure(figsize=(20, 28), facecolor=BACKGROUND_COLOR)

        gs = GridSpec(6, 2, figure=fig, height_ratios=[0.8, 3, 3, 3, 2, 0.5],
                      hspace=0.15, wspace=0.1)

        # Header
        ax_header = fig.add_subplot(gs[0, :])
        self._draw_header(ax_header, "BUILD-UP PHASE",
                         "In Possession - Own Third (0-33%)")

        # Home build-up map
        ax_home_buildup = fig.add_subplot(gs[1, 0])
        self._draw_buildup_map(ax_home_buildup, is_home=True)

        # Away build-up map
        ax_away_buildup = fig.add_subplot(gs[1, 1])
        self._draw_buildup_map(ax_away_buildup, is_home=False)

        # Home progressive passes
        ax_home_prog = fig.add_subplot(gs[2, 0])
        self._draw_progressive_passes(ax_home_prog, is_home=True)

        # Away progressive passes
        ax_away_prog = fig.add_subplot(gs[2, 1])
        self._draw_progressive_passes(ax_away_prog, is_home=False)

        # Home corridor analysis
        ax_home_corridor = fig.add_subplot(gs[3, 0])
        self._draw_corridor_analysis(ax_home_corridor, is_home=True)

        # Away corridor analysis
        ax_away_corridor = fig.add_subplot(gs[3, 1])
        self._draw_corridor_analysis(ax_away_corridor, is_home=False)

        # Stats row
        ax_stats = fig.add_subplot(gs[4, :])
        self._draw_buildup_stats(ax_stats)

        # Footer
        ax_footer = fig.add_subplot(gs[5, :])
        self._draw_footer(ax_footer)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())

        return fig

    def _draw_buildup_map(self, ax: plt.Axes, is_home: bool):
        """Draw passes originating from own third"""
        self._draw_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        passes = self.home_passes if is_home else self.away_passes

        # Filter passes starting in own third (x < 33.33)
        buildup_passes = passes[
            (passes['x'] < 33.33) &
            (passes['end_x'].notna()) &
            (passes['end_y'].notna())
        ].copy()

        # Highlight own third zone
        rect = Rectangle((0, 0), 33.33, 100, fill=True, facecolor=color, alpha=0.1)
        ax.add_patch(rect)
        ax.axvline(x=33.33, color=color, linestyle='--', alpha=0.5, lw=1.5)

        if len(buildup_passes) == 0:
            ax.text(50, 50, "No data", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nBuild-Up Passes", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Draw passes
        for _, p in buildup_passes.iterrows():
            x, y = p['x'], p['y']
            end_x, end_y = p['end_x'], p['end_y']

            if not is_home:
                x, y = 100 - x, 100 - y
                end_x, end_y = 100 - end_x, 100 - end_y

            alpha = 0.7 if p['is_successful'] else 0.3
            ax.annotate('', xy=(end_x, end_y), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color=color, alpha=alpha, lw=1))

        success_rate = buildup_passes['is_successful'].mean() * 100
        ax.set_title(f"{team_name}\nBuild-Up Passes ({len(buildup_passes)}) - {success_rate:.0f}% Success",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_progressive_passes(self, ax: plt.Axes, is_home: bool):
        """Draw progressive passes from build-up (gaining >10m towards goal)"""
        self._draw_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        passes = self.home_passes if is_home else self.away_passes

        # Filter passes starting in own third that are progressive
        buildup_passes = passes[
            (passes['x'] < 33.33) &
            (passes['end_x'].notna()) &
            (passes['is_successful'] == True)
        ].copy()

        # Progressive = gains at least 10 units (roughly 10m) towards goal
        buildup_passes['progress'] = buildup_passes['end_x'] - buildup_passes['x']
        progressive = buildup_passes[buildup_passes['progress'] >= 10]

        if len(progressive) == 0:
            ax.text(50, 50, "No progressive passes", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nProgressive Passes from Build-Up", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Draw progressive passes with width based on progress
        for _, p in progressive.iterrows():
            x, y = p['x'], p['y']
            end_x, end_y = p['end_x'], p['end_y']

            if not is_home:
                x, y = 100 - x, 100 - y
                end_x, end_y = 100 - end_x, 100 - end_y

            width = min(2.5, p['progress'] / 15)
            ax.annotate('', xy=(end_x, end_y), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color=color, alpha=0.8, lw=width))

        avg_progress = progressive['progress'].mean()
        ax.set_title(f"{team_name}\nProgressive Passes ({len(progressive)}) - Avg {avg_progress:.1f}m",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_corridor_analysis(self, ax: plt.Axes, is_home: bool):
        """Draw corridor preference analysis (left/center/right)"""
        ax.set_facecolor(BACKGROUND_COLOR)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        passes = self.home_passes if is_home else self.away_passes

        # Filter build-up passes
        buildup_passes = passes[
            (passes['x'] < 33.33) &
            (passes['end_x'].notna()) &
            (passes['is_successful'] == True)
        ].copy()

        # Define corridors based on end_y
        buildup_passes['corridor'] = pd.cut(
            buildup_passes['end_y'],
            bins=[0, 30, 70, 100],
            labels=['Right', 'Center', 'Left']
        )

        corridor_counts = buildup_passes['corridor'].value_counts()
        total = corridor_counts.sum()

        # Draw horizontal bar chart
        corridors = ['Left', 'Center', 'Right']
        values = [corridor_counts.get(c, 0) for c in corridors]
        pcts = [v/total*100 if total > 0 else 0 for v in values]

        y_pos = [2, 1, 0]
        bars = ax.barh(y_pos, pcts, color=color, alpha=0.7, height=0.6)

        for i, (bar, pct, val) in enumerate(zip(bars, pcts, values)):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{pct:.0f}% ({val})', va='center', fontsize=10, color=TEXT_COLOR)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(corridors, fontsize=11, color=TEXT_COLOR)
        ax.set_xlim(0, 100)
        ax.set_xlabel('% of Build-Up Passes', fontsize=10, color=TEXT_COLOR)
        ax.set_title(f"{team_name}\nBuild-Up Corridor Preference", fontsize=12, color=TEXT_COLOR, pad=8)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=TEXT_COLOR)

    def _draw_buildup_stats(self, ax: plt.Axes):
        """Draw build-up phase statistics"""
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis('off')

        # Calculate stats for both teams
        home_buildup = self.home_passes[self.home_passes['x'] < 33.33]
        away_buildup = self.away_passes[self.away_passes['x'] < 33.33]

        home_success = home_buildup['is_successful'].mean() * 100 if len(home_buildup) > 0 else 0
        away_success = away_buildup['is_successful'].mean() * 100 if len(away_buildup) > 0 else 0

        home_prog = len(home_buildup[(home_buildup['end_x'] - home_buildup['x']) >= 10])
        away_prog = len(away_buildup[(away_buildup['end_x'] - away_buildup['x']) >= 10])

        home_long = home_buildup['is_long_ball'].sum() if 'is_long_ball' in home_buildup.columns else 0
        away_long = away_buildup['is_long_ball'].sum() if 'is_long_ball' in away_buildup.columns else 0

        # Draw stats
        stats = [
            (len(home_buildup), len(away_buildup), "Build-Up Passes"),
            (f"{home_success:.0f}", f"{away_success:.0f}", "Pass Success %"),
            (home_prog, away_prog, "Progressive Passes"),
            (home_long, away_long, "Long Balls"),
        ]

        y_start = 0.7
        y_step = 0.18

        ax.text(0.5, 0.95, "BUILD-UP STATISTICS", fontsize=14, fontweight='bold',
                color=TEXT_COLOR, ha='center', va='center', transform=ax.transAxes)

        # Team headers (with more space before first stat)
        ax.text(0.2, 0.85, self.match_info.home_team_name, fontsize=11,
                color=self.colors.home, ha='center', va='center', transform=ax.transAxes, fontweight='bold')
        ax.text(0.8, 0.85, self.match_info.away_team_name, fontsize=11,
                color=self.colors.away, ha='center', va='center', transform=ax.transAxes, fontweight='bold')

        for i, (home_val, away_val, label) in enumerate(stats):
            y = y_start - i * y_step
            ax.text(0.2, y, str(home_val), fontsize=14, color=self.colors.home,
                    ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.text(0.5, y, label, fontsize=10, color=TEXT_COLOR,
                    ha='center', va='center', transform=ax.transAxes)
            ax.text(0.8, y, str(away_val), fontsize=14, color=self.colors.away,
                    ha='center', va='center', transform=ax.transAxes, fontweight='bold')


class MiddleProgressionPhasePage(PhasePage):
    """
    Middle Progression Phase Analysis (Middle Third - x: 33.33-66.66)

    Visualizations:
    1. Passes in middle third
    2. Line-breaking passes
    3. Progression into final third
    4. Take-ons in middle third
    5. Key stats: passes completed, line breaks, final third entries
    """

    def generate(self, save_path: str = None) -> plt.Figure:
        fig = plt.figure(figsize=(20, 28), facecolor=BACKGROUND_COLOR)

        gs = GridSpec(6, 2, figure=fig, height_ratios=[0.8, 3, 3, 3, 2, 0.5],
                      hspace=0.15, wspace=0.1)

        # Header
        ax_header = fig.add_subplot(gs[0, :])
        self._draw_header(ax_header, "MIDDLE PROGRESSION PHASE",
                         "In Possession - Middle Third (33-66%)")

        # Home middle third map
        ax_home_middle = fig.add_subplot(gs[1, 0])
        self._draw_middle_third_map(ax_home_middle, is_home=True)

        # Away middle third map
        ax_away_middle = fig.add_subplot(gs[1, 1])
        self._draw_middle_third_map(ax_away_middle, is_home=False)

        # Home final third entries
        ax_home_ft = fig.add_subplot(gs[2, 0])
        self._draw_final_third_entries(ax_home_ft, is_home=True)

        # Away final third entries
        ax_away_ft = fig.add_subplot(gs[2, 1])
        self._draw_final_third_entries(ax_away_ft, is_home=False)

        # Home take-ons
        ax_home_takeons = fig.add_subplot(gs[3, 0])
        self._draw_middle_takeons(ax_home_takeons, is_home=True)

        # Away take-ons
        ax_away_takeons = fig.add_subplot(gs[3, 1])
        self._draw_middle_takeons(ax_away_takeons, is_home=False)

        # Stats row
        ax_stats = fig.add_subplot(gs[4, :])
        self._draw_middle_stats(ax_stats)

        # Footer
        ax_footer = fig.add_subplot(gs[5, :])
        self._draw_footer(ax_footer)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())

        return fig

    def _draw_middle_third_map(self, ax: plt.Axes, is_home: bool):
        """Draw passes in middle third"""
        self._draw_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        passes = self.home_passes if is_home else self.away_passes

        # Filter passes in middle third
        middle_passes = passes[
            (passes['x'] >= 33.33) & (passes['x'] < 66.66) &
            (passes['end_x'].notna())
        ].copy()

        # Highlight middle third
        rect = Rectangle((33.33, 0), 33.33, 100, fill=True, facecolor=color, alpha=0.1)
        ax.add_patch(rect)

        if len(middle_passes) == 0:
            ax.text(50, 50, "No data", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nMiddle Third Passes", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Draw passes
        for _, p in middle_passes.iterrows():
            x, y = p['x'], p['y']
            end_x, end_y = p['end_x'], p['end_y']

            if not is_home:
                x, y = 100 - x, 100 - y
                end_x, end_y = 100 - end_x, 100 - end_y

            alpha = 0.6 if p['is_successful'] else 0.2
            ax.annotate('', xy=(end_x, end_y), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color=color, alpha=alpha, lw=0.8))

        success_rate = middle_passes['is_successful'].mean() * 100
        ax.set_title(f"{team_name}\nMiddle Third Passes ({len(middle_passes)}) - {success_rate:.0f}%",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_final_third_entries(self, ax: plt.Axes, is_home: bool):
        """Draw passes entering the final third from middle third"""
        self._draw_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        passes = self.home_passes if is_home else self.away_passes

        # Filter passes from middle third into final third
        ft_entries = passes[
            (passes['x'] >= 33.33) & (passes['x'] < 66.66) &
            (passes['end_x'] >= 66.66) &
            (passes['is_successful'] == True)
        ].copy()

        # Draw final third line
        ax.axvline(x=66.66 if is_home else 33.33, color=color, linestyle='--', alpha=0.7, lw=2)

        if len(ft_entries) == 0:
            ax.text(50, 50, "No entries", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nFinal Third Entries", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Draw entry passes
        for _, p in ft_entries.iterrows():
            x, y = p['x'], p['y']
            end_x, end_y = p['end_x'], p['end_y']

            if not is_home:
                x, y = 100 - x, 100 - y
                end_x, end_y = 100 - end_x, 100 - end_y

            ax.annotate('', xy=(end_x, end_y), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color=color, alpha=0.7, lw=1.5))

        ax.set_title(f"{team_name}\nFinal Third Entries ({len(ft_entries)})",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_middle_takeons(self, ax: plt.Axes, is_home: bool):
        """Draw take-ons in the middle third"""
        self._draw_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        take_ons = self.home_take_ons if is_home else self.away_take_ons

        # Filter take-ons in middle third
        middle_takeons = take_ons[
            (take_ons['x'] >= 33.33) & (take_ons['x'] < 66.66)
        ].copy()

        # Highlight middle third
        rect = Rectangle((33.33 if is_home else 33.33, 0), 33.33, 100,
                         fill=True, facecolor=color, alpha=0.05)
        ax.add_patch(rect)

        if len(middle_takeons) == 0:
            ax.text(50, 50, "No take-ons", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nTake-Ons in Middle Third", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Draw take-ons
        for _, t in middle_takeons.iterrows():
            x, y = t['x'], t['y']
            if not is_home:
                x, y = 100 - x, 100 - y

            marker_color = '#2ecc71' if t['is_successful'] else '#e74c3c'
            ax.scatter(x, y, s=100, c=marker_color, marker='o',
                       edgecolors='white', linewidths=1.5, alpha=0.8, zorder=10)

        success_rate = middle_takeons['is_successful'].mean() * 100
        ax.set_title(f"{team_name}\nTake-Ons ({len(middle_takeons)}) - {success_rate:.0f}% Success",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_middle_stats(self, ax: plt.Axes):
        """Draw middle progression statistics"""
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis('off')

        # Calculate stats
        home_middle = self.home_passes[(self.home_passes['x'] >= 33.33) & (self.home_passes['x'] < 66.66)]
        away_middle = self.away_passes[(self.away_passes['x'] >= 33.33) & (self.away_passes['x'] < 66.66)]

        home_ft_entries = len(home_middle[(home_middle['end_x'] >= 66.66) & (home_middle['is_successful'] == True)])
        away_ft_entries = len(away_middle[(away_middle['end_x'] >= 66.66) & (away_middle['is_successful'] == True)])

        home_takeons = self.home_take_ons[(self.home_take_ons['x'] >= 33.33) & (self.home_take_ons['x'] < 66.66)]
        away_takeons = self.away_take_ons[(self.away_take_ons['x'] >= 33.33) & (self.away_take_ons['x'] < 66.66)]

        stats = [
            (len(home_middle), len(away_middle), "Middle Third Passes"),
            (home_ft_entries, away_ft_entries, "Final Third Entries"),
            (len(home_takeons), len(away_takeons), "Take-Ons Attempted"),
        ]

        y_start = 0.7
        y_step = 0.18

        ax.text(0.5, 0.95, "MIDDLE PROGRESSION STATISTICS", fontsize=14, fontweight='bold',
                color=TEXT_COLOR, ha='center', va='center', transform=ax.transAxes)

        # Team headers (with more space before first stat)
        ax.text(0.2, 0.85, self.match_info.home_team_name, fontsize=11,
                color=self.colors.home, ha='center', va='center', transform=ax.transAxes, fontweight='bold')
        ax.text(0.8, 0.85, self.match_info.away_team_name, fontsize=11,
                color=self.colors.away, ha='center', va='center', transform=ax.transAxes, fontweight='bold')

        for i, (home_val, away_val, label) in enumerate(stats):
            y = y_start - i * y_step
            ax.text(0.2, y, str(home_val), fontsize=14, color=self.colors.home,
                    ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.text(0.5, y, label, fontsize=10, color=TEXT_COLOR,
                    ha='center', va='center', transform=ax.transAxes)
            ax.text(0.8, y, str(away_val), fontsize=14, color=self.colors.away,
                    ha='center', va='center', transform=ax.transAxes, fontweight='bold')


class FinalThirdPhasePage(PhasePage):
    """
    Final Third Phase Analysis (Attacking Third - x: 66.66-100)

    Visualizations:
    1. Passes in final third
    2. Zone 14 and half-space entries
    3. Crosses and cutbacks
    4. Shot map and xG
    5. Key stats: passes, crosses, shots, xG
    """

    def generate(self, save_path: str = None) -> plt.Figure:
        fig = plt.figure(figsize=(20, 28), facecolor=BACKGROUND_COLOR)

        gs = GridSpec(6, 2, figure=fig, height_ratios=[0.8, 3, 3, 3, 2, 0.5],
                      hspace=0.15, wspace=0.1)

        # Header
        ax_header = fig.add_subplot(gs[0, :])
        self._draw_header(ax_header, "FINAL THIRD PHASE",
                         "In Possession - Attacking Third (66-100%)")

        # Home final third map
        ax_home_ft = fig.add_subplot(gs[1, 0])
        self._draw_final_third_map(ax_home_ft, is_home=True)

        # Away final third map
        ax_away_ft = fig.add_subplot(gs[1, 1])
        self._draw_final_third_map(ax_away_ft, is_home=False)

        # Home crosses and cutbacks
        ax_home_cross = fig.add_subplot(gs[2, 0])
        self._draw_crosses_map(ax_home_cross, is_home=True)

        # Away crosses and cutbacks
        ax_away_cross = fig.add_subplot(gs[2, 1])
        self._draw_crosses_map(ax_away_cross, is_home=False)

        # Home shot map
        ax_home_shots = fig.add_subplot(gs[3, 0])
        self._draw_shot_map(ax_home_shots, is_home=True)

        # Away shot map
        ax_away_shots = fig.add_subplot(gs[3, 1])
        self._draw_shot_map(ax_away_shots, is_home=False)

        # Stats row
        ax_stats = fig.add_subplot(gs[4, :])
        self._draw_final_third_stats(ax_stats)

        # Footer
        ax_footer = fig.add_subplot(gs[5, :])
        self._draw_footer(ax_footer)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())

        return fig

    def _draw_final_third_map(self, ax: plt.Axes, is_home: bool):
        """Draw passes in the final third (vertical orientation, goal at top)"""
        self._draw_vertical_pitch(ax, third='attacking', goal_at_top=True)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        passes = self.home_passes if is_home else self.away_passes

        # Filter passes in final third
        ft_passes = passes[
            (passes['x'] >= 66.66) &
            (passes['end_x'].notna())
        ].copy()

        if len(ft_passes) == 0:
            ax.text(50, 25, "No data", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nFinal Third Passes", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Draw passes with vertical transformation (goal at top)
        # Half-pitch: x=50 -> y=0 (halfway), x=100 -> y=50 (goal at top)
        for _, p in ft_passes.iterrows():
            x, y = p['x'], p['y']
            end_x, end_y = p['end_x'], p['end_y']

            # Transform to vertical half-pitch: y becomes new_x, x maps to new_y
            new_x = y
            new_y = x - 50  # x: 50-100 -> y: 0-50
            new_end_x = end_y
            new_end_y = max(0, end_x - 50)

            alpha = 0.6 if p['is_successful'] else 0.2
            ax.annotate('', xy=(new_end_x, new_end_y), xytext=(new_x, new_y),
                        arrowprops=dict(arrowstyle='->', color=color, alpha=alpha, lw=0.8))

        success_rate = ft_passes['is_successful'].mean() * 100
        ax.set_title(f"{team_name}\nFinal Third Passes ({len(ft_passes)}) - {success_rate:.0f}%",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_crosses_map(self, ax: plt.Axes, is_home: bool):
        """Draw crosses from wide areas (vertical orientation, goal at top)"""
        self._draw_vertical_pitch(ax, third='attacking', goal_at_top=True)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        passes = self.home_passes if is_home else self.away_passes

        # Filter crosses
        crosses = passes[
            (passes['is_cross'] == True) &
            (passes['end_x'].notna())
        ].copy()

        if len(crosses) == 0:
            ax.text(50, 25, "No crosses", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nCrosses", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Highlight penalty box (rotated for half-pitch: y=33 to y=50)
        rect = Rectangle((21.1, 33), 57.8, 17, fill=True, facecolor=color, alpha=0.1)
        ax.add_patch(rect)

        # Draw crosses with vertical transformation (half-pitch)
        for _, p in crosses.iterrows():
            x, y = p['x'], p['y']
            end_x, end_y = p['end_x'], p['end_y']

            # Transform to vertical half-pitch: x=50->y=0, x=100->y=50
            new_x = y
            new_y = max(0, x - 50)
            new_end_x = end_y
            new_end_y = max(0, end_x - 50)

            cross_color = '#2ecc71' if p['is_successful'] else '#e74c3c'
            ax.annotate('', xy=(new_end_x, new_end_y), xytext=(new_x, new_y),
                        arrowprops=dict(arrowstyle='->', color=cross_color, alpha=0.7, lw=1.5))

        success_rate = crosses['is_successful'].mean() * 100
        ax.set_title(f"{team_name}\nCrosses ({len(crosses)}) - {success_rate:.0f}% Accurate",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_shot_map(self, ax: plt.Axes, is_home: bool):
        """Draw shot map with xG (vertical orientation, goal at top)"""
        self._draw_vertical_pitch(ax, third='attacking', goal_at_top=True)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        shots = self.home_shots if is_home else self.away_shots

        if len(shots) == 0:
            ax.text(50, 25, "No shots", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nShot Map", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Draw shots with vertical transformation (half-pitch)
        for _, shot in shots.iterrows():
            x, y = shot['x'], shot['y']

            # Transform to vertical half-pitch: x=50->y=0, x=100->y=50
            new_x = y
            new_y = max(0, x - 50)

            size = shot.get('xG', 0.1) * 600 + 80

            if shot.get('is_goal', False):
                marker = '*'
                marker_color = '#2ecc71'
                size *= 1.5
            elif shot.get('is_on_target', False):
                marker = 'o'
                marker_color = '#f39c12'
            else:
                marker = 'o'
                marker_color = '#95a5a6'

            ax.scatter(new_x, new_y, s=size, c=marker_color, marker=marker,
                       edgecolors='white', linewidths=1.5, alpha=0.8, zorder=10)

        total_xg = shots['xG'].sum()
        goals = shots['is_goal'].sum()
        ax.set_title(f"{team_name}\nShots ({len(shots)}) - {goals} Goals - xG: {total_xg:.2f}",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_final_third_stats(self, ax: plt.Axes):
        """Draw final third statistics"""
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis('off')

        # Calculate stats
        home_ft = self.home_passes[self.home_passes['x'] >= 66.66]
        away_ft = self.away_passes[self.away_passes['x'] >= 66.66]

        home_crosses = self.home_passes['is_cross'].sum()
        away_crosses = self.away_passes['is_cross'].sum()

        home_xg = self.home_shots['xG'].sum()
        away_xg = self.away_shots['xG'].sum()

        home_big_chances = self.home_shots['is_big_chance'].sum() if 'is_big_chance' in self.home_shots.columns else 0
        away_big_chances = self.away_shots['is_big_chance'].sum() if 'is_big_chance' in self.away_shots.columns else 0

        stats = [
            (len(home_ft), len(away_ft), "Final Third Passes"),
            (home_crosses, away_crosses, "Crosses"),
            (len(self.home_shots), len(self.away_shots), "Shots"),
            (f"{home_xg:.2f}", f"{away_xg:.2f}", "Expected Goals (xG)"),
            (home_big_chances, away_big_chances, "Big Chances"),
        ]

        y_start = 0.72
        y_step = 0.15

        ax.text(0.5, 0.98, "FINAL THIRD STATISTICS", fontsize=14, fontweight='bold',
                color=TEXT_COLOR, ha='center', va='center', transform=ax.transAxes)

        # Team headers (with more space before first stat)
        ax.text(0.2, 0.88, self.match_info.home_team_name, fontsize=11,
                color=self.colors.home, ha='center', va='center', transform=ax.transAxes, fontweight='bold')
        ax.text(0.8, 0.88, self.match_info.away_team_name, fontsize=11,
                color=self.colors.away, ha='center', va='center', transform=ax.transAxes, fontweight='bold')

        for i, (home_val, away_val, label) in enumerate(stats):
            y = y_start - i * y_step
            ax.text(0.2, y, str(home_val), fontsize=14, color=self.colors.home,
                    ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.text(0.5, y, label, fontsize=10, color=TEXT_COLOR,
                    ha='center', va='center', transform=ax.transAxes)
            ax.text(0.8, y, str(away_val), fontsize=14, color=self.colors.away,
                    ha='center', va='center', transform=ax.transAxes, fontweight='bold')


class HighPressPhasePage(PhasePage):
    """
    High Press Phase Analysis (Out of Possession - Opponent's Third)

    Visualizations:
    1. High press triggers (where pressing starts)
    2. Defensive actions in opponent's third
    3. Ball recoveries high up the pitch
    4. PPDA and pressing intensity
    5. Key stats: high recoveries, tackles won, pressing success
    """

    def generate(self, save_path: str = None) -> plt.Figure:
        fig = plt.figure(figsize=(20, 28), facecolor=BACKGROUND_COLOR)

        gs = GridSpec(6, 2, figure=fig, height_ratios=[0.8, 3, 3, 3, 2, 0.5],
                      hspace=0.15, wspace=0.1)

        # Header
        ax_header = fig.add_subplot(gs[0, :])
        self._draw_header(ax_header, "HIGH PRESS PHASE",
                         "Out of Possession - High Block / Pressing")

        # Home high press map
        ax_home_press = fig.add_subplot(gs[1, 0])
        self._draw_high_press_map(ax_home_press, is_home=True)

        # Away high press map
        ax_away_press = fig.add_subplot(gs[1, 1])
        self._draw_high_press_map(ax_away_press, is_home=False)

        # Home high recoveries
        ax_home_rec = fig.add_subplot(gs[2, 0])
        self._draw_high_recoveries(ax_home_rec, is_home=True)

        # Away high recoveries
        ax_away_rec = fig.add_subplot(gs[2, 1])
        self._draw_high_recoveries(ax_away_rec, is_home=False)

        # Home pressing heatmap
        ax_home_heat = fig.add_subplot(gs[3, 0])
        self._draw_pressing_heatmap(ax_home_heat, is_home=True)

        # Away pressing heatmap
        ax_away_heat = fig.add_subplot(gs[3, 1])
        self._draw_pressing_heatmap(ax_away_heat, is_home=False)

        # Stats row
        ax_stats = fig.add_subplot(gs[4, :])
        self._draw_high_press_stats(ax_stats)

        # Footer
        ax_footer = fig.add_subplot(gs[5, :])
        self._draw_footer(ax_footer)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())

        return fig

    def _draw_high_press_map(self, ax: plt.Axes, is_home: bool):
        """Draw defensive actions in opponent's third (high press) - vertical orientation, goal at top"""
        self._draw_vertical_pitch(ax, third='attacking', goal_at_top=True)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        defensive = self.home_defensive if is_home else self.away_defensive

        # High press = defensive actions in opponent's third (x > 66.66 in Opta coords)
        high_press = defensive[defensive['x'] > 66.66].copy()

        # Highlight the final third zone (y: 33.33-50 on half-pitch scale)
        rect = Rectangle((0, 33.33), 100, 16.67, fill=True, facecolor=color, alpha=0.1)
        ax.add_patch(rect)

        if len(high_press) == 0:
            ax.text(50, 25, "No high press actions", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nHigh Press Actions", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Plot by action type with vertical transformation (half-pitch)
        markers = {'Tackle': 'o', 'Interception': 's', 'Clearance': '^'}
        colors_map = {'Tackle': '#3498db', 'Interception': '#e74c3c', 'Clearance': '#f39c12'}

        for action_type in high_press['action_type'].unique():
            mask = high_press['action_type'] == action_type
            x_vals = high_press.loc[mask, 'x'].values
            y_vals = high_press.loc[mask, 'y'].values

            # Transform to vertical half-pitch: x=50->y=0, x=100->y=50
            new_x_vals = y_vals  # y becomes x
            new_y_vals = x_vals - 50  # x: 50-100 -> y: 0-50

            ax.scatter(new_x_vals, new_y_vals,
                       marker=markers.get(action_type, 'o'),
                       c=colors_map.get(action_type, color),
                       s=80, alpha=0.7, edgecolors='white', linewidths=0.8, zorder=10,
                       label=action_type)

        ax.legend(loc='upper left', fontsize=8)
        ax.set_title(f"{team_name}\nHigh Press Actions ({len(high_press)})",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_high_recoveries(self, ax: plt.Axes, is_home: bool):
        """Draw ball recoveries in the attacking third (vertical orientation, goal at top)"""
        self._draw_vertical_pitch(ax, third='attacking', goal_at_top=True)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        recoveries = self.home_recoveries if is_home else self.away_recoveries

        # High recoveries = x > 66.66
        high_recoveries = recoveries[recoveries['x'] > 66.66].copy()

        # Highlight the final third zone (y: 33.33-50 on half-pitch scale)
        rect = Rectangle((0, 33.33), 100, 16.67, fill=True, facecolor='#2ecc71', alpha=0.1)
        ax.add_patch(rect)

        if len(high_recoveries) == 0:
            ax.text(50, 25, "No high recoveries", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nHigh Recoveries", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Draw recoveries with vertical transformation (half-pitch)
        for _, r in high_recoveries.iterrows():
            x, y = r['x'], r['y']

            # Transform to vertical half-pitch: x=50->y=0, x=100->y=50
            new_x = y
            new_y = x - 50

            ax.scatter(new_x, new_y, s=100, c='#2ecc71', marker='o',
                       edgecolors='white', linewidths=1.5, alpha=0.8, zorder=10)

        ax.set_title(f"{team_name}\nHigh Recoveries ({len(high_recoveries)})",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_pressing_heatmap(self, ax: plt.Axes, is_home: bool):
        """Draw heatmap of all defensive actions"""
        self._draw_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        defensive = self.home_defensive if is_home else self.away_defensive

        if len(defensive) == 0:
            ax.text(50, 50, "No data", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nDefensive Action Heatmap", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        x_vals = defensive['x'].values.copy()
        y_vals = defensive['y'].values.copy()

        if not is_home:
            x_vals = 100 - x_vals
            y_vals = 100 - y_vals

        # Create heatmap
        heatmap, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=[10, 10],
                                                   range=[[0, 100], [0, 100]])

        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list('custom', ['#f5f5f0', color], N=100)

        extent = [0, 100, 0, 100]
        ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap, alpha=0.6, aspect='auto')

        # Calculate average defensive line
        avg_x = x_vals.mean()
        ax.axvline(x=avg_x, color=TEXT_COLOR, linestyle='--', alpha=0.7, lw=2)
        ax.text(avg_x + 2, 95, f'Avg: {avg_x:.0f}', fontsize=8, color=TEXT_COLOR)

        ax.set_title(f"{team_name}\nDefensive Action Heatmap",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_high_press_stats(self, ax: plt.Axes):
        """Draw high press statistics"""
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis('off')

        # Calculate stats
        home_high_def = self.home_defensive[self.home_defensive['x'] > 66.66]
        away_high_def = self.away_defensive[self.away_defensive['x'] > 66.66]

        home_high_rec = len(self.home_recoveries[self.home_recoveries['x'] > 66.66])
        away_high_rec = len(self.away_recoveries[self.away_recoveries['x'] > 66.66])

        home_tackles_high = len(home_high_def[home_high_def['action_type'] == 'Tackle'])
        away_tackles_high = len(away_high_def[away_high_def['action_type'] == 'Tackle'])

        home_int_high = len(home_high_def[home_high_def['action_type'] == 'Interception'])
        away_int_high = len(away_high_def[away_high_def['action_type'] == 'Interception'])

        # Calculate PPDA (opponent passes / team defensive actions)
        away_passes_own_half = len(self.away_passes[self.away_passes['x'] < 50])
        home_passes_own_half = len(self.home_passes[self.home_passes['x'] < 50])
        home_def_opp_half = len(self.home_defensive[self.home_defensive['x'] > 50])
        away_def_opp_half = len(self.away_defensive[self.away_defensive['x'] > 50])

        home_ppda = away_passes_own_half / home_def_opp_half if home_def_opp_half > 0 else 0
        away_ppda = home_passes_own_half / away_def_opp_half if away_def_opp_half > 0 else 0

        stats = [
            (len(home_high_def), len(away_high_def), "High Press Actions"),
            (home_high_rec, away_high_rec, "High Recoveries"),
            (home_tackles_high, away_tackles_high, "High Tackles"),
            (home_int_high, away_int_high, "High Interceptions"),
            (f"{home_ppda:.1f}", f"{away_ppda:.1f}", "PPDA (lower = more intense)"),
        ]

        y_start = 0.72
        y_step = 0.15

        ax.text(0.5, 0.98, "HIGH PRESS STATISTICS", fontsize=14, fontweight='bold',
                color=TEXT_COLOR, ha='center', va='center', transform=ax.transAxes)

        # Team headers (with more space before first stat)
        ax.text(0.2, 0.88, self.match_info.home_team_name, fontsize=11,
                color=self.colors.home, ha='center', va='center', transform=ax.transAxes, fontweight='bold')
        ax.text(0.8, 0.88, self.match_info.away_team_name, fontsize=11,
                color=self.colors.away, ha='center', va='center', transform=ax.transAxes, fontweight='bold')

        for i, (home_val, away_val, label) in enumerate(stats):
            y = y_start - i * y_step
            ax.text(0.2, y, str(home_val), fontsize=14, color=self.colors.home,
                    ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.text(0.5, y, label, fontsize=10, color=TEXT_COLOR,
                    ha='center', va='center', transform=ax.transAxes)
            ax.text(0.8, y, str(away_val), fontsize=14, color=self.colors.away,
                    ha='center', va='center', transform=ax.transAxes, fontweight='bold')


class MidLowBlockPhasePage(PhasePage):
    """
    Mid/Low Block Phase Analysis (Out of Possession - Own Half)

    Visualizations:
    1. Defensive actions in own half
    2. Defensive line position
    3. Blocks and clearances
    4. Recovery positions
    5. Key stats: clearances, blocks, tackles in own half
    """

    def generate(self, save_path: str = None) -> plt.Figure:
        fig = plt.figure(figsize=(20, 28), facecolor=BACKGROUND_COLOR)

        gs = GridSpec(6, 2, figure=fig, height_ratios=[0.8, 3, 3, 3, 2, 0.5],
                      hspace=0.15, wspace=0.1)

        # Header
        ax_header = fig.add_subplot(gs[0, :])
        self._draw_header(ax_header, "MID / LOW BLOCK PHASE",
                         "Out of Possession - Defensive Shape")

        # Home defensive shape
        ax_home_def = fig.add_subplot(gs[1, 0])
        self._draw_defensive_shape(ax_home_def, is_home=True)

        # Away defensive shape
        ax_away_def = fig.add_subplot(gs[1, 1])
        self._draw_defensive_shape(ax_away_def, is_home=False)

        # Home clearances and blocks
        ax_home_clear = fig.add_subplot(gs[2, 0])
        self._draw_clearances_blocks(ax_home_clear, is_home=True)

        # Away clearances and blocks
        ax_away_clear = fig.add_subplot(gs[2, 1])
        self._draw_clearances_blocks(ax_away_clear, is_home=False)

        # Home defensive third recoveries
        ax_home_rec = fig.add_subplot(gs[3, 0])
        self._draw_defensive_recoveries(ax_home_rec, is_home=True)

        # Away defensive third recoveries
        ax_away_rec = fig.add_subplot(gs[3, 1])
        self._draw_defensive_recoveries(ax_away_rec, is_home=False)

        # Stats row
        ax_stats = fig.add_subplot(gs[4, :])
        self._draw_block_stats(ax_stats)

        # Footer
        ax_footer = fig.add_subplot(gs[5, :])
        self._draw_footer(ax_footer)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())

        return fig

    def _draw_defensive_shape(self, ax: plt.Axes, is_home: bool):
        """Draw defensive actions in own half showing shape"""
        self._draw_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        defensive = self.home_defensive if is_home else self.away_defensive

        # Own half = x < 50
        own_half_def = defensive[defensive['x'] < 50].copy()

        # Highlight own half
        if is_home:
            rect = Rectangle((0, 0), 50, 100, fill=True, facecolor=color, alpha=0.05)
        else:
            rect = Rectangle((50, 0), 50, 100, fill=True, facecolor=color, alpha=0.05)
        ax.add_patch(rect)

        if len(own_half_def) == 0:
            ax.text(50, 50, "No data", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nDefensive Shape", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        x_vals = own_half_def['x'].values.copy()
        y_vals = own_half_def['y'].values.copy()

        if not is_home:
            x_vals = 100 - x_vals
            y_vals = 100 - y_vals

        # Plot by action type
        markers = {'Tackle': 'o', 'Interception': 's', 'Clearance': '^', 'Save': 'D'}
        colors_map = {'Tackle': '#3498db', 'Interception': '#e74c3c',
                      'Clearance': '#f39c12', 'Save': '#2ecc71'}

        for action_type in own_half_def['action_type'].unique():
            mask = own_half_def['action_type'] == action_type
            ax_x = x_vals[mask]
            ax_y = y_vals[mask]

            ax.scatter(ax_x, ax_y,
                       marker=markers.get(action_type, 'o'),
                       c=colors_map.get(action_type, color),
                       s=70, alpha=0.7, edgecolors='white', linewidths=0.8, zorder=10,
                       label=action_type)

        # Draw defensive line
        avg_def_line = x_vals.mean()
        ax.axvline(x=avg_def_line, color=color, linestyle='-', alpha=0.8, lw=2)
        ax.text(avg_def_line + 2, 5, f'Def Line: {avg_def_line:.0f}', fontsize=8, color=color, fontweight='bold')

        ax.legend(loc='upper right', fontsize=7)
        ax.set_title(f"{team_name}\nDefensive Actions - Own Half ({len(own_half_def)})",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_clearances_blocks(self, ax: plt.Axes, is_home: bool):
        """Draw clearances in the defensive third (vertical half-pitch, goal at bottom)"""
        self._draw_vertical_pitch(ax, third='defensive', goal_at_top=False)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        defensive = self.home_defensive if is_home else self.away_defensive

        # Highlight the defensive third zone (y: 0-33.33 on half-pitch, goal at y=0)
        rect = Rectangle((0, 0), 100, 33.33, fill=True, facecolor='#f39c12', alpha=0.1)
        ax.add_patch(rect)

        # Clearances in defensive third
        clearances = defensive[
            (defensive['action_type'] == 'Clearance') &
            (defensive['x'] < 33.33)
        ].copy()

        if len(clearances) == 0:
            ax.text(50, 25, "No clearances", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nClearances in Defensive Third", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Draw clearances with vertical transformation (goal at bottom: x=0->y=0, x=50->y=50)
        for _, c in clearances.iterrows():
            x, y = c['x'], c['y']

            # Transform to vertical half-pitch (goal at bottom)
            new_x = y
            new_y = x  # x: 0-50 maps to y: 0-50

            ax.scatter(new_x, new_y, s=100, c='#f39c12', marker='^',
                       edgecolors='white', linewidths=1.5, alpha=0.8, zorder=10)

        ax.set_title(f"{team_name}\nClearances ({len(clearances)})",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_defensive_recoveries(self, ax: plt.Axes, is_home: bool):
        """Draw recoveries in the defensive third (vertical half-pitch, goal at bottom)"""
        self._draw_vertical_pitch(ax, third='defensive', goal_at_top=False)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        recoveries = self.home_recoveries if is_home else self.away_recoveries

        # Highlight the defensive third zone (y: 0-33.33 on half-pitch, goal at y=0)
        rect = Rectangle((0, 0), 100, 33.33, fill=True, facecolor=color, alpha=0.1)
        ax.add_patch(rect)

        # Defensive third recoveries
        def_recoveries = recoveries[recoveries['x'] < 33.33].copy()

        if len(def_recoveries) == 0:
            ax.text(50, 25, "No recoveries", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nDefensive Third Recoveries", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Draw recoveries with vertical transformation (goal at bottom: x=0->y=0, x=50->y=50)
        for _, r in def_recoveries.iterrows():
            x, y = r['x'], r['y']

            # Transform to vertical half-pitch (goal at bottom)
            new_x = y
            new_y = x  # x: 0-50 maps to y: 0-50

            ax.scatter(new_x, new_y, s=80, c=color, marker='o',
                       edgecolors='white', linewidths=1.2, alpha=0.7, zorder=10)

        ax.set_title(f"{team_name}\nDefensive Third Recoveries ({len(def_recoveries)})",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_block_stats(self, ax: plt.Axes):
        """Draw mid/low block statistics"""
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis('off')

        # Calculate stats
        home_def_own = self.home_defensive[self.home_defensive['x'] < 50]
        away_def_own = self.away_defensive[self.away_defensive['x'] < 50]

        home_clearances = len(home_def_own[home_def_own['action_type'] == 'Clearance'])
        away_clearances = len(away_def_own[away_def_own['action_type'] == 'Clearance'])

        home_tackles_own = len(home_def_own[home_def_own['action_type'] == 'Tackle'])
        away_tackles_own = len(away_def_own[away_def_own['action_type'] == 'Tackle'])

        home_int_own = len(home_def_own[home_def_own['action_type'] == 'Interception'])
        away_int_own = len(away_def_own[away_def_own['action_type'] == 'Interception'])

        home_def_rec = len(self.home_recoveries[self.home_recoveries['x'] < 33.33])
        away_def_rec = len(self.away_recoveries[self.away_recoveries['x'] < 33.33])

        # Calculate average defensive line
        home_def_line = home_def_own['x'].mean() if len(home_def_own) > 0 else 0
        away_def_line = away_def_own['x'].mean() if len(away_def_own) > 0 else 0

        stats = [
            (len(home_def_own), len(away_def_own), "Own Half Defensive Actions"),
            (home_clearances, away_clearances, "Clearances"),
            (home_tackles_own, away_tackles_own, "Tackles in Own Half"),
            (home_int_own, away_int_own, "Interceptions in Own Half"),
            (f"{home_def_line:.0f}", f"{away_def_line:.0f}", "Avg Defensive Line"),
        ]

        y_start = 0.72
        y_step = 0.15

        ax.text(0.5, 0.98, "MID/LOW BLOCK STATISTICS", fontsize=14, fontweight='bold',
                color=TEXT_COLOR, ha='center', va='center', transform=ax.transAxes)

        # Team headers (with more space before first stat)
        ax.text(0.2, 0.88, self.match_info.home_team_name, fontsize=11,
                color=self.colors.home, ha='center', va='center', transform=ax.transAxes, fontweight='bold')
        ax.text(0.8, 0.88, self.match_info.away_team_name, fontsize=11,
                color=self.colors.away, ha='center', va='center', transform=ax.transAxes, fontweight='bold')

        for i, (home_val, away_val, label) in enumerate(stats):
            y = y_start - i * y_step
            ax.text(0.2, y, str(home_val), fontsize=14, color=self.colors.home,
                    ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.text(0.5, y, label, fontsize=10, color=TEXT_COLOR,
                    ha='center', va='center', transform=ax.transAxes)
            ax.text(0.8, y, str(away_val), fontsize=14, color=self.colors.away,
                    ha='center', va='center', transform=ax.transAxes, fontweight='bold')


class PositiveTransitionPage(PhasePage):
    """
    Positive Transition Phase (Winning the Ball -> Attack)

    Visualizations:
    1. Ball recovery locations
    2. Counter-attack passes (fast progression after recovery)
    3. Shots from counter-attacks
    4. Key stats: recoveries leading to shots, counter-attack xG
    """

    def generate(self, save_path: str = None) -> plt.Figure:
        fig = plt.figure(figsize=(20, 28), facecolor=BACKGROUND_COLOR)

        gs = GridSpec(6, 2, figure=fig, height_ratios=[0.8, 3, 3, 3, 2, 0.5],
                      hspace=0.15, wspace=0.1)

        # Header
        ax_header = fig.add_subplot(gs[0, :])
        self._draw_header(ax_header, "POSITIVE TRANSITIONS",
                         "Winning Possession -> Attacking")

        # Home recovery locations
        ax_home_rec = fig.add_subplot(gs[1, 0])
        self._draw_recovery_locations(ax_home_rec, is_home=True)

        # Away recovery locations
        ax_away_rec = fig.add_subplot(gs[1, 1])
        self._draw_recovery_locations(ax_away_rec, is_home=False)

        # Home counter-attack progressions
        ax_home_counter = fig.add_subplot(gs[2, 0])
        self._draw_counter_progressions(ax_home_counter, is_home=True)

        # Away counter-attack progressions
        ax_away_counter = fig.add_subplot(gs[2, 1])
        self._draw_counter_progressions(ax_away_counter, is_home=False)

        # Home direct play
        ax_home_direct = fig.add_subplot(gs[3, 0])
        self._draw_direct_play(ax_home_direct, is_home=True)

        # Away direct play
        ax_away_direct = fig.add_subplot(gs[3, 1])
        self._draw_direct_play(ax_away_direct, is_home=False)

        # Stats row
        ax_stats = fig.add_subplot(gs[4, :])
        self._draw_transition_stats(ax_stats)

        # Footer
        ax_footer = fig.add_subplot(gs[5, :])
        self._draw_footer(ax_footer)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())

        return fig

    def _draw_recovery_locations(self, ax: plt.Axes, is_home: bool):
        """Draw ball recovery locations by pitch third"""
        self._draw_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        recoveries = self.home_recoveries if is_home else self.away_recoveries

        if len(recoveries) == 0:
            ax.text(50, 50, "No recoveries", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nBall Recoveries", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Color by pitch third
        for _, r in recoveries.iterrows():
            x, y = r['x'], r['y']

            # Determine third
            if x < 33.33:
                marker_color = '#3498db'  # Defensive
            elif x < 66.66:
                marker_color = '#f39c12'  # Middle
            else:
                marker_color = '#2ecc71'  # Attacking

            if not is_home:
                x, y = 100 - x, 100 - y

            ax.scatter(x, y, s=60, c=marker_color, marker='o',
                       edgecolors='white', linewidths=0.8, alpha=0.7, zorder=10)

        # Legend
        ax.scatter([], [], c='#3498db', label='Defensive', s=50)
        ax.scatter([], [], c='#f39c12', label='Middle', s=50)
        ax.scatter([], [], c='#2ecc71', label='Attacking', s=50)
        ax.legend(loc='upper left', fontsize=8)

        ax.set_title(f"{team_name}\nBall Recoveries ({len(recoveries)})",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_counter_progressions(self, ax: plt.Axes, is_home: bool):
        """Draw progressive passes (potential counter-attacks)"""
        self._draw_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        passes = self.home_passes if is_home else self.away_passes

        # Progressive passes = gain at least 25 units towards goal
        prog_passes = passes[
            (passes['is_successful'] == True) &
            (passes['end_x'].notna()) &
            ((passes['end_x'] - passes['x']) >= 25)
        ].copy()

        if len(prog_passes) == 0:
            ax.text(50, 50, "No fast progressions", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nFast Progressions", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Draw passes
        for _, p in prog_passes.iterrows():
            x, y = p['x'], p['y']
            end_x, end_y = p['end_x'], p['end_y']

            if not is_home:
                x, y = 100 - x, 100 - y
                end_x, end_y = 100 - end_x, 100 - end_y

            progress = abs(end_x - x) if is_home else abs(x - end_x)
            width = min(3, progress / 15)

            ax.annotate('', xy=(end_x, end_y), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color=color, alpha=0.7, lw=width))

        avg_progress = (prog_passes['end_x'] - prog_passes['x']).mean()
        ax.set_title(f"{team_name}\nFast Progressions ({len(prog_passes)}) - Avg {avg_progress:.0f}m",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_direct_play(self, ax: plt.Axes, is_home: bool):
        """Draw direct play - long balls"""
        self._draw_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        passes = self.home_passes if is_home else self.away_passes

        # Direct play = long balls
        direct_passes = passes[
            (passes['is_long_ball'] == True) &
            (passes['end_x'].notna())
        ].copy()

        if len(direct_passes) == 0:
            ax.text(50, 50, "No direct passes", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nDirect Play", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Draw passes
        for _, p in direct_passes.iterrows():
            x, y = p['x'], p['y']
            end_x, end_y = p['end_x'], p['end_y']

            if not is_home:
                x, y = 100 - x, 100 - y
                end_x, end_y = 100 - end_x, 100 - end_y

            alpha = 0.8 if p['is_successful'] else 0.3

            ax.annotate('', xy=(end_x, end_y), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color='#3498db', alpha=alpha, lw=1.5))

        success_rate = direct_passes['is_successful'].mean() * 100
        ax.set_title(f"{team_name}\nDirect Play ({len(direct_passes)}) - {success_rate:.0f}% Success",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_transition_stats(self, ax: plt.Axes):
        """Draw positive transition statistics"""
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis('off')

        # Calculate stats
        home_recoveries = len(self.home_recoveries)
        away_recoveries = len(self.away_recoveries)

        home_high_rec = len(self.home_recoveries[self.home_recoveries['x'] > 66.66])
        away_high_rec = len(self.away_recoveries[self.away_recoveries['x'] > 66.66])

        home_prog = len(self.home_passes[
            (self.home_passes['is_successful'] == True) &
            ((self.home_passes['end_x'] - self.home_passes['x']) >= 25)
        ])
        away_prog = len(self.away_passes[
            (self.away_passes['is_successful'] == True) &
            ((self.away_passes['end_x'] - self.away_passes['x']) >= 25)
        ])

        home_long = self.home_passes['is_long_ball'].sum()
        away_long = self.away_passes['is_long_ball'].sum()

        stats = [
            (home_recoveries, away_recoveries, "Ball Recoveries"),
            (home_high_rec, away_high_rec, "High Recoveries (Att 3rd)"),
            (home_prog, away_prog, "Fast Progressions (25m+)"),
            (home_long, away_long, "Long Balls"),
        ]

        y_start = 0.72
        y_step = 0.15

        ax.text(0.5, 0.98, "POSITIVE TRANSITION STATISTICS", fontsize=14, fontweight='bold',
                color=TEXT_COLOR, ha='center', va='center', transform=ax.transAxes)

        # Team headers (with more space before first stat)
        ax.text(0.2, 0.88, self.match_info.home_team_name, fontsize=11,
                color=self.colors.home, ha='center', va='center', transform=ax.transAxes, fontweight='bold')
        ax.text(0.8, 0.88, self.match_info.away_team_name, fontsize=11,
                color=self.colors.away, ha='center', va='center', transform=ax.transAxes, fontweight='bold')

        for i, (home_val, away_val, label) in enumerate(stats):
            y = y_start - i * y_step
            ax.text(0.2, y, str(home_val), fontsize=14, color=self.colors.home,
                    ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.text(0.5, y, label, fontsize=10, color=TEXT_COLOR,
                    ha='center', va='center', transform=ax.transAxes)
            ax.text(0.8, y, str(away_val), fontsize=14, color=self.colors.away,
                    ha='center', va='center', transform=ax.transAxes, fontweight='bold')


class NegativeTransitionPage(PhasePage):
    """
    Negative Transition Phase (Losing the Ball -> Defending)

    Visualizations:
    1. Ball loss locations
    2. Counter-press actions (defensive actions after losing ball)
    3. Fouls to stop counters
    4. Key stats: ball losses, counter-press success
    """

    def generate(self, save_path: str = None) -> plt.Figure:
        fig = plt.figure(figsize=(20, 28), facecolor=BACKGROUND_COLOR)

        gs = GridSpec(6, 2, figure=fig, height_ratios=[0.8, 3, 3, 3, 2, 0.5],
                      hspace=0.15, wspace=0.1)

        # Header
        ax_header = fig.add_subplot(gs[0, :])
        self._draw_header(ax_header, "NEGATIVE TRANSITIONS",
                         "Losing Possession -> Defending")

        # Home ball losses
        ax_home_loss = fig.add_subplot(gs[1, 0])
        self._draw_ball_losses(ax_home_loss, is_home=True)

        # Away ball losses
        ax_away_loss = fig.add_subplot(gs[1, 1])
        self._draw_ball_losses(ax_away_loss, is_home=False)

        # Home counter-press
        ax_home_cpress = fig.add_subplot(gs[2, 0])
        self._draw_counter_press(ax_home_cpress, is_home=True)

        # Away counter-press
        ax_away_cpress = fig.add_subplot(gs[2, 1])
        self._draw_counter_press(ax_away_cpress, is_home=False)

        # Home tactical fouls
        ax_home_fouls = fig.add_subplot(gs[3, 0])
        self._draw_fouls_map(ax_home_fouls, is_home=True)

        # Away tactical fouls
        ax_away_fouls = fig.add_subplot(gs[3, 1])
        self._draw_fouls_map(ax_away_fouls, is_home=False)

        # Stats row
        ax_stats = fig.add_subplot(gs[4, :])
        self._draw_negative_transition_stats(ax_stats)

        # Footer
        ax_footer = fig.add_subplot(gs[5, :])
        self._draw_footer(ax_footer)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())

        return fig

    def _draw_ball_losses(self, ax: plt.Axes, is_home: bool):
        """Draw locations where possession was lost (failed passes)"""
        self._draw_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        passes = self.home_passes if is_home else self.away_passes

        # Ball losses = failed passes
        losses = passes[passes['is_successful'] == False].copy()

        if len(losses) == 0:
            ax.text(50, 50, "No ball losses", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nBall Losses", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Draw losses colored by pitch third
        for _, l in losses.iterrows():
            x, y = l['x'], l['y']

            # Color by danger level (losing ball high up is less dangerous)
            if x > 66.66:
                marker_color = '#2ecc71'  # Low danger
            elif x > 33.33:
                marker_color = '#f39c12'  # Medium danger
            else:
                marker_color = '#e74c3c'  # High danger

            if not is_home:
                x, y = 100 - x, 100 - y

            ax.scatter(x, y, s=50, c=marker_color, marker='x',
                       alpha=0.6, linewidths=1.5, zorder=10)

        # Legend
        ax.scatter([], [], c='#e74c3c', marker='x', label='High Danger', s=50)
        ax.scatter([], [], c='#f39c12', marker='x', label='Medium Danger', s=50)
        ax.scatter([], [], c='#2ecc71', marker='x', label='Low Danger', s=50)
        ax.legend(loc='upper left', fontsize=8)

        ax.set_title(f"{team_name}\nBall Losses ({len(losses)})",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_counter_press(self, ax: plt.Axes, is_home: bool):
        """Draw counter-pressing actions (tackles/interceptions in opponent's half)"""
        self._draw_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        defensive = self.home_defensive if is_home else self.away_defensive

        # Counter-press = tackles/interceptions in opponent's half (immediate reaction to loss)
        counter_press = defensive[
            (defensive['action_type'].isin(['Tackle', 'Interception'])) &
            (defensive['x'] > 50)
        ].copy()

        if len(counter_press) == 0:
            ax.text(50, 50, "No counter-press actions", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nCounter-Press", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Highlight opponent's half
        if is_home:
            rect = Rectangle((50, 0), 50, 100, fill=True, facecolor=color, alpha=0.05)
        else:
            rect = Rectangle((0, 0), 50, 100, fill=True, facecolor=color, alpha=0.05)
        ax.add_patch(rect)

        # Draw actions
        for _, cp in counter_press.iterrows():
            x, y = cp['x'], cp['y']
            if not is_home:
                x, y = 100 - x, 100 - y

            marker = 'o' if cp['action_type'] == 'Tackle' else 's'
            marker_color = '#3498db' if cp['action_type'] == 'Tackle' else '#e74c3c'

            ax.scatter(x, y, s=80, c=marker_color, marker=marker,
                       edgecolors='white', linewidths=1, alpha=0.7, zorder=10)

        ax.scatter([], [], c='#3498db', marker='o', label='Tackle', s=50)
        ax.scatter([], [], c='#e74c3c', marker='s', label='Interception', s=50)
        ax.legend(loc='upper left', fontsize=8)

        ax.set_title(f"{team_name}\nCounter-Press Actions ({len(counter_press)})",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_fouls_map(self, ax: plt.Axes, is_home: bool):
        """Draw fouls committed (potential tactical fouls)"""
        self._draw_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away

        # Get fouls committed by this team (foul_committed = True for committing team)
        from opta_api import EventProcessor
        fouls_df = self.events_df[self.events_df['typeId'] == 4].copy()

        if is_home:
            team_fouls = fouls_df[fouls_df['contestantId'] == self.match_info.home_team_id]
        else:
            team_fouls = fouls_df[fouls_df['contestantId'] == self.match_info.away_team_id]

        # Fouls committed (outcome = 1)
        fouls_committed = team_fouls[team_fouls['outcome'] == 1]

        if len(fouls_committed) == 0:
            ax.text(50, 50, "No fouls committed", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nFouls Committed", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Draw fouls
        for _, f in fouls_committed.iterrows():
            x, y = f['x'], f['y']
            if not is_home:
                x, y = 100 - x, 100 - y

            ax.scatter(x, y, s=80, c='#e74c3c', marker='X',
                       edgecolors='white', linewidths=1, alpha=0.7, zorder=10)

        ax.set_title(f"{team_name}\nFouls Committed ({len(fouls_committed)})",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_negative_transition_stats(self, ax: plt.Axes):
        """Draw negative transition statistics"""
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis('off')

        # Calculate stats
        home_losses = len(self.home_passes[self.home_passes['is_successful'] == False])
        away_losses = len(self.away_passes[self.away_passes['is_successful'] == False])

        # Dangerous losses (in own third)
        home_danger_losses = len(self.home_passes[
            (self.home_passes['is_successful'] == False) &
            (self.home_passes['x'] < 33.33)
        ])
        away_danger_losses = len(self.away_passes[
            (self.away_passes['is_successful'] == False) &
            (self.away_passes['x'] < 33.33)
        ])

        # Counter-press actions
        home_cpress = len(self.home_defensive[
            (self.home_defensive['action_type'].isin(['Tackle', 'Interception'])) &
            (self.home_defensive['x'] > 50)
        ])
        away_cpress = len(self.away_defensive[
            (self.away_defensive['action_type'].isin(['Tackle', 'Interception'])) &
            (self.away_defensive['x'] > 50)
        ])

        # Fouls committed
        fouls_df = self.events_df[self.events_df['typeId'] == 4]
        home_fouls = len(fouls_df[
            (fouls_df['contestantId'] == self.match_info.home_team_id) &
            (fouls_df['outcome'] == 1)
        ])
        away_fouls = len(fouls_df[
            (fouls_df['contestantId'] == self.match_info.away_team_id) &
            (fouls_df['outcome'] == 1)
        ])

        stats = [
            (home_losses, away_losses, "Ball Losses (Failed Passes)"),
            (home_danger_losses, away_danger_losses, "Dangerous Losses (Own 3rd)"),
            (home_cpress, away_cpress, "Counter-Press Actions"),
            (home_fouls, away_fouls, "Fouls Committed"),
        ]

        y_start = 0.7
        y_step = 0.18

        ax.text(0.5, 0.95, "NEGATIVE TRANSITION STATISTICS", fontsize=14, fontweight='bold',
                color=TEXT_COLOR, ha='center', va='center', transform=ax.transAxes)

        # Team headers (with more space before first stat)
        ax.text(0.2, 0.85, self.match_info.home_team_name, fontsize=11,
                color=self.colors.home, ha='center', va='center', transform=ax.transAxes, fontweight='bold')
        ax.text(0.8, 0.85, self.match_info.away_team_name, fontsize=11,
                color=self.colors.away, ha='center', va='center', transform=ax.transAxes, fontweight='bold')

        for i, (home_val, away_val, label) in enumerate(stats):
            y = y_start - i * y_step
            ax.text(0.2, y, str(home_val), fontsize=14, color=self.colors.home,
                    ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.text(0.5, y, label, fontsize=10, color=TEXT_COLOR,
                    ha='center', va='center', transform=ax.transAxes)
            ax.text(0.8, y, str(away_val), fontsize=14, color=self.colors.away,
                    ha='center', va='center', transform=ax.transAxes, fontweight='bold')


class FullPhaseReport:
    """
    Generate complete phase-by-phase match report (7 pages)
    """

    def __init__(self, events_df: pd.DataFrame, match_info: MatchInfo, match_stats: dict = None):
        self.events_df = events_df
        self.match_info = match_info
        self.match_stats = match_stats

        # Process events
        self.processor = EventProcessor(events_df, match_info)
        self.passes = self.processor.get_passes()
        self.shots = self.processor.get_shots()
        self.defensive = self.processor.get_defensive_actions()
        self.recoveries = self.processor.get_ball_recoveries()
        self.take_ons = self.processor.get_take_ons()

    def generate_all_pages(self, output_dir: str = None) -> List[plt.Figure]:
        """Generate all 7 phase pages"""

        common_args = {
            'events_df': self.events_df,
            'match_info': self.match_info,
            'passes': self.passes,
            'shots': self.shots,
            'defensive': self.defensive,
            'recoveries': self.recoveries,
            'take_ons': self.take_ons,
            'match_stats': self.match_stats
        }

        pages = [
            ('1_buildup', BuildUpPhasePage(**common_args)),
            ('2_middle_progression', MiddleProgressionPhasePage(**common_args)),
            ('3_final_third', FinalThirdPhasePage(**common_args)),
            ('4_high_press', HighPressPhasePage(**common_args)),
            ('5_mid_low_block', MidLowBlockPhasePage(**common_args)),
            ('6_positive_transition', PositiveTransitionPage(**common_args)),
            ('7_negative_transition', NegativeTransitionPage(**common_args)),
        ]

        figures = []
        for name, page in pages:
            save_path = None
            if output_dir:
                save_path = f"{output_dir}/{name}.png"

            fig = page.generate(save_path=save_path)
            figures.append(fig)

        return figures

    def generate_pdf(self, output_path: str) -> None:
        """Generate all pages as a single PDF"""
        from matplotlib.backends.backend_pdf import PdfPages

        figures = self.generate_all_pages()

        with PdfPages(output_path) as pdf:
            for fig in figures:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)


def generate_phase_report(match_id: str, league_key: str = None,
                          output_dir: str = None) -> List[plt.Figure]:
    """
    Convenience function to generate phase report from match ID

    Args:
        match_id: Opta match ID
        league_key: League key (optional)
        output_dir: Directory to save PNG files

    Returns:
        List of matplotlib figures
    """
    if league_key:
        api = OptaAPI.for_league(league_key)
    else:
        api = OptaAPI()

    events_df, match_info = api.get_match_events(match_id)
    match_stats = api.get_match_stats(match_id)

    report = FullPhaseReport(events_df, match_info, match_stats)
    return report.generate_all_pages(output_dir=output_dir)


if __name__ == "__main__":
    # Example usage
    print("Phase Report Generator")
    print("Usage: generate_phase_report(match_id, league_key, output_dir)")
