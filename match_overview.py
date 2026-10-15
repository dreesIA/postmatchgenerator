"""
Match Overview One-Pager Generator
Creates a comprehensive match analysis PNG with multiple visualizations
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.gridspec import GridSpec
import pandas as pd
from typing import Tuple, Dict
from dataclasses import dataclass

from pitch_viz import PitchColors
from opta_api import OptaAPI, EventProcessor, MatchInfo, PassingNetworkBuilder


@dataclass
class TeamColors:
    """Color scheme for home and away teams"""
    home: str = '#e63946'
    away: str = '#457b9d'
    home_light: str = '#f4a5ab'
    away_light: str = '#a8c5d9'


# Background color for the report (off-white)
BACKGROUND_COLOR = '#f5f5f0'
TEXT_COLOR = '#1a1a1a'
TEXT_SECONDARY = '#666666'


class MatchOverview:
    """
    Generate a comprehensive one-page match overview
    """

    def __init__(self, events_df: pd.DataFrame, match_info: MatchInfo, match_stats: dict = None):
        """
        Initialize match overview generator

        Args:
            events_df: Raw events DataFrame
            match_info: MatchInfo object
            match_stats: Optional match stats dictionary from API (for accurate possession)
        """
        self.events_df = events_df.copy()
        self.match_info = match_info
        self.match_stats = match_stats
        self.processor = EventProcessor(events_df, match_info)
        self.colors = TeamColors()
        self.pitch_colors = PitchColors()

        # Pre-process all data
        self._process_data()

    def _process_data(self):
        """Process all event data"""
        self.passes = self.processor.get_passes()
        self.shots = self.processor.get_shots()
        self.defensive = self.processor.get_defensive_actions()
        self.recoveries = self.processor.get_ball_recoveries()
        self.take_ons = self.processor.get_take_ons()
        self.aerials = self.processor.get_aerials()
        self.fouls = self.processor.get_fouls()

        # Split by team
        self.home_passes = self.passes[self.passes['is_home'] == True]
        self.away_passes = self.passes[self.passes['is_home'] == False]
        self.home_shots = self.shots[self.shots['is_home'] == True]
        self.away_shots = self.shots[self.shots['is_home'] == False]
        self.home_defensive = self.defensive[self.defensive['is_home'] == True]
        self.away_defensive = self.defensive[self.defensive['is_home'] == False]

        # Build passing networks (pass events_df to identify substitutes)
        network_builder = PassingNetworkBuilder(self.passes, self.match_info, self.events_df)
        self.home_nodes, self.home_edges = network_builder.build_network(
            team_id=self.match_info.home_team_id
        )
        self.away_nodes, self.away_edges = network_builder.build_network(
            team_id=self.match_info.away_team_id
        )

    def generate(self, save_path: str = None, figsize: Tuple[int, int] = (28, 40)) -> plt.Figure:
        """
        Generate the complete match overview

        Args:
            save_path: Path to save the figure
            figsize: Figure size (width, height)

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize, facecolor=BACKGROUND_COLOR)

        # Create grid layout: 4 columns for flexible positioning
        # Outer columns (0, 3) for team visualizations, inner columns (1, 2) for center content and flow maps
        gs = GridSpec(
            10, 4,
            figure=fig,
            height_ratios=[0.6, 0.4, 3, 3, 3, 3, 3, 0.3, 3.5, 1],
            width_ratios=[1.2, 0.8, 0.8, 1.2],
            hspace=0.12,
            wspace=0.06
        )

        # Title row (spans all columns)
        ax_title = fig.add_subplot(gs[0, :])
        self._draw_title(ax_title)

        # Subtitle row
        ax_subtitle = fig.add_subplot(gs[1, :])
        self._draw_subtitle(ax_subtitle)

        # === LEFT COLUMN (Home Team) - Column 0 ===
        # Passing Network
        ax_home_pass = fig.add_subplot(gs[2, 0])
        self._draw_passing_network(ax_home_pass, is_home=True)

        # Defensive Actions
        ax_home_def = fig.add_subplot(gs[3, 0])
        self._draw_defensive_actions(ax_home_def, is_home=True)

        # Final Third Entries
        ax_home_ft = fig.add_subplot(gs[4, 0])
        self._draw_final_third_entries(ax_home_ft, is_home=True)

        # Half Space & Zone 14 Entries
        ax_home_hz = fig.add_subplot(gs[5, 0])
        self._draw_halfspace_zone14(ax_home_hz, is_home=True)

        # Chance Location & Origination
        ax_home_chance = fig.add_subplot(gs[6, 0])
        self._draw_chance_origins(ax_home_chance, is_home=True)

        # === RIGHT COLUMN (Away Team) - Column 3 ===
        # Passing Network
        ax_away_pass = fig.add_subplot(gs[2, 3])
        self._draw_passing_network(ax_away_pass, is_home=False)

        # Defensive Actions
        ax_away_def = fig.add_subplot(gs[3, 3])
        self._draw_defensive_actions(ax_away_def, is_home=False)

        # Final Third Entries
        ax_away_ft = fig.add_subplot(gs[4, 3])
        self._draw_final_third_entries(ax_away_ft, is_home=False)

        # Half Space & Zone 14 Entries
        ax_away_hz = fig.add_subplot(gs[5, 3])
        self._draw_halfspace_zone14(ax_away_hz, is_home=False)

        # Chance Location & Origination
        ax_away_chance = fig.add_subplot(gs[6, 3])
        self._draw_chance_origins(ax_away_chance, is_home=False)

        # === CENTER COLUMNS (1-2) ===
        # Stats Table
        ax_stats = fig.add_subplot(gs[2:4, 1:3])
        self._draw_stats_table(ax_stats)

        # Combined Shot Map
        ax_shots = fig.add_subplot(gs[4:6, 1:3])
        self._draw_combined_shot_map(ax_shots)

        # Game Control Index
        ax_control = fig.add_subplot(gs[6, 1:3])
        self._draw_game_control(ax_control)

        # === BOTTOM ROW - Pass Flow Maps Side by Side in Center ===
        ax_home_flow = fig.add_subplot(gs[8, 0:2])
        self._draw_pass_flow(ax_home_flow, is_home=True)

        ax_away_flow = fig.add_subplot(gs[8, 2:4])
        self._draw_pass_flow(ax_away_flow, is_home=False)

        # Footer
        ax_footer = fig.add_subplot(gs[9, :])
        self._draw_footer(ax_footer)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor(), edgecolor='none')

        return fig

    def _draw_title(self, ax: plt.Axes):
        """Draw the title with team names and score"""
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis('off')

        # Home team (left)
        ax.text(0.25, 0.5, self.match_info.home_team_name,
                fontsize=32, fontweight='bold', color=self.colors.home,
                ha='center', va='center', transform=ax.transAxes)

        # Score (center)
        ax.text(0.5, 0.5, f"{self.match_info.home_score} - {self.match_info.away_score}",
                fontsize=42, fontweight='bold', color=TEXT_COLOR,
                ha='center', va='center', transform=ax.transAxes)

        # Away team (right)
        ax.text(0.75, 0.5, self.match_info.away_team_name,
                fontsize=32, fontweight='bold', color=self.colors.away,
                ha='center', va='center', transform=ax.transAxes)

    def _draw_subtitle(self, ax: plt.Axes):
        """Draw subtitle with competition, date, and credit"""
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis('off')

        subtitle = f"{self.match_info.competition}  •  {self.match_info.date}  •  {self.match_info.venue}"
        ax.text(0.5, 0.7, subtitle,
                fontsize=12, color=TEXT_SECONDARY,
                ha='center', va='center', transform=ax.transAxes)

        ax.text(0.5, 0.2, "Generated by Idrees Ihsanullah",
                fontsize=10, color=TEXT_SECONDARY, style='italic',
                ha='center', va='center', transform=ax.transAxes)

    def _draw_mini_pitch(self, ax: plt.Axes):
        """Draw a mini pitch on the axes"""
        ax.set_facecolor('#c8d6a0')  # Light grass green

        lw = 1.2
        lc = '#2d5a27'  # Dark green lines

        # Pitch outline
        ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color=lc, lw=lw)
        ax.plot([50, 50], [0, 100], color=lc, lw=lw)

        # Center circle
        center_circle = Circle((50, 50), 9.15, fill=False, color=lc, lw=lw)
        ax.add_patch(center_circle)

        # Penalty areas
        ax.plot([0, 17, 17, 0], [21.1, 21.1, 78.9, 78.9], color=lc, lw=lw)
        ax.plot([100, 83, 83, 100], [21.1, 21.1, 78.9, 78.9], color=lc, lw=lw)

        # 6-yard boxes
        ax.plot([0, 5.8, 5.8, 0], [36.8, 36.8, 63.2, 63.2], color=lc, lw=lw)
        ax.plot([100, 94.2, 94.2, 100], [36.8, 36.8, 63.2, 63.2], color=lc, lw=lw)

        ax.set_xlim(-3, 103)
        ax.set_ylim(-5, 105)
        # Use adjustable aspect to allow pitch to fill more space
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

    def _draw_passing_network(self, ax: plt.Axes, is_home: bool):
        """Draw passing network for a team"""
        self._draw_mini_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        nodes = self.home_nodes if is_home else self.away_nodes
        edges = self.home_edges if is_home else self.away_edges

        if len(nodes) == 0:
            ax.text(50, 50, "No data", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nPassing Network", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Flip coordinates for away team (attacking left to right)
        if not is_home:
            nodes = nodes.copy()
            nodes['avg_x'] = 100 - nodes['avg_x']
            nodes['avg_y'] = 100 - nodes['avg_y']

        # Draw edges
        min_passes = 3
        filtered_edges = edges[edges['pass_count'] >= min_passes]

        node_pos = {row['player_id']: (row['avg_x'], row['avg_y'])
                    for _, row in nodes.iterrows()}

        for _, edge in filtered_edges.iterrows():
            passer_id = edge['passer_id']
            receiver_id = edge['receiver_id']

            if passer_id in node_pos and receiver_id in node_pos:
                start = node_pos[passer_id]
                end = node_pos[receiver_id]

                # Flip for away
                if not is_home:
                    # Already flipped in nodes
                    pass

                width = min(edge['pass_count'] * 0.5, 5)
                alpha = min(0.8, 0.3 + edge['pass_count'] * 0.05)

                ax.plot([start[0], end[0]], [start[1], end[1]],
                        color=color, alpha=alpha, linewidth=width, zorder=5)

        # Draw nodes - larger sizes, circles for starters, squares for subs
        for _, node in nodes.iterrows():
            size = node['pass_count'] * 25 + 80
            is_starter = node.get('is_starter', True)
            marker = 'o' if is_starter else 's'  # circle for starters, square for subs

            ax.scatter(node['avg_x'], node['avg_y'],
                       s=size, c=color, marker=marker, edgecolors='white', linewidths=2, zorder=10)

        # Player labels - larger font
        for _, node in nodes.iterrows():
            name = node.get('player_name', '')
            if isinstance(name, str) and len(name) > 0:
                short_name = name.split()[-1][:8] if ' ' in name else name[:8]
                ax.text(node['avg_x'], node['avg_y'] - 7, short_name,
                        fontsize=8, color=TEXT_COLOR, ha='center', va='top', zorder=11)

        ax.set_title(f"{team_name}\nPassing Network", fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_defensive_actions(self, ax: plt.Axes, is_home: bool):
        """Draw defensive actions map"""
        self._draw_mini_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        defensive = self.home_defensive if is_home else self.away_defensive

        if len(defensive) == 0:
            ax.text(50, 50, "No data", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nDefensive Actions", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Flip for away team
        x_vals = defensive['x'].values.copy()
        y_vals = defensive['y'].values.copy()
        if not is_home:
            x_vals = 100 - x_vals
            y_vals = 100 - y_vals

        # Plot by action type
        markers = {'Tackle': 'o', 'Interception': 's', 'Clearance': '^', 'Save': 'D'}
        colors_map = {'Tackle': '#3498db', 'Interception': '#e74c3c',
                      'Clearance': '#f39c12', 'Save': '#2ecc71'}

        for action_type in defensive['action_type'].unique():
            mask = defensive['action_type'] == action_type
            ax.scatter(x_vals[mask], y_vals[mask],
                       marker=markers.get(action_type, 'o'),
                       c=colors_map.get(action_type, color),
                       s=70, alpha=0.7, edgecolors='white', linewidths=0.8, zorder=10)

        # Calculate defensive metrics using ORIGINAL coordinates (before flip)
        # In Opta, both teams attack towards x=100, so x > 66.66 is opponent's final third
        orig_x = defensive['x'].values.copy()

        # Calculate and draw defensive line height (all actions except clearances - overall defensive shape)
        defensive_line_mask = defensive['action_type'] != 'Clearance'
        if defensive_line_mask.sum() > 0:
            avg_def_line = x_vals[defensive_line_mask].mean()
            ax.axvline(x=avg_def_line, color='#3498db', linestyle='-', alpha=0.8, lw=2, zorder=5)
            ax.text(avg_def_line + 2, 5, f'Def Line: {avg_def_line:.0f}', fontsize=7,
                    color='#3498db', ha='left', va='bottom', fontweight='bold')

        # Calculate and draw height of confrontation (tackles + interceptions in middle and final third)
        # Use original Opta coordinates: x > 33.33 means middle third or higher (attacking direction)
        confrontation_mask = defensive['action_type'].isin(['Tackle', 'Interception'])
        confrontation_orig_x = orig_x[confrontation_mask]
        # Filter for actions in middle and final third (original Opta x > 33.33)
        press_mask = confrontation_orig_x > 33.33
        if press_mask.sum() > 0:
            # Get the visualization x values for the filtered actions
            press_viz_x = x_vals[confrontation_mask][press_mask]
            avg_confrontation = press_viz_x.mean()
            ax.axvline(x=avg_confrontation, color='#e74c3c', linestyle='--', alpha=0.8, lw=2, zorder=5)
            ax.text(avg_confrontation + 2, 95, f'Confrontation: {avg_confrontation:.0f}', fontsize=7,
                    color='#e74c3c', ha='left', va='top', fontweight='bold')

        ax.set_title(f"{team_name}\nDefensive Actions", fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_final_third_entries(self, ax: plt.Axes, is_home: bool):
        """Draw final third entry passes"""
        self._draw_mini_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        passes = self.home_passes if is_home else self.away_passes
        color = self.colors.home if is_home else self.colors.away

        # Filter for final third entries (pass starts outside final third, ends in final third)
        # Opta normalizes coordinates so both teams attack towards x=100
        ft_entries = passes[
            (passes['x'] < 66.66) &
            (passes['end_x'] >= 66.66) &
            (passes['is_successful'] == True)
        ]

        if len(ft_entries) == 0:
            ax.text(50, 50, "No entries", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nFinal Third Entries", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Draw passes
        for _, p in ft_entries.iterrows():
            x, y = p['x'], p['y']
            end_x, end_y = p['end_x'], p['end_y']

            # Flip coordinates for away team to show them attacking left
            if not is_home:
                x, y = 100 - x, 100 - y
                end_x, end_y = 100 - end_x, 100 - end_y

            ax.annotate('', xy=(end_x, end_y), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color=color, alpha=0.6, lw=1.5))

        # Draw final third line (at 66.66 for home attacking right, 33.33 for away attacking left)
        ax.axvline(x=66.66 if is_home else 33.33, color=TEXT_COLOR, linestyle='--', alpha=0.5, lw=1.5)

        ax.set_title(f"{team_name}\nFinal Third Entries ({len(ft_entries)})",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_halfspace_zone14(self, ax: plt.Axes, is_home: bool):
        """Draw half space and zone 14 entries"""
        self._draw_mini_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        passes = self.home_passes if is_home else self.away_passes

        # Zone 14: Central area just outside the box (roughly x: 66-83, y: 30-70)
        # Half spaces: Wide areas between center and wings, extending to byline (y: 15-30 and 70-85, x: 66-100)
        # Opta normalizes coordinates so both teams attack towards x=100

        # Filter passes ending in these zones (same filter for both teams)
        zone14 = passes[
            (passes['end_x'] >= 66) & (passes['end_x'] <= 83) &
            (passes['end_y'] >= 30) & (passes['end_y'] <= 70) &
            (passes['is_successful'] == True)
        ]
        halfspace_left = passes[
            (passes['end_x'] >= 66) & (passes['end_x'] <= 100) &
            (passes['end_y'] >= 70) & (passes['end_y'] <= 85) &
            (passes['is_successful'] == True)
        ]
        halfspace_right = passes[
            (passes['end_x'] >= 66) & (passes['end_x'] <= 100) &
            (passes['end_y'] >= 15) & (passes['end_y'] <= 30) &
            (passes['is_successful'] == True)
        ]

        # Draw zones (positioned based on attacking direction in visualization)
        if is_home:
            # Home attacks right - zones on the right side, halfspaces extend to byline
            rect_z14 = Rectangle((66, 30), 17, 40, fill=True, facecolor='#f39c12', alpha=0.2)
            rect_hs_l = Rectangle((66, 70), 34, 15, fill=True, facecolor='#9b59b6', alpha=0.2)
            rect_hs_r = Rectangle((66, 15), 34, 15, fill=True, facecolor='#9b59b6', alpha=0.2)
        else:
            # Away attacks left - zones on the left side (flipped), halfspaces extend to byline
            rect_z14 = Rectangle((17, 30), 17, 40, fill=True, facecolor='#f39c12', alpha=0.2)
            rect_hs_l = Rectangle((0, 15), 34, 15, fill=True, facecolor='#9b59b6', alpha=0.2)
            rect_hs_r = Rectangle((0, 70), 34, 15, fill=True, facecolor='#9b59b6', alpha=0.2)

        ax.add_patch(rect_z14)
        ax.add_patch(rect_hs_l)
        ax.add_patch(rect_hs_r)

        # Plot entry points
        all_entries = pd.concat([zone14, halfspace_left, halfspace_right])

        for _, p in all_entries.iterrows():
            end_x, end_y = p['end_x'], p['end_y']
            # Flip coordinates for away team to show them attacking left
            if not is_home:
                end_x, end_y = 100 - end_x, 100 - end_y

            ax.scatter(end_x, end_y, c=color, s=50, alpha=0.7, edgecolors='white', linewidths=0.8)

        z14_count = len(zone14)
        hs_count = len(halfspace_left) + len(halfspace_right)

        ax.set_title(f"{team_name}\nZone 14: {z14_count} | Half Spaces: {hs_count}",
                     fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_chance_origins(self, ax: plt.Axes, is_home: bool):
        """Draw chance creation origins and shot locations"""
        self._draw_mini_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        shots = self.home_shots if is_home else self.away_shots
        passes = self.home_passes if is_home else self.away_passes

        # Get key passes (assists and shot-creating passes)
        key_passes = passes[passes['is_key_pass'] == True]

        if len(shots) == 0:
            ax.text(50, 50, "No chances", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nChance Origins", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Draw shot locations
        for _, shot in shots.iterrows():
            x, y = shot['x'], shot['y']
            if not is_home:
                x, y = 100 - x, 100 - y

            size = shot.get('xG', 0.1) * 400 + 50

            if shot.get('is_goal', False):
                marker_color = '#2ecc71'
            elif shot.get('is_on_target', False):
                marker_color = '#f39c12'
            else:
                marker_color = '#95a5a6'

            ax.scatter(x, y, s=size, c=marker_color, edgecolors='white',
                       linewidths=1.2, alpha=0.8, zorder=10)

        # Draw key pass origins
        for _, p in key_passes.iterrows():
            x, y = p['x'], p['y']
            end_x, end_y = p['end_x'], p['end_y']

            if pd.isna(end_x) or pd.isna(end_y):
                continue

            if not is_home:
                x, y = 100 - x, 100 - y
                end_x, end_y = 100 - end_x, 100 - end_y

            ax.annotate('', xy=(end_x, end_y), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color=TEXT_COLOR, alpha=0.5, lw=1.2))
            ax.scatter(x, y, s=35, c=TEXT_COLOR, marker='o', alpha=0.6, zorder=9)

        ax.set_title(f"{team_name}\nChance Origins", fontsize=12, color=TEXT_COLOR, pad=8)

    def _calculate_stats(self) -> Dict:
        """Calculate all match statistics"""
        stats = {}

        # xG
        stats['home_xg'] = self.home_shots['xG'].sum() if len(self.home_shots) > 0 else 0
        stats['away_xg'] = self.away_shots['xG'].sum() if len(self.away_shots) > 0 else 0

        # Possession - try to get real data from match stats first
        stats['home_possession'] = 50
        stats['away_possession'] = 50

        if self.match_stats:
            try:
                lineups = self.match_stats.get('liveData', {}).get('lineUp', [])
                for team_data in lineups:
                    team_id = team_data.get('contestantId')
                    team_stats = team_data.get('stat', [])

                    for stat in team_stats:
                        if stat.get('type') == 'possessionPercentage':
                            possession = float(stat.get('value', 50))
                            if team_id == self.match_info.home_team_id:
                                stats['home_possession'] = possession
                                stats['away_possession'] = 100 - possession
                            elif team_id == self.match_info.away_team_id:
                                stats['away_possession'] = possession
                                stats['home_possession'] = 100 - possession
                            break
            except (KeyError, ValueError, TypeError):
                pass  # Fall back to pass-based calculation

        # If we didn't get real possession data, calculate from passes
        if stats['home_possession'] == 50 and stats['away_possession'] == 50:
            home_pass_count = len(self.home_passes[self.home_passes['is_successful'] == True])
            away_pass_count = len(self.away_passes[self.away_passes['is_successful'] == True])
            total_passes = home_pass_count + away_pass_count
            if total_passes > 0:
                stats['home_possession'] = home_pass_count / total_passes * 100
                stats['away_possession'] = 100 - stats['home_possession']

        # Try to get stats from API first, fall back to event-based calculations
        home_api_stats = {}
        away_api_stats = {}

        if self.match_stats:
            try:
                lineups = self.match_stats.get('liveData', {}).get('lineUp', [])
                for team_data in lineups:
                    team_id = team_data.get('contestantId')
                    team_stat_list = team_data.get('stat', [])
                    team_stat_dict = {}
                    for s in team_stat_list:
                        try:
                            team_stat_dict[s['type']] = float(s['value'])
                        except (ValueError, TypeError):
                            team_stat_dict[s['type']] = s['value']

                    if team_id == self.match_info.home_team_id:
                        home_api_stats = team_stat_dict
                    elif team_id == self.match_info.away_team_id:
                        away_api_stats = team_stat_dict
            except (KeyError, TypeError):
                pass

        # Pass counts for various calculations
        home_pass_count = len(self.home_passes[self.home_passes['is_successful'] == True])
        away_pass_count = len(self.away_passes[self.away_passes['is_successful'] == True])

        # Field Tilt - use final third entries from API if available
        if home_api_stats and away_api_stats:
            home_ft_entries = home_api_stats.get('finalThirdEntries', 0)
            away_ft_entries = away_api_stats.get('finalThirdEntries', 0)
            total_ft = home_ft_entries + away_ft_entries
            if total_ft > 0:
                stats['home_field_tilt'] = (home_ft_entries / total_ft * 100)
                stats['away_field_tilt'] = (away_ft_entries / total_ft * 100)
            else:
                stats['home_field_tilt'] = 50
                stats['away_field_tilt'] = 50
        else:
            # Fall back to event-based calculation
            home_opp_half = len(self.home_passes[self.home_passes['x'] > 50])
            away_opp_half = len(self.away_passes[self.away_passes['x'] < 50])
            total_opp = home_opp_half + away_opp_half
            stats['home_field_tilt'] = (home_opp_half / total_opp * 100) if total_opp > 0 else 50
            stats['away_field_tilt'] = 100 - stats['home_field_tilt']

        # Passes in opponent's half - calculate from event data
        # Opta normalizes coordinates so both teams attack towards x=100
        # Opponent's half is where x > 50
        home_opp_half = len(self.home_passes[self.home_passes['x'] > 50])
        away_opp_half = len(self.away_passes[self.away_passes['x'] > 50])
        stats['home_passes_opp_half'] = home_opp_half
        stats['away_passes_opp_half'] = away_opp_half

        # PPDA (Passes Per Defensive Action) - use API stats for more accurate calculation
        if home_api_stats and away_api_stats:
            # Defensive actions = tackles + interceptions + fouls
            home_def_actions = (home_api_stats.get('totalTackle', 0) +
                               home_api_stats.get('interceptionWon', 0) +
                               home_api_stats.get('fkFoulLost', 0))
            away_def_actions = (away_api_stats.get('totalTackle', 0) +
                               away_api_stats.get('interceptionWon', 0) +
                               away_api_stats.get('fkFoulLost', 0))

            # PPDA = opponent's passes / team's defensive actions
            away_passes = away_api_stats.get('accuratePass', away_pass_count)
            home_passes = home_api_stats.get('accuratePass', home_pass_count)

            stats['home_ppda'] = away_passes / home_def_actions if home_def_actions > 0 else 0
            stats['away_ppda'] = home_passes / away_def_actions if away_def_actions > 0 else 0
        else:
            # Fall back to event-based calculation
            home_def_own_half = len(self.home_defensive[self.home_defensive['x'] < 50])
            away_def_own_half = len(self.away_defensive[self.away_defensive['x'] > 50])
            stats['home_ppda'] = away_pass_count / home_def_own_half if home_def_own_half > 0 else 0
            stats['away_ppda'] = home_pass_count / away_def_own_half if away_def_own_half > 0 else 0

        # High Regains - use possWonAtt3rd from API if available
        if home_api_stats and away_api_stats:
            stats['home_high_regains'] = int(home_api_stats.get('possWonAtt3rd', 0))
            stats['away_high_regains'] = int(away_api_stats.get('possWonAtt3rd', 0))
        else:
            home_high_regains = len(self.recoveries[
                (self.recoveries['is_home'] == True) & (self.recoveries['x'] > 66.66)
            ])
            away_high_regains = len(self.recoveries[
                (self.recoveries['is_home'] == False) & (self.recoveries['x'] < 33.33)
            ])
            stats['home_high_regains'] = home_high_regains
            stats['away_high_regains'] = away_high_regains

        # Crosses
        stats['home_crosses'] = len(self.home_passes[self.home_passes['is_cross'] == True])
        stats['away_crosses'] = len(self.away_passes[self.away_passes['is_cross'] == True])

        # Corners
        stats['home_corners'] = len(self.home_passes[self.home_passes['is_corner'] == True])
        stats['away_corners'] = len(self.away_passes[self.away_passes['is_corner'] == True])

        # Fouls won
        stats['home_fouls_won'] = len(self.fouls[
            (self.fouls['is_home'] == False) & (self.fouls['foul_committed'] == True)
        ])
        stats['away_fouls_won'] = len(self.fouls[
            (self.fouls['is_home'] == True) & (self.fouls['foul_committed'] == True)
        ])

        # Tackles
        home_tackles = self.home_defensive[self.home_defensive['action_type'] == 'Tackle']
        away_tackles = self.away_defensive[self.away_defensive['action_type'] == 'Tackle']
        stats['home_tackles'] = len(home_tackles)
        stats['away_tackles'] = len(away_tackles)
        stats['home_tackle_pct'] = (home_tackles['is_successful'].mean() * 100) if len(home_tackles) > 0 else 0
        stats['away_tackle_pct'] = (away_tackles['is_successful'].mean() * 100) if len(away_tackles) > 0 else 0

        # Aerial duels
        home_aerials = self.aerials[self.aerials['is_home'] == True]
        away_aerials = self.aerials[self.aerials['is_home'] == False]
        stats['home_aerials'] = len(home_aerials)
        stats['away_aerials'] = len(away_aerials)
        stats['home_aerial_pct'] = (home_aerials['is_won'].mean() * 100) if len(home_aerials) > 0 else 0
        stats['away_aerial_pct'] = (away_aerials['is_won'].mean() * 100) if len(away_aerials) > 0 else 0

        return stats

    def _draw_stats_table(self, ax: plt.Axes):
        """Draw the statistics comparison table"""
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis('off')

        stats = self._calculate_stats()

        # Define rows: (label, home_value, away_value, format_type)
        rows = [
            ('xG', f"{stats['home_xg']:.2f}", f"{stats['away_xg']:.2f}", 'normal'),
            ('Possession', f"{stats['home_possession']:.0f}%", f"{stats['away_possession']:.0f}%", 'bar'),
            ('Field Tilt', f"{stats['home_field_tilt']:.0f}%", f"{stats['away_field_tilt']:.0f}%", 'bar'),
            ('Passes (Opp Half)', str(stats['home_passes_opp_half']), str(stats['away_passes_opp_half']), 'normal'),
            ('PPDA', f"{stats['home_ppda']:.1f}", f"{stats['away_ppda']:.1f}", 'normal'),
            ('High Regains', str(stats['home_high_regains']), str(stats['away_high_regains']), 'normal'),
            ('Crosses', str(stats['home_crosses']), str(stats['away_crosses']), 'normal'),
            ('Corners', str(stats['home_corners']), str(stats['away_corners']), 'normal'),
            ('Fouls Won', str(stats['home_fouls_won']), str(stats['away_fouls_won']), 'normal'),
            ('Tackles', f"{stats['home_tackles']} ({stats['home_tackle_pct']:.0f}%)",
             f"{stats['away_tackles']} ({stats['away_tackle_pct']:.0f}%)", 'normal'),
            ('Aerials', f"{stats['home_aerials']} ({stats['home_aerial_pct']:.0f}%)",
             f"{stats['away_aerials']} ({stats['away_aerial_pct']:.0f}%)", 'normal'),
        ]

        # Draw table
        y_start = 0.95
        y_step = 0.08

        # Header
        ax.text(0.5, y_start, "MATCH STATISTICS", fontsize=12, fontweight='bold',
                color=TEXT_COLOR, ha='center', va='center', transform=ax.transAxes)

        y_start -= 0.05

        for i, (label, home_val, away_val, fmt_type) in enumerate(rows):
            y = y_start - i * y_step

            # Home value
            ax.text(0.15, y, home_val, fontsize=10, color=self.colors.home,
                    ha='center', va='center', transform=ax.transAxes, fontweight='bold')

            # Label
            ax.text(0.5, y, label, fontsize=9, color=TEXT_COLOR,
                    ha='center', va='center', transform=ax.transAxes)

            # Away value
            ax.text(0.85, y, away_val, fontsize=10, color=self.colors.away,
                    ha='center', va='center', transform=ax.transAxes, fontweight='bold')

            # Draw bar for percentage stats
            if fmt_type == 'bar':
                bar_y = y - 0.02
                bar_width = 0.3
                bar_height = 0.015

                # Home bar (from center going left)
                try:
                    home_pct = float(home_val.replace('%', '')) / 100
                except:
                    home_pct = 0.5

                rect_home = Rectangle((0.5 - bar_width * home_pct, bar_y - bar_height / 2),
                                       bar_width * home_pct, bar_height,
                                       transform=ax.transAxes, facecolor=self.colors.home, alpha=0.7)
                rect_away = Rectangle((0.5, bar_y - bar_height / 2),
                                       bar_width * (1 - home_pct), bar_height,
                                       transform=ax.transAxes, facecolor=self.colors.away, alpha=0.7)
                ax.add_patch(rect_home)
                ax.add_patch(rect_away)

    def _draw_combined_shot_map(self, ax: plt.Axes):
        """Draw combined shot map for both teams"""
        self._draw_mini_pitch(ax)

        # Plot home shots (attacking right)
        for _, shot in self.home_shots.iterrows():
            x, y = shot['x'], shot['y']
            size = shot.get('xG', 0.1) * 500 + 60

            if shot.get('is_goal', False):
                marker = '*'
                size *= 1.5
            else:
                marker = 'o'

            ax.scatter(x, y, s=size, c=self.colors.home, marker=marker,
                       edgecolors='white', linewidths=1.5, alpha=0.8, zorder=10)

        # Plot away shots (attacking left, so flip)
        for _, shot in self.away_shots.iterrows():
            x, y = 100 - shot['x'], 100 - shot['y']
            size = shot.get('xG', 0.1) * 500 + 60

            if shot.get('is_goal', False):
                marker = '*'
                size *= 1.5
            else:
                marker = 'o'

            ax.scatter(x, y, s=size, c=self.colors.away, marker=marker,
                       edgecolors='white', linewidths=1.5, alpha=0.8, zorder=10)

        # xG labels
        home_xg = self.home_shots['xG'].sum() if len(self.home_shots) > 0 else 0
        away_xg = self.away_shots['xG'].sum() if len(self.away_shots) > 0 else 0

        ax.text(25, 95, f"xG: {home_xg:.2f}", fontsize=12, color=self.colors.home,
                fontweight='bold', ha='center')
        ax.text(75, 95, f"xG: {away_xg:.2f}", fontsize=12, color=self.colors.away,
                fontweight='bold', ha='center')

        ax.set_title("Shot Map", fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_game_control(self, ax: plt.Axes):
        """Draw game control timeline"""
        ax.set_facecolor(BACKGROUND_COLOR)

        # Calculate control by time period (15-minute windows)
        max_time = self.events_df['timeMin'].max()
        if pd.isna(max_time) or max_time == 0:
            max_time = 90

        # Use 15-minute intervals: 0, 15, 30, 45, 60, 75, 90
        windows = list(range(0, int(max_time) + 15, 15))
        home_control = []
        away_control = []

        for start in windows[:-1]:
            end = start + 15

            # Events in window
            window_events = self.events_df[
                (self.events_df['timeMin'] >= start) &
                (self.events_df['timeMin'] < end)
            ]

            # Possession proxy: successful passes
            home_events = len(window_events[window_events['contestantId'] == self.match_info.home_team_id])
            away_events = len(window_events[window_events['contestantId'] == self.match_info.away_team_id])

            total = home_events + away_events
            if total > 0:
                home_control.append(home_events / total)
                away_control.append(away_events / total)
            else:
                home_control.append(0.5)
                away_control.append(0.5)

        # Plot stacked area using step function for 15-min blocks
        x = list(windows[:-1])
        ax.fill_between(x, 0, home_control, color=self.colors.home, alpha=0.7, label='Home', step='post')
        ax.fill_between(x, home_control, 1, color=self.colors.away, alpha=0.7, label='Away', step='post')

        # Add 50% line
        ax.axhline(y=0.5, color=TEXT_SECONDARY, linestyle='--', alpha=0.5, lw=0.5)

        # Period markers at 15-minute intervals
        for t in [15, 30, 45, 60, 75]:
            if t <= max_time:
                ax.axvline(x=t, color=TEXT_SECONDARY, linestyle='-', alpha=0.3, lw=1)

        # Add goal markers
        goals = self.shots[self.shots['is_goal'] == True]
        for _, goal in goals.iterrows():
            goal_time = goal['timeMin']
            is_home_goal = goal['is_home']
            goal_color = self.colors.home if is_home_goal else self.colors.away
            # Draw thin vertical bar at goal time
            ax.axvline(x=goal_time, color=goal_color, linestyle='-', alpha=0.9, lw=2, zorder=10)

        ax.set_xlim(0, max(max_time, 90))
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax.set_xlabel('Minutes', fontsize=10, color=TEXT_COLOR)
        ax.set_ylabel('Control', fontsize=10, color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.set_title("Game Control Index", fontsize=12, color=TEXT_COLOR, pad=8)

        for spine in ax.spines.values():
            spine.set_color(TEXT_SECONDARY)
            spine.set_alpha(0.3)

    def _draw_pass_flow(self, ax: plt.Axes, is_home: bool):
        """Draw pass flow map showing most common passes between 30 zones (5 lanes x 6 zones)"""
        self._draw_mini_pitch(ax)

        team_name = self.match_info.home_team_name if is_home else self.match_info.away_team_name
        color = self.colors.home if is_home else self.colors.away
        passes = self.home_passes if is_home else self.away_passes

        # Filter successful passes with valid end coordinates
        valid_passes = passes[
            (passes['is_successful'] == True) &
            (passes['end_x'].notna()) &
            (passes['end_y'].notna())
        ].copy()

        if len(valid_passes) == 0:
            ax.text(50, 50, "No data", color=TEXT_COLOR, ha='center', va='center', fontsize=12)
            ax.set_title(f"{team_name}\nPass Flow", fontsize=12, color=TEXT_COLOR, pad=8)
            return

        # Create 30 zones: 6 horizontal zones x 5 vertical lanes
        # x_bins: 6 horizontal zones (0-16.67, 16.67-33.33, 33.33-50, 50-66.67, 66.67-83.33, 83.33-100)
        # y_bins: 5 vertical lanes (0-20, 20-40, 40-60, 60-80, 80-100)
        x_bins = [0, 16.67, 33.33, 50, 66.67, 83.33, 100]
        y_bins = [0, 20, 40, 60, 80, 100]

        valid_passes['start_zone_x'] = pd.cut(valid_passes['x'], bins=x_bins, labels=range(6))
        valid_passes['start_zone_y'] = pd.cut(valid_passes['y'], bins=y_bins, labels=range(5))
        valid_passes['end_zone_x'] = pd.cut(valid_passes['end_x'], bins=x_bins, labels=range(6))
        valid_passes['end_zone_y'] = pd.cut(valid_passes['end_y'], bins=y_bins, labels=range(5))

        # Group by zone pairs and count
        zone_flows = valid_passes.groupby(
            ['start_zone_x', 'start_zone_y', 'end_zone_x', 'end_zone_y'],
            observed=True
        ).size().reset_index(name='count')

        # Get top flows (most common pass connections between zones)
        zone_flows = zone_flows.nlargest(20, 'count')

        # Zone centers for 6x5 grid
        x_centers = [8.33, 25, 41.67, 58.33, 75, 91.67]
        y_centers = [10, 30, 50, 70, 90]

        # Draw zone grid lines for reference (subtle)
        for x_line in x_bins[1:-1]:
            viz_x = x_line if is_home else 100 - x_line
            ax.axvline(x=viz_x, color=TEXT_SECONDARY, linestyle=':', alpha=0.2, lw=0.5)
        for y_line in y_bins[1:-1]:
            viz_y = y_line if is_home else 100 - y_line
            ax.axhline(y=viz_y, color=TEXT_SECONDARY, linestyle=':', alpha=0.2, lw=0.5)

        # Draw arrows for most common passes
        max_count = zone_flows['count'].max()

        for _, flow in zone_flows.iterrows():
            try:
                sx = x_centers[int(flow['start_zone_x'])]
                sy = y_centers[int(flow['start_zone_y'])]
                ex = x_centers[int(flow['end_zone_x'])]
                ey = y_centers[int(flow['end_zone_y'])]
            except (ValueError, IndexError):
                continue

            # Flip for away team
            if not is_home:
                sx, sy = 100 - sx, 100 - sy
                ex, ey = 100 - ex, 100 - ey

            # Arrow properties based on count
            width = (flow['count'] / max_count) * 4 + 0.8
            alpha = min(0.9, (flow['count'] / max_count) * 0.6 + 0.3)

            ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle='->', color=color, alpha=alpha,
                                        lw=width, connectionstyle='arc3,rad=0.1'))

        ax.set_title(f"{team_name}\nPass Flow", fontsize=12, color=TEXT_COLOR, pad=8)

    def _draw_footer(self, ax: plt.Axes):
        """Draw footer with data source and credits"""
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis('off')

        ax.text(0.5, 0.5, "Generated by Idrees Ihsanullah",
                fontsize=9, color=TEXT_SECONDARY, ha='center', va='center', transform=ax.transAxes)


def generate_match_overview(match_id: str, league_key: str = None,
                            save_path: str = None) -> plt.Figure:
    """
    Convenience function to generate match overview from match ID

    Args:
        match_id: Opta match ID
        league_key: League key for API (optional if using default)
        save_path: Path to save PNG

    Returns:
        Matplotlib figure
    """
    if league_key:
        api = OptaAPI.for_league(league_key)
    else:
        api = OptaAPI()

    events_df, match_info = api.get_match_events(match_id)
    match_stats = api.get_match_stats(match_id)
    overview = MatchOverview(events_df, match_info, match_stats)

    if save_path is None:
        # Generate filename from team names
        home_short = match_info.home_team_name.replace(' ', '_')[:15]
        away_short = match_info.away_team_name.replace(' ', '_')[:15]
        save_path = f"match_overview_{home_short}_vs_{away_short}.png"

    return overview.generate(save_path=save_path)
