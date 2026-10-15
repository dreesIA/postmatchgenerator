"""
Pitch visualization components for scouting reports.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, Rectangle, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# Color scheme
PITCH_COLOR = '#1a472a'
PITCH_LINES = '#ffffff'
BACKGROUND = '#0e1117'


class PitchDrawer:
    """
    Draws football pitch visualizations.
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        line_color: str = PITCH_LINES,
        pitch_color: str = PITCH_COLOR
    ):
        """
        Initialize pitch drawer.

        Args:
            pitch_length: Pitch length in meters
            pitch_width: Pitch width in meters
            line_color: Color for pitch lines
            pitch_color: Color for pitch background
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.line_color = line_color
        self.pitch_color = pitch_color

    def draw_pitch(
        self,
        ax: plt.Axes,
        orientation: str = 'horizontal',
        half: str = None,
        linewidth: float = 1.5
    ) -> plt.Axes:
        """
        Draw a football pitch on the given axes.

        Args:
            ax: Matplotlib axes
            orientation: 'horizontal' or 'vertical'
            half: 'left', 'right', or None for full pitch
            linewidth: Line width

        Returns:
            Modified axes
        """
        ax.set_facecolor(self.pitch_color)
        lc = self.line_color
        lw = linewidth

        if orientation == 'horizontal':
            # Full pitch dimensions (0-100 scale)
            if half == 'left':
                x_range = (0, 50)
            elif half == 'right':
                x_range = (50, 100)
            else:
                x_range = (0, 100)

            y_range = (0, 100)

            # Pitch outline
            ax.plot([x_range[0], x_range[1]], [0, 0], color=lc, lw=lw)
            ax.plot([x_range[0], x_range[1]], [100, 100], color=lc, lw=lw)
            ax.plot([x_range[0], x_range[0]], [0, 100], color=lc, lw=lw)
            ax.plot([x_range[1], x_range[1]], [0, 100], color=lc, lw=lw)

            # Center line and circle
            if half is None:
                ax.plot([50, 50], [0, 100], color=lc, lw=lw)
                center_circle = plt.Circle((50, 50), 9.15, fill=False, color=lc, lw=lw)
                ax.add_patch(center_circle)
                ax.plot(50, 50, 'o', color=lc, markersize=3)

            # Left penalty area
            if x_range[0] <= 17:
                # 18-yard box
                ax.plot([0, 17], [21.1, 21.1], color=lc, lw=lw)
                ax.plot([0, 17], [78.9, 78.9], color=lc, lw=lw)
                ax.plot([17, 17], [21.1, 78.9], color=lc, lw=lw)
                # 6-yard box
                ax.plot([0, 5.8], [36.8, 36.8], color=lc, lw=lw)
                ax.plot([0, 5.8], [63.2, 63.2], color=lc, lw=lw)
                ax.plot([5.8, 5.8], [36.8, 63.2], color=lc, lw=lw)
                # Penalty spot
                ax.plot(11, 50, 'o', color=lc, markersize=3)
                # Penalty arc
                arc = Arc((11, 50), 18.3, 18.3, angle=0, theta1=-53, theta2=53, color=lc, lw=lw)
                ax.add_patch(arc)

            # Right penalty area
            if x_range[1] >= 83:
                # 18-yard box
                ax.plot([100, 83], [21.1, 21.1], color=lc, lw=lw)
                ax.plot([100, 83], [78.9, 78.9], color=lc, lw=lw)
                ax.plot([83, 83], [21.1, 78.9], color=lc, lw=lw)
                # 6-yard box
                ax.plot([100, 94.2], [36.8, 36.8], color=lc, lw=lw)
                ax.plot([100, 94.2], [63.2, 63.2], color=lc, lw=lw)
                ax.plot([94.2, 94.2], [36.8, 63.2], color=lc, lw=lw)
                # Penalty spot
                ax.plot(89, 50, 'o', color=lc, markersize=3)
                # Penalty arc
                arc = Arc((89, 50), 18.3, 18.3, angle=0, theta1=127, theta2=233, color=lc, lw=lw)
                ax.add_patch(arc)

            # Corner arcs
            if x_range[0] == 0:
                ax.add_patch(Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=90, color=lc, lw=lw))
                ax.add_patch(Arc((0, 100), 2, 2, angle=0, theta1=270, theta2=360, color=lc, lw=lw))
            if x_range[1] == 100:
                ax.add_patch(Arc((100, 0), 2, 2, angle=0, theta1=90, theta2=180, color=lc, lw=lw))
                ax.add_patch(Arc((100, 100), 2, 2, angle=0, theta1=180, theta2=270, color=lc, lw=lw))

            ax.set_xlim(x_range[0] - 3, x_range[1] + 3)
            ax.set_ylim(-3, 103)

        ax.set_aspect('equal')
        ax.axis('off')

        return ax

    def draw_vertical_half(
        self,
        ax: plt.Axes,
        goal_at_top: bool = True,
        linewidth: float = 1.5
    ) -> plt.Axes:
        """
        Draw a vertical half-pitch.

        Args:
            ax: Matplotlib axes
            goal_at_top: Whether goal is at top
            linewidth: Line width

        Returns:
            Modified axes
        """
        ax.set_facecolor(self.pitch_color)
        lc = self.line_color
        lw = linewidth

        # Pitch outline (width=100, height=50)
        ax.plot([0, 100, 100, 0, 0], [0, 0, 50, 50, 0], color=lc, lw=lw)

        if goal_at_top:
            # Penalty area at top
            ax.plot([21.1, 21.1], [33, 50], color=lc, lw=lw)
            ax.plot([78.9, 78.9], [33, 50], color=lc, lw=lw)
            ax.plot([21.1, 78.9], [33, 33], color=lc, lw=lw)
            # 6-yard box
            ax.plot([36.8, 36.8], [44.2, 50], color=lc, lw=lw)
            ax.plot([63.2, 63.2], [44.2, 50], color=lc, lw=lw)
            ax.plot([36.8, 63.2], [44.2, 44.2], color=lc, lw=lw)
            # Goal
            ax.plot([44, 56], [50, 50], color=lc, lw=2.5)
            # Penalty spot
            ax.plot(50, 39, 'o', color=lc, markersize=3)
            # Center circle arc
            theta = np.linspace(0, np.pi, 50)
            ax.plot(50 + 9.15 * np.cos(theta), 9.15 * np.sin(theta), color=lc, lw=lw)
        else:
            # Penalty area at bottom
            ax.plot([21.1, 21.1], [0, 17], color=lc, lw=lw)
            ax.plot([78.9, 78.9], [0, 17], color=lc, lw=lw)
            ax.plot([21.1, 78.9], [17, 17], color=lc, lw=lw)
            # 6-yard box
            ax.plot([36.8, 36.8], [0, 5.8], color=lc, lw=lw)
            ax.plot([63.2, 63.2], [0, 5.8], color=lc, lw=lw)
            ax.plot([36.8, 63.2], [5.8, 5.8], color=lc, lw=lw)
            # Goal
            ax.plot([44, 56], [0, 0], color=lc, lw=2.5)
            # Penalty spot
            ax.plot(50, 11, 'o', color=lc, markersize=3)
            # Center circle arc
            theta = np.linspace(np.pi, 2 * np.pi, 50)
            ax.plot(50 + 9.15 * np.cos(theta), 50 + 9.15 * np.sin(theta), color=lc, lw=lw)

        ax.set_xlim(-5, 105)
        ax.set_ylim(-3, 53)
        ax.set_aspect('equal')
        ax.axis('off')

        return ax


class PitchVisualizations:
    """
    Creates pitch-based visualizations for player scouting.
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize visualizations.

        Args:
            output_dir: Directory to save figures
        """
        self.pitch_drawer = PitchDrawer()
        self.output_dir = Path(output_dir) if output_dir else Path('outputs/figures')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_heatmap(
        self,
        df: pd.DataFrame,
        player_name: str = "",
        title: str = "Touch Heatmap",
        save_path: str = None,
        cmap: str = 'hot'
    ) -> plt.Figure:
        """
        Create touch density heatmap.

        Args:
            df: Events DataFrame with x, y columns
            player_name: Player name for title
            title: Figure title
            save_path: Path to save figure
            cmap: Colormap name

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8), facecolor=BACKGROUND)
        self.pitch_drawer.draw_pitch(ax)

        if df.empty or 'x' not in df.columns or 'y' not in df.columns:
            ax.text(50, 50, "No data", ha='center', va='center', color='white', fontsize=14)
        else:
            # Create heatmap using 2D histogram
            x = df['x'].values
            y = df['y'].values

            # Create bins
            x_bins = np.linspace(0, 100, 25)
            y_bins = np.linspace(0, 100, 17)

            heatmap, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])

            # Apply Gaussian smoothing
            heatmap = gaussian_filter(heatmap, sigma=1.5)

            # Create custom colormap
            colors = ['#0e1117', '#1a472a', '#2ecc71', '#f1c40f', '#e74c3c']
            custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)

            # Plot heatmap
            extent = [0, 100, 0, 100]
            ax.imshow(
                heatmap.T, extent=extent, origin='lower',
                cmap=custom_cmap, alpha=0.7, aspect='auto', zorder=1
            )

        # Title
        full_title = f"{player_name}\n{title}" if player_name else title
        ax.set_title(full_title, fontsize=16, color='white', fontweight='bold', pad=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)

        return fig

    def create_pass_map(
        self,
        df: pd.DataFrame,
        player_name: str = "",
        title: str = "Pass Map",
        highlight_progressive: bool = True,
        color: str = '#3498db',
        save_path: str = None
    ) -> plt.Figure:
        """
        Create pass map visualization.

        Args:
            df: Events DataFrame with passes
            player_name: Player name for title
            title: Figure title
            highlight_progressive: Whether to highlight progressive passes
            color: Primary color
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8), facecolor=BACKGROUND)
        self.pitch_drawer.draw_pitch(ax)

        passes = df[df['is_pass']] if 'is_pass' in df.columns else df

        if passes.empty:
            ax.text(50, 50, "No passes", ha='center', va='center', color='white', fontsize=14)
        else:
            for _, p in passes.iterrows():
                x, y = p['x'], p['y']
                end_x = p.get('end_x', x)
                end_y = p.get('end_y', y)

                if pd.isna(end_x) or pd.isna(end_y):
                    continue

                # Determine color
                is_progressive = p.get('is_progressive_pass', False)
                is_successful = p.get('outcome_success', True)

                if highlight_progressive and is_progressive:
                    pass_color = '#2ecc71'
                    alpha = 0.9
                    lw = 1.5
                elif is_successful:
                    pass_color = color
                    alpha = 0.5
                    lw = 1.0
                else:
                    pass_color = '#e74c3c'
                    alpha = 0.3
                    lw = 0.8

                # Draw arrow
                ax.annotate(
                    '', xy=(end_x, end_y), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color=pass_color, alpha=alpha, lw=lw)
                )

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='#2ecc71', label='Progressive'),
            mpatches.Patch(facecolor=color, label='Successful'),
            mpatches.Patch(facecolor='#e74c3c', label='Unsuccessful'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

        # Title
        full_title = f"{player_name}\n{title}" if player_name else title
        ax.set_title(full_title, fontsize=16, color='white', fontweight='bold', pad=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)

        return fig

    def create_shot_map(
        self,
        df: pd.DataFrame,
        player_name: str = "",
        title: str = "Shot Map",
        include_xg: bool = True,
        save_path: str = None
    ) -> plt.Figure:
        """
        Create shot map with xG sizing.

        Args:
            df: Events DataFrame with shots
            player_name: Player name for title
            title: Figure title
            include_xg: Whether to size by xG
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 10), facecolor=BACKGROUND)
        self.pitch_drawer.draw_vertical_half(ax, goal_at_top=True)

        shots = df[df['is_shot']] if 'is_shot' in df.columns else df

        if shots.empty:
            ax.text(50, 25, "No shots", ha='center', va='center', color='white', fontsize=14)
        else:
            total_xg = 0
            goals = 0

            for _, shot in shots.iterrows():
                x, y = shot['x'], shot['y']

                # Transform to vertical half-pitch
                new_x = y
                new_y = max(0, x - 50)

                # Get xG for sizing
                xg = shot.get('xG', 0.1)
                if pd.isna(xg):
                    xg = 0.1
                total_xg += xg

                # Size based on xG
                size = xg * 600 + 80

                # Determine marker and color
                is_goal = shot.get('is_goal', False)
                is_on_target = shot.get('is_on_target', False)

                if is_goal:
                    marker = '*'
                    color = '#2ecc71'
                    size *= 1.5
                    goals += 1
                elif is_on_target:
                    marker = 'o'
                    color = '#f39c12'
                else:
                    marker = 'o'
                    color = '#95a5a6'

                ax.scatter(
                    new_x, new_y, s=size, c=color, marker=marker,
                    edgecolors='white', linewidths=1.5, alpha=0.8, zorder=10
                )

            # Add summary
            summary_text = f"Shots: {len(shots)} | Goals: {goals} | xG: {total_xg:.2f}"
            ax.text(50, -8, summary_text, ha='center', va='center',
                   color='white', fontsize=11, fontweight='bold')

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#2ecc71',
                      markersize=12, label='Goal', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12',
                      markersize=10, label='On Target', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#95a5a6',
                      markersize=10, label='Off Target', linestyle='None'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        # Title
        full_title = f"{player_name}\n{title}" if player_name else title
        ax.set_title(full_title, fontsize=16, color='white', fontweight='bold', pad=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)

        return fig

    def create_defensive_actions_map(
        self,
        df: pd.DataFrame,
        player_name: str = "",
        title: str = "Defensive Actions",
        save_path: str = None
    ) -> plt.Figure:
        """
        Create defensive actions map.

        Args:
            df: Events DataFrame
            player_name: Player name for title
            title: Figure title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8), facecolor=BACKGROUND)
        self.pitch_drawer.draw_pitch(ax)

        # Filter defensive actions
        defensive = df[
            (df['is_tackle']) | (df['is_interception']) |
            (df['is_clearance']) | (df['is_blocked'])
        ] if all(col in df.columns for col in ['is_tackle', 'is_interception']) else pd.DataFrame()

        if defensive.empty:
            ax.text(50, 50, "No defensive actions", ha='center', va='center',
                   color='white', fontsize=14)
        else:
            markers = {
                'is_tackle': ('o', '#3498db', 'Tackle'),
                'is_interception': ('s', '#e74c3c', 'Interception'),
                'is_clearance': ('^', '#f39c12', 'Clearance'),
                'is_blocked': ('D', '#9b59b6', 'Block'),
            }

            legend_handles = []

            for action_type, (marker, color, label) in markers.items():
                if action_type in defensive.columns:
                    action_df = defensive[defensive[action_type]]
                    if not action_df.empty:
                        ax.scatter(
                            action_df['x'], action_df['y'],
                            marker=marker, c=color, s=80,
                            edgecolors='white', linewidths=1, alpha=0.8, zorder=10
                        )
                        legend_handles.append(
                            plt.Line2D([0], [0], marker=marker, color='w',
                                      markerfacecolor=color, markersize=10,
                                      label=f"{label} ({len(action_df)})", linestyle='None')
                        )

            if legend_handles:
                ax.legend(handles=legend_handles, loc='upper left', fontsize=8)

        # Title
        full_title = f"{player_name}\n{title}" if player_name else title
        ax.set_title(full_title, fontsize=16, color='white', fontweight='bold', pad=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)

        return fig

    def create_carry_map(
        self,
        df: pd.DataFrame,
        player_name: str = "",
        title: str = "Carry Map",
        highlight_progressive: bool = True,
        color: str = '#9b59b6',
        save_path: str = None
    ) -> plt.Figure:
        """
        Create carry/dribble map.

        Args:
            df: Events DataFrame
            player_name: Player name for title
            title: Figure title
            highlight_progressive: Whether to highlight progressive carries
            color: Primary color
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8), facecolor=BACKGROUND)
        self.pitch_drawer.draw_pitch(ax)

        carries = df[df['is_dribble']] if 'is_dribble' in df.columns else pd.DataFrame()

        if carries.empty:
            ax.text(50, 50, "No carries", ha='center', va='center', color='white', fontsize=14)
        else:
            for _, c in carries.iterrows():
                x, y = c['x'], c['y']
                end_x = c.get('end_x', x + 5)
                end_y = c.get('end_y', y)

                if pd.isna(end_x) or pd.isna(end_y):
                    continue

                # Determine color
                is_progressive = c.get('is_progressive_carry', False)
                is_successful = c.get('outcome_success', True)

                if highlight_progressive and is_progressive:
                    carry_color = '#2ecc71'
                    alpha = 0.9
                    lw = 2.0
                elif is_successful:
                    carry_color = color
                    alpha = 0.6
                    lw = 1.5
                else:
                    carry_color = '#e74c3c'
                    alpha = 0.4
                    lw = 1.0

                # Draw line with dot at end
                ax.plot([x, end_x], [y, end_y], color=carry_color, alpha=alpha, lw=lw)
                ax.scatter(end_x, end_y, c=carry_color, s=30, alpha=alpha, zorder=5)

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='#2ecc71', label='Progressive'),
            mpatches.Patch(facecolor=color, label='Successful'),
            mpatches.Patch(facecolor='#e74c3c', label='Failed'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

        # Title
        full_title = f"{player_name}\n{title}" if player_name else title
        ax.set_title(full_title, fontsize=16, color='white', fontweight='bold', pad=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)

        return fig

    def create_average_position(
        self,
        df: pd.DataFrame,
        player_name: str = "",
        title: str = "Average Position",
        color: str = '#3498db',
        save_path: str = None
    ) -> plt.Figure:
        """
        Create average position with confidence ellipse.

        Args:
            df: Events DataFrame
            player_name: Player name for title
            title: Figure title
            color: Point color
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8), facecolor=BACKGROUND)
        self.pitch_drawer.draw_pitch(ax)

        if df.empty or 'x' not in df.columns:
            ax.text(50, 50, "No data", ha='center', va='center', color='white', fontsize=14)
        else:
            x = df['x'].dropna().values
            y = df['y'].dropna().values

            if len(x) > 2:
                # Calculate mean and covariance
                mean_x = np.mean(x)
                mean_y = np.mean(y)

                # Calculate covariance for ellipse
                cov = np.cov(x, y)
                eigenvalues, eigenvectors = np.linalg.eig(cov)

                # Sort by eigenvalue
                order = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]

                # Ellipse parameters
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                width = 2 * np.sqrt(eigenvalues[0]) * 2  # 2 std deviations
                height = 2 * np.sqrt(eigenvalues[1]) * 2

                # Draw confidence ellipse
                ellipse = mpatches.Ellipse(
                    (mean_x, mean_y), width, height, angle=angle,
                    fill=True, facecolor=color, alpha=0.3, edgecolor=color, lw=2
                )
                ax.add_patch(ellipse)

                # Draw center point
                ax.scatter(mean_x, mean_y, c=color, s=200, edgecolors='white',
                          linewidths=2, zorder=10)

                # Add coordinates text
                ax.text(mean_x, mean_y - 8, f"({mean_x:.1f}, {mean_y:.1f})",
                       ha='center', va='top', color='white', fontsize=10)

        # Title
        full_title = f"{player_name}\n{title}" if player_name else title
        ax.set_title(full_title, fontsize=16, color='white', fontweight='bold', pad=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)

        return fig

    def create_key_passes_map(
        self,
        df: pd.DataFrame,
        player_name: str = "",
        title: str = "Key Passes & Assists",
        save_path: str = None
    ) -> plt.Figure:
        """
        Create key passes and assists map.

        Args:
            df: Events DataFrame
            player_name: Player name for title
            title: Figure title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 10), facecolor=BACKGROUND)
        self.pitch_drawer.draw_vertical_half(ax, goal_at_top=True)

        # Filter key passes and assists
        key_passes = df[
            (df['is_key_pass']) | (df['is_assist'])
        ] if 'is_key_pass' in df.columns else pd.DataFrame()

        if key_passes.empty:
            ax.text(50, 25, "No key passes", ha='center', va='center',
                   color='white', fontsize=14)
        else:
            assists = 0
            for _, p in key_passes.iterrows():
                x, y = p['x'], p['y']
                end_x = p.get('end_x', x)
                end_y = p.get('end_y', y)

                # Transform to vertical half
                new_x, new_y = y, max(0, x - 50)
                new_end_x = end_y if not pd.isna(end_y) else new_x
                new_end_y = max(0, end_x - 50) if not pd.isna(end_x) else new_y

                is_assist = p.get('is_assist', False)
                if is_assist:
                    color = '#2ecc71'
                    lw = 2.5
                    assists += 1
                else:
                    color = '#f39c12'
                    lw = 1.5

                ax.annotate(
                    '', xy=(new_end_x, new_end_y), xytext=(new_x, new_y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw)
                )

            # Summary
            summary = f"Key Passes: {len(key_passes)} | Assists: {assists}"
            ax.text(50, -8, summary, ha='center', va='center',
                   color='white', fontsize=11, fontweight='bold')

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor='#2ecc71', label='Assist'),
            mpatches.Patch(facecolor='#f39c12', label='Key Pass'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        # Title
        full_title = f"{player_name}\n{title}" if player_name else title
        ax.set_title(full_title, fontsize=16, color='white', fontweight='bold', pad=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)

        return fig
