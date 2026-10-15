"""
Non-pitch chart visualizations for scouting reports.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

# Color scheme
BACKGROUND = '#0e1117'
TEXT_COLOR = '#fafafa'
TEXT_SECONDARY = '#8b949e'
GRID_COLOR = '#30363d'


class ChartVisualizations:
    """
    Creates non-pitch chart visualizations for player scouting.
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize visualizations.

        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir) if output_dir else Path('outputs/figures')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_radar_chart(
        self,
        values: Dict[str, float],
        labels: List[str] = None,
        player_name: str = "",
        title: str = "Percentile Radar",
        color: str = '#3498db',
        fill_alpha: float = 0.3,
        save_path: str = None
    ) -> plt.Figure:
        """
        Create percentile radar chart.

        Args:
            values: Dictionary of metric -> percentile (0-100)
            labels: Optional custom labels (defaults to metric names)
            player_name: Player name for title
            title: Chart title
            color: Primary color
            fill_alpha: Fill transparency
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if not values:
            fig, ax = plt.subplots(figsize=(10, 10), facecolor=BACKGROUND)
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                   color='white', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return fig

        # Prepare data
        metrics = list(values.keys())
        percentiles = [values[m] for m in metrics]
        labels = labels or [m.replace('_', ' ').title() for m in metrics]

        # Number of variables
        num_vars = len(metrics)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Complete the data circle
        percentiles += percentiles[:1]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True),
                               facecolor=BACKGROUND)
        ax.set_facecolor(BACKGROUND)

        # Draw the outline
        ax.plot(angles, percentiles, 'o-', linewidth=2, color=color, markersize=6)
        ax.fill(angles, percentiles, alpha=fill_alpha, color=color)

        # Draw reference lines at 25, 50, 75 percentiles
        for ref_line in [25, 50, 75]:
            ax.plot(angles, [ref_line] * len(angles), '--', color=GRID_COLOR, lw=0.8, alpha=0.5)

        # Fix axis to go in the right direction
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw axis lines for each angle
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10, color=TEXT_COLOR)

        # Set limits
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25', '50', '75', '100'], fontsize=8, color=TEXT_SECONDARY)

        # Style
        ax.spines['polar'].set_color(GRID_COLOR)
        ax.grid(color=GRID_COLOR, alpha=0.3)

        # Title
        full_title = f"{player_name}\n{title}" if player_name else title
        ax.set_title(full_title, fontsize=16, color=TEXT_COLOR, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)

        return fig

    def create_comparison_radar(
        self,
        player_values: Dict[str, float],
        comparison_values: Dict[str, float],
        player_name: str = "Player",
        comparison_name: str = "League Average",
        title: str = "Comparison Radar",
        player_color: str = '#3498db',
        comparison_color: str = '#e74c3c',
        save_path: str = None
    ) -> plt.Figure:
        """
        Create comparison radar with two players/benchmarks.

        Args:
            player_values: Player's percentiles
            comparison_values: Comparison percentiles
            player_name: Player name
            comparison_name: Comparison label
            title: Chart title
            player_color: Player color
            comparison_color: Comparison color
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        # Get common metrics
        metrics = list(set(player_values.keys()) & set(comparison_values.keys()))

        if not metrics:
            fig, ax = plt.subplots(figsize=(10, 10), facecolor=BACKGROUND)
            ax.text(0.5, 0.5, "No common metrics", ha='center', va='center',
                   color='white', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return fig

        labels = [m.replace('_', ' ').title() for m in metrics]
        player_vals = [player_values[m] for m in metrics]
        comparison_vals = [comparison_values[m] for m in metrics]

        # Number of variables
        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        player_vals += player_vals[:1]
        comparison_vals += comparison_vals[:1]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True),
                               facecolor=BACKGROUND)
        ax.set_facecolor(BACKGROUND)

        # Draw both radars
        ax.plot(angles, player_vals, 'o-', linewidth=2, color=player_color,
               markersize=6, label=player_name)
        ax.fill(angles, player_vals, alpha=0.25, color=player_color)

        ax.plot(angles, comparison_vals, 'o-', linewidth=2, color=comparison_color,
               markersize=6, label=comparison_name)
        ax.fill(angles, comparison_vals, alpha=0.15, color=comparison_color)

        # Reference lines
        for ref_line in [25, 50, 75]:
            ax.plot(angles, [ref_line] * len(angles), '--', color=GRID_COLOR, lw=0.8, alpha=0.5)

        # Styling
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9, color=TEXT_COLOR)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25', '50', '75', '100'], fontsize=8, color=TEXT_SECONDARY)
        ax.spines['polar'].set_color(GRID_COLOR)
        ax.grid(color=GRID_COLOR, alpha=0.3)

        # Legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=10)

        # Title
        ax.set_title(title, fontsize=16, color=TEXT_COLOR, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)

        return fig

    def create_percentile_bar(
        self,
        values: Dict[str, float],
        player_name: str = "",
        title: str = "Percentile Rankings",
        color: str = '#3498db',
        save_path: str = None
    ) -> plt.Figure:
        """
        Create horizontal percentile bar chart.

        Args:
            values: Dictionary of metric -> percentile (0-100)
            player_name: Player name for title
            title: Chart title
            color: Bar color
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if not values:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor=BACKGROUND)
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                   color='white', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return fig

        # Sort by percentile
        sorted_items = sorted(values.items(), key=lambda x: x[1], reverse=True)
        metrics = [item[0].replace('_', ' ').title() for item in sorted_items]
        percentiles = [item[1] for item in sorted_items]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(6, len(metrics) * 0.4)), facecolor=BACKGROUND)
        ax.set_facecolor(BACKGROUND)

        # Create bars
        y_pos = np.arange(len(metrics))
        bars = ax.barh(y_pos, percentiles, color=color, alpha=0.8, height=0.6)

        # Color bars based on percentile
        for bar, pct in zip(bars, percentiles):
            if pct >= 75:
                bar.set_color('#2ecc71')
            elif pct >= 50:
                bar.set_color('#3498db')
            elif pct >= 25:
                bar.set_color('#f39c12')
            else:
                bar.set_color('#e74c3c')

        # Add percentile labels
        for i, (bar, pct) in enumerate(zip(bars, percentiles)):
            ax.text(pct + 2, i, f'{pct:.0f}%', va='center', ha='left',
                   fontsize=10, color=TEXT_COLOR, fontweight='bold')

        # Reference lines
        for ref in [25, 50, 75]:
            ax.axvline(x=ref, color=GRID_COLOR, linestyle='--', lw=1, alpha=0.5)

        # Styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics, fontsize=10, color=TEXT_COLOR)
        ax.set_xlim(0, 110)
        ax.set_xlabel('Percentile', fontsize=12, color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.invert_yaxis()

        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

        # Title
        full_title = f"{player_name}\n{title}" if player_name else title
        ax.set_title(full_title, fontsize=16, color=TEXT_COLOR, fontweight='bold', pad=15)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)

        return fig

    def create_rolling_line_chart(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        player_name: str = "",
        title: str = "Rolling Performance",
        window: int = 5,
        colors: List[str] = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        Create rolling average line chart.

        Args:
            df: Match-by-match DataFrame
            metrics: Metrics to plot
            player_name: Player name for title
            title: Chart title
            window: Rolling window size
            colors: Colors for each metric
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if df.empty or not metrics:
            fig, ax = plt.subplots(figsize=(12, 6), facecolor=BACKGROUND)
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                   color='white', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return fig

        default_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        colors = colors or default_colors[:len(metrics)]

        fig, ax = plt.subplots(figsize=(12, 6), facecolor=BACKGROUND)
        ax.set_facecolor(BACKGROUND)

        x = range(len(df))

        for i, metric in enumerate(metrics):
            if metric not in df.columns:
                continue

            # Calculate rolling average
            rolling_col = f'{metric}_rolling_{window}'
            if rolling_col in df.columns:
                values = df[rolling_col]
            else:
                values = df[metric].rolling(window=window, min_periods=1).mean()

            color = colors[i % len(colors)]
            label = metric.replace('_', ' ').title()

            # Plot line
            ax.plot(x, values, color=color, lw=2, label=label)

            # Plot actual values as dots
            ax.scatter(x, df[metric], color=color, s=30, alpha=0.5, zorder=5)

        # Styling
        ax.set_xlabel('Match', fontsize=12, color=TEXT_COLOR)
        ax.set_ylabel('Value (per 90)', fontsize=12, color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.grid(True, alpha=0.3, color=GRID_COLOR)
        ax.legend(loc='upper right', fontsize=9)

        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

        # Title
        full_title = f"{player_name}\n{title} ({window}-Match Rolling Average)" if player_name else f"{title} ({window}-Match Rolling Average)"
        ax.set_title(full_title, fontsize=14, color=TEXT_COLOR, fontweight='bold', pad=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)

        return fig

    def create_scatter_plot(
        self,
        df: pd.DataFrame,
        x_metric: str,
        y_metric: str,
        player_value: Tuple[float, float] = None,
        player_name: str = "",
        title: str = None,
        highlight_color: str = '#e74c3c',
        save_path: str = None
    ) -> plt.Figure:
        """
        Create scatter plot with player highlighted.

        Args:
            df: Cohort DataFrame
            x_metric: X-axis metric
            y_metric: Y-axis metric
            player_value: (x, y) for subject player
            player_name: Player name
            title: Chart title
            highlight_color: Color for player point
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if df.empty or x_metric not in df.columns or y_metric not in df.columns:
            fig, ax = plt.subplots(figsize=(10, 8), facecolor=BACKGROUND)
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                   color='white', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return fig

        fig, ax = plt.subplots(figsize=(10, 8), facecolor=BACKGROUND)
        ax.set_facecolor(BACKGROUND)

        # Plot cohort
        ax.scatter(df[x_metric], df[y_metric], c='#3498db', s=60, alpha=0.6,
                  edgecolors='white', linewidths=0.5, label='Cohort')

        # Highlight player
        if player_value:
            ax.scatter(player_value[0], player_value[1], c=highlight_color, s=200,
                      edgecolors='white', linewidths=2, zorder=10, label=player_name)

        # Add trend line
        if len(df) >= 3:
            z = np.polyfit(df[x_metric], df[y_metric], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df[x_metric].min(), df[x_metric].max(), 100)
            ax.plot(x_line, p(x_line), '--', color=TEXT_SECONDARY, lw=1.5, alpha=0.5)

        # Styling
        ax.set_xlabel(x_metric.replace('_', ' ').title(), fontsize=12, color=TEXT_COLOR)
        ax.set_ylabel(y_metric.replace('_', ' ').title(), fontsize=12, color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.grid(True, alpha=0.3, color=GRID_COLOR)
        ax.legend(loc='upper left', fontsize=9)

        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

        # Title
        title = title or f"{x_metric.replace('_', ' ').title()} vs {y_metric.replace('_', ' ').title()}"
        ax.set_title(title, fontsize=14, color=TEXT_COLOR, fontweight='bold', pad=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)

        return fig

    def create_zone_bar_chart(
        self,
        zone_values: Dict[str, float],
        player_name: str = "",
        title: str = "Touches by Zone",
        colors: Dict[str, str] = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        Create bar chart for zone distribution.

        Args:
            zone_values: Dictionary of zone -> value
            player_name: Player name for title
            title: Chart title
            colors: Colors for each zone
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if not zone_values:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor=BACKGROUND)
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                   color='white', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return fig

        default_colors = {
            'defensive': '#3498db',
            'middle': '#f39c12',
            'attacking': '#2ecc71',
            'left': '#9b59b6',
            'central': '#e74c3c',
            'right': '#1abc9c',
        }
        colors = colors or {}

        zones = list(zone_values.keys())
        values = list(zone_values.values())
        bar_colors = [colors.get(z, default_colors.get(z.lower(), '#3498db')) for z in zones]

        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BACKGROUND)
        ax.set_facecolor(BACKGROUND)

        bars = ax.bar(zones, values, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=10,
                   color=TEXT_COLOR, fontweight='bold')

        # Styling
        ax.set_xlabel('Zone', fontsize=12, color=TEXT_COLOR)
        ax.set_ylabel('Value', fontsize=12, color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=10)

        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

        # Title
        full_title = f"{player_name}\n{title}" if player_name else title
        ax.set_title(full_title, fontsize=14, color=TEXT_COLOR, fontweight='bold', pad=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)

        return fig

    def create_game_state_chart(
        self,
        game_state_values: Dict[str, Dict[str, float]],
        player_name: str = "",
        title: str = "Performance by Game State",
        save_path: str = None
    ) -> plt.Figure:
        """
        Create grouped bar chart for game state splits.

        Args:
            game_state_values: {metric: {game_state: value}}
            player_name: Player name for title
            title: Chart title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if not game_state_values:
            fig, ax = plt.subplots(figsize=(12, 6), facecolor=BACKGROUND)
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                   color='white', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return fig

        metrics = list(game_state_values.keys())
        game_states = list(list(game_state_values.values())[0].keys())

        x = np.arange(len(metrics))
        width = 0.25
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Winning, Drawing, Losing

        fig, ax = plt.subplots(figsize=(12, 6), facecolor=BACKGROUND)
        ax.set_facecolor(BACKGROUND)

        for i, (state, color) in enumerate(zip(game_states, colors)):
            values = [game_state_values[m].get(state, 0) for m in metrics]
            ax.bar(x + i * width, values, width, label=state, color=color, alpha=0.8)

        # Styling
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics],
                          fontsize=10, color=TEXT_COLOR)
        ax.set_ylabel('Value (per 90)', fontsize=12, color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, color=GRID_COLOR)

        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

        # Title
        full_title = f"{player_name}\n{title}" if player_name else title
        ax.set_title(full_title, fontsize=14, color=TEXT_COLOR, fontweight='bold', pad=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)

        return fig

    def create_histogram(
        self,
        values: np.ndarray,
        player_value: float = None,
        player_name: str = "",
        title: str = "Distribution",
        xlabel: str = "Value",
        color: str = '#3498db',
        bins: int = 20,
        save_path: str = None
    ) -> plt.Figure:
        """
        Create histogram with player value highlighted.

        Args:
            values: Array of values for distribution
            player_value: Player's value to highlight
            player_name: Player name
            title: Chart title
            xlabel: X-axis label
            color: Histogram color
            bins: Number of bins
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if len(values) == 0:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor=BACKGROUND)
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                   color='white', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return fig

        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BACKGROUND)
        ax.set_facecolor(BACKGROUND)

        # Plot histogram
        ax.hist(values, bins=bins, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)

        # Highlight player value
        if player_value is not None:
            ax.axvline(x=player_value, color='#e74c3c', lw=2.5, linestyle='--',
                      label=f'{player_name}: {player_value:.2f}')
            ax.legend(fontsize=10)

        # Styling
        ax.set_xlabel(xlabel, fontsize=12, color=TEXT_COLOR)
        ax.set_ylabel('Frequency', fontsize=12, color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR)

        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

        # Title
        ax.set_title(title, fontsize=14, color=TEXT_COLOR, fontweight='bold', pad=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)

        return fig

    def create_dot_plot(
        self,
        player_value: float,
        cohort_mean: float,
        cohort_values: np.ndarray,
        metric_name: str,
        player_name: str = "",
        save_path: str = None
    ) -> plt.Figure:
        """
        Create dot plot comparing player to cohort.

        Args:
            player_value: Player's value
            cohort_mean: Cohort mean
            cohort_values: All cohort values
            metric_name: Metric name
            player_name: Player name
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 3), facecolor=BACKGROUND)
        ax.set_facecolor(BACKGROUND)

        # Plot cohort distribution
        ax.scatter(cohort_values, [0] * len(cohort_values), c='#3498db', s=40,
                  alpha=0.5, edgecolors='white', linewidths=0.5)

        # Plot cohort mean
        ax.axvline(x=cohort_mean, color=TEXT_SECONDARY, lw=2, linestyle='--',
                  label=f'Cohort Mean: {cohort_mean:.2f}')

        # Plot player value
        ax.scatter([player_value], [0], c='#e74c3c', s=200, edgecolors='white',
                  linewidths=2, zorder=10, label=f'{player_name}: {player_value:.2f}')

        # Styling
        ax.set_xlabel(metric_name.replace('_', ' ').title(), fontsize=12, color=TEXT_COLOR)
        ax.set_yticks([])
        ax.tick_params(colors=TEXT_COLOR)
        ax.legend(loc='upper right', fontsize=9)

        for spine in ['top', 'left', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color(GRID_COLOR)

        # Title
        ax.set_title(f"{metric_name.replace('_', ' ').title()} vs Cohort",
                    fontsize=12, color=TEXT_COLOR, fontweight='bold', pad=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BACKGROUND)

        return fig
