"""Role and usage patterns section."""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .base import BaseSection, SectionContent, Takeaway


class RoleUsageSection(BaseSection):
    """Analysis of player role, position heatmaps, and usage patterns."""

    @property
    def section_title(self) -> str:
        return "Role & Usage"

    def generate(self) -> SectionContent:
        """Generate the role and usage section."""
        content = SectionContent(title=self.section_title)

        # Create heatmap
        heatmap_fig = self._create_position_heatmap()
        if heatmap_fig:
            content.add_figure(heatmap_fig, "Positional Heatmap (All Touches)")

        # Create average position by match
        avg_pos_fig = self._create_average_positions()
        if avg_pos_fig:
            content.add_figure(avg_pos_fig, "Average Position by Match")

        # Create minutes distribution chart
        minutes_fig = self._create_minutes_distribution()
        if minutes_fig:
            content.add_figure(minutes_fig, "Minutes Played Distribution")

        # Create zone activity chart
        zone_fig = self._create_zone_activity()
        if zone_fig:
            content.add_figure(zone_fig, "Activity by Pitch Zone")

        # Generate takeaways
        takeaways = self._generate_takeaways()
        for takeaway in takeaways:
            content.takeaways.append(takeaway)

        # Add usage statistics
        content.raw_data['usage_stats'] = self._get_usage_stats()

        return content

    def _create_position_heatmap(self) -> Figure:
        """Create positional heatmap from touch data."""
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None

        if events_df is None or events_df.empty:
            return None

        # Filter to touches/actions with valid coordinates
        touch_events = events_df[
            events_df['x'].notna() &
            events_df['y'].notna()
        ].copy()

        if touch_events.empty:
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor('#0a4d0a')  # Pitch green

        # Draw pitch
        self._draw_pitch(ax)

        # Create heatmap
        x = touch_events['x'].values
        y = touch_events['y'].values

        # Use hexbin for heatmap
        hb = ax.hexbin(x, y, gridsize=20, cmap='YlOrRd', alpha=0.8,
                       extent=[0, 100, 0, 100], mincnt=1)

        # Colorbar
        cb = plt.colorbar(hb, ax=ax, shrink=0.8)
        cb.set_label('Touch Density', color=self.colors['text'])
        cb.ax.yaxis.set_tick_params(color=self.colors['text'])
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=self.colors['text'])

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.axis('off')

        ax.set_title(f"{self.player_data.info.name} - Touch Heatmap",
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _draw_pitch(self, ax) -> None:
        """Draw a basic pitch on the axes."""
        # Pitch outline
        ax.plot([0, 100], [0, 0], 'white', linewidth=1)
        ax.plot([0, 100], [100, 100], 'white', linewidth=1)
        ax.plot([0, 0], [0, 100], 'white', linewidth=1)
        ax.plot([100, 100], [0, 100], 'white', linewidth=1)

        # Halfway line
        ax.plot([50, 50], [0, 100], 'white', linewidth=1)

        # Center circle
        circle = plt.Circle((50, 50), 9.15, fill=False, color='white', linewidth=1)
        ax.add_patch(circle)

        # 18-yard boxes
        ax.plot([0, 16.5], [21.1, 21.1], 'white', linewidth=1)
        ax.plot([0, 16.5], [78.9, 78.9], 'white', linewidth=1)
        ax.plot([16.5, 16.5], [21.1, 78.9], 'white', linewidth=1)

        ax.plot([83.5, 100], [21.1, 21.1], 'white', linewidth=1)
        ax.plot([83.5, 100], [78.9, 78.9], 'white', linewidth=1)
        ax.plot([83.5, 83.5], [21.1, 78.9], 'white', linewidth=1)

        # 6-yard boxes
        ax.plot([0, 5.5], [36.8, 36.8], 'white', linewidth=1)
        ax.plot([0, 5.5], [63.2, 63.2], 'white', linewidth=1)
        ax.plot([5.5, 5.5], [36.8, 63.2], 'white', linewidth=1)

        ax.plot([94.5, 100], [36.8, 36.8], 'white', linewidth=1)
        ax.plot([94.5, 100], [63.2, 63.2], 'white', linewidth=1)
        ax.plot([94.5, 94.5], [36.8, 63.2], 'white', linewidth=1)

    def _create_average_positions(self) -> Figure:
        """Create average position visualization by match."""
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None

        if events_df is None or events_df.empty or 'match_id' not in events_df.columns:
            return None

        # Calculate average position per match
        avg_positions = events_df.groupby('match_id').agg({
            'x': 'mean',
            'y': 'mean'
        }).reset_index()

        if avg_positions.empty:
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor('#0a4d0a')

        self._draw_pitch(ax)

        # Plot average positions
        scatter = ax.scatter(avg_positions['x'], avg_positions['y'],
                            s=150, c=range(len(avg_positions)),
                            cmap='coolwarm', alpha=0.8, edgecolor='white', linewidth=1)

        # Overall average position
        overall_x = avg_positions['x'].mean()
        overall_y = avg_positions['y'].mean()
        ax.scatter(overall_x, overall_y, s=300, c=self.colors['highlight'],
                  marker='*', edgecolor='white', linewidth=2, zorder=10)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.axis('off')

        ax.set_title(f"Average Position by Match (★ = Season Average)",
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_minutes_distribution(self) -> Figure:
        """Create minutes played distribution chart."""
        appearances = self.player_data.appearances if hasattr(self.player_data, 'appearances') else []

        if not appearances:
            return None

        minutes_list = [app.minutes_played for app in appearances if app.minutes_played > 0]

        if not minutes_list:
            return None

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        # Bar chart of minutes per match
        x = range(len(minutes_list))
        colors = [self.colors['highlight'] if m >= 60 else self.colors['neutral'] for m in minutes_list]
        bars = ax.bar(x, minutes_list, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)

        # Add average line
        avg_minutes = np.mean(minutes_list)
        ax.axhline(y=avg_minutes, color=self.colors['positive'], linestyle='--',
                  linewidth=2, label=f'Average: {avg_minutes:.0f} mins')

        ax.set_xlabel('Match', color=self.colors['text'])
        ax.set_ylabel('Minutes Played', color=self.colors['text'])
        ax.set_title('Minutes Played Distribution', color=self.colors['text'],
                    fontsize=12, fontweight='bold')

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc='upper right', facecolor=self.colors['secondary'],
                 edgecolor=self.colors['neutral'], labelcolor=self.colors['text'])

        plt.tight_layout()
        return fig

    def _create_zone_activity(self) -> Figure:
        """Create zone activity breakdown chart."""
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None

        if events_df is None or events_df.empty or 'zone' not in events_df.columns:
            return None

        # Count actions by zone
        zone_counts = events_df['zone'].value_counts()

        if zone_counts.empty:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        # Define zone order
        zone_order = ['defensive_third', 'middle_third', 'final_third', 'box']
        zone_labels = ['Defensive Third', 'Middle Third', 'Final Third', 'Penalty Area']

        values = [zone_counts.get(z, 0) for z in zone_order]
        total = sum(values)
        percentages = [v / total * 100 if total > 0 else 0 for v in values]

        colors = [self.colors['neutral'], self.colors['accent'],
                 self.colors['highlight'], self.colors['positive']]

        bars = ax.barh(zone_labels, percentages, color=colors, alpha=0.8,
                      edgecolor='white', linewidth=0.5)

        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{pct:.1f}%', va='center', color=self.colors['text'], fontsize=10)

        ax.set_xlabel('% of Actions', color=self.colors['text'])
        ax.set_title('Activity Distribution by Zone', color=self.colors['text'],
                    fontsize=12, fontweight='bold')

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlim(0, max(percentages) * 1.15)

        plt.tight_layout()
        return fig

    def _generate_takeaways(self) -> List[Takeaway]:
        """Generate takeaways about role and usage."""
        takeaways = []

        # Minutes analysis
        appearances = self.player_data.appearances if hasattr(self.player_data, 'appearances') else []
        if appearances:
            minutes_list = [app.minutes_played for app in appearances]
            total_minutes = sum(minutes_list)
            avg_minutes = np.mean(minutes_list)
            full_games = sum(1 for m in minutes_list if m >= 85)
            sub_appearances = sum(1 for m in minutes_list if 0 < m < 45)

            takeaways.append(Takeaway(
                text=f"Availability: {len(appearances)} appearances, {total_minutes:.0f} total minutes",
                is_positive=True
            ))

            if full_games > len(appearances) * 0.6:
                takeaways.append(Takeaway(
                    text=f"Regular starter: {full_games}/{len(appearances)} full games ({full_games/len(appearances)*100:.0f}%)",
                    is_positive=True
                ))
            elif sub_appearances > len(appearances) * 0.4:
                takeaways.append(Takeaway(
                    text=f"Often used as substitute: {sub_appearances}/{len(appearances)} sub appearances",
                    is_positive=False
                ))

        # Zone analysis
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None
        if events_df is not None and 'zone' in events_df.columns:
            zone_counts = events_df['zone'].value_counts(normalize=True) * 100

            if 'final_third' in zone_counts and zone_counts['final_third'] > 40:
                takeaways.append(Takeaway(
                    text=f"Highly involved in final third: {zone_counts['final_third']:.1f}% of actions",
                    is_positive=True
                ))
            elif 'defensive_third' in zone_counts and zone_counts['defensive_third'] > 40:
                takeaways.append(Takeaway(
                    text=f"Deep-lying role: {zone_counts['defensive_third']:.1f}% of actions in defensive third",
                    is_positive=True
                ))

        return takeaways

    def _get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics."""
        appearances = self.player_data.appearances if hasattr(self.player_data, 'appearances') else []

        if not appearances:
            return {}

        minutes_list = [app.minutes_played for app in appearances]

        return {
            'total_appearances': len(appearances),
            'total_minutes': sum(minutes_list),
            'avg_minutes': np.mean(minutes_list) if minutes_list else 0,
            'median_minutes': np.median(minutes_list) if minutes_list else 0,
            'full_90s': sum(1 for m in minutes_list if m >= 85),
            'starts': sum(1 for app in appearances if getattr(app, 'is_starter', m >= 45)),
            'sub_appearances': sum(1 for m in minutes_list if 0 < m < 45),
        }
