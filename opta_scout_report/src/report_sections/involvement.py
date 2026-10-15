"""Involvement and ball reception section."""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .base import BaseSection, SectionContent, Takeaway


class InvolvementSection(BaseSection):
    """Analysis of player involvement, touches, and ball reception patterns."""

    @property
    def section_title(self) -> str:
        return "Involvement & Ball Reception"

    def generate(self) -> SectionContent:
        """Generate the involvement section."""
        content = SectionContent(title=self.section_title)

        # Create touch zone chart
        touch_zone_fig = self._create_touch_zone_chart()
        if touch_zone_fig:
            content.add_figure(touch_zone_fig, "Touches by Zone")

        # Create reception heatmap
        reception_fig = self._create_reception_heatmap()
        if reception_fig:
            content.add_figure(reception_fig, "Pass Reception Locations")

        # Create involvement metrics comparison
        involvement_fig = self._create_involvement_comparison()
        if involvement_fig:
            content.add_figure(involvement_fig, "Involvement Metrics vs Position Average")

        # Create touch trend chart
        trend_fig = self._create_touch_trend()
        if trend_fig:
            content.add_figure(trend_fig, "Touches Per 90 Trend")

        # Generate takeaways
        takeaways = self._generate_takeaways()
        for takeaway in takeaways:
            content.takeaways.append(takeaway)

        # Add involvement table
        involvement_table = self._get_involvement_table()
        content.add_table(involvement_table, "Involvement Metrics")

        return content

    def _create_touch_zone_chart(self) -> Figure:
        """Create touches by zone horizontal bar chart."""
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None

        if events_df is None or events_df.empty:
            return None

        # Count touches by zone (horizontal thirds)
        zones = {
            'Defensive Third': (0, 33.33),
            'Middle Third': (33.33, 66.66),
            'Final Third': (66.66, 100),
        }

        zone_counts = {}
        for zone_name, (x_min, x_max) in zones.items():
            count = len(events_df[(events_df['x'] >= x_min) & (events_df['x'] < x_max)])
            zone_counts[zone_name] = count

        total = sum(zone_counts.values())
        if total == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        labels = list(zone_counts.keys())
        values = [zone_counts[z] for z in labels]
        percentages = [v / total * 100 for v in values]

        colors = [self.colors['neutral'], self.colors['accent'], self.colors['highlight']]

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, percentages, color=colors, alpha=0.8, edgecolor='white')

        # Add count and percentage labels
        for i, (bar, count, pct) in enumerate(zip(bars, values, percentages)):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{pct:.1f}% ({count})', va='center',
                   color=self.colors['text'], fontsize=10)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color=self.colors['text'])
        ax.set_xlabel('Percentage of Touches', color=self.colors['text'])

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlim(0, max(percentages) * 1.25)
        ax.set_title('Touch Distribution by Third', color=self.colors['text'],
                    fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_reception_heatmap(self) -> Figure:
        """Create heatmap of pass reception locations."""
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None

        if events_df is None or events_df.empty:
            return None

        # Filter to receptions (passes received)
        # In Opta data, we look for passes where this player is the recipient
        receptions = events_df[
            (events_df['is_recipient'] == True) |
            (events_df['event_type'] == 'pass_received')
        ].copy() if 'is_recipient' in events_df.columns else None

        if receptions is None or receptions.empty:
            # Fallback to all touches
            receptions = events_df[events_df['x'].notna() & events_df['y'].notna()].copy()

        if receptions.empty:
            return None

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor('#0a4d0a')

        # Draw pitch
        self._draw_pitch(ax)

        # Create heatmap
        x = receptions['x'].values
        y = receptions['y'].values

        hb = ax.hexbin(x, y, gridsize=15, cmap='Blues', alpha=0.8,
                      extent=[0, 100, 0, 100], mincnt=1)

        cb = plt.colorbar(hb, ax=ax, shrink=0.8)
        cb.set_label('Reception Density', color=self.colors['text'])
        cb.ax.yaxis.set_tick_params(color=self.colors['text'])
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=self.colors['text'])

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.axis('off')

        ax.set_title('Pass Reception Locations', color=self.colors['text'],
                    fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _draw_pitch(self, ax) -> None:
        """Draw a basic pitch."""
        ax.plot([0, 100], [0, 0], 'white', linewidth=1)
        ax.plot([0, 100], [100, 100], 'white', linewidth=1)
        ax.plot([0, 0], [0, 100], 'white', linewidth=1)
        ax.plot([100, 100], [0, 100], 'white', linewidth=1)
        ax.plot([50, 50], [0, 100], 'white', linewidth=1)

        circle = plt.Circle((50, 50), 9.15, fill=False, color='white', linewidth=1)
        ax.add_patch(circle)

        # 18-yard boxes
        ax.plot([0, 16.5], [21.1, 21.1], 'white', linewidth=1)
        ax.plot([0, 16.5], [78.9, 78.9], 'white', linewidth=1)
        ax.plot([16.5, 16.5], [21.1, 78.9], 'white', linewidth=1)

        ax.plot([83.5, 100], [21.1, 21.1], 'white', linewidth=1)
        ax.plot([83.5, 100], [78.9, 78.9], 'white', linewidth=1)
        ax.plot([83.5, 83.5], [21.1, 78.9], 'white', linewidth=1)

    def _create_involvement_comparison(self) -> Figure:
        """Create involvement metrics comparison chart."""
        metrics = {
            'Touches': 'touches',
            'Passes Received': 'passes_received',
            'Progressive Passes Recv': 'progressive_passes_received',
        }

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        labels = []
        values = []
        percentiles = []

        for display_name, metric_key in metrics.items():
            value = self._get_metric_value(metric_key)
            if value > 0:
                labels.append(display_name)
                values.append(value)
                percentiles.append(self._get_percentile(metric_key))

        if not values:
            plt.close(fig)
            return None

        x = np.arange(len(labels))
        width = 0.6

        # Color by percentile
        colors = []
        for pct in percentiles:
            if pct >= 75:
                colors.append(self.colors['positive'])
            elif pct >= 50:
                colors.append(self.colors['highlight'])
            elif pct >= 25:
                colors.append(self.colors['neutral'])
            else:
                colors.append(self.colors['negative'])

        bars = ax.bar(x, values, width, color=colors, alpha=0.8, edgecolor='white')

        # Add percentile labels
        for bar, pct, val in zip(bars, percentiles, values):
            height = bar.get_height()
            ax.annotate(f'{pct:.0f}th\n({val:.1f})',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom',
                       color=self.colors['text'], fontsize=9)

        ax.set_ylabel('Per 90', color=self.colors['text'])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=self.colors['text'])

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title('Involvement Metrics with Percentile Rankings',
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_touch_trend(self) -> Figure:
        """Create touch count trend over matches."""
        match_metrics = getattr(self.metrics, 'match_by_match', None)

        if match_metrics is None or not match_metrics:
            return None

        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        touches = []
        for match in match_metrics:
            if isinstance(match, dict):
                touches.append(match.get('touches', 0))
            else:
                touches.append(getattr(match, 'touches', 0))

        if sum(touches) == 0:
            plt.close(fig)
            return None

        x = range(len(touches))
        ax.plot(x, touches, color=self.colors['highlight'], marker='o',
               linewidth=2, markersize=6)

        # Add trend line
        z = np.polyfit(list(x), touches, 1)
        p = np.poly1d(z)
        ax.plot(x, p(list(x)), linestyle='--', color=self.colors['positive'],
               linewidth=1.5, alpha=0.7, label='Trend')

        # Add average line
        avg = np.mean(touches)
        ax.axhline(y=avg, color=self.colors['neutral'], linestyle=':',
                  linewidth=1, label=f'Average: {avg:.1f}')

        ax.set_xlabel('Match', color=self.colors['text'])
        ax.set_ylabel('Touches per 90', color=self.colors['text'])

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc='upper right', facecolor=self.colors['secondary'],
                 edgecolor=self.colors['neutral'], labelcolor=self.colors['text'])

        ax.set_title('Touches Per 90 Over Season', color=self.colors['text'],
                    fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _generate_takeaways(self) -> List[Takeaway]:
        """Generate involvement takeaways."""
        takeaways = []

        # Touches analysis
        touches = self._get_metric_value('touches')
        touches_pct = self._get_percentile('touches')
        if touches > 0:
            descriptor = self._percentile_descriptor(touches_pct)
            takeaways.append(Takeaway(
                text=f"Touch frequency: {touches:.1f} per 90 ({descriptor}, {touches_pct:.0f}th percentile)",
                metric='touches',
                value=touches,
                percentile=touches_pct,
                is_positive=touches_pct >= 50
            ))

        # Passes received
        received = self._get_metric_value('passes_received')
        received_pct = self._get_percentile('passes_received')
        if received > 0:
            takeaways.append(Takeaway(
                text=f"Passes received: {received:.1f} per 90 ({received_pct:.0f}th percentile)",
                metric='passes_received',
                value=received,
                percentile=received_pct,
                is_positive=received_pct >= 50
            ))

        # Progressive passes received
        prog_received = self._get_metric_value('progressive_passes_received')
        prog_received_pct = self._get_percentile('progressive_passes_received')
        if prog_received > 0 and prog_received_pct >= 70:
            takeaways.append(Takeaway(
                text=f"Strong target for progressive passes: {prog_received:.1f} per 90 ({prog_received_pct:.0f}th percentile)",
                metric='progressive_passes_received',
                value=prog_received,
                percentile=prog_received_pct,
                is_positive=True
            ))
        elif prog_received > 0 and prog_received_pct <= 30:
            takeaways.append(Takeaway(
                text=f"Rarely receives progressive passes: {prog_received:.1f} per 90 ({prog_received_pct:.0f}th percentile)",
                metric='progressive_passes_received',
                value=prog_received,
                percentile=prog_received_pct,
                is_positive=False
            ))

        return takeaways

    def _get_involvement_table(self) -> Dict[str, Any]:
        """Get involvement metrics table data."""
        return {
            'touches': {
                'value': self._get_metric_value('touches'),
                'percentile': self._get_percentile('touches'),
            },
            'passes_received': {
                'value': self._get_metric_value('passes_received'),
                'percentile': self._get_percentile('passes_received'),
            },
            'progressive_passes_received': {
                'value': self._get_metric_value('progressive_passes_received'),
                'percentile': self._get_percentile('progressive_passes_received'),
            },
        }
