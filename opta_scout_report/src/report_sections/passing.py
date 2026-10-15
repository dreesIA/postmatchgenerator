"""Passing analysis section."""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .base import BaseSection, SectionContent, Takeaway


class PassingSection(BaseSection):
    """Comprehensive passing analysis."""

    @property
    def section_title(self) -> str:
        return "Passing"

    def generate(self) -> SectionContent:
        """Generate the passing section."""
        content = SectionContent(title=self.section_title)

        # Create pass map
        pass_map_fig = self._create_pass_map()
        if pass_map_fig:
            content.add_figure(pass_map_fig, "Completed Passes")

        # Create progressive passes map
        prog_pass_fig = self._create_progressive_pass_map()
        if prog_pass_fig:
            content.add_figure(prog_pass_fig, "Progressive Passes")

        # Create passing metrics comparison
        metrics_fig = self._create_passing_metrics()
        if metrics_fig:
            content.add_figure(metrics_fig, "Passing Metrics vs Position Average")

        # Create pass type breakdown
        pass_type_fig = self._create_pass_type_breakdown()
        if pass_type_fig:
            content.add_figure(pass_type_fig, "Pass Type Breakdown")

        # Create pass direction chart
        direction_fig = self._create_pass_direction_chart()
        if direction_fig:
            content.add_figure(direction_fig, "Pass Direction Distribution")

        # Generate takeaways
        takeaways = self._generate_takeaways()
        for takeaway in takeaways:
            content.takeaways.append(takeaway)

        # Add passing table
        passing_table = self._get_passing_table()
        content.add_table(passing_table, "Passing Metrics")

        return content

    def _create_pass_map(self) -> Figure:
        """Create pass map visualization."""
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None

        if events_df is None or events_df.empty:
            return None

        # Filter to successful passes
        passes = events_df[
            (events_df['event_type'] == 'pass') &
            (events_df['outcome'] == 1)
        ].copy() if 'event_type' in events_df.columns else None

        # Fallback: try type_id for Opta
        if passes is None or passes.empty:
            passes = events_df[
                (events_df.get('type_id', '') == 1) &
                (events_df.get('outcome', 1) == 1)
            ].copy() if 'type_id' in events_df.columns else None

        if passes is None or passes.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor('#0a4d0a')

        self._draw_pitch(ax)

        # Sample passes if too many
        if len(passes) > 200:
            passes = passes.sample(n=200)

        # Draw passes
        for _, row in passes.iterrows():
            if 'end_x' in row and 'end_y' in row:
                x_start, y_start = row['x'], row['y']
                x_end, y_end = row['end_x'], row['end_y']

                if np.isnan(x_start) or np.isnan(y_start) or np.isnan(x_end) or np.isnan(y_end):
                    continue

                # Color based on pass distance/progression
                dx = x_end - x_start
                if dx > 10:  # Progressive
                    color = self.colors['positive']
                    alpha = 0.7
                else:
                    color = self.colors['text']
                    alpha = 0.4

                ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                           arrowprops=dict(arrowstyle='->', color=color,
                                          alpha=alpha, linewidth=0.8))

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.axis('off')

        ax.set_title(f"Completed Passes (sample of {len(passes)})",
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_progressive_pass_map(self) -> Figure:
        """Create progressive passes map."""
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None

        if events_df is None or events_df.empty:
            return None

        # Filter to progressive passes
        prog_passes = events_df[
            (events_df.get('is_progressive_pass', False) == True)
        ].copy() if 'is_progressive_pass' in events_df.columns else None

        if prog_passes is None or prog_passes.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor('#0a4d0a')

        self._draw_pitch(ax)

        # Draw progressive passes
        for _, row in prog_passes.iterrows():
            if 'end_x' in row and 'end_y' in row:
                x_start, y_start = row['x'], row['y']
                x_end, y_end = row['end_x'], row['end_y']

                if np.isnan(x_start) or np.isnan(y_start) or np.isnan(x_end) or np.isnan(y_end):
                    continue

                ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                           arrowprops=dict(arrowstyle='->', color=self.colors['positive'],
                                          alpha=0.7, linewidth=1.2))

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.axis('off')

        ax.set_title(f"Progressive Passes ({len(prog_passes)} total)",
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _draw_pitch(self, ax) -> None:
        """Draw pitch markings."""
        ax.plot([0, 100], [0, 0], 'white', linewidth=1)
        ax.plot([0, 100], [100, 100], 'white', linewidth=1)
        ax.plot([0, 0], [0, 100], 'white', linewidth=1)
        ax.plot([100, 100], [0, 100], 'white', linewidth=1)
        ax.plot([50, 50], [0, 100], 'white', linewidth=1)

        circle = plt.Circle((50, 50), 9.15, fill=False, color='white', linewidth=1)
        ax.add_patch(circle)

        ax.plot([0, 16.5], [21.1, 21.1], 'white', linewidth=1)
        ax.plot([0, 16.5], [78.9, 78.9], 'white', linewidth=1)
        ax.plot([16.5, 16.5], [21.1, 78.9], 'white', linewidth=1)

        ax.plot([83.5, 100], [21.1, 21.1], 'white', linewidth=1)
        ax.plot([83.5, 100], [78.9, 78.9], 'white', linewidth=1)
        ax.plot([83.5, 83.5], [21.1, 78.9], 'white', linewidth=1)

    def _create_passing_metrics(self) -> Figure:
        """Create passing metrics comparison chart."""
        metrics = {
            'Pass Accuracy': ('pass_accuracy', '%'),
            'Progressive Passes': ('progressive_passes', 'p90'),
            'Final Third Entries': ('final_third_entries', 'p90'),
            'Penalty Area Entries': ('penalty_area_entries', 'p90'),
            'Key Passes': ('key_passes', 'p90'),
            'Through Balls': ('through_balls', 'p90'),
        }

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        labels = []
        values = []
        percentiles = []
        formats = []

        for display_name, (metric_key, fmt) in metrics.items():
            value = self._get_metric_value(metric_key)
            labels.append(display_name)
            values.append(value)
            percentiles.append(self._get_percentile(metric_key))
            formats.append(fmt)

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

        bars = ax.bar(x, percentiles, width, color=colors, alpha=0.8, edgecolor='white')

        # Add value labels
        for bar, val, fmt, pct in zip(bars, values, formats, percentiles):
            height = bar.get_height()
            if fmt == '%':
                label = f'{val:.1f}%'
            else:
                label = f'{val:.1f}'
            ax.annotate(f'{label}\n({pct:.0f}th)',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom',
                       color=self.colors['text'], fontsize=8)

        ax.set_ylabel('Percentile', color=self.colors['text'])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=self.colors['text'], rotation=20, ha='right')
        ax.set_ylim(0, 105)

        # Add reference lines
        ax.axhline(y=50, color=self.colors['neutral'], linestyle='--', alpha=0.5)
        ax.axhline(y=75, color=self.colors['positive'], linestyle=':', alpha=0.3)
        ax.axhline(y=25, color=self.colors['negative'], linestyle=':', alpha=0.3)

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title('Passing Metrics - Percentile Rankings',
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_pass_type_breakdown(self) -> Figure:
        """Create pass type breakdown pie chart."""
        # Calculate pass type distribution
        short = self._get_metric_value('passes_completed') - self._get_metric_value('long_passes_attempted')
        long_passes = self._get_metric_value('long_passes_attempted')
        crosses = self._get_metric_value('crosses')
        through_balls = self._get_metric_value('through_balls')

        pass_types = {
            'Short/Medium': max(0, short - crosses - through_balls),
            'Long': long_passes,
            'Crosses': crosses,
            'Through Balls': through_balls,
        }

        # Filter out zeros
        pass_types = {k: v for k, v in pass_types.items() if v > 0.1}

        if not pass_types or sum(pass_types.values()) < 1:
            return None

        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor(self.colors['primary'])

        colors = [self.colors['accent'], self.colors['highlight'],
                 self.colors['positive'], self.colors['neutral']][:len(pass_types)]

        wedges, texts, autotexts = ax.pie(
            pass_types.values(),
            labels=pass_types.keys(),
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.75,
            explode=[0.02] * len(pass_types)
        )

        # Style
        for text in texts:
            text.set_color(self.colors['text'])
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title('Pass Type Distribution (Per 90)',
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_pass_direction_chart(self) -> Figure:
        """Create pass direction distribution chart."""
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None

        if events_df is None or events_df.empty:
            return None

        # Filter to passes with end coordinates
        passes = events_df[
            events_df['end_x'].notna() & events_df['end_y'].notna()
        ].copy()

        if passes.empty:
            return None

        # Calculate directions
        passes['dx'] = passes['end_x'] - passes['x']
        passes['dy'] = passes['end_y'] - passes['y']

        # Categorize
        forward = len(passes[passes['dx'] > 5])
        backward = len(passes[passes['dx'] < -5])
        sideways = len(passes[(passes['dx'] >= -5) & (passes['dx'] <= 5)])

        total = forward + backward + sideways
        if total == 0:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        categories = ['Forward', 'Sideways', 'Backward']
        values = [forward/total*100, sideways/total*100, backward/total*100]
        colors = [self.colors['positive'], self.colors['neutral'], self.colors['negative']]

        bars = ax.barh(categories, values, color=colors, alpha=0.8, edgecolor='white')

        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', va='center', color=self.colors['text'])

        ax.set_xlabel('Percentage', color=self.colors['text'])
        ax.set_xlim(0, max(values) * 1.2)

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title('Pass Direction Distribution', color=self.colors['text'],
                    fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _generate_takeaways(self) -> List[Takeaway]:
        """Generate passing takeaways."""
        takeaways = []

        # Pass accuracy
        accuracy = self._get_metric_value('pass_accuracy')
        accuracy_pct = self._get_percentile('pass_accuracy')
        if accuracy > 0:
            descriptor = self._percentile_descriptor(accuracy_pct)
            takeaways.append(Takeaway(
                text=f"Pass accuracy: {accuracy:.1f}% ({descriptor}, {accuracy_pct:.0f}th percentile)",
                metric='pass_accuracy',
                value=accuracy,
                percentile=accuracy_pct,
                is_positive=accuracy_pct >= 50
            ))

        # Progressive passes
        prog = self._get_metric_value('progressive_passes')
        prog_pct = self._get_percentile('progressive_passes')
        if prog > 0:
            if prog_pct >= 70:
                takeaways.append(Takeaway(
                    text=f"Strong ball progression: {prog:.1f} progressive passes per 90 ({prog_pct:.0f}th percentile)",
                    metric='progressive_passes',
                    value=prog,
                    percentile=prog_pct,
                    is_positive=True
                ))
            elif prog_pct <= 30:
                takeaways.append(Takeaway(
                    text=f"Limited ball progression: {prog:.1f} progressive passes per 90 ({prog_pct:.0f}th percentile)",
                    metric='progressive_passes',
                    value=prog,
                    percentile=prog_pct,
                    is_positive=False
                ))

        # Final third entries
        ft_entries = self._get_metric_value('final_third_entries')
        ft_pct = self._get_percentile('final_third_entries')
        if ft_entries > 0 and (ft_pct >= 70 or ft_pct <= 30):
            descriptor = self._percentile_descriptor(ft_pct)
            takeaways.append(Takeaway(
                text=f"Final third entries: {ft_entries:.1f} per 90 ({descriptor})",
                metric='final_third_entries',
                value=ft_entries,
                percentile=ft_pct,
                is_positive=ft_pct >= 50
            ))

        # Key passes
        key = self._get_metric_value('key_passes')
        key_pct = self._get_percentile('key_passes')
        if key > 0 and key_pct >= 70:
            takeaways.append(Takeaway(
                text=f"Strong chance creator: {key:.1f} key passes per 90 ({key_pct:.0f}th percentile)",
                metric='key_passes',
                value=key,
                percentile=key_pct,
                is_positive=True
            ))

        # Long pass accuracy
        long_acc = self._get_metric_value('long_pass_accuracy')
        long_pct = self._get_percentile('long_pass_accuracy')
        if long_acc > 0 and (long_pct >= 75 or long_pct <= 25):
            descriptor = self._percentile_descriptor(long_pct)
            takeaways.append(Takeaway(
                text=f"Long pass accuracy: {long_acc:.1f}% ({descriptor})",
                metric='long_pass_accuracy',
                value=long_acc,
                percentile=long_pct,
                is_positive=long_pct >= 50
            ))

        return takeaways

    def _get_passing_table(self) -> Dict[str, Any]:
        """Get passing metrics table."""
        metrics = [
            'passes_attempted', 'passes_completed', 'pass_accuracy',
            'progressive_passes', 'line_breaking_passes', 'final_third_entries',
            'penalty_area_entries', 'long_passes_attempted', 'long_pass_accuracy',
            'switches', 'through_balls', 'crosses', 'cross_accuracy', 'key_passes'
        ]

        result = {}
        for metric in metrics:
            result[metric] = {
                'value': self._get_metric_value(metric),
                'percentile': self._get_percentile(metric),
            }
        return result
