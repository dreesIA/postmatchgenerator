"""Carrying and dribbling analysis section."""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .base import BaseSection, SectionContent, Takeaway


class CarryingSection(BaseSection):
    """Analysis of ball carrying, dribbling, and progressive runs."""

    @property
    def section_title(self) -> str:
        return "Carrying & Dribbling"

    def generate(self) -> SectionContent:
        """Generate the carrying section."""
        content = SectionContent(title=self.section_title)

        # Create carry map
        carry_map_fig = self._create_carry_map()
        if carry_map_fig:
            content.add_figure(carry_map_fig, "Ball Carries")

        # Create progressive carries map
        prog_carry_fig = self._create_progressive_carry_map()
        if prog_carry_fig:
            content.add_figure(prog_carry_fig, "Progressive Carries")

        # Create dribbling metrics
        dribble_fig = self._create_dribbling_metrics()
        if dribble_fig:
            content.add_figure(dribble_fig, "Dribbling Metrics")

        # Create carry end zones
        end_zone_fig = self._create_carry_end_zones()
        if end_zone_fig:
            content.add_figure(end_zone_fig, "Carry End Zone Distribution")

        # Create turnover analysis
        turnover_fig = self._create_turnover_analysis()
        if turnover_fig:
            content.add_figure(turnover_fig, "Ball Retention Analysis")

        # Generate takeaways
        takeaways = self._generate_takeaways()
        for takeaway in takeaways:
            content.takeaways.append(takeaway)

        # Add carrying table
        carrying_table = self._get_carrying_table()
        content.add_table(carrying_table, "Carrying & Dribbling Metrics")

        return content

    def _create_carry_map(self) -> Figure:
        """Create ball carry map."""
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None

        if events_df is None or events_df.empty:
            return None

        # Filter to carries
        carries = events_df[
            (events_df.get('event_type', '') == 'carry') |
            (events_df.get('type_id', 0) == 3)
        ].copy() if 'event_type' in events_df.columns or 'type_id' in events_df.columns else None

        if carries is None or carries.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor('#0a4d0a')

        self._draw_pitch(ax)

        # Sample if too many
        if len(carries) > 150:
            carries = carries.sample(n=150)

        # Draw carries
        for _, row in carries.iterrows():
            if 'end_x' not in row or 'end_y' not in row:
                continue

            x_start, y_start = row['x'], row['y']
            x_end, y_end = row['end_x'], row['end_y']

            if np.isnan(x_start) or np.isnan(y_start) or np.isnan(x_end) or np.isnan(y_end):
                continue

            # Color based on progression
            dx = x_end - x_start
            if dx > 10:  # Progressive
                color = self.colors['positive']
                alpha = 0.8
                linewidth = 1.5
            else:
                color = self.colors['text']
                alpha = 0.4
                linewidth = 1

            ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                       arrowprops=dict(arrowstyle='->', color=color,
                                      alpha=alpha, linewidth=linewidth))

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.axis('off')

        ax.set_title(f"Ball Carries (sample of {len(carries)})",
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_progressive_carry_map(self) -> Figure:
        """Create progressive carries map."""
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None

        if events_df is None or events_df.empty:
            return None

        # Filter to progressive carries
        prog_carries = events_df[
            events_df.get('is_progressive_carry', False) == True
        ].copy() if 'is_progressive_carry' in events_df.columns else None

        if prog_carries is None or prog_carries.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor('#0a4d0a')

        self._draw_pitch(ax)

        for _, row in prog_carries.iterrows():
            if 'end_x' not in row or 'end_y' not in row:
                continue

            x_start, y_start = row['x'], row['y']
            x_end, y_end = row['end_x'], row['end_y']

            if np.isnan(x_start) or np.isnan(y_start) or np.isnan(x_end) or np.isnan(y_end):
                continue

            ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                       arrowprops=dict(arrowstyle='->', color=self.colors['positive'],
                                      alpha=0.8, linewidth=1.5))

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.axis('off')

        ax.set_title(f"Progressive Carries ({len(prog_carries)} total)",
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

    def _create_dribbling_metrics(self) -> Figure:
        """Create dribbling/take-on metrics comparison."""
        metrics = {
            'Dribbles Attempted': 'dribbles_attempted',
            'Dribble Success %': 'dribble_success_rate',
            'Progressive Carries': 'progressive_carries',
            'Carries into Box': 'carries_into_box',
            'Carries into Final 3rd': 'carries_into_final_third',
        }

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        labels = []
        percentiles = []
        values = []

        for display_name, metric_key in metrics.items():
            value = self._get_metric_value(metric_key)
            pct = self._get_percentile(metric_key)
            labels.append(display_name)
            values.append(value)
            percentiles.append(pct)

        x = np.arange(len(labels))
        width = 0.6

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

        for bar, val, pct in zip(bars, values, percentiles):
            height = bar.get_height()
            if 'rate' in labels[bars.index(bar)].lower() or '%' in labels[bars.index(bar)]:
                label = f'{val:.1f}%'
            else:
                label = f'{val:.1f}'
            ax.annotate(f'{label}\n({pct:.0f}th)',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom',
                       color=self.colors['text'], fontsize=9)

        ax.set_ylabel('Percentile', color=self.colors['text'])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=self.colors['text'], rotation=15, ha='right')
        ax.set_ylim(0, 105)

        ax.axhline(y=50, color=self.colors['neutral'], linestyle='--', alpha=0.5)

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title('Dribbling Metrics - Percentile Rankings',
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_carry_end_zones(self) -> Figure:
        """Create carry end zone distribution."""
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None

        if events_df is None or events_df.empty:
            return None

        # Filter to carries
        carries = events_df[
            (events_df.get('event_type', '') == 'carry') |
            (events_df.get('type_id', 0) == 3)
        ].copy() if 'event_type' in events_df.columns or 'type_id' in events_df.columns else None

        if carries is None or carries.empty or 'end_x' not in carries.columns:
            return None

        # Categorize end zones
        zones = {
            'Defensive Third': carries[carries['end_x'] < 33.33],
            'Middle Third': carries[(carries['end_x'] >= 33.33) & (carries['end_x'] < 66.66)],
            'Final Third': carries[(carries['end_x'] >= 66.66) & (carries['end_x'] < 83)],
            'Penalty Area': carries[carries['end_x'] >= 83],
        }

        zone_counts = {k: len(v) for k, v in zones.items()}
        total = sum(zone_counts.values())

        if total == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        labels = list(zone_counts.keys())
        values = [zone_counts[z] / total * 100 for z in labels]
        colors = [self.colors['neutral'], self.colors['accent'],
                 self.colors['highlight'], self.colors['positive']]

        bars = ax.barh(labels, values, color=colors, alpha=0.8, edgecolor='white')

        for bar, val, count in zip(bars, values, zone_counts.values()):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}% ({count})', va='center', color=self.colors['text'])

        ax.set_xlabel('Percentage of Carries', color=self.colors['text'])
        ax.set_xlim(0, max(values) * 1.25)

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title('Carry End Zone Distribution', color=self.colors['text'],
                    fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_turnover_analysis(self) -> Figure:
        """Create ball retention/turnover analysis."""
        metrics = {
            'Dispossessions': ('dispossessions', False),  # Lower is better
            'Miscontrols': ('miscontrols', False),
            'Dribble Success %': ('dribble_success_rate', True),  # Higher is better
        }

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor(self.colors['primary'])

        for ax, (display_name, (metric_key, higher_better)) in zip(axes, metrics.items()):
            ax.set_facecolor(self.colors['secondary'])

            value = self._get_metric_value(metric_key)
            pct = self._get_percentile(metric_key)

            # For metrics where lower is better, invert the color logic
            if higher_better:
                is_good = pct >= 50
            else:
                is_good = pct <= 50

            color = self.colors['positive'] if is_good else self.colors['negative']

            # Create simple gauge
            ax.barh([0], [pct], color=color, alpha=0.8, height=0.5)
            ax.barh([0], [100], color=self.colors['neutral'], alpha=0.2, height=0.5)

            ax.axvline(x=50, color=self.colors['text'], linestyle='--', alpha=0.5)

            # Value annotation
            if '%' in display_name or 'rate' in metric_key.lower():
                label = f'{value:.1f}%'
            else:
                label = f'{value:.2f}'

            ax.text(50, -0.4, label, ha='center', va='top',
                   color=self.colors['text'], fontsize=14, fontweight='bold')
            ax.text(50, 0.5, f'{pct:.0f}th', ha='center', va='bottom',
                   color=self.colors['text'], fontsize=10)

            ax.set_xlim(0, 100)
            ax.set_ylim(-0.6, 0.6)
            ax.axis('off')
            ax.set_title(display_name, color=self.colors['text'], fontsize=11)

        plt.suptitle('Ball Retention Analysis', color=self.colors['text'],
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig

    def _generate_takeaways(self) -> List[Takeaway]:
        """Generate carrying/dribbling takeaways."""
        takeaways = []

        # Dribble success rate
        success_rate = self._get_metric_value('dribble_success_rate')
        success_pct = self._get_percentile('dribble_success_rate')
        dribbles = self._get_metric_value('dribbles_attempted')

        if dribbles > 0:
            if success_pct >= 70:
                takeaways.append(Takeaway(
                    text=f"Effective dribbler: {success_rate:.1f}% success on {dribbles:.1f} attempts per 90",
                    metric='dribble_success_rate',
                    value=success_rate,
                    percentile=success_pct,
                    is_positive=True
                ))
            elif success_pct <= 30:
                takeaways.append(Takeaway(
                    text=f"Low dribble success: {success_rate:.1f}% on {dribbles:.1f} attempts per 90",
                    metric='dribble_success_rate',
                    value=success_rate,
                    percentile=success_pct,
                    is_positive=False
                ))

        # Progressive carries
        prog = self._get_metric_value('progressive_carries')
        prog_pct = self._get_percentile('progressive_carries')
        if prog > 0:
            if prog_pct >= 70:
                takeaways.append(Takeaway(
                    text=f"Strong ball carrier: {prog:.1f} progressive carries per 90 ({prog_pct:.0f}th percentile)",
                    metric='progressive_carries',
                    value=prog,
                    percentile=prog_pct,
                    is_positive=True
                ))
            elif prog_pct <= 30:
                takeaways.append(Takeaway(
                    text=f"Limited ball carrying: {prog:.1f} progressive carries per 90 ({prog_pct:.0f}th percentile)",
                    metric='progressive_carries',
                    value=prog,
                    percentile=prog_pct,
                    is_positive=False
                ))

        # Carries into box
        box_carries = self._get_metric_value('carries_into_box')
        box_pct = self._get_percentile('carries_into_box')
        if box_carries > 0 and box_pct >= 70:
            takeaways.append(Takeaway(
                text=f"Dangerous runner: {box_carries:.1f} carries into the box per 90",
                metric='carries_into_box',
                value=box_carries,
                percentile=box_pct,
                is_positive=True
            ))

        # Dispossessions
        disp = self._get_metric_value('dispossessions')
        disp_pct = self._get_percentile('dispossessions')
        if disp > 0 and disp_pct >= 70:  # High dispossession is bad
            takeaways.append(Takeaway(
                text=f"Possession security concern: {disp:.1f} dispossessions per 90 ({disp_pct:.0f}th percentile)",
                metric='dispossessions',
                value=disp,
                percentile=disp_pct,
                is_positive=False
            ))

        return takeaways

    def _get_carrying_table(self) -> Dict[str, Any]:
        """Get carrying metrics table."""
        metrics = [
            'carries', 'progressive_carries', 'carry_distance',
            'dribbles_attempted', 'dribble_success_rate',
            'dispossessions', 'miscontrols',
            'carries_into_final_third', 'carries_into_box'
        ]

        result = {}
        for metric in metrics:
            result[metric] = {
                'value': self._get_metric_value(metric),
                'percentile': self._get_percentile(metric),
            }
        return result
