"""Defensive analysis section."""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .base import BaseSection, SectionContent, Takeaway


class DefendingSection(BaseSection):
    """Analysis of defensive actions, pressures, and duels."""

    @property
    def section_title(self) -> str:
        return "Defending"

    def generate(self) -> SectionContent:
        """Generate the defending section."""
        content = SectionContent(title=self.section_title)

        # Create defensive actions map
        def_actions_fig = self._create_defensive_actions_map()
        if def_actions_fig:
            content.add_figure(def_actions_fig, "Defensive Actions Location")

        # Create pressure heatmap
        pressure_fig = self._create_pressure_heatmap()
        if pressure_fig:
            content.add_figure(pressure_fig, "Pressing Activity")

        # Create defensive metrics comparison
        metrics_fig = self._create_defensive_metrics()
        if metrics_fig:
            content.add_figure(metrics_fig, "Defensive Metrics vs Position Average")

        # Create duel success breakdown
        duel_fig = self._create_duel_breakdown()
        if duel_fig:
            content.add_figure(duel_fig, "Duel Success Rates")

        # Create pressure zone breakdown
        pressure_zone_fig = self._create_pressure_zones()
        if pressure_zone_fig:
            content.add_figure(pressure_zone_fig, "Pressing by Zone")

        # Generate takeaways
        takeaways = self._generate_takeaways()
        for takeaway in takeaways:
            content.takeaways.append(takeaway)

        # Add defending table
        table = self._get_defending_table()
        content.add_table(table, "Defensive Metrics")

        return content

    def _create_defensive_actions_map(self) -> Figure:
        """Create defensive actions location map."""
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None

        if events_df is None or events_df.empty:
            return None

        # Filter to defensive actions
        def_types = ['tackle', 'interception', 'block', 'clearance']
        def_type_ids = [7, 8, 10, 12]

        def_actions = events_df[
            (events_df.get('event_type', '').isin(def_types)) |
            (events_df.get('type_id', 0).isin(def_type_ids))
        ].copy() if 'event_type' in events_df.columns or 'type_id' in events_df.columns else None

        if def_actions is None or def_actions.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor('#0a4d0a')

        self._draw_pitch(ax)

        # Color code by action type
        action_colors = {
            'tackle': self.colors['highlight'],
            'interception': self.colors['positive'],
            'block': self.colors['neutral'],
            'clearance': self.colors['accent'],
            7: self.colors['highlight'],
            8: self.colors['positive'],
            10: self.colors['neutral'],
            12: self.colors['accent'],
        }

        for _, row in def_actions.iterrows():
            x, y = row.get('x', 0), row.get('y', 0)
            if np.isnan(x) or np.isnan(y):
                continue

            action_type = row.get('event_type', row.get('type_id', 7))
            color = action_colors.get(action_type, self.colors['highlight'])

            ax.scatter(x, y, s=60, c=color, alpha=0.7, edgecolor='white', linewidth=0.5)

        # Legend
        for action, color in [('Tackle', self.colors['highlight']),
                              ('Interception', self.colors['positive']),
                              ('Block', self.colors['neutral']),
                              ('Clearance', self.colors['accent'])]:
            ax.scatter([], [], s=60, c=color, label=action, edgecolor='white')

        ax.legend(loc='upper left', facecolor=self.colors['secondary'],
                 edgecolor=self.colors['neutral'], labelcolor=self.colors['text'])

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.axis('off')

        ax.set_title(f"Defensive Actions ({len(def_actions)} total)",
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_pressure_heatmap(self) -> Figure:
        """Create pressure location heatmap."""
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None

        if events_df is None or events_df.empty:
            return None

        # Filter to pressures
        pressures = events_df[
            (events_df.get('event_type', '') == 'pressure') |
            (events_df.get('is_pressure', False) == True)
        ].copy() if 'event_type' in events_df.columns or 'is_pressure' in events_df.columns else None

        if pressures is None or pressures.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor('#0a4d0a')

        self._draw_pitch(ax)

        x = pressures['x'].dropna().values
        y = pressures['y'].dropna().values

        if len(x) > 0:
            hb = ax.hexbin(x, y, gridsize=15, cmap='Reds', alpha=0.8,
                          extent=[0, 100, 0, 100], mincnt=1)

            cb = plt.colorbar(hb, ax=ax, shrink=0.8)
            cb.set_label('Pressure Density', color=self.colors['text'])
            cb.ax.yaxis.set_tick_params(color=self.colors['text'])
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=self.colors['text'])

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.axis('off')

        ax.set_title(f"Pressing Activity ({len(pressures)} pressures)",
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

    def _create_defensive_metrics(self) -> Figure:
        """Create defensive metrics comparison."""
        metrics = {
            'Tackles Won': 'tackles_won',
            'Tackle Success %': 'tackle_success_rate',
            'Interceptions': 'interceptions',
            'Blocks': 'blocks',
            'Clearances': 'clearances',
            'Ball Recoveries': 'ball_recoveries',
            'Pressures': 'pressures',
            'Pressure Success %': 'pressure_success_rate',
        }

        fig, ax = plt.subplots(figsize=(12, 6))
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

        for bar, val, pct, label in zip(bars, values, percentiles, labels):
            height = bar.get_height()
            if '%' in label:
                val_str = f'{val:.1f}%'
            else:
                val_str = f'{val:.1f}'
            ax.annotate(f'{val_str}\n({pct:.0f}th)',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom',
                       color=self.colors['text'], fontsize=8)

        ax.set_ylabel('Percentile', color=self.colors['text'])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=self.colors['text'], rotation=25, ha='right')
        ax.set_ylim(0, 105)

        ax.axhline(y=50, color=self.colors['neutral'], linestyle='--', alpha=0.5)

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title('Defensive Metrics - Percentile Rankings',
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_duel_breakdown(self) -> Figure:
        """Create duel success rate breakdown."""
        duels = {
            'Ground Duels': ('tackle_success_rate', self._get_metric_value('tackle_success_rate')),
            'Aerial Duels': ('aerial_duel_win_rate', self._get_metric_value('aerial_duel_win_rate')),
            'Pressures': ('pressure_success_rate', self._get_metric_value('pressure_success_rate')),
        }

        # Filter out zeros
        duels = {k: v for k, v in duels.items() if v[1] > 0}

        if not duels:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        labels = list(duels.keys())
        values = [v[1] for v in duels.values()]
        percentiles = [self._get_percentile(v[0]) for v in duels.values()]

        x = np.arange(len(labels))
        width = 0.5

        # Color based on success rate (50% = average)
        colors = []
        for val in values:
            if val >= 60:
                colors.append(self.colors['positive'])
            elif val >= 50:
                colors.append(self.colors['highlight'])
            elif val >= 40:
                colors.append(self.colors['neutral'])
            else:
                colors.append(self.colors['negative'])

        bars = ax.bar(x, values, width, color=colors, alpha=0.8, edgecolor='white')

        # Add percentile annotations
        for bar, pct in zip(bars, percentiles):
            height = bar.get_height()
            ax.annotate(f'{pct:.0f}th %ile',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom',
                       color=self.colors['text'], fontsize=10)

        # Add 50% reference line
        ax.axhline(y=50, color=self.colors['text'], linestyle='--', alpha=0.5,
                  label='50% (break-even)')

        ax.set_ylabel('Success Rate (%)', color=self.colors['text'])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=self.colors['text'])
        ax.set_ylim(0, max(values) * 1.2)

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc='upper right', facecolor=self.colors['secondary'],
                 edgecolor=self.colors['neutral'], labelcolor=self.colors['text'])

        ax.set_title('Duel Success Rates', color=self.colors['text'],
                    fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_pressure_zones(self) -> Figure:
        """Create pressure zone breakdown."""
        zones = {
            'Defensive Third': 'defensive_third_pressures',
            'Middle Third': 'middle_third_pressures',
            'Attacking Third': 'attacking_third_pressures',
        }

        values = {k: self._get_metric_value(v) for k, v in zones.items()}
        total = sum(values.values())

        if total == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        labels = list(values.keys())
        percentages = [v / total * 100 for v in values.values()]
        counts = list(values.values())

        colors = [self.colors['neutral'], self.colors['accent'], self.colors['highlight']]

        bars = ax.barh(labels, percentages, color=colors, alpha=0.8, edgecolor='white')

        for bar, pct, count in zip(bars, percentages, counts):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{pct:.1f}% ({count:.1f} p90)', va='center',
                   color=self.colors['text'], fontsize=10)

        ax.set_xlabel('Percentage of Pressures', color=self.colors['text'])
        ax.set_xlim(0, max(percentages) * 1.3)

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title('Pressing Distribution by Zone', color=self.colors['text'],
                    fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _generate_takeaways(self) -> List[Takeaway]:
        """Generate defensive takeaways."""
        takeaways = []

        # Tackles
        tackles = self._get_metric_value('tackles_won')
        tackles_pct = self._get_percentile('tackles_won')
        tackle_rate = self._get_metric_value('tackle_success_rate')

        if tackles > 0:
            if tackles_pct >= 70:
                takeaways.append(Takeaway(
                    text=f"Active tackler: {tackles:.1f} tackles won per 90 ({tackles_pct:.0f}th percentile)",
                    metric='tackles_won',
                    value=tackles,
                    percentile=tackles_pct,
                    is_positive=True
                ))
            if tackle_rate >= 65:
                takeaways.append(Takeaway(
                    text=f"Efficient in challenges: {tackle_rate:.1f}% tackle success rate",
                    metric='tackle_success_rate',
                    value=tackle_rate,
                    is_positive=True
                ))

        # Interceptions
        interceptions = self._get_metric_value('interceptions')
        int_pct = self._get_percentile('interceptions')
        if interceptions > 0 and int_pct >= 70:
            takeaways.append(Takeaway(
                text=f"Strong reading of play: {interceptions:.1f} interceptions per 90 ({int_pct:.0f}th percentile)",
                metric='interceptions',
                value=interceptions,
                percentile=int_pct,
                is_positive=True
            ))

        # Aerial duels
        aerial_rate = self._get_metric_value('aerial_duel_win_rate')
        aerial_pct = self._get_percentile('aerial_duel_win_rate')
        if aerial_rate > 0:
            if aerial_pct >= 70:
                takeaways.append(Takeaway(
                    text=f"Dominant in the air: {aerial_rate:.1f}% aerial duel win rate",
                    metric='aerial_duel_win_rate',
                    value=aerial_rate,
                    percentile=aerial_pct,
                    is_positive=True
                ))
            elif aerial_pct <= 30:
                takeaways.append(Takeaway(
                    text=f"Struggles aerially: {aerial_rate:.1f}% aerial duel win rate ({aerial_pct:.0f}th percentile)",
                    metric='aerial_duel_win_rate',
                    value=aerial_rate,
                    percentile=aerial_pct,
                    is_positive=False
                ))

        # Pressures
        pressures = self._get_metric_value('pressures')
        press_pct = self._get_percentile('pressures')
        if pressures > 0 and press_pct >= 70:
            takeaways.append(Takeaway(
                text=f"High pressing intensity: {pressures:.1f} pressures per 90 ({press_pct:.0f}th percentile)",
                metric='pressures',
                value=pressures,
                percentile=press_pct,
                is_positive=True
            ))

        # Ball recoveries
        recoveries = self._get_metric_value('ball_recoveries')
        rec_pct = self._get_percentile('ball_recoveries')
        if recoveries > 0 and rec_pct >= 70:
            takeaways.append(Takeaway(
                text=f"Effective at regaining possession: {recoveries:.1f} ball recoveries per 90",
                metric='ball_recoveries',
                value=recoveries,
                percentile=rec_pct,
                is_positive=True
            ))

        return takeaways

    def _get_defending_table(self) -> Dict[str, Any]:
        """Get defending metrics table."""
        metrics = [
            'tackles', 'tackles_won', 'tackle_success_rate',
            'interceptions', 'blocks', 'clearances',
            'aerial_duels', 'aerial_duel_win_rate',
            'pressures', 'pressure_success_rate',
            'defensive_actions', 'ball_recoveries',
            'defensive_third_pressures', 'middle_third_pressures', 'attacking_third_pressures'
        ]

        result = {}
        for metric in metrics:
            result[metric] = {
                'value': self._get_metric_value(metric),
                'percentile': self._get_percentile(metric),
            }
        return result
