"""Chance creation and shooting analysis section."""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .base import BaseSection, SectionContent, Takeaway


class ChanceCreationSection(BaseSection):
    """Analysis of shooting, xG, assists, and chance creation."""

    @property
    def section_title(self) -> str:
        return "Chance Creation & Shooting"

    def generate(self) -> SectionContent:
        """Generate the chance creation section."""
        content = SectionContent(title=self.section_title)

        # Create shot map
        shot_map_fig = self._create_shot_map()
        if shot_map_fig:
            content.add_figure(shot_map_fig, "Shot Map with xG")

        # Create key passes map
        key_pass_fig = self._create_key_passes_map()
        if key_pass_fig:
            content.add_figure(key_pass_fig, "Key Passes & Assists")

        # Create xG timeline
        xg_timeline_fig = self._create_xg_timeline()
        if xg_timeline_fig:
            content.add_figure(xg_timeline_fig, "xG Accumulation Over Season")

        # Create chance creation metrics
        metrics_fig = self._create_chance_creation_metrics()
        if metrics_fig:
            content.add_figure(metrics_fig, "Chance Creation Metrics")

        # Create shot quality analysis
        shot_quality_fig = self._create_shot_quality_analysis()
        if shot_quality_fig:
            content.add_figure(shot_quality_fig, "Shot Quality Analysis")

        # Generate takeaways
        takeaways = self._generate_takeaways()
        for takeaway in takeaways:
            content.takeaways.append(takeaway)

        # Add chance creation table
        table = self._get_chance_creation_table()
        content.add_table(table, "Chance Creation & Shooting Metrics")

        return content

    def _create_shot_map(self) -> Figure:
        """Create shot map with xG values."""
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None

        if events_df is None or events_df.empty:
            return None

        # Filter to shots
        shots = events_df[
            (events_df.get('event_type', '').isin(['shot', 'goal'])) |
            (events_df.get('type_id', 0).isin([13, 14, 15, 16]))
        ].copy() if 'event_type' in events_df.columns or 'type_id' in events_df.columns else None

        if shots is None or shots.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor('#0a4d0a')

        self._draw_pitch(ax)

        # Draw shots
        for _, row in shots.iterrows():
            x, y = row.get('x', 0), row.get('y', 0)
            if np.isnan(x) or np.isnan(y):
                continue

            xg = row.get('xG', row.get('expected_goals', 0.05))
            is_goal = row.get('is_goal', False) or row.get('type_id', 0) == 16

            # Size based on xG
            size = max(50, min(500, xg * 1500))

            if is_goal:
                color = self.colors['positive']
                marker = '*'
                size *= 1.5
            else:
                color = self.colors['highlight']
                marker = 'o'

            ax.scatter(x, y, s=size, c=color, marker=marker,
                      alpha=0.8, edgecolor='white', linewidth=1)

        # Add legend
        ax.scatter([], [], s=100, c=self.colors['highlight'], marker='o',
                  label='Shot', edgecolor='white')
        ax.scatter([], [], s=150, c=self.colors['positive'], marker='*',
                  label='Goal', edgecolor='white')
        ax.legend(loc='upper left', facecolor=self.colors['secondary'],
                 edgecolor=self.colors['neutral'], labelcolor=self.colors['text'])

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.axis('off')

        total_xg = shots.get('xG', shots.get('expected_goals', 0)).sum()
        goals = len(shots[shots.get('is_goal', False) == True]) if 'is_goal' in shots.columns else 0

        ax.set_title(f"Shot Map - {len(shots)} shots, {goals} goals, {total_xg:.2f} xG",
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_key_passes_map(self) -> Figure:
        """Create key passes and assists map."""
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None

        if events_df is None or events_df.empty:
            return None

        # Filter to key passes and assists
        key_passes = events_df[
            (events_df.get('is_key_pass', False) == True) |
            (events_df.get('is_assist', False) == True)
        ].copy() if 'is_key_pass' in events_df.columns else None

        if key_passes is None or key_passes.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor('#0a4d0a')

        self._draw_pitch(ax)

        assists = 0
        for _, row in key_passes.iterrows():
            if 'end_x' not in row or 'end_y' not in row:
                continue

            x_start, y_start = row['x'], row['y']
            x_end, y_end = row['end_x'], row['end_y']

            if np.isnan(x_start) or np.isnan(y_start) or np.isnan(x_end) or np.isnan(y_end):
                continue

            is_assist = row.get('is_assist', False)

            if is_assist:
                color = self.colors['positive']
                linewidth = 2.5
                assists += 1
            else:
                color = self.colors['highlight']
                linewidth = 1.5

            ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                       arrowprops=dict(arrowstyle='->', color=color,
                                      alpha=0.8, linewidth=linewidth))

            # Mark end point (shot location)
            ax.scatter(x_end, y_end, s=30, c=color, alpha=0.6)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.axis('off')

        ax.set_title(f"Key Passes ({len(key_passes)}) & Assists ({assists})",
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

    def _create_xg_timeline(self) -> Figure:
        """Create cumulative xG timeline over season."""
        match_metrics = getattr(self.metrics, 'match_by_match', None)

        if match_metrics is None or not match_metrics:
            return None

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        xg_values = []
        goals = []

        for match in match_metrics:
            if isinstance(match, dict):
                xg_values.append(match.get('xG', 0))
                goals.append(match.get('goals', 0))
            else:
                xg_values.append(getattr(match, 'xG', 0))
                goals.append(getattr(match, 'goals', 0))

        if sum(xg_values) == 0:
            plt.close(fig)
            return None

        x = range(len(xg_values))
        cumulative_xg = np.cumsum(xg_values)
        cumulative_goals = np.cumsum(goals)

        ax.plot(x, cumulative_xg, color=self.colors['highlight'], linewidth=2,
               label='Cumulative xG', marker='o', markersize=4)
        ax.plot(x, cumulative_goals, color=self.colors['positive'], linewidth=2,
               label='Cumulative Goals', marker='s', markersize=4)

        # Fill between
        ax.fill_between(x, cumulative_xg, cumulative_goals,
                       where=(np.array(cumulative_goals) > np.array(cumulative_xg)),
                       color=self.colors['positive'], alpha=0.2,
                       label='Overperforming xG')
        ax.fill_between(x, cumulative_xg, cumulative_goals,
                       where=(np.array(cumulative_goals) <= np.array(cumulative_xg)),
                       color=self.colors['negative'], alpha=0.2,
                       label='Underperforming xG')

        ax.set_xlabel('Match', color=self.colors['text'])
        ax.set_ylabel('Cumulative Value', color=self.colors['text'])

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc='upper left', facecolor=self.colors['secondary'],
                 edgecolor=self.colors['neutral'], labelcolor=self.colors['text'])

        final_xg = cumulative_xg[-1] if len(cumulative_xg) > 0 else 0
        final_goals = cumulative_goals[-1] if len(cumulative_goals) > 0 else 0
        diff = final_goals - final_xg

        ax.set_title(f"xG vs Goals Over Season (Diff: {diff:+.2f})",
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_chance_creation_metrics(self) -> Figure:
        """Create chance creation metrics comparison."""
        metrics = {
            'xG': 'xG',
            'xA': 'xA',
            'Shots': 'shots',
            'Shots on Target': 'shots_on_target',
            'Key Passes': 'key_passes',
            'Big Chances Created': 'big_chances_created',
            'Shot Creating Actions': 'shot_creating_actions',
            'Goal Creating Actions': 'goal_creating_actions',
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

        for bar, val, pct in zip(bars, values, percentiles):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}\n({pct:.0f}th)',
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

        ax.set_title('Chance Creation Metrics - Percentile Rankings',
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_shot_quality_analysis(self) -> Figure:
        """Create shot quality analysis."""
        xg = self._get_metric_value('xG')
        shots = self._get_metric_value('shots')
        xg_per_shot = self._get_metric_value('xG_per_shot')
        goals = self._get_metric_value('goals')

        if shots == 0:
            return None

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor(self.colors['primary'])

        # Shot volume gauge
        ax1 = axes[0]
        ax1.set_facecolor(self.colors['secondary'])
        shots_pct = self._get_percentile('shots')
        self._draw_simple_gauge(ax1, shots, shots_pct, 'Shots p90')

        # Shot quality gauge
        ax2 = axes[1]
        ax2.set_facecolor(self.colors['secondary'])
        xg_per_shot_pct = self._get_percentile('xG_per_shot')
        self._draw_simple_gauge(ax2, xg_per_shot, xg_per_shot_pct, 'xG per Shot')

        # Conversion gauge
        ax3 = axes[2]
        ax3.set_facecolor(self.colors['secondary'])
        conversion = (goals / xg * 100) if xg > 0 else 0
        # Conversion over 100% is overperforming
        conversion_color = self.colors['positive'] if conversion >= 100 else self.colors['negative']
        self._draw_simple_gauge(ax3, conversion, min(conversion, 100),
                               'Goals vs xG %', value_color=conversion_color)

        plt.suptitle('Shot Quality Analysis', color=self.colors['text'],
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig

    def _draw_simple_gauge(self, ax, value: float, percentile: float, label: str,
                          value_color: str = None) -> None:
        """Draw a simple horizontal gauge."""
        if value_color is None:
            if percentile >= 75:
                value_color = self.colors['positive']
            elif percentile >= 50:
                value_color = self.colors['highlight']
            elif percentile >= 25:
                value_color = self.colors['neutral']
            else:
                value_color = self.colors['negative']

        # Background bar
        ax.barh([0], [100], color=self.colors['neutral'], alpha=0.2, height=0.4)
        # Value bar
        ax.barh([0], [percentile], color=value_color, alpha=0.8, height=0.4)

        # Reference line at 50th
        ax.axvline(x=50, color=self.colors['text'], linestyle='--', alpha=0.5)

        # Labels
        ax.text(50, -0.4, f'{value:.2f}', ha='center', va='top',
               color=self.colors['text'], fontsize=14, fontweight='bold')
        ax.text(50, 0.4, f'{percentile:.0f}th', ha='center', va='bottom',
               color=self.colors['text_secondary'], fontsize=10)

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, 0.6)
        ax.axis('off')
        ax.set_title(label, color=self.colors['text'], fontsize=11)

    def _generate_takeaways(self) -> List[Takeaway]:
        """Generate chance creation takeaways."""
        takeaways = []

        # xG analysis
        xg = self._get_metric_value('xG')
        xg_pct = self._get_percentile('xG')
        goals = self._get_metric_value('goals')

        if xg > 0:
            if xg_pct >= 70:
                takeaways.append(Takeaway(
                    text=f"High goal threat: {xg:.2f} xG per 90 ({xg_pct:.0f}th percentile)",
                    metric='xG',
                    value=xg,
                    percentile=xg_pct,
                    is_positive=True
                ))

            # xG vs Goals comparison
            diff = goals - xg
            if diff > 0.1:
                takeaways.append(Takeaway(
                    text=f"Clinical finisher: {goals:.2f} goals vs {xg:.2f} xG (+{diff:.2f} overperformance)",
                    is_positive=True
                ))
            elif diff < -0.1:
                takeaways.append(Takeaway(
                    text=f"Underperforming xG: {goals:.2f} goals vs {xg:.2f} xG ({diff:.2f})",
                    is_positive=False
                ))

        # xA analysis
        xa = self._get_metric_value('xA')
        xa_pct = self._get_percentile('xA')
        if xa > 0 and xa_pct >= 70:
            takeaways.append(Takeaway(
                text=f"Strong creator: {xa:.2f} xA per 90 ({xa_pct:.0f}th percentile)",
                metric='xA',
                value=xa,
                percentile=xa_pct,
                is_positive=True
            ))

        # Shot volume
        shots = self._get_metric_value('shots')
        shots_pct = self._get_percentile('shots')
        if shots > 0:
            if shots_pct >= 75:
                takeaways.append(Takeaway(
                    text=f"High shot volume: {shots:.1f} shots per 90 ({shots_pct:.0f}th percentile)",
                    metric='shots',
                    value=shots,
                    percentile=shots_pct,
                    is_positive=True
                ))
            elif shots_pct <= 25:
                takeaways.append(Takeaway(
                    text=f"Low shot volume: {shots:.1f} shots per 90 ({shots_pct:.0f}th percentile)",
                    metric='shots',
                    value=shots,
                    percentile=shots_pct,
                    is_positive=False
                ))

        # Key passes
        key = self._get_metric_value('key_passes')
        key_pct = self._get_percentile('key_passes')
        if key > 0 and key_pct >= 70:
            takeaways.append(Takeaway(
                text=f"Regular chance creator: {key:.1f} key passes per 90",
                metric='key_passes',
                value=key,
                percentile=key_pct,
                is_positive=True
            ))

        return takeaways

    def _get_chance_creation_table(self) -> Dict[str, Any]:
        """Get chance creation metrics table."""
        metrics = [
            'shots', 'shots_on_target', 'goals', 'xG', 'npxG', 'xG_per_shot',
            'assists', 'xA', 'key_passes', 'shot_creating_actions',
            'goal_creating_actions', 'big_chances_created'
        ]

        result = {}
        for metric in metrics:
            result[metric] = {
                'value': self._get_metric_value(metric),
                'percentile': self._get_percentile(metric),
            }
        return result
