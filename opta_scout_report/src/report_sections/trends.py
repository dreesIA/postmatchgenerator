"""Performance trends section."""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .base import BaseSection, SectionContent, Takeaway


class TrendsSection(BaseSection):
    """Analysis of performance trends over time."""

    @property
    def section_title(self) -> str:
        return "Performance Trends"

    def generate(self) -> SectionContent:
        """Generate the trends section."""
        content = SectionContent(title=self.section_title)

        # Create rolling average chart
        rolling_fig = self._create_rolling_performance()
        if rolling_fig:
            content.add_figure(rolling_fig, "Rolling Performance (5-Game Average)")

        # Create form chart
        form_fig = self._create_form_chart()
        if form_fig:
            content.add_figure(form_fig, "Recent Form Analysis")

        # Create key metrics trend
        key_trend_fig = self._create_key_metrics_trend()
        if key_trend_fig:
            content.add_figure(key_trend_fig, "Key Metrics Over Time")

        # Create consistency analysis
        consistency_fig = self._create_consistency_analysis()
        if consistency_fig:
            content.add_figure(consistency_fig, "Consistency Analysis")

        # Create game state performance
        game_state_fig = self._create_game_state_performance()
        if game_state_fig:
            content.add_figure(game_state_fig, "Performance by Game State")

        # Generate takeaways
        takeaways = self._generate_takeaways()
        for takeaway in takeaways:
            content.takeaways.append(takeaway)

        # Add trend data
        content.raw_data['trend_analysis'] = self._get_trend_analysis()

        return content

    def _create_rolling_performance(self) -> Figure:
        """Create rolling average performance chart."""
        match_metrics = getattr(self.metrics, 'match_by_match', None)

        if match_metrics is None or len(match_metrics) < 5:
            return None

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        # Get key metrics for rolling average
        position_group = self.player_data.info.position_group

        if position_group in ['ST', 'W', 'AM']:
            metric_key = 'xG'
            metric_label = 'xG'
        elif position_group in ['CM', 'DM']:
            metric_key = 'progressive_passes'
            metric_label = 'Progressive Passes'
        else:
            metric_key = 'defensive_actions'
            metric_label = 'Defensive Actions'

        values = []
        for match in match_metrics:
            if isinstance(match, dict):
                values.append(match.get(metric_key, 0))
            else:
                values.append(getattr(match, metric_key, 0))

        if sum(values) == 0:
            plt.close(fig)
            return None

        x = range(len(values))

        # Raw values
        ax.plot(x, values, color=self.colors['neutral'], alpha=0.4, linewidth=1,
               marker='o', markersize=4, label='Per Game')

        # Rolling 5-game average
        window = min(5, len(values))
        rolling_avg = np.convolve(values, np.ones(window)/window, mode='valid')
        rolling_x = range(window-1, len(values))
        ax.plot(rolling_x, rolling_avg, color=self.colors['highlight'], linewidth=2.5,
               label=f'{window}-Game Rolling Avg')

        # Season average
        season_avg = np.mean(values)
        ax.axhline(y=season_avg, color=self.colors['positive'], linestyle='--',
                  linewidth=1.5, alpha=0.7, label=f'Season Avg: {season_avg:.2f}')

        ax.set_xlabel('Match', color=self.colors['text'])
        ax.set_ylabel(f'{metric_label} per 90', color=self.colors['text'])

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc='upper left', facecolor=self.colors['secondary'],
                 edgecolor=self.colors['neutral'], labelcolor=self.colors['text'])

        ax.set_title(f'{metric_label} - Rolling Performance',
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_form_chart(self) -> Figure:
        """Create recent form analysis chart."""
        match_metrics = getattr(self.metrics, 'match_by_match', None)

        if match_metrics is None or len(match_metrics) < 3:
            return None

        # Compare last 5 games to season average
        recent = match_metrics[-5:] if len(match_metrics) >= 5 else match_metrics
        all_games = match_metrics

        # Key metrics to compare
        metrics_to_compare = ['xG', 'xA', 'progressive_passes', 'tackles_won', 'pressures']

        recent_avgs = {}
        season_avgs = {}

        for metric in metrics_to_compare:
            recent_values = []
            all_values = []

            for match in recent:
                if isinstance(match, dict):
                    recent_values.append(match.get(metric, 0))
                else:
                    recent_values.append(getattr(match, metric, 0))

            for match in all_games:
                if isinstance(match, dict):
                    all_values.append(match.get(metric, 0))
                else:
                    all_values.append(getattr(match, metric, 0))

            recent_avgs[metric] = np.mean(recent_values) if recent_values else 0
            season_avgs[metric] = np.mean(all_values) if all_values else 0

        # Filter out metrics where both are 0
        valid_metrics = {k: (recent_avgs[k], season_avgs[k])
                        for k in metrics_to_compare
                        if recent_avgs[k] > 0 or season_avgs[k] > 0}

        if not valid_metrics:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        labels = [m.replace('_', ' ').title() for m in valid_metrics.keys()]
        recent_vals = [v[0] for v in valid_metrics.values()]
        season_vals = [v[1] for v in valid_metrics.values()]

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, recent_vals, width, label='Last 5 Games',
                      color=self.colors['highlight'], alpha=0.8, edgecolor='white')
        bars2 = ax.bar(x + width/2, season_vals, width, label='Season Average',
                      color=self.colors['neutral'], alpha=0.8, edgecolor='white')

        ax.set_ylabel('Per 90', color=self.colors['text'])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=self.colors['text'], rotation=15, ha='right')

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc='upper right', facecolor=self.colors['secondary'],
                 edgecolor=self.colors['neutral'], labelcolor=self.colors['text'])

        ax.set_title('Recent Form vs Season Average',
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_key_metrics_trend(self) -> Figure:
        """Create multi-metric trend chart."""
        match_metrics = getattr(self.metrics, 'match_by_match', None)

        if match_metrics is None or len(match_metrics) < 3:
            return None

        position_group = self.player_data.info.position_group

        # Select metrics based on position
        if position_group in ['ST', 'W', 'AM']:
            metrics = ['xG', 'xA', 'shots']
            labels = ['xG', 'xA', 'Shots']
        elif position_group in ['CM', 'DM']:
            metrics = ['progressive_passes', 'tackles_won', 'interceptions']
            labels = ['Prog. Passes', 'Tackles', 'Interceptions']
        else:
            metrics = ['tackles_won', 'interceptions', 'clearances']
            labels = ['Tackles', 'Interceptions', 'Clearances']

        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3 * len(metrics)), sharex=True)
        fig.patch.set_facecolor(self.colors['primary'])

        colors = [self.colors['highlight'], self.colors['positive'], self.colors['accent']]

        for ax, metric, label, color in zip(axes, metrics, labels, colors):
            ax.set_facecolor(self.colors['secondary'])

            values = []
            for match in match_metrics:
                if isinstance(match, dict):
                    values.append(match.get(metric, 0))
                else:
                    values.append(getattr(match, metric, 0))

            x = range(len(values))
            ax.plot(x, values, color=color, linewidth=2, marker='o', markersize=4)

            # Add trend line
            if len(values) > 2:
                z = np.polyfit(list(x), values, 1)
                p = np.poly1d(z)
                ax.plot(x, p(list(x)), linestyle='--', color=color, alpha=0.5, linewidth=1)

            ax.set_ylabel(label, color=self.colors['text'])
            ax.tick_params(colors=self.colors['text'])
            ax.spines['bottom'].set_color(self.colors['neutral'])
            ax.spines['left'].set_color(self.colors['neutral'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axes[-1].set_xlabel('Match', color=self.colors['text'])

        plt.suptitle('Key Metrics Over Time', color=self.colors['text'],
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig

    def _create_consistency_analysis(self) -> Figure:
        """Create consistency/variance analysis."""
        match_metrics = getattr(self.metrics, 'match_by_match', None)

        if match_metrics is None or len(match_metrics) < 5:
            return None

        # Calculate coefficient of variation for key metrics
        metrics_to_analyze = ['touches', 'passes_completed', 'progressive_passes',
                             'tackles_won', 'interceptions']

        consistency_data = {}

        for metric in metrics_to_analyze:
            values = []
            for match in match_metrics:
                if isinstance(match, dict):
                    values.append(match.get(metric, 0))
                else:
                    values.append(getattr(match, metric, 0))

            values = [v for v in values if v > 0]  # Filter zeros

            if len(values) >= 3:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = (std_val / mean_val * 100) if mean_val > 0 else 0
                consistency_data[metric] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv
                }

        if not consistency_data:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        labels = [m.replace('_', ' ').title() for m in consistency_data.keys()]
        cvs = [consistency_data[m]['cv'] for m in consistency_data.keys()]

        # Color by consistency (lower CV = more consistent = better)
        colors = []
        for cv in cvs:
            if cv <= 30:
                colors.append(self.colors['positive'])
            elif cv <= 50:
                colors.append(self.colors['highlight'])
            else:
                colors.append(self.colors['negative'])

        x = np.arange(len(labels))
        bars = ax.bar(x, cvs, color=colors, alpha=0.8, edgecolor='white')

        # Add reference line
        ax.axhline(y=30, color=self.colors['positive'], linestyle='--', alpha=0.5,
                  label='Consistent (<30%)')
        ax.axhline(y=50, color=self.colors['negative'], linestyle='--', alpha=0.5,
                  label='Inconsistent (>50%)')

        ax.set_ylabel('Coefficient of Variation (%)', color=self.colors['text'])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=self.colors['text'], rotation=15, ha='right')

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc='upper right', facecolor=self.colors['secondary'],
                 edgecolor=self.colors['neutral'], labelcolor=self.colors['text'])

        ax.set_title('Consistency Analysis (Lower = More Consistent)',
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_game_state_performance(self) -> Figure:
        """Create performance by game state chart."""
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None

        if events_df is None or events_df.empty or 'game_state' not in events_df.columns:
            return None

        # Get key metrics by game state
        game_states = ['winning', 'drawing', 'losing']
        metrics = ['progressive_passes', 'shots', 'tackles_won']

        state_data = {}
        for state in game_states:
            state_events = events_df[events_df['game_state'] == state]
            if len(state_events) > 0:
                # This is simplified - in reality would need minutes adjustment
                state_data[state] = {
                    'count': len(state_events),
                    'progressive_passes': len(state_events[state_events.get('is_progressive_pass', False) == True]),
                    'shots': len(state_events[state_events.get('event_type', '') == 'shot']),
                    'tackles': len(state_events[state_events.get('event_type', '') == 'tackle']),
                }

        if not state_data:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        states = [s.title() for s in state_data.keys()]
        counts = [state_data[s.lower()]['count'] for s in states]

        colors = [self.colors['positive'], self.colors['neutral'], self.colors['negative']]

        bars = ax.bar(states, counts, color=colors[:len(states)], alpha=0.8, edgecolor='white')

        ax.set_ylabel('Actions', color=self.colors['text'])
        ax.set_xlabel('Game State', color=self.colors['text'])

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title('Activity by Game State', color=self.colors['text'],
                    fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _generate_takeaways(self) -> List[Takeaway]:
        """Generate trend takeaways."""
        takeaways = []

        match_metrics = getattr(self.metrics, 'match_by_match', None)

        if match_metrics is None or len(match_metrics) < 5:
            takeaways.append(Takeaway(
                text="Insufficient data for comprehensive trend analysis",
                is_positive=True
            ))
            return takeaways

        # Recent form analysis
        recent = match_metrics[-5:]

        position_group = self.player_data.info.position_group
        key_metric = 'xG' if position_group in ['ST', 'W', 'AM'] else 'progressive_passes'

        recent_values = []
        all_values = []

        for match in recent:
            if isinstance(match, dict):
                recent_values.append(match.get(key_metric, 0))
            else:
                recent_values.append(getattr(match, key_metric, 0))

        for match in match_metrics:
            if isinstance(match, dict):
                all_values.append(match.get(key_metric, 0))
            else:
                all_values.append(getattr(match, key_metric, 0))

        if recent_values and all_values:
            recent_avg = np.mean(recent_values)
            season_avg = np.mean(all_values)

            diff = recent_avg - season_avg
            pct_diff = (diff / season_avg * 100) if season_avg > 0 else 0

            if pct_diff > 15:
                takeaways.append(Takeaway(
                    text=f"Strong recent form: {key_metric.replace('_', ' ')} up {pct_diff:.0f}% vs season average",
                    is_positive=True
                ))
            elif pct_diff < -15:
                takeaways.append(Takeaway(
                    text=f"Declining form: {key_metric.replace('_', ' ')} down {abs(pct_diff):.0f}% vs season average",
                    is_positive=False
                ))
            else:
                takeaways.append(Takeaway(
                    text=f"Consistent form: {key_metric.replace('_', ' ')} within 15% of season average",
                    is_positive=True
                ))

        # Trend direction
        if len(all_values) >= 5:
            x = list(range(len(all_values)))
            z = np.polyfit(x, all_values, 1)
            trend = z[0]  # Slope

            if trend > 0.05:
                takeaways.append(Takeaway(
                    text=f"Improving trajectory: {key_metric.replace('_', ' ')} trending upward over season",
                    is_positive=True
                ))
            elif trend < -0.05:
                takeaways.append(Takeaway(
                    text=f"Declining trajectory: {key_metric.replace('_', ' ')} trending downward over season",
                    is_positive=False
                ))

        return takeaways

    def _get_trend_analysis(self) -> Dict[str, Any]:
        """Get comprehensive trend analysis data."""
        match_metrics = getattr(self.metrics, 'match_by_match', None)

        if match_metrics is None:
            return {}

        return {
            'total_matches': len(match_metrics),
            'recent_sample': min(5, len(match_metrics)),
            'has_sufficient_data': len(match_metrics) >= 5,
        }
