"""
Pitch Visualization Module for Opta F24 Data
Creates shot maps, passing networks, and on-pitch stat diagrams
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, Circle, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PitchColors:
    """Color scheme for pitch visualizations"""
    pitch: str = '#1a472a'
    lines: str = '#ffffff'
    home: str = '#e63946'
    away: str = '#457b9d'
    goal: str = '#2ecc71'
    miss: str = '#95a5a6'
    save: str = '#f39c12'
    post: str = '#9b59b6'


class Pitch:
    """
    Football pitch drawing class
    Opta coordinates: 0-100 for both x and y
    """
    
    def __init__(self, 
                 pitch_color: str = '#1a472a',
                 line_color: str = '#ffffff',
                 figsize: Tuple[int, int] = (12, 8),
                 half: bool = False,
                 orientation: str = 'horizontal'):
        """
        Initialize pitch
        
        Args:
            pitch_color: Background color
            line_color: Line color
            figsize: Figure size
            half: Show only half pitch
            orientation: 'horizontal' or 'vertical'
        """
        self.pitch_color = pitch_color
        self.line_color = line_color
        self.figsize = figsize
        self.half = half
        self.orientation = orientation
        self.linewidth = 1.5
        
    def draw(self, ax: plt.Axes = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Draw the pitch
        
        Returns:
            Tuple of (figure, axes)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        ax.set_facecolor(self.pitch_color)
        
        if self.orientation == 'horizontal':
            self._draw_horizontal(ax)
        else:
            self._draw_vertical(ax)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        return fig, ax
    
    def _draw_horizontal(self, ax: plt.Axes):
        """Draw horizontal pitch (0-100 x, 0-100 y)"""
        lw = self.linewidth
        lc = self.line_color
        
        # Pitch outline
        if self.half:
            ax.plot([50, 100, 100, 50], [0, 0, 100, 100], color=lc, lw=lw)
            ax.set_xlim(45, 105)
        else:
            ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color=lc, lw=lw)
            ax.set_xlim(-5, 105)
        
        ax.set_ylim(-5, 105)
        
        # Halfway line
        if not self.half:
            ax.plot([50, 50], [0, 100], color=lc, lw=lw)
        
        # Center circle
        if not self.half:
            center_circle = Circle((50, 50), 9.15, fill=False, color=lc, lw=lw)
            ax.add_patch(center_circle)
            ax.scatter([50], [50], color=lc, s=20, zorder=5)
        
        # Left penalty area (scaled to Opta 0-100)
        # Real: 16.5m from goal line, 40.3m wide -> Opta: ~16.5, 21.1-78.9
        if not self.half:
            ax.plot([0, 17, 17, 0], [21.1, 21.1, 78.9, 78.9], color=lc, lw=lw)
            ax.plot([0, 5.8, 5.8, 0], [36.8, 36.8, 63.2, 63.2], color=lc, lw=lw)
            ax.scatter([11], [50], color=lc, s=20, zorder=5)  # Penalty spot
            
            # D arc
            left_arc = Arc((11, 50), 18.3, 18.3, angle=0, theta1=-53, theta2=53, 
                          color=lc, lw=lw)
            ax.add_patch(left_arc)
        
        # Right penalty area
        ax.plot([100, 83, 83, 100], [21.1, 21.1, 78.9, 78.9], color=lc, lw=lw)
        ax.plot([100, 94.2, 94.2, 100], [36.8, 36.8, 63.2, 63.2], color=lc, lw=lw)
        ax.scatter([89], [50], color=lc, s=20, zorder=5)  # Penalty spot
        
        # D arc
        right_arc = Arc((89, 50), 18.3, 18.3, angle=0, theta1=127, theta2=233, 
                       color=lc, lw=lw)
        ax.add_patch(right_arc)
        
        # Goals
        if not self.half:
            ax.plot([0, -2, -2, 0], [44.2, 44.2, 55.8, 55.8], color=lc, lw=lw*1.5)
        ax.plot([100, 102, 102, 100], [44.2, 44.2, 55.8, 55.8], color=lc, lw=lw*1.5)
        
        # Corner arcs
        if not self.half:
            ax.add_patch(Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=90, color=lc, lw=lw))
            ax.add_patch(Arc((0, 100), 2, 2, angle=0, theta1=270, theta2=360, color=lc, lw=lw))
        ax.add_patch(Arc((100, 0), 2, 2, angle=0, theta1=90, theta2=180, color=lc, lw=lw))
        ax.add_patch(Arc((100, 100), 2, 2, angle=0, theta1=180, theta2=270, color=lc, lw=lw))
    
    def _draw_vertical(self, ax: plt.Axes):
        """Draw vertical pitch (0-100 y becomes x, 0-100 x becomes y)"""
        lw = self.linewidth
        lc = self.line_color
        
        # Pitch outline
        if self.half:
            ax.plot([0, 0, 100, 100], [50, 100, 100, 50], color=lc, lw=lw)
            ax.set_ylim(45, 105)
        else:
            ax.plot([0, 0, 100, 100, 0], [0, 100, 100, 0, 0], color=lc, lw=lw)
            ax.set_ylim(-5, 105)
        
        ax.set_xlim(-5, 105)
        
        # Halfway line
        if not self.half:
            ax.plot([0, 100], [50, 50], color=lc, lw=lw)
        
        # Center circle
        if not self.half:
            center_circle = Circle((50, 50), 9.15, fill=False, color=lc, lw=lw)
            ax.add_patch(center_circle)
            ax.scatter([50], [50], color=lc, s=20, zorder=5)
        
        # Top penalty area
        ax.plot([21.1, 21.1, 78.9, 78.9], [100, 83, 83, 100], color=lc, lw=lw)
        ax.plot([36.8, 36.8, 63.2, 63.2], [100, 94.2, 94.2, 100], color=lc, lw=lw)
        ax.scatter([50], [89], color=lc, s=20, zorder=5)
        
        top_arc = Arc((50, 89), 18.3, 18.3, angle=0, theta1=217, theta2=323, 
                     color=lc, lw=lw)
        ax.add_patch(top_arc)
        
        # Goal
        ax.plot([44.2, 44.2, 55.8, 55.8], [100, 102, 102, 100], color=lc, lw=lw*1.5)


class ShotMap:
    """
    Create shot map visualizations with xG
    """
    
    def __init__(self, shots_df: pd.DataFrame, team_name: str = None):
        self.shots = shots_df.copy()
        self.team_name = team_name
        self.colors = PitchColors()
    
    def plot(self, 
             ax: plt.Axes = None,
             show_xg: bool = True,
             half_pitch: bool = True,
             title: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create shot map
        
        Args:
            ax: Matplotlib axes
            show_xg: Size markers by xG
            half_pitch: Show only attacking half
            title: Plot title
            
        Returns:
            Tuple of (figure, axes)
        """
        pitch = Pitch(half=half_pitch, figsize=(10, 8))
        fig, ax = pitch.draw(ax)
        
        # Color by outcome
        colors = []
        for _, shot in self.shots.iterrows():
            if shot.get('is_goal', False):
                colors.append(self.colors.goal)
            elif shot.get('is_on_target', False):
                colors.append(self.colors.save)
            elif shot.get('hit_post', False):
                colors.append(self.colors.post)
            else:
                colors.append(self.colors.miss)
        
        # Size by xG
        if show_xg and 'xG' in self.shots.columns:
            sizes = self.shots['xG'] * 800 + 50
        else:
            sizes = 100
        
        # Plot shots
        scatter = ax.scatter(
            self.shots['x'], 
            self.shots['y'],
            c=colors,
            s=sizes,
            alpha=0.8,
            edgecolors='white',
            linewidths=1.5,
            zorder=10
        )
        
        # Add xG labels for goals
        for _, shot in self.shots[self.shots.get('is_goal', False) == True].iterrows():
            ax.annotate(
                f"{shot.get('xG', 0):.2f}",
                (shot['x'], shot['y']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                color='white',
                fontweight='bold'
            )
        
        # Legend
        legend_elements = [
            plt.scatter([], [], c=self.colors.goal, s=100, label='Goal', edgecolors='white'),
            plt.scatter([], [], c=self.colors.save, s=100, label='Saved', edgecolors='white'),
            plt.scatter([], [], c=self.colors.post, s=100, label='Post', edgecolors='white'),
            plt.scatter([], [], c=self.colors.miss, s=100, label='Miss', edgecolors='white'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
        
        # Stats annotation
        total_xg = self.shots['xG'].sum() if 'xG' in self.shots.columns else 0
        goals = self.shots.get('is_goal', pd.Series([False])).sum()
        
        stats_text = f"Shots: {len(self.shots)} | Goals: {goals} | xG: {total_xg:.2f}"
        ax.text(97, 3, stats_text, fontsize=10, color='white', 
                ha='right', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
        elif self.team_name:
            ax.set_title(f"{self.team_name} - Shot Map", fontsize=14, 
                        fontweight='bold', color='white', pad=10)
        
        fig.patch.set_facecolor('#1a1a2e')
        
        return fig, ax


class PassingNetwork:
    """
    Create passing network visualizations
    """
    
    def __init__(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, team_name: str = None):
        self.nodes = nodes_df.copy()
        self.edges = edges_df.copy()
        self.team_name = team_name
        self.colors = PitchColors()
    
    def plot(self,
             ax: plt.Axes = None,
             min_passes: int = 2,
             node_size_factor: float = 50,
             edge_width_factor: float = 0.3,
             title: str = None,
             color: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create passing network visualization
        
        Args:
            ax: Matplotlib axes
            min_passes: Minimum passes for edge to show
            node_size_factor: Size multiplier for nodes
            edge_width_factor: Width multiplier for edges
            title: Plot title
            color: Team color
            
        Returns:
            Tuple of (figure, axes)
        """
        pitch = Pitch(figsize=(12, 8))
        fig, ax = pitch.draw(ax)
        
        team_color = color or self.colors.home
        
        # Filter edges by minimum passes
        edges = self.edges[self.edges['pass_count'] >= min_passes]
        
        # Create node position lookup
        node_pos = {row['player_id']: (row['avg_x'], row['avg_y']) 
                   for _, row in self.nodes.iterrows()}
        
        # Draw edges
        for _, edge in edges.iterrows():
            passer_id = edge['passer_id']
            receiver_id = edge['receiver_id']
            
            if passer_id in node_pos and receiver_id in node_pos:
                start = node_pos[passer_id]
                end = node_pos[receiver_id]
                
                width = edge['pass_count'] * edge_width_factor
                alpha = min(0.8, 0.2 + edge['pass_count'] * 0.1)
                
                ax.annotate(
                    '',
                    xy=end,
                    xytext=start,
                    arrowprops=dict(
                        arrowstyle='-',
                        color=team_color,
                        alpha=alpha,
                        linewidth=width,
                        connectionstyle='arc3,rad=0.1'
                    ),
                    zorder=5
                )
        
        # Draw nodes
        sizes = self.nodes['pass_count'] * node_size_factor
        
        scatter = ax.scatter(
            self.nodes['avg_x'],
            self.nodes['avg_y'],
            s=sizes,
            c=team_color,
            edgecolors='white',
            linewidths=2,
            zorder=10,
            alpha=0.9
        )
        
        # Add player names
        for _, node in self.nodes.iterrows():
            # Truncate name
            name = node['player_name']
            if ' ' in name:
                name = name.split()[-1][:10]  # Last name, max 10 chars
            
            ax.annotate(
                name,
                (node['avg_x'], node['avg_y']),
                xytext=(0, -15),
                textcoords='offset points',
                fontsize=8,
                color='white',
                ha='center',
                fontweight='bold',
                path_effects=[]
            )
        
        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
        elif self.team_name:
            ax.set_title(f"{self.team_name} - Passing Network", fontsize=14,
                        fontweight='bold', color='white', pad=10)
        
        fig.patch.set_facecolor('#1a1a2e')
        
        return fig, ax


class PossessionWonMap:
    """
    Create possession won / ball recovery heat maps
    """
    
    def __init__(self, recoveries_df: pd.DataFrame, team_name: str = None):
        self.recoveries = recoveries_df.copy()
        self.team_name = team_name
        self.colors = PitchColors()
    
    def plot_scatter(self,
                     ax: plt.Axes = None,
                     color: str = None,
                     title: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create scatter plot of possession won locations
        
        Returns:
            Tuple of (figure, axes)
        """
        pitch = Pitch(figsize=(12, 8))
        fig, ax = pitch.draw(ax)
        
        team_color = color or self.colors.home
        
        ax.scatter(
            self.recoveries['x'],
            self.recoveries['y'],
            c=team_color,
            s=80,
            alpha=0.7,
            edgecolors='white',
            linewidths=1,
            zorder=10
        )
        
        # Add pitch third lines
        ax.axvline(x=33.33, color='white', linestyle='--', alpha=0.3, lw=1)
        ax.axvline(x=66.66, color='white', linestyle='--', alpha=0.3, lw=1)
        
        # Stats by third
        thirds = self.recoveries['pitch_third'].value_counts()
        
        stats_text = "Ball Recoveries by Third:\n"
        for third in ['Defensive', 'Middle', 'Attacking']:
            count = thirds.get(third, 0)
            pct = count / len(self.recoveries) * 100 if len(self.recoveries) > 0 else 0
            stats_text += f"{third}: {count} ({pct:.0f}%)\n"
        
        ax.text(2, 95, stats_text.strip(), fontsize=9, color='white',
                va='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
        elif self.team_name:
            ax.set_title(f"{self.team_name} - Ball Recoveries", fontsize=14,
                        fontweight='bold', color='white', pad=10)
        
        fig.patch.set_facecolor('#1a1a2e')
        
        return fig, ax
    
    def plot_heatmap(self,
                     ax: plt.Axes = None,
                     title: str = None,
                     bins: int = 10) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create heat map of possession won locations
        
        Returns:
            Tuple of (figure, axes)
        """
        pitch = Pitch(figsize=(12, 8))
        fig, ax = pitch.draw(ax)
        
        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            self.recoveries['x'],
            self.recoveries['y'],
            bins=bins,
            range=[[0, 100], [0, 100]]
        )
        
        # Custom colormap
        cmap = LinearSegmentedColormap.from_list(
            'custom', ['#1a472a', '#2ecc71', '#f1c40f', '#e74c3c']
        )
        
        # Plot heatmap
        extent = [0, 100, 0, 100]
        im = ax.imshow(
            heatmap.T,
            origin='lower',
            extent=extent,
            cmap=cmap,
            alpha=0.6,
            aspect='auto',
            zorder=2
        )
        
        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
        elif self.team_name:
            ax.set_title(f"{self.team_name} - Ball Recovery Heatmap", fontsize=14,
                        fontweight='bold', color='white', pad=10)
        
        fig.patch.set_facecolor('#1a1a2e')
        
        return fig, ax


class DefensiveActionsMap:
    """
    Create defensive actions visualization
    """
    
    def __init__(self, defensive_df: pd.DataFrame, team_name: str = None):
        self.defensive = defensive_df.copy()
        self.team_name = team_name
    
    def plot(self,
             ax: plt.Axes = None,
             title: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot defensive actions by type
        
        Returns:
            Tuple of (figure, axes)
        """
        pitch = Pitch(figsize=(12, 8))
        fig, ax = pitch.draw(ax)
        
        # Colors by action type
        action_colors = {
            'Tackle': '#3498db',
            'Interception': '#e74c3c',
            'Clearance': '#f39c12',
            'Save': '#2ecc71'
        }
        
        action_markers = {
            'Tackle': 'o',
            'Interception': 's',
            'Clearance': '^',
            'Save': 'D'
        }
        
        # Plot each action type, in a fixed order so legends match across charts
        present = list(self.defensive['action_type'].unique())
        ordered = [t for t in action_colors if t in present] + [t for t in present if t not in action_colors]
        for action_type in ordered:
            actions = self.defensive[self.defensive['action_type'] == action_type]
            
            ax.scatter(
                actions['x'],
                actions['y'],
                c=action_colors.get(action_type, '#ffffff'),
                marker=action_markers.get(action_type, 'o'),
                s=80,
                alpha=0.8,
                edgecolors='white',
                linewidths=1,
                zorder=10,
                label=f"{action_type} ({len(actions)})"
            )
        
        ax.legend(loc='upper left', framealpha=0.9)
        
        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
        elif self.team_name:
            ax.set_title(f"{self.team_name} - Defensive Actions", fontsize=14,
                        fontweight='bold', color='white', pad=10)
        
        fig.patch.set_facecolor('#1a1a2e')
        
        return fig, ax


class MatchReport:
    """
    Generate complete match report with multiple visualizations
    """
    
    def __init__(self, 
                 match_info,
                 home_shots: pd.DataFrame,
                 away_shots: pd.DataFrame,
                 home_passes: pd.DataFrame,
                 away_passes: pd.DataFrame,
                 home_recoveries: pd.DataFrame,
                 away_recoveries: pd.DataFrame,
                 home_nodes: pd.DataFrame = None,
                 away_nodes: pd.DataFrame = None,
                 home_edges: pd.DataFrame = None,
                 away_edges: pd.DataFrame = None):
        
        self.match_info = match_info
        self.home_shots = home_shots
        self.away_shots = away_shots
        self.home_passes = home_passes
        self.away_passes = away_passes
        self.home_recoveries = home_recoveries
        self.away_recoveries = away_recoveries
        self.home_nodes = home_nodes
        self.away_nodes = away_nodes
        self.home_edges = home_edges
        self.away_edges = away_edges
    
    def generate_shot_comparison(self, save_path: str = None) -> plt.Figure:
        """Generate side-by-side shot maps"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Home shots
        shot_map_home = ShotMap(self.home_shots, self.match_info.home_team_name)
        shot_map_home.plot(ax=axes[0], title=f"{self.match_info.home_team_name}")
        
        # Away shots (flip coordinates for visualization)
        away_shots_flipped = self.away_shots.copy()
        away_shots_flipped['x'] = 100 - away_shots_flipped['x']
        away_shots_flipped['y'] = 100 - away_shots_flipped['y']
        
        shot_map_away = ShotMap(away_shots_flipped, self.match_info.away_team_name)
        shot_map_away.plot(ax=axes[1], title=f"{self.match_info.away_team_name}")
        
        fig.suptitle(
            f"{self.match_info.home_team_name} {self.match_info.home_score} - "
            f"{self.match_info.away_score} {self.match_info.away_team_name}",
            fontsize=16, fontweight='bold', color='white', y=0.98
        )
        
        fig.patch.set_facecolor('#1a1a2e')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', 
                       facecolor=fig.get_facecolor())
        
        return fig


if __name__ == "__main__":
    # Test with sample data
    print("Pitch visualization module loaded successfully!")
