# --- Standard Library ---
import os
import sys
import csv
import json
import random
from functools import partial
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Callable, Optional
from itertools import combinations, product
from collections import Counter

# --- Numerical & Data Manipulation ---
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, wilcoxon

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from collections import defaultdict
import networkx as nx
import matplotlib.colors as mcolors
from colorsys import hls_to_rgb

def plot_fitness(fitness_dfs, title_suffix=""):

    # Each selection method has a specific color to be easier to distinguish
    method_palettes = {
        'ranking': [
            "#49006a", "#7a0177", "#ae017e",
            "#67001f", "#980043", "#ce1256",
            "#df65b0", "#88419d"
        ],
        'fitness_proportionate': [
            "#023858", "#045a8d", "#0570b0", "#3690c0", "#41b6c4", "#253494",
            "#377eb8", "#02818a"
        ],
        'tournament': [
            "#67000d", "#cb181d", "#ef3b2c", "#fc4e2a", "#d73027", "#f46d43",
            "#ff7f00", "#993404"

        ]
    }

    # This section was made with the help of ChatGPT, basically it counts how many times a color has appeared to not repeat it
    method_color_index = defaultdict(int)

    def get_color_for_config(name):
        for method in method_palettes:
            if name.startswith(method):
                color_list = method_palettes[method]
                color = color_list[method_color_index[method] % len(color_list)]
                method_color_index[method] += 1
                return color
        return '#999999'  # fallback gray

    # Sorting items by fitness so they appear in a cleaner format
    sorted_items = sorted(
        fitness_dfs.items(),
        key=lambda x: x[1].mean(axis=0).iloc[-1],
        reverse=True
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Mean Fitness", "Median Fitness"),
        shared_yaxes=True,
        horizontal_spacing=0.1
    )

    for i, (config_name, df) in enumerate(sorted_items):
        mean_fitness = df.mean(axis=0)
        median_fitness = df.median(axis=0)
        display_name = config_name.replace("_", " ")
        color = get_color_for_config(config_name)

        show_legend = i < 10

        fig.add_trace(go.Scatter(
            x=mean_fitness.index,
            y=mean_fitness.values,
            mode='lines',
            name=display_name,
            legendgroup=config_name,
            showlegend=show_legend,
            line=dict(color=color),
            hovertemplate=f"{display_name}<br>Gen: %{{x}}<br>Fitness: %{{y:.4f}}<extra></extra>"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=median_fitness.index,
            y=median_fitness.values,
            mode='lines',
            name=display_name,
            legendgroup=config_name,
            showlegend=False,
            line=dict(color=color),
            hovertemplate=f"{display_name}<br>Gen: %{{x}}<br>Fitness: %{{y:.4f}}<extra></extra>"
        ), row=1, col=2)

    fig.update_layout(
            title_text=f"Fitness Across Generations - {title_suffix}",
            template="simple_white",
            height=600,
            width=1700,
            margin=dict(l=40, r=250, t=60, b=60),
            legend=dict(
            title="Configurations",
            orientation="v",
            y=1,
            x=1.02,
            font=dict(size=10)
        )
    )

    fig.update_xaxes(title_text="Generation", row=1, col=1)
    fig.update_xaxes(title_text="Generation", row=1, col=2)
    fig.update_yaxes(title_text="Fitness", row=1, col=1)

    fig.show()

def plot_final_fitness_boxplots(fitness_dfs, title_suffix=''):
    data = []
    for config_label, df in fitness_dfs.items():
        final_gen_fitness = df.iloc[:, -1].values  # final generation fitness
        for value in final_gen_fitness:
            data.append({'Fitness': value, 'Configuration': config_label})
    df_long = pd.DataFrame(data)

    medians = df_long.groupby('Configuration')['Fitness'].median().sort_values(ascending=False)
    df_long['Configuration'] = pd.Categorical(df_long['Configuration'], categories=medians.index, ordered=True)

    method_palettes = {
        'ranking': [
            "#49006a", "#7a0177", "#ae017e",
            "#67001f", "#980043", "#ce1256",
            "#df65b0", "#88419d"
        ],
        'fitness_proportionate': [
            "#023858", "#045a8d", "#0570b0", "#3690c0", "#41b6c4", "#253494",
            "#377eb8", "#02818a"
        ],
        'tournament': [
            "#67000d", "#cb181d", "#ef3b2c", "#fc4e2a", "#d73027", "#f46d43",
            "#ff7f00", "#993404"

        ]
    }

    method_color_index = defaultdict(int)
    config_colors = {}

    for config in medians.index:
        for method in method_palettes:
            if config.startswith(method):
                palette = method_palettes[method]
                color = palette[method_color_index[method] % len(palette)]
                config_colors[config] = color
                method_color_index[method] += 1
                break
        else:
            config_colors[config] = '#999999'  # fallback gray

    height = max(6, 0.4 * len(medians))
    plt.figure(figsize=(14, height))
    
    ax = sns.boxplot(
        y='Configuration',
        x='Fitness',
        data=df_long,
        palette=config_colors,
        linewidth=2,
        fliersize=3,
        width=0.6,
        orient='h'
    )

    title = 'Final Generation Fitness per Run'
    if title_suffix:
        title += f' ({title_suffix})'
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel('Final Generation Fitness', fontsize=13)
    plt.ylabel('Configuration', fontsize=13)

    # Styling
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_component_comparisons(selection_fit_dfs, crossover_fit_dfs, mutation_fit_dfs):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Selection Comparison", "Crossover Comparison", "Mutation Comparison"),
        shared_yaxes=True,
        horizontal_spacing=0.08
    )

    # Selection
    for name, df in selection_fit_dfs.items():
        median_fitness = df.median(axis=0)
        fig.add_trace(go.Scatter(
            x=median_fitness.index,
            y=median_fitness.values,
            mode='lines',
            name=f"[Selection] {name.replace('_', ' ')}",
            legendgroup="selection",
            showlegend=True,
            hovertemplate=f"{name}<br>Gen: %{{x}}<br>Fitness: %{{y:.4f}}<extra></extra>"
        ), row=1, col=1)

    # Crossover
    for name, df in crossover_fit_dfs.items():
        median_fitness = df.median(axis=0)
        fig.add_trace(go.Scatter(
            x=median_fitness.index,
            y=median_fitness.values,
            mode='lines',
            name=f"[Crossover] {name.replace('_', ' ')}",
            legendgroup="crossover",
            showlegend=True,
            hovertemplate=f"{name}<br>Gen: %{{x}}<br>Fitness: %{{y:.4f}}<extra></extra>"
        ), row=1, col=2)

    # Mutation
    for name, df in mutation_fit_dfs.items():
        median_fitness = df.median(axis=0)
        fig.add_trace(go.Scatter(
            x=median_fitness.index,
            y=median_fitness.values,
            mode='lines',
            name=f"[Mutation] {name.replace('_', ' ')}",
            legendgroup="mutation",
            showlegend=True,
            hovertemplate=f"{name}<br>Gen: %{{x}}<br>Fitness: %{{y:.4f}}<extra></extra>"
        ), row=1, col=3)

    fig.update_layout(
        height=600,
        width=1700,
        title="Operator Comparison: Median Fitness Over Generations",
        template="simple_white",
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            title="Legend",
            font=dict(size=11)
        ),
        margin=dict(l=60, r=250, t=60, b=50),
        showlegend=True
    )

    # Remove grid from axes
    for i in range(1, 4):
        fig.update_xaxes(showgrid=False, row=1, col=i)
        fig.update_yaxes(showgrid=False, row=1, col=i)

    fig.update_xaxes(title_text="Generation", row=1, col=1)
    fig.update_xaxes(title_text="Generation", row=1, col=2)
    fig.update_xaxes(title_text="Generation", row=1, col=3)
    fig.update_yaxes(title_text="Fitness", row=1, col=1)

    fig.show()


def plot_component_comparisons_from_curves(selection_curves, crossover_curves, mutation_curves):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Selection Comparison", "Crossover Comparison", "Mutation Comparison"),
        shared_yaxes=True,
        horizontal_spacing=0.08
    )

    #  Selection 
    for name, curve in selection_curves.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(curve))),
            y=curve,
            mode='lines',
            name=f"[Selection] {name.replace('_', ' ')}",
            legendgroup="selection",
            showlegend=True,
            hovertemplate=f"{name}<br>Gen: %{{x}}<br>Fitness: %{{y:.4f}}<extra></extra>"
        ), row=1, col=1)

    #  Crossover 
    for name, curve in crossover_curves.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(curve))),
            y=curve,
            mode='lines',
            name=f"[Crossover] {name.replace('_', ' ')}",
            legendgroup="crossover",
            showlegend=True,
            hovertemplate=f"{name}<br>Gen: %{{x}}<br>Fitness: %{{y:.4f}}<extra></extra>"
        ), row=1, col=2)

    #  Mutation 
    for name, curve in mutation_curves.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(curve))),
            y=curve,
            mode='lines',
            name=f"[Mutation] {name.replace('_', ' ')}",
            legendgroup="mutation",
            showlegend=True,
            hovertemplate=f"{name}<br>Gen: %{{x}}<br>Fitness: %{{y:.4f}}<extra></extra>"
        ), row=1, col=3)

    fig.update_layout(
        height=600,
        width=1700,
        title="Operator Comparison: Median of Medians Over Generations",
        template="simple_white",
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            title="Legend",
            font=dict(size=11)
        ),
        margin=dict(l=60, r=250, t=60, b=50),
        showlegend=True
    )

    for i in range(1, 4):
        fig.update_xaxes(showgrid=False, title_text="Generation", row=1, col=i)
        fig.update_yaxes(showgrid=False, title_text="Fitness" if i == 1 else "", row=1, col=i)

    fig.show()

def plot_statistical_distance_graph(best_configs, final_gen_fitness, p_values_df):
    # Mean fitness
    mean_fitness = {cfg: np.mean(final_gen_fitness[cfg]) for cfg in best_configs}

    # Group by selection method
    def get_selection_method(cfg):
        return cfg.split('_')[0]

    grouped_configs = defaultdict(list)
    for cfg in best_configs:
        method = get_selection_method(cfg)
        grouped_configs[method].append(cfg)

    # Strong color shades per group
    def get_strong_colors(hue_deg, n_colors):
        hue = hue_deg / 360
        return [
            mcolors.to_hex(hls_to_rgb(hue, l, 0.95))
            for l in np.linspace(0.35, 0.6, n_colors)
        ]

    hue_map = {
        'fitness': 120,
        'ranking': 270,
        'tournament': 30
    }

    color_map = {}
    for method, cfgs in grouped_configs.items():
        shades = get_strong_colors(hue_map[method], len(cfgs))
        for cfg, color in zip(sorted(cfgs), shades):
            color_map[cfg] = color

    sorted_configs = [cfg for method in sorted(grouped_configs) for cfg in sorted(grouped_configs[method])]

    # Fully connected graph with distance = 1 - p_value
    G = nx.Graph()
    for cfg1, cfg2 in combinations(best_configs, 2):
        p = p_values_df.loc[cfg1, cfg2]
        # Protect against NaNs or invalid p-values
        if not np.isnan(p):
            G.add_edge(cfg1, cfg2, weight=1 - p)

    # Layout with distance = dissimilarity
    pos = nx.spring_layout(G, weight='weight', seed=42)

    # Edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='lightgray'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )

    # Nodes
    node_traces = []
    for cfg in sorted_configs:
        if cfg not in G.nodes:
            continue
        x, y = pos[cfg]
        node_traces.append(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=22, color=color_map[cfg], line=dict(color='black', width=1)),
            hovertemplate=f"{cfg}<br>Mean Final Fitness: {mean_fitness[cfg]:.4f}<extra></extra>",
            name=cfg,
            legendgroup=get_selection_method(cfg),
            showlegend=True
        ))

    # Layout
    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        title="Statistical Distance Between Configurations (Node Distance ∝ Dissimilarity)",
        title_font_size=16,
        showlegend=True,
        legend_title="Configurations Grouped by Selection Method",
        legend=dict(
            font=dict(size=10),
            traceorder='normal',
            itemsizing='trace',
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0)',
        ),
        margin=dict(l=40, r=250, t=60, b=40),
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    fig.show()


def plot_fine_tune(df_hp_tuning, title_suffix=''):
    # Parse JSON strings if needed
    if isinstance(df_hp_tuning['fitness_history'].iloc[0], str):
        df_hp_tuning['fitness_history'] = df_hp_tuning['fitness_history'].apply(json.loads)

    # Build long-form DataFrame
    records = []
    for _, row in df_hp_tuning.iterrows():
        config = f"xo={row['xo_prob']}, mut={row['mut_prob']}, swaps={row['n_swaps']}, tournament_size={row['tournament_size']}"
        final_idx = len(row['fitness_history'][0]) - 1
        for run_id, run in enumerate(row['fitness_history']):
            records.append({
                'Configuration': config,
                'Fitness': run[final_idx],
                'n_swaps': int(row['n_swaps']),
                'tournament_size': int(row['tournament_size']),
            })

    df_final = pd.DataFrame(records)

    # Define complexity tiers
    def classify(value, low, high):
        if value <= low:
            return 'Low'
        elif value >= high:
            return 'High'
        else:
            return 'Medium'

    df_final['Swap Tier'] = df_final['n_swaps'].apply(lambda x: classify(x, 2, 5))
    df_final['Tournament Tier'] = df_final['tournament_size'].apply(lambda x: classify(x, 2, 5))
    df_final['Complexity'] = df_final['Swap Tier'] + '-' + df_final['Tournament Tier']

    # Set palette for complexity
    complexity_palette = {
        'Low-Low': '#1a9850',
        'Low-Medium': '#66bd63',
        'Low-High': '#a6d96a',
        'Medium-Low': '#fdae61',
        'Medium-Medium': '#f46d43',
        'Medium-High': '#d73027',
        'High-Low': '#4575b4',
        'High-Medium': '#74add1',
        'High-High': '#abd9e9'
    }

    # Map color to each configuration
    config_to_complexity = df_final.groupby("Configuration")["Complexity"].first().to_dict()
    median_order = df_final.groupby("Configuration")["Fitness"].median().sort_values(ascending=False)
    df_final['Configuration'] = pd.Categorical(df_final['Configuration'], categories=median_order.index, ordered=True)
    config_colors = {cfg: complexity_palette.get(config_to_complexity[cfg], '#999999') for cfg in median_order.index}

    # Plot
    height = max(6, 0.4 * len(median_order))
    plt.figure(figsize=(12, height))
    ax = sns.boxplot(
        y='Configuration',
        x='Fitness',
        data=df_final,
        palette=[config_colors[cfg] for cfg in median_order.index],
        order=median_order.index,
        linewidth=1.5,
        fliersize=3,
        width=0.6,
        orient='h'
    )

    # Title and styling
    title = "Final Generation Fitness per Run"
    if title_suffix:
        title += f" ({title_suffix})"

    plt.title(title, fontsize=15, weight='bold')
    plt.xlabel("Final Generation Fitness", fontsize=12)
    plt.ylabel("Configuration", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    sns.despine()
    plt.tight_layout()

    # Legend for complexity
    handles = [plt.Line2D([0], [0], marker='s', color='w', label=key, markersize=10, markerfacecolor=val)
               for key, val in complexity_palette.items()]
    ax.legend(handles=handles, title='Complexity Tier', loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


def plot_final_fitness_adaptive_ga(df_adaptive_ga, title='Final Generation Fitness per Configuration'):
    # Extract final generation (last row) and reshape
    last_generation = df_adaptive_ga.tail(1).T
    last_generation.columns = ["Fitness"]
    last_generation["Configuration"] = last_generation.index.str.replace(r"\.\d+$", "", regex=True)

    # Sort configurations by median
    medians = last_generation.groupby("Configuration")["Fitness"].median().sort_values(ascending=False)
    last_generation["Configuration"] = pd.Categorical(last_generation["Configuration"], categories=medians.index, ordered=True)

    # Flat list of colors inspired by your palettes
    palette = [
        "#67000d", "#cb181d", "#ef3b2c", "#fc4e2a", "#ff7f00", "#993404",
        "#023858", "#045a8d", "#0570b0", "#3690c0", "#41b6c4", "#253494",
        "#49006a", "#7a0177", "#ae017e", "#ce1256", "#df65b0", "#88419d"
    ]

    # Map colors to each configuration (cyclically if needed)
    unique_configs = medians.index.tolist()
    config_colors = {config: palette[i % len(palette)] for i, config in enumerate(unique_configs)}

    # Plot
    height = max(6, 0.45 * len(medians))
    plt.figure(figsize=(14, height))

    sns.boxplot(
        data=last_generation,
        y="Configuration",
        x="Fitness",
        palette=config_colors,
        linewidth=2,
        fliersize=3,
        width=0.6,
        orient='h',
        showmeans=False
    )

    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel("Final Generation Fitness", fontsize=13)
    plt.ylabel("Configuration", fontsize=13)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()

def plot_median_fitness_over_generations(fitness_histories, title="Median Fitness per Generation across Configurations"):
    """
    Plots median fitness over generations for each configuration/scenario.

    """
    plt.figure(figsize=(12, 6))
    palette = [
        "#67000d", "#cb181d", "#ef3b2c", "#fc4e2a", "#ff7f00", "#993404",
        "#023858", "#045a8d", "#0570b0", "#3690c0", "#41b6c4", "#253494",
        "#49006a", "#7a0177", "#ae017e", "#ce1256", "#df65b0", "#88419d"
    ]
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(fitness_histories.keys())}

    for name, df in fitness_histories.items():
        median_fitness = df.median(axis=1)
        plt.plot(median_fitness, label=name, linewidth=2, color=color_map[name])

    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Median Fitness", fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Configuration")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_median_fitness_over_generations(df_adaptive_ga, title="Median Fitness per Generation Across Configurations"):
    """
    Plots the median fitness per generation for each configuration.
    """

    df_clean = df_adaptive_ga.copy()
    df_clean.columns = df_clean.columns.str.replace(r"\.\d+$", "", regex=True)

    # Group by configuration name and compute median across runs
    grouped = df_clean.groupby(axis=1, level=0)
    median_per_config = grouped.median()

    # Sort by final generation fitness (descending)
    final_medians = median_per_config.iloc[-1].sort_values(ascending=False)
    median_per_config = median_per_config[final_medians.index]


    palette = [
        "#67000d", "#cb181d", "#ef3b2c", "#fc4e2a", "#ff7f00", "#993404",
        "#023858", "#045a8d", "#0570b0", "#3690c0", "#41b6c4", "#253494",
        "#49006a", "#7a0177", "#ae017e", "#ce1256", "#df65b0", "#88419d"
    ]

    color_map = {config: palette[i % len(palette)] for i, config in enumerate(median_per_config.columns)}

    # Plot
    plt.figure(figsize=(14, 6))
    for config in median_per_config.columns:
        plt.plot(median_per_config.index, median_per_config[config], label=config,
                 color=color_map[config], linewidth=2)

    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Median Fitness", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Configuration", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()