import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def save_plot(fig, filename, figures_path):
    path = os.path.join(figures_path, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_histograms(df, figures_path):
    numeric_cols = df.select_dtypes("number").columns
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(df[col], bins=50, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        save_plot(fig, f"hist_{col}.png", figures_path)

def correlation_heatmap(df, figures_path):
    fig, ax = plt.subplots(figsize=(15,12))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    save_plot(fig, "correlation_heatmap.png", figures_path)

def sensor_degradation_plot(df, sensor, sample_engines, figures_path):
    fig, ax = plt.subplots(figsize=(12,6))
    for eng in sample_engines:
        subset = df[df["unit"]==eng]
        ax.plot(subset["cycle"], subset[sensor], label=f"Engine {eng}")
    ax.set_xlabel("Cycle")
    ax.set_ylabel(sensor)
    ax.set_title(f"{sensor} Degradation Over Cycles (Sample Engines)")
    ax.legend()
    save_plot(fig, f"{sensor}_degradation.png", figures_path)

def compute_rul(df):
    max_cycles = df.groupby("unit")["cycle"].max().reset_index()
    max_cycles.columns = ["unit", "max_cycle"]
    df = df.merge(max_cycles, on="unit")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df
