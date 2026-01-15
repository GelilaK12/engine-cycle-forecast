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

def sensor_degradation_plot(df, sensor, engines, figures_path, highlight_outliers=False):
    plt.figure(figsize=(10,6))
    for engine in engines:
        engine_df = df[df["unit"] == engine]
        plt.plot(engine_df["cycle"], engine_df[sensor], alpha=0.7)
    if highlight_outliers:
        plt.title(f"{sensor} degradation (Outlier Engines Highlighted)")
    else:
        plt.title(f"{sensor} degradation")
    plt.xlabel("Cycle")
    plt.ylabel(sensor)
    plt.tight_layout()

    filename = f"{sensor}_degradation.png"
    if highlight_outliers:
         filename = f"{sensor}_degradation_outliers.png"

    plt.savefig(os.path.join(figures_path, filename), dpi=300)
    plt.close()


def compute_rul(df):
    max_cycles = df.groupby("unit")["cycle"].max().reset_index()
    max_cycles.columns = ["unit", "max_cycle"]
    df = df.merge(max_cycles, on="unit")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df
