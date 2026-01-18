import os
import pandas as pd
from scripts.utils import plot_histograms, correlation_heatmap, sensor_degradation_plot

PROJECT_ROOT = r"C:\Users\kewtiiiii\Desktop\Projects\engine-cycle-forecast"
raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "train_FD001.txt")
processed_path = os.path.join(PROJECT_ROOT, "data", "processed", "fd001_processed.csv")
figures_path = os.path.join(PROJECT_ROOT, "artifacts", "figures")
os.makedirs(os.path.join(PROJECT_ROOT, "data", "processed"), exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

def load_raw():
    df = pd.read_csv(raw_path, sep=r"\s+", header=None)
    cols = ["unit", "cycle"] + [f"op_setting_{i}" for i in range(1,4)] + [f"sensor_{i}" for i in range(1,22)]
    df.columns = cols

    max_cycles = df.groupby("unit")["cycle"].max().reset_index()
    max_cycles.columns = ["unit", "max_cycle"]
    df = df.merge(max_cycles, on="unit")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df

