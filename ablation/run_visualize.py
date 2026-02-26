"""Run visualization — edit the variables below then: python run_visualize.py"""
import subprocess, sys

CITY       = "Bologna, Italy"
CHECKPOINT = "outputs/film_efficientnet_b3/best_model.pth"
MODEL      = "film"   # single | dual | film | crossattn
JSON       = "/workspace/PopulationDataset/final_clustered_samples.json"
BASE       = "/workspace/PopulationDataset"
CACHE      = "/workspace/PopulationDataset/cache"
OUT        = "/workspace/visualizations"
STRIDE     = "16"

subprocess.run([
    sys.executable, "visualize.py",
    "--city",       CITY,
    "--checkpoint", CHECKPOINT,
    "--model",      MODEL,
    "--json",       JSON,
    "--base",       BASE,
    "--cache",      CACHE,
    "--out",        OUT,
    "--stride",     STRIDE,
])
