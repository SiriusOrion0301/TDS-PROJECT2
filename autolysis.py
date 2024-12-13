# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "numpy",
#   "scikit-learn",
#   "openai",
#   "rich",
#   "python-dotenv",
# ]
# ///

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()
console = Console()

def visualize_data(df):
    """Visualize the data with multiple plots."""
    # Histogram of numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(f'{col}_distribution.png')
        plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()

def analyze_data(df):
    """Perform analysis on the dataset."""
    # Example analysis: Summary statistics
    summary = df.describe(include='all')
    console.log(f"Summary Statistics:\n{summary}")

    # Outlier detection
    iso = IsolationForest(contamination=0.05)
    outliers = iso.fit_predict(df.select_dtypes(include=[np.number]))
    df['Outliers'] = outliers
    console.log(f"Outlier detection completed. Outliers marked in the dataset.")

    return summary

def main():
    if len(sys.argv) != 2:
        console.log("[red]Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)

    file_path = sys.argv[1]
    console.log(f"[yellow]Reading dataset: {file_path}[/]")
    
    try:
        df = pd.read_csv(file_path)
        console.log("[green]Dataset loaded successfully.[/]")
        
        # Analyze the data
        analyze_data(df)

        # Visualize the data
        visualize_data(df)

        # Save results to README.md
        with open('README.md', 'w') as f:
            f.write("# Automated Data Analysis Report\n")
            f.write("## Summary Statistics\n")
            f.write(f"{df.describe(include='all')}\n")
            f.write("## Visualizations\n")
            f.write("![Correlation Heatmap](correlation_heatmap.png)\n")
            for col in df.select_dtypes(include=[np.number]).columns:
                f.write(f"![{col} Distribution]({col}_distribution.png)\n")

        console.log("[green]Analysis and visualizations completed. Results saved to README.md.[/]")
    
    except Exception as e:
        console.log(f"[red]Error: {e}[/]")
        sys.exit(1)

if __name__ == "__main__":
    main()
