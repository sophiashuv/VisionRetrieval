import argparse
import os
import json
import matplotlib.pyplot as plt
import cv2
import random
import pandas as pd


def plot_retrieval(query_image_path, query_class, top5_paths, method, model, output_path, idx):
    fig, axs = plt.subplots(1, 6, figsize=(18, 4))
    plt.tight_layout()

    query_img = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
    axs[0].imshow(query_img, cmap='gray')
    axs[0].set_title(f"Query\n(Class {query_class})")
    axs[0].axis('off')

    for i, path in enumerate(top5_paths):
        retrieved_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        retrieved_class = os.path.basename(os.path.dirname(path))
        axs[i + 1].imshow(retrieved_img, cmap='gray')
        axs[i + 1].set_title(f"Top {i + 1}\n(Class {retrieved_class})")
        axs[i + 1].axis('off')

    model_suffix = model if model != "N/A" else "none"
    filename = f"{method}_{model_suffix}_viz_{idx + 1}.png"
    plt.suptitle(f"Method: {method} | Model: {model_suffix}", fontsize=14)
    plt.savefig(os.path.join(output_path, filename))
    plt.close()


def visualize_retrievals(json_path, query_root, output_folder, n_samples=5):
    os.makedirs(output_folder, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    valid_entries = [
        entry for entry in data
        if "top_5" in entry and len(entry["top_5"]) == 5
    ]

    sampled_entries = random.sample(valid_entries, min(n_samples, len(valid_entries)))

    for idx, entry in enumerate(sampled_entries):
        query_class = entry["query_folder"]
        query_image_path = os.path.join(query_root, query_class, entry["filename"])
        top5_paths = entry["top_5"]
        method = entry["method"]
        model = entry.get("model", "none")

        if os.path.exists(query_image_path) and all(os.path.exists(p) for p in top5_paths):
            plot_retrieval(query_image_path, query_class, top5_paths, method, model, output_folder, idx)

    print(f"Saved {len(sampled_entries)} visualizations to: {output_folder}")


def plot_metrics(csv_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Rename columns to consistent format
    df = df.rename(columns={
        "Top-1 Accuracy": "Top-1 Acc",
        "Top-5 Accuracy": "Top-5 Acc",
        "Avg Retrieval Time": "Avg Time (s)"
    })

    metrics = ["Top-1 Acc", "Top-5 Acc", "Precision@5", "Recall@5", "mAP", "Avg Time (s)"]
    methods = df["Method"].tolist()

    for metric in metrics:
        plt.figure(figsize=(14, 6))
        values = df[metric].tolist()
        plt.bar(methods, values)
        plt.xticks(rotation=90)
        plt.ylabel(metric)
        plt.title(f"{metric} Comparison Across Methods")
        plt.tight_layout()

        filename = f"{metric.replace('@', 'at').replace(' ', '_').lower()}_barplot.png"
        plt.savefig(os.path.join(output_folder, filename))
        plt.close()

    print(f"Bar plots saved to: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize retrieval results or metric plots.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Handler 1: Retrieval visualizations
    retrieval_parser = subparsers.add_parser("retrieval", help="Visualize top-5 retrieval results from JSON")
    retrieval_parser.add_argument("--json_path", required=True, help="Path to the retrieval results JSON file")
    retrieval_parser.add_argument("--query_root", required=True, help="Root path to query image folders")
    retrieval_parser.add_argument("--output_folder", required=True, help="Folder to save the visualizations")
    retrieval_parser.add_argument("--n_samples", type=int, default=5, help="Number of visualizations to create")

    # Handler 2: Metric plots
    metric_parser = subparsers.add_parser("metrics", help="Plot metric comparisons from CSV")
    metric_parser.add_argument("--csv_path", required=True, help="Path to retrieval results CSV file")
    metric_parser.add_argument("--output_folder", required=True, help="Folder to save the metric plots")

    args = parser.parse_args()

    if args.command == "retrieval":
        visualize_retrievals(args.json_path, args.query_root, args.output_folder, args.n_samples)
    elif args.command == "metrics":
        plot_metrics(args.csv_path, args.output_folder)
