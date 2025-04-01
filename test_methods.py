import argparse
import os
import json
import torch
import numpy as np
import datetime
import cv2
import pandas as pd
import faiss
import time
from PIL import Image
from torchvision import transforms
from autoencoder import Autoencoder
from siamese import SiameseNetwork
from hash import dhash, phash, hash_to_bitvector
from finetuning import extract_features, initialize_model


def load_faiss_index_and_metadata(folder, method, model_name=None):
    if method == "autoencoder":
        index_path = os.path.join(folder, "autoencoder_faiss.index")
        metadata_path = os.path.join(folder, "autoencoder_metadata.csv")
    elif method == "siamese":
        index_path = os.path.join(folder, "siamese_faiss.index")
        metadata_path = os.path.join(folder, "siamese_metadata.csv")
    elif method in ["dhash", "phash"]:
        index_path = os.path.join(folder, f"{method}_faiss.index")
        metadata_path = os.path.join(folder, f"{method}_metadata.csv")
    elif method == "finetuned":
        index_path = os.path.join(folder, f"{model_name}_faiss.index")
        metadata_path = os.path.join(folder, f"{model_name}_metadata.csv")
    else:
        raise ValueError("Unknown method")

    if method in ["dhash", "phash"]:
        index = faiss.read_index_binary(index_path)
    else:
        index = faiss.read_index(index_path)
    metadata = pd.read_csv(metadata_path)
    return index, metadata


def evaluate_retrieval(query_folder, database_folder, save_folder, method, model_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if method == "autoencoder":
        autoencoder = Autoencoder(256).to(device)
        autoencoder.load_state_dict(torch.load(os.path.join(database_folder, "autoencoder.pth"), map_location=device))
        autoencoder.eval()
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    elif method == "siamese":
        siamese_model = SiameseNetwork(256).to(device)
        siamese_model.load_state_dict(
            torch.load(os.path.join(database_folder, "siamese_model.pth"), map_location=device))
        siamese_model.eval()
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    elif method == "finetuned":
        model, transform = initialize_model(model_name)
    else:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    index, metadata = load_faiss_index_and_metadata(database_folder, method, model_name)

    results = []
    output_data = []
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_top1 = 0
    total_top5 = 0
    total_queries = 0
    total_time = 0.0

    for subfolder in sorted(os.listdir(query_folder)):
        subfolder_path = os.path.join(query_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        class_top1 = 0
        class_top5 = 0
        class_times = []
        class_queries = 0

        for filename in sorted(os.listdir(subfolder_path)):
            file_path = os.path.join(subfolder_path, filename)
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):
                try:
                    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        raise ValueError("Error loading image with OpenCV")

                    start = time.time()

                    if method == "autoencoder":
                        image_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            query_embedding, _ = autoencoder(image_tensor)
                        query_vector = query_embedding.squeeze().cpu().numpy().astype(np.float32).reshape(1, -1)

                    elif method == "siamese":
                        image_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            query_embedding = siamese_model.forward_once(image_tensor)
                        query_vector = query_embedding.squeeze().cpu().numpy().astype(np.float32).reshape(1, -1)

                    elif method == "dhash":
                        hash_val = dhash(image)
                        query_vector = hash_to_bitvector(hash_val).reshape(1, -1).astype(np.uint8)

                    elif method == "phash":
                        hash_val = phash(image)
                        query_vector = hash_to_bitvector(hash_val).reshape(1, -1).astype(np.uint8)

                    elif method == "finetuned":
                        query_embedding = extract_features(file_path, model, transform, device)
                        query_vector = query_embedding.astype(np.float32).reshape(1, -1)

                    distances, indices = index.search(query_vector, 5)
                    top_5 = metadata.iloc[indices[0]]["image_path"].tolist()
                    top_5_folders = [os.path.basename(os.path.dirname(path)) for path in top_5]

                    elapsed = time.time() - start
                    class_times.append(elapsed)
                    total_time += elapsed

                    if top_5_folders[0] == subfolder:
                        total_top1 += 1
                        class_top1 += 1
                    if any(folder == subfolder for folder in top_5_folders):
                        total_top5 += 1
                        class_top5 += 1

                    total_queries += 1
                    class_queries += 1

                    output_data.append({
                        "filename": filename,
                        "method": method,
                        "model": model_name if model_name else "N/A",
                        "query_folder": subfolder,
                        "top_5": top_5,
                        "top1_match": top_5_folders[0] == subfolder,
                        "top5_match": any(folder == subfolder for folder in top_5_folders),
                        "retrieval_time_sec": round(elapsed, 4)
                    })
                except Exception as e:
                    output_data.append({
                        "filename": filename,
                        "error": str(e)
                    })

        avg_time = np.mean(class_times) if class_times else 0.0
        results.append([
            method,
            model_name if model_name else "N/A",
            subfolder,
            class_top1 / class_queries if class_queries else 0.0,
            class_top5 / class_queries if class_queries else 0.0,
            round(avg_time, 4),
            timestamp
        ])

    df = pd.DataFrame(results, columns=[
        "Method", "Model", "Class", "Top-1 Acc", "Top-5 Acc", "Avg Time (s)", "Timestamp"
    ])

    csv_path = os.path.join(save_folder, "retrieval_results.csv")
    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

    json_path = os.path.join(save_folder, f"{method}_{model_name if model_name else 'none'}_retrieval_results.json")
    with open(json_path, "w") as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"[{method}] Retrieval complete. Results saved to:")
    print(f"- {csv_path}")
    print(f"- {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image retrieval methods.")
    parser.add_argument("--query_folder", required=True, help="Folder containing query images")
    parser.add_argument("--database_folder", required=True, help="Folder containing FAISS index and metadata")
    parser.add_argument("--save_folder", required=True, help="Folder to save results")
    parser.add_argument("--method", choices=["autoencoder", "dhash", "phash", "finetuned", "siamese"], required=True,
                        help="Retrieval method to use")
    parser.add_argument("--model_name", choices=["resnet", "efficientnet", "mobilenet"], default=None,
                        help="Model name for finetuned method")
    args = parser.parse_args()

    evaluate_retrieval(args.query_folder, args.database_folder, args.save_folder, args.method, args.model_name)
