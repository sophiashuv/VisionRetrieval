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
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from autoencoder import Autoencoder
from siamese import SiameseNetwork, build_encoder
from hash import dhash, phash, hash_to_bitvector
from pretrained import extract_features, initialize_model


def visualize_reconstruction(image_path, model, transform, device, encoder_type="basic", save_path=None):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        _, reconstruction = model(image_tensor)

    original_np = image_tensor.squeeze().cpu().numpy()
    recon_np = reconstruction.squeeze().cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_np, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(recon_np, cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

    plt.suptitle(f"Autoencoder Reconstruction ({encoder_type})", fontsize=14)

    if save_path:
        plt.savefig(save_path)
        print(f"Reconstruction saved at {save_path}")
    else:
        plt.show()

    plt.close()


def load_faiss_index_and_metadata(folder, method):
    index_path = os.path.join(folder, f"{method}_faiss.index")
    metadata_path = os.path.join(folder, f"{method}_metadata.csv")
    index = faiss.read_index_binary(index_path) if method in ["dhash", "phash"] else faiss.read_index(index_path)
    metadata = pd.read_csv(metadata_path)
    return index, metadata


def parse_siamese_filename(filename):
    autoencoder_based = "autoencoder" in filename
    encoder = filename.replace(".pth", "").split("_")[-1]
    return autoencoder_based, encoder


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters())


def compute_relevance_metrics(y_true, y_pred_top5):
    binary_relevance = [
        [1 if os.path.basename(os.path.dirname(p)) == true else 0 for p in preds]
        for true, preds in zip(y_true, y_pred_top5)
    ]

    precision_at_5 = np.mean([np.sum(rel) / 5.0 for rel in binary_relevance])
    recall_at_5 = np.mean([min(np.sum(rel), 1.0) for rel in binary_relevance])

    ap_list = []
    for rel in binary_relevance:
        num_relevant = 0
        ap = 0.0
        for i, r in enumerate(rel):
            if r:
                num_relevant += 1
                ap += num_relevant / (i + 1)
        ap = ap / num_relevant if num_relevant else 0.0
        ap_list.append(ap)

    mAP = np.mean(ap_list)
    return precision_at_5, recall_at_5, mAP


def save_summary_csv(save_folder, method, top1, top5, precision, recall, mAP, avg_time, param_count, timestamp):
    summary_df = pd.DataFrame([{
        "Method": method,
        "Top-1 Accuracy": top1,
        "Top-5 Accuracy": top5,
        "Precision@5": precision,
        "Recall@5": recall,
        "mAP": mAP,
        "Avg Retrieval Time": avg_time,
        "Num Parameters": param_count,
        "Timestamp": timestamp
    }])
    csv_path = os.path.join(save_folder, "retrieval_metrics.csv")
    summary_df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)


def evaluate_retrieval(query_folder, database_folder, save_folder, method, embedding_dim=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    param_count = 0
    if "siamese" in method:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        models = method.split("_")
        encoder = build_encoder(models[-1], embedding_dim, None, device)
        use_head = models[-1] in ["basic", "better"]
        if models[-1] == "mobilenet":
            encoder_output_dim = 1280
        elif models[-1] == "resnet":
            encoder_output_dim = 512
        elif models[-1] == "efficientnet":
            encoder_output_dim = 1280
        else:
            encoder_output_dim = embedding_dim

        use_head = True
        siamese_model = SiameseNetwork(encoder=encoder, embedding_dim=embedding_dim, use_head=use_head,
                               encoder_output_dim=encoder_output_dim).to(device)


        siamese_model.load_state_dict(torch.load(os.path.join(database_folder, f"{method}.pth"), map_location=device))
        siamese_model.eval()
        model = siamese_model
    elif "autoencoder" in method:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        _,  encoder_type = method.split("_")
        model = Autoencoder(embedding_dim=embedding_dim, encoder_type=encoder_type).to(device)
        model.load_state_dict(torch.load(os.path.join(database_folder, f"{method}.pth"), map_location=device))
        model.eval()
    elif "pretrained" in method:
        model, transform = initialize_model(method.split("_")[-1])
        model.eval()
    if model:
        param_count = count_model_parameters(model)

    index, metadata = load_faiss_index_and_metadata(database_folder, method)

    output_data, y_true, y_pred_top5 = [], [], []
    total_top1, total_top5, total_queries, total_time = 0, 0, 0, 0.0
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for subfolder in sorted(os.listdir(query_folder)):
        subfolder_path = os.path.join(query_folder, subfolder)
        if not os.path.isdir(subfolder_path): continue

        for filename in sorted(os.listdir(subfolder_path)):
            file_path = os.path.join(subfolder_path, filename)
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")): continue
            try:
                start = time.time()

                if "siamese" in method:
                    image = Image.open(file_path).convert("RGB")
                    image_tensor = transform(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        query_embedding = model.forward_once(image_tensor)
                    query_vector = query_embedding.squeeze().cpu().numpy()
                    query_vector = query_vector / np.linalg.norm(query_vector)
                    query_vector = query_vector.astype(np.float32).reshape(1, -1)

                elif "autoencoder" in method:
                    image = Image.open(file_path).convert("RGB")
                    image_tensor = transform(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        query_embedding, _ = model(image_tensor, current_epoch=999)
                        query_vector = query_embedding.squeeze().cpu().numpy()
                        query_vector = query_vector / np.linalg.norm(query_vector)
                        query_vector = query_vector.astype(np.float32).reshape(1, -1)
                elif method in ["dhash", "phash"]:
                    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    hash_func = dhash if method == "dhash" else phash
                    query_vector = hash_to_bitvector(hash_func(image)).reshape(1, -1).astype(np.uint8)
                elif "pretrained" in method:
                    query_vector = extract_features(file_path, model, transform, device)
                    query_vector = query_vector / np.linalg.norm(query_vector)
                    query_vector = query_vector.astype(np.float32).reshape(1, -1)

                distances, indices = index.search(query_vector, 5)
                retrieved_paths = metadata.iloc[indices[0]]['image_path'].tolist()
                query_basename = os.path.basename(filename)
                top_5 = [p for p in retrieved_paths if os.path.basename(p) != query_basename]

                if len(top_5) < 1:
                    print(f"Skipped: all top results for {filename} are the same image.")
                    continue
                top_5 = top_5[:5]
                top_5_folders = [os.path.basename(os.path.dirname(p)) for p in top_5]

                elapsed = time.time() - start
                total_time += elapsed

                top1 = top_5_folders[0] == subfolder
                top5 = subfolder in top_5_folders

                y_true.append(subfolder)
                y_pred_top5.append(top_5)
                total_top1 += int(top1)
                total_top5 += int(top5)
                total_queries += 1

                output_data.append({
                    "filename": filename,
                    "method": method,
                    "query_folder": subfolder,
                    "top_5": top_5,
                    "distances": [round(float(d), 6) for d in distances[0]],
                    "top1_match": top1,
                    "top5_match": top5,
                    "retrieval_time_sec": round(elapsed, 4)
                })

            except Exception as e:
                output_data.append({"filename": filename, "error": str(e)})

    top1_acc = total_top1 / total_queries if total_queries else 0.0
    top5_acc = total_top5 / total_queries if total_queries else 0.0
    avg_time = total_time / total_queries if total_queries else 0.0
    precision, recall, mAP = compute_relevance_metrics(y_true, y_pred_top5)

    save_summary_csv(save_folder, method, top1_acc, top5_acc, precision, recall, mAP, avg_time, param_count, timestamp)

    with open(os.path.join(save_folder, f"{method}_retrieval_results.json"), "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"[{method}] Evaluation complete. Metrics and logs saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image retrieval methods.")
    parser.add_argument("--query_folder", required=True)
    parser.add_argument("--database_folder", required=True)
    parser.add_argument("--save_folder", required=True)
    parser.add_argument("--method", choices=["dhash", "phash",
                                             "pretrained_resnet", "pretrained_mobilenet", "pretrained_efficientnet",
                                             "autoencoder_basic",  "autoencoder_better",
                                             "autoencoder_resnet", "autoencoder_mobilenet", "autoencoder_efficientnet",
                                             "siamese_basic", "siamese_better",
                                             "siamese_resnet", "siamese_mobilenet", "siamese_efficientnet",
                                             "siamese_autoencoder_basic", "siamese_autoencoder_better",
                                             "siamese_autoencoder_resnet", "siamese_autoencoder_mobilenet",
                                             "siamese_autoencoder_efficientnet"],
                        required=True)
    args = parser.parse_args()

    evaluate_retrieval(
        query_folder=args.query_folder,
        database_folder=args.database_folder,
        save_folder=args.save_folder,
        method=args.method
    )
