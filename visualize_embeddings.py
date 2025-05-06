# visualize_embeddings.py

import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from autoencoder import Autoencoder
from siamese import SiameseNetwork, build_encoder
from hash import dhash, phash, hash_to_bitvector
from pretrained import extract_features, initialize_model

SUPPORTED_METHODS = [
"siamese_better"
    # "dhash", "phash",
    # "pretrained_resnet", "pretrained_mobilenet", "pretrained_efficientnet",
    # "autoencoder_basic", "autoencoder_resnet", "autoencoder_mobilenet", "autoencoder_efficientnet",
    # "siamese_basic", "siamese_resnet", "siamese_mobilenet", "siamese_efficientnet",
    # "siamese_autoencoder_basic",  "siamese_autoencoder_better", "siamese_autoencoder_resnet",
    # "siamese_autoencoder_mobilenet", "siamese_autoencoder_efficientnet", "siamese_better"
]


def load_model(method, database_folder, embedding_dim, device):
    model_path = os.path.join(database_folder, f"{method}.pth")

    if "siamese" in method:
        encoder_type = method.split("_")[-1]
        encoder = build_encoder(encoder_type, embedding_dim, None, device)
        if not os.path.exists(model_path):
            return None, None
        model = SiameseNetwork(encoder=encoder, embedding_dim=embedding_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    elif "autoencoder" in method:
        # Split to get encoder type (e.g., autoencoder_resnet)
        encoder_type = "_".join(method.split("_")[1:])
        if not os.path.exists(model_path):
            return None, None
        model = Autoencoder(embedding_dim=embedding_dim, encoder_type=encoder_type).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    elif method in ["dhash", "phash"]:
        model = method
        transform = None

    elif "pretrained" in method:
        model, transform = initialize_model(method.split("_")[-1])
        model.eval()

    else:
        return None, None

    return model, transform


def get_embedding(file_path, method, model, transform, device):
    if "siamese" in method:
        image = Image.open(file_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.forward_once(image_tensor)
        return emb.squeeze().cpu().numpy()
    elif "autoencoder" in method:
        image = Image.open(file_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb, _ = model(image_tensor)
        return emb.squeeze().cpu().numpy()
    elif method in ["dhash", "phash"]:
        import cv2
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        hash_func = dhash if method == "dhash" else phash
        return hash_to_bitvector(hash_func(image))
    elif "pretrained" in method:
        return extract_features(file_path, model, transform, device)
    else:
        raise ValueError("Unsupported method.")


def collect_embeddings(query_folder, model, method, transform, device):
    embeddings, labels = [], []
    for subfolder in sorted(os.listdir(query_folder)):
        subfolder_path = os.path.join(query_folder, subfolder)
        if not os.path.isdir(subfolder_path): continue

        for filename in sorted(os.listdir(subfolder_path)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):
                continue
            file_path = os.path.join(subfolder_path, filename)
            try:
                emb = get_embedding(file_path, method, model, transform, device)
                embeddings.append(emb)
                labels.append(subfolder)
            except Exception as e:
                print(f"[{method}] Error processing {file_path}: {e}")
    return np.array(embeddings), labels


def plot_pca_3d(embeddings, labels, method_name, save_folder):
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(embeddings)

    label_set = sorted(set(labels))
    label_to_idx = {label: i for i, label in enumerate(label_set)}
    colors = [label_to_idx[label] for label in labels]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=colors, cmap='tab10', alpha=0.7)

    # Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=plt.cm.tab10(i / len(label_set)), markersize=10)
               for i, label in enumerate(label_set)]
    ax.legend(handles=handles, title="Labels")

    ax.set_title(f"3D PCA Visualization ({method_name})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{method_name}_pca_3d.png")
    plt.savefig(save_path)
    print(f"[{method_name}] PCA plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize embeddings for all available methods with PCA.")
    parser.add_argument("--query_folder", required=True)
    parser.add_argument("--database_folder", required=True)
    parser.add_argument("--save_folder", required=True)
    parser.add_argument("--embedding_dim", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for method in SUPPORTED_METHODS:
        print(f"\n[{method}] Processing...")
        model, transform = load_model(method, args.database_folder, args.embedding_dim, device)
        if model is None and method not in ["dhash", "phash"]:
            print(f"[{method}] Model not found or unsupported. Skipping.")
            continue

        try:
            embeddings, labels = collect_embeddings(args.query_folder, model, method, transform, device)
            if len(embeddings) == 0:
                print(f"[{method}] No embeddings extracted. Skipping.")
                continue
            plot_pca_3d(embeddings, labels, method, args.save_folder)
        except Exception as e:
            print(f"[{method}] Failed with error: {e}")


if __name__ == "__main__":
    main()
