import argparse
import os
import faiss
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import pandas as pd
import time


def extract_features(image_path, model, transform, device):
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(image).squeeze().cpu().numpy()
        return features.flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def initialize_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    models_dict = {
        "resnet": models.resnet50(pretrained=True),
        "efficientnet": models.efficientnet_b0(pretrained=True),
        "mobilenet": models.mobilenet_v3_large(pretrained=True)
    }

    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(models_dict.keys())}")

    model = models_dict[model_name]
    model.eval()
    model = model.to(device)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    return model, transform


def compute_embeddings(base_folder, save_folder, model_name):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = initialize_model(model_name)

    image_paths = []
    embeddings = []

    for root, _, files in os.walk(base_folder):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):
                file_path = os.path.join(root, filename)
                features = extract_features(file_path, model, transform, device)
                if features is not None:
                    image_paths.append(file_path)
                    embeddings.append(features)

    if len(embeddings) == 0:
        print("No valid embeddings found. Exiting.")
        return

    embeddings = np.array(embeddings, dtype=np.float32)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index.add(embeddings)

    model_prefix = f"pretrained_{model_name}"
    index_file = os.path.join(save_folder, f"{model_prefix}_faiss.index")
    metadata_file = os.path.join(save_folder, f"{model_prefix}_metadata.csv")

    faiss.write_index(index, index_file)
    metadata_df = pd.DataFrame({"index": range(len(image_paths)), "image_path": image_paths})
    metadata_df.to_csv(metadata_file, index=False)

    print(f"Embeddings saved: {index_file}")
    print(f"Metadata saved: {metadata_file}")
    total_encoding_time = time.time() - start_time
    print(f"Total encoding time: {total_encoding_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute image embeddings for a dataset.")
    parser.add_argument("--base_folder", help="Base folder containing images", required=True)
    parser.add_argument("--save_folder", help="Where to save embeddings", required=True)
    parser.add_argument("--model", choices=["resnet", "efficientnet", "mobilenet"],
                        help="Model to use for feature extraction", required=True)

    args = parser.parse_args()

    compute_embeddings(args.base_folder, args.save_folder, args.model)
