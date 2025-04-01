import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import faiss
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class SiameseDataset(Dataset):
    def __init__(self, image_folder_dataset, transform):
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
        self.class_to_imgs = self._group_by_class()

    def _group_by_class(self):
        class_to_imgs = {}
        for path, target in self.image_folder_dataset.samples:
            class_name = self.image_folder_dataset.classes[target]
            class_to_imgs.setdefault(class_name, []).append(path)
        return class_to_imgs

    def __getitem__(self, index):
        anchor_path, anchor_label = self.image_folder_dataset.samples[index]
        anchor_class = self.image_folder_dataset.classes[anchor_label]
        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            positive_path = random.choice(self.class_to_imgs[anchor_class])
            label = 1
        else:
            other_classes = [cls for cls in self.class_to_imgs if cls != anchor_class]
            negative_class = random.choice(other_classes)
            positive_path = random.choice(self.class_to_imgs[negative_class])
            label = 0

        anchor_image = Image.open(anchor_path).convert("L")
        positive_image = Image.open(positive_path).convert("L")

        return (
            self.transform(anchor_image),
            self.transform(positive_image),
            torch.tensor([label], dtype=torch.float32)
        )

    def __len__(self):
        return len(self.image_folder_dataset)


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1),  # -> [32, 14, 14]
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> [64, 7, 7]
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),  # -> [128, 4, 4]
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128 * 28 * 28, embedding_dim)
        )

    def forward_once(self, x):
        return self.cnn(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distances = torch.nn.functional.pairwise_distance(output1, output2, keepdim=True)
        loss = 0.5 * (label * distances.pow(2) + (1 - label) * torch.clamp(self.margin - distances, min=0.0).pow(2))
        return loss.mean()


def train_siamese_network(database_folder, save_folder, embedding_dim=256, num_epochs=20, batch_size=16, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image_folder_dataset = datasets.ImageFolder(root=database_folder)
    siamese_dataset = SiameseDataset(image_folder_dataset, transform)
    dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=batch_size)

    model = SiameseNetwork(embedding_dim).to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        total_loss = 0
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

    os.makedirs(save_folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_folder, "siamese_model.pth"))
    return model, transform, device


def extract_embeddings(model, transform, device, database_folder, save_folder, embedding_dim=256):
    model.eval()
    embeddings = []
    image_paths = []

    for class_folder in os.listdir(database_folder):
        class_path = os.path.join(database_folder, class_folder)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                file_path = os.path.join(class_path, filename)
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):
                    try:
                        image = Image.open(file_path).convert("L")
                        image_tensor = transform(image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            emb = model.forward_once(image_tensor)
                        embeddings.append(emb.squeeze().cpu().numpy())
                        image_paths.append(file_path)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    if len(embeddings) == 0:
        print("No valid embeddings found. Exiting.")
        return

    embeddings = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, os.path.join(save_folder, "siamese_faiss.index"))
    pd.DataFrame({"index": range(len(image_paths)), "image_path": image_paths}).to_csv(
        os.path.join(save_folder, "siamese_metadata.csv"), index=False
    )
    print(f"Saved FAISS index and metadata to {save_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Siamese network and extract embeddings.")
    parser.add_argument("--base_folder", required=True, help="Dataset base folder")
    parser.add_argument("--save_folder", required=True, help="Folder to save model and embeddings")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--embedding_dim", type=int, default=256)
    args = parser.parse_args()

    model, transform, device = train_siamese_network(
        database_folder=args.base_folder,
        save_folder=args.save_folder,
        embedding_dim = args.embedding_dim,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    extract_embeddings(
        model=model,
        transform=transform,
        device=device,
        database_folder=args.base_folder,
        save_folder=args.save_folder,
        embedding_dim=args.embedding_dim
    )
