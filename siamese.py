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
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from autoencoder import Autoencoder


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
    def __init__(self, encoder=None, embedding_dim=256):
        super(SiameseNetwork, self).__init__()
        if encoder is not None:
            self.cnn = encoder
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
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


def build_encoder(encoder_type, embedding_dim, encoder_path, device):
    if encoder_path:
        autoencoder = Autoencoder(embedding_dim=embedding_dim, encoder_type=encoder_type if encoder_type != "autoencoder" else "basic").to(device)
        autoencoder.load_state_dict(torch.load(encoder_path, map_location=device))
        autoencoder.eval()
        return autoencoder.encoder

    elif encoder_type == "resnet":
        model = models.resnet18(pretrained=True)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        encoder = torch.nn.Sequential(*list(model.children())[:-1])
        return torch.nn.Sequential(
            encoder,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(model.fc.in_features, embedding_dim)
        )

    elif encoder_type == "mobilenet":
        model = models.mobilenet_v2(pretrained=True)
        model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        encoder = model.features
        return torch.nn.Sequential(
            encoder,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(1280, embedding_dim)
        )

    elif encoder_type == "efficientnet":
        model = models.efficientnet_b0(pretrained=True)
        model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        encoder = model.features
        return torch.nn.Sequential(
            encoder,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(1280, embedding_dim)
        )

    elif encoder_type == "basic":
        return torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 28 * 28, embedding_dim)
        )

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def get_model_suffix(encoder_type, encoder_path):
    if encoder_type == "autoencoder" or encoder_path:
        model_name = os.path.splitext(os.path.basename(encoder_path))[0]
        return f"{encoder_type}_{model_name}"
    return encoder_type


def train_siamese_network(database_folder, save_folder, embedding_dim=256, num_epochs=20, batch_size=16,
                          learning_rate=0.001, encoder_type="basic", encoder_path=None, model_suffix="basic",
                          early_stopping_patience=5):

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

    encoder = build_encoder(encoder_type, embedding_dim, encoder_path, device)
    print(f"Using encoder: {encoder_type}")

    model = SiameseNetwork(encoder=encoder, embedding_dim=embedding_dim).to(device)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    log_dir = os.path.join(save_folder, f"siamese_tensorboard_{model_suffix}")
    writer = SummaryWriter(log_dir=log_dir)

    best_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(save_folder, f"siamese_model_{model_suffix}.pth")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch + 1)

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model at epoch {epoch+1}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    writer.close()
    return model, transform, device


def extract_embeddings(model, transform, device, database_folder, save_folder, embedding_dim=256, model_suffix="basic"):
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

    faiss.write_index(index, os.path.join(save_folder, f"siamese_faiss_{model_suffix}.index"))
    pd.DataFrame({"index": range(len(image_paths)), "image_path": image_paths}).to_csv(
        os.path.join(save_folder, f"siamese_metadata_{model_suffix}.csv"), index=False
    )
    print(f"Saved FAISS index and metadata to {save_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Siamese network and extract embeddings.")
    parser.add_argument("--base_folder", required=True, help="Dataset base folder")
    parser.add_argument("--save_folder", required=True, help="Folder to save model and embeddings")
    parser.add_argument("--encoder_type", type=str, default="basic",
                        choices=["basic", "autoencoder", "resnet", "mobilenet", "efficientnet"],
                        help="Type of encoder to use")
    parser.add_argument("--encoder_path", type=str, default=None,
                        help="Path to pretrained encoder.pth weights")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--es_patience", type=int, default=5, help="Number of epochs to wait before early stopping")
    args = parser.parse_args()

    model_suffix = get_model_suffix(args.encoder_type, args.encoder_path)

    model, transform, device = train_siamese_network(
        database_folder=args.base_folder,
        save_folder=args.save_folder,
        embedding_dim=args.embedding_dim,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        encoder_type=args.encoder_type,
        encoder_path=args.encoder_path,
        model_suffix=model_suffix,
        early_stopping_patience=args.es_patience
    )

    extract_embeddings(
        model=model,
        transform=transform,
        device=device,
        database_folder=args.base_folder,
        save_folder=args.save_folder,
        embedding_dim=args.embedding_dim,
        model_suffix=model_suffix
    )

