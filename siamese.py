import argparse
import os
import random
import torch
import torch.optim as optim
import faiss
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from autoencoder import Autoencoder


class SiameseDataset(Dataset):
    def __init__(self, dataset, transform):
        self.transform = transform
        self.class_to_imgs = {}
        self.samples = []

        if isinstance(dataset, ConcatDataset):
            for sub_dataset in dataset.datasets:
                self.samples.extend(sub_dataset.samples)
                for path, target in sub_dataset.samples:
                    class_name = sub_dataset.classes[target]
                    self.class_to_imgs.setdefault(class_name, []).append(path)
        else:
            self.samples = dataset.samples
            for path, target in dataset.samples:
                class_name = dataset.classes[target]
                self.class_to_imgs.setdefault(class_name, []).append(path)

        self.class_names = list(self.class_to_imgs.keys())
        self.dataset_len = len(self.samples) * 2

    def _group_by_class(self):
        class_to_imgs = {}

        if isinstance(self.image_folder_dataset, ConcatDataset):
            for sub_dataset in self.image_folder_dataset.datasets:
                for path, target in sub_dataset.samples:
                    class_name = sub_dataset.classes[target]
                    class_to_imgs.setdefault(class_name, []).append(path)
        else:
            for path, target in self.image_folder_dataset.samples:
                class_name = self.image_folder_dataset.classes[target]
                class_to_imgs.setdefault(class_name, []).append(path)

        return class_to_imgs

    def __getitem__(self, index):
        real_index = index // 2
        is_positive = index % 2 == 0

        anchor_path, anchor_label = self.samples[real_index]
        anchor_class = os.path.basename(os.path.dirname(anchor_path))

        if is_positive:
            positive_candidates = [p for p in self.class_to_imgs[anchor_class] if p != anchor_path]
            if not positive_candidates:
                return self.__getitem__((index + 1) % self.__len__())
            positive_path = random.choice(positive_candidates)
            label = 1
        else:
            other_classes = [cls for cls in self.class_to_imgs if cls != anchor_class]
            negative_class = random.choice(other_classes)
            positive_path = random.choice(self.class_to_imgs[negative_class])
            label = 0

        anchor_image = Image.open(anchor_path).convert("RGB")
        positive_image = Image.open(positive_path).convert("RGB")

        return (
            self.transform(anchor_image),
            self.transform(positive_image),
            torch.tensor([label], dtype=torch.float32)
        )

    def __len__(self):
        return self.dataset_len


class SiameseNetwork(nn.Module):
    def __init__(self, encoder=None, embedding_dim=256):
        super(SiameseNetwork, self).__init__()
        if encoder is not None:
            self.cnn = encoder
        else:
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(128 * 28 * 28, embedding_dim)
            )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        x = self.head(x)
        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)


class BetterEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super(BetterEncoder, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distances = torch.nn.functional.pairwise_distance(output1, output2, keepdim=True)
        loss = 0.5 * (label * distances.pow(2) + (1 - label) * torch.clamp(self.margin - distances, min=0.0).pow(2))
        return loss.mean()


def build_encoder(encoder_type, embedding_dim, encoder_path, device):
    if "autoencoder" in encoder_type:
        _, encoder = encoder_type.split("_")
        autoencoder = Autoencoder(embedding_dim=embedding_dim,
                                  encoder_type=encoder).to(device)
        autoencoder.load_state_dict(torch.load(encoder_path, map_location=device))
        autoencoder.eval()
        return autoencoder.encoder
    elif encoder_type == "resnet":
        model = models.resnet18(pretrained=True)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        encoder = torch.nn.Sequential(*list(model.children())[:-1])
        return torch.nn.Sequential(
            encoder,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(model.fc.in_features, embedding_dim)
        )

    elif encoder_type == "mobilenet":
        model = models.mobilenet_v2(pretrained=True)
        model.features[0][0] = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        encoder = model.features
        return torch.nn.Sequential(
            encoder,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(1280, embedding_dim)
        )

    elif encoder_type == "efficientnet":
        model = models.efficientnet_b0(pretrained=True)
        model.features[0][0] = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        encoder = model.features
        return torch.nn.Sequential(
            encoder,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(1280, embedding_dim)
        )

    elif encoder_type == "basic":
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 28 * 28, embedding_dim)
        )
    elif encoder_type == "better":
        return BetterEncoder(embedding_dim=embedding_dim)

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def train_siamese_network(database_folders, save_folder, embedding_dim=256, num_epochs=20, batch_size=16,
                          learning_rate=0.001, encoder_type="basic", encoder_path=None,
                          early_stopping_patience=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    datasets_list = [datasets.ImageFolder(root=folder) for folder in database_folders]
    concat_dataset = ConcatDataset(datasets_list)
    siamese_dataset = SiameseDataset(concat_dataset, transform)

    print(f"Dataset loaded from {', '.join(database_folders)}: {len(siamese_dataset.samples)} images found.")

    val_size = int(0.2 * len(siamese_dataset))
    train_size = len(siamese_dataset) - val_size
    train_dataset, val_dataset = random_split(siamese_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    encoder = build_encoder(encoder_type, embedding_dim, encoder_path, device)
    print(f"Using encoder: {encoder_type}")

    model = SiameseNetwork(encoder=encoder, embedding_dim=embedding_dim).to(device)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    log_dir = os.path.join(save_folder, f"siamese_tensorboard_{encoder_type}")
    writer = SummaryWriter(log_dir=log_dir)

    best_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(save_folder, f"siamese_{encoder_type}.pth")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, label)
                val_loss += loss.item()

                distances = torch.nn.functional.pairwise_distance(output1, output2)
                preds = (distances < 0.5).float()
                correct += (preds == label.view(-1)).sum().item()
                total += label.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.4f}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
        writer.add_scalar("Accuracy/val", accuracy, epoch + 1)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model at epoch {epoch + 1}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    writer.close()
    return model, transform, device


def extract_embeddings(model, transform, device, database_folders, save_folder, embedding_dim=256, encoder_type="basic"):
    model.eval()
    embeddings = []
    image_paths = []

    for base_folder in database_folders:
        for class_folder in os.listdir(base_folder):
            class_path = os.path.join(base_folder, class_folder)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    file_path = os.path.join(class_path, filename)
                    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):
                        try:
                            image = Image.open(file_path).convert("RGB")
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

    faiss.write_index(index, os.path.join(save_folder, f"siamese_{encoder_type}_faiss.index"))
    pd.DataFrame({"index": range(len(image_paths)), "image_path": image_paths}).to_csv(
        os.path.join(save_folder, f"siamese_{encoder_type}_metadata.csv"), index=False
    )
    print(f"Saved FAISS index and metadata to {save_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Siamese network and extract embeddings.")
    parser.add_argument("--base_folders", nargs='+', required=True, help="One or more dataset base folders")
    parser.add_argument("--save_folder", required=True, help="Folder to save model and embeddings")
    parser.add_argument("--encoder_type", type=str, default="basic",
                        choices=["basic", "resnet", "mobilenet", "efficientnet", "autoencoder_basic", "autoencoder_better"
                                 "autoencoder_resnet", "autoencoder_mobilenet", "autoencoder_efficientnet", "better"],
                        help="Type of encoder to use")
    parser.add_argument("--encoder_path", type=str, default=None,
                        help="Path to pretrained encoder.pth weights")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--es_patience", type=int, default=5, help="Number of epochs to wait before early stopping")
    args = parser.parse_args()

    model, transform, device = train_siamese_network(
        database_folders=args.base_folders,
        save_folder=args.save_folder,
        embedding_dim=args.embedding_dim,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        encoder_type=args.encoder_type,
        encoder_path=args.encoder_path,
        early_stopping_patience=args.es_patience
    )

    extract_embeddings(
        model=model,
        transform=transform,
        device=device,
        database_folders=args.base_folders,
        save_folder=args.save_folder,
        embedding_dim=args.embedding_dim,
        encoder_type=args.encoder_type
    )

