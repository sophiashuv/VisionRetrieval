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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time


from autoencoder import Autoencoder, BetterEncoder
from tqdm import tqdm



class CSVPairSiameseDataset(Dataset):
    def __init__(self, folders, transform):
        self.transform = transform
        self.pair_data = []

        for folder in folders:
            csv_path = os.path.join(folder, "pairs.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"{csv_path} not found.")
            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                pair_id = row["pair_id"]
                anchor_path = os.path.join(folder, f"pair_{pair_id:05d}_anchor.jpg")
                pair_path = os.path.join(folder, f"pair_{pair_id:05d}_pair.jpg")
                label = int(row["label"])

                if os.path.exists(anchor_path) and os.path.exists(pair_path):
                    self.pair_data.append((anchor_path, pair_path, label))

        print(f"Loaded {len(self.pair_data)} image pairs from {len(folders)} folder(s).")

    def __len__(self):
        return len(self.pair_data)

    def __getitem__(self, index):
        anchor_path, pair_path, label = self.pair_data[index]

        anchor_img = Image.open(anchor_path).convert("RGB")
        pair_img = Image.open(pair_path).convert("RGB")

        return (
            self.transform(anchor_img),
            self.transform(pair_img),
            torch.tensor([label], dtype=torch.float32)
        )

class UnlabeledCLIPSiameseDataset(Dataset):
    def __init__(self, image_dir, transform, device):
        import open_clip
        from sklearn.metrics.pairwise import cosine_similarity
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.transform = transform
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model = self.model.to(device).eval()

        self.embeddings = self._compute_clip_embeddings()
        self.similarity_matrix = cosine_similarity(self.embeddings)

    def _compute_clip_embeddings(self):
        embeddings = []
        with torch.no_grad():
            for path in tqdm(self.image_paths, desc="Computing CLIP embeddings"):
                image = Image.open(path).convert("RGB")
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                emb = self.model.encode_image(image_tensor).squeeze().cpu().numpy()
                embeddings.append(emb)
        return np.array(embeddings)

    def __len__(self):
        return len(self.image_paths) * 2

    def __getitem__(self, index):
        real_idx = index // 2
        is_positive = index % 2 == 0

        anchor_path = self.image_paths[real_idx]
        similarities = self.similarity_matrix[real_idx]

        sorted_indices = np.argsort(-similarities)

        if is_positive:
            candidates = sorted_indices[1:6]  # skip self (0), take top 5
        else:
            candidates = sorted_indices[200:]  # low similarity

        if len(candidates) == 0:
            return self.__getitem__((index + 1) % len(self))  # fallback

        pair_idx = np.random.choice(candidates)
        pair_path = self.image_paths[pair_idx]
        label = 1 if is_positive else 0

        anchor_img = Image.open(anchor_path).convert("RGB")
        pair_img = Image.open(pair_path).convert("RGB")

        return self.transform(anchor_img), self.transform(pair_img), torch.tensor([label], dtype=torch.float32)


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
    def __init__(self, encoder, use_head=False, embedding_dim=256, encoder_output_dim=1280):
        super(SiameseNetwork, self).__init__()
        self.cnn = encoder
        self.head = None

        if use_head:
            self.head = nn.Sequential(
                nn.Linear(encoder_output_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
        self.encoder_output_dim = encoder_output_dim

    def forward_once(self, x):
        x = self.cnn(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        if self.head:
            x = self.head(x)
        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)


# class BetterEncoder(nn.Module):
#     def __init__(self, embedding_dim=256):
#         super(BetterEncoder, self).__init__()
#
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(256, embedding_dim)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.pool(x)  # Now x has shape [B, 256, 1, 1]
#         x = x.view(x.size(0), -1)  # Flatten to [B, 256]
#         x = self.fc(x)  # Output shape [B, embedding_dim]
#         return x


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
        autoencoder = Autoencoder(embedding_dim=embedding_dim, encoder_type=encoder).to(device)
        autoencoder.load_state_dict(torch.load(encoder_path, map_location=device))
        autoencoder.eval()
        return autoencoder.encoder

    elif encoder_type == "resnet":
        model = models.resnet50(pretrained=True)
        modules = list(model.children())[:-1]
        encoder = nn.Sequential(
            *modules,
            nn.Flatten(),
            # nn.Linear(model.fc.in_features, embedding_dim)
        )
        return encoder

    elif encoder_type == "mobilenet":
        model = models.mobilenet_v3_large(pretrained=True)

        encoder = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # nn.Linear(model.last_channel, embedding_dim)
        )
        return encoder

    elif encoder_type == "efficientnet":
        model = models.efficientnet_b0(pretrained=True)
        encoder = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # nn.Linear(model.classifier[1].in_features, embedding_dim)
        )
        return encoder

    elif encoder_type == "basic":
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, embedding_dim)
        )

    elif encoder_type == "better":
        return BetterEncoder(embedding_dim=embedding_dim)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def train_siamese_network(database_folders, save_folder, embedding_dim=256, num_epochs=20, batch_size=16,
                          learning_rate=0.0001, encoder_type="basic", encoder_path=None,
                          early_stopping_patience=5, use_clip_loader=False, use_pair_csv_loader=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if use_pair_csv_loader:
        full_dataset = CSVPairSiameseDataset(folders=database_folders, transform=train_transform)
    elif use_clip_loader:
        full_dataset = UnlabeledCLIPSiameseDataset(
            image_dir=database_folders[0], transform=train_transform, device=device)
    else:
        datasets_list = [datasets.ImageFolder(root=folder) for folder in database_folders]
        concat_dataset = ConcatDataset(datasets_list)
        full_dataset = SiameseDataset(concat_dataset, transform=train_transform)

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    if hasattr(val_dataset, 'dataset'):
        val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    encoder = build_encoder(encoder_type, embedding_dim, encoder_path, device)

    if encoder_type == "mobilenet":
        encoder_output_dim = 1280
    elif encoder_type == "resnet":
        encoder_output_dim = 512
    elif encoder_type == "efficientnet":
        encoder_output_dim = 1280
    else:
        encoder_output_dim = embedding_dim

    use_head = encoder_type in ["basic", "better"]
    model = SiameseNetwork(encoder=encoder, embedding_dim=embedding_dim, use_head=use_head,
                           encoder_output_dim=encoder_output_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    log_dir = os.path.join(save_folder, f"siamese_tensorboard_{encoder_type}")
    writer = SummaryWriter(log_dir=log_dir)

    best_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(save_folder, f"siamese_{encoder_type}.pth")
    training_start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = ContrastiveLoss()(output1, output2, label)
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
                loss = ContrastiveLoss()(output1, output2, label)
                val_loss += loss.item()

                distances = torch.nn.functional.pairwise_distance(output1, output2)
                preds = (distances < 0.5).float()
                correct += (preds == label.view(-1)).sum().item()
                total += label.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.4f}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
        writer.add_scalar("Accuracy/val", accuracy, epoch + 1)
        writer.add_scalar("LearningRate", current_lr, epoch + 1)

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

    total_training_time = time.time() - training_start_time
    print(f"Total training time: {total_training_time:.2f} seconds")
    writer.add_scalar("Time/total_training_time_sec", total_training_time)
    writer.close()

    return model, val_transform, device


def extract_embeddings(model, transform, device, database_folders, save_folder, embedding_dim=256, encoder_type="basic"):
    start_time = time.time()
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
                            model.eval()
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

    index = faiss.IndexFlatIP(embeddings.shape[1])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(save_folder, f"siamese_{encoder_type}_faiss.index"))
    pd.DataFrame({"index": range(len(image_paths)), "image_path": image_paths}).to_csv(
        os.path.join(save_folder, f"siamese_{encoder_type}_metadata.csv"), index=False
    )
    print(f"Saved FAISS index and metadata to {save_folder}")
    total_encoding_time = time.time() - start_time
    print(f"Total encoding time: {total_encoding_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siamese Network Training and Embedding Extraction")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the Siamese network")
    train_parser.add_argument("--base_folders", nargs='+', required=True)
    train_parser.add_argument("--save_folder", required=True)
    train_parser.add_argument("--encoder_type", type=str, default="basic", choices=[
        "basic", "resnet", "mobilenet", "efficientnet",
        "autoencoder_basic", "autoencoder_better", "autoencoder_resnet", "autoencoder_mobilenet",
        "autoencoder_efficientnet",
        "better"
    ])
    train_parser.add_argument("--encoder_path", type=str, default=None)
    train_parser.add_argument("--num_epochs", type=int, default=100)
    train_parser.add_argument("--batch_size", type=int, default=16)
    train_parser.add_argument("--learning_rate", type=float, default=0.0005)
    train_parser.add_argument("--embedding_dim", type=int, default=256)
    train_parser.add_argument("--es_patience", type=int, default=5)
    train_parser.add_argument("--use_clip_loader", action="store_true")
    train_parser.add_argument("--use_pair_csv_loader", action="store_true")
    train_parser.add_argument("--test_folder", nargs='+', help="Optional: extract embeddings on test data")

    # Extract subcommand
    extract_parser = subparsers.add_parser("extract", help="Extract embeddings with a pretrained model")
    extract_parser.add_argument("--base_folders", nargs='+', required=True)
    extract_parser.add_argument("--save_folder", required=True)
    extract_parser.add_argument("--encoder_type", type=str, required=True)
    extract_parser.add_argument("--encoder_path", required=True)
    extract_parser.add_argument("--embedding_dim", type=int, default=256)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        model, transform, device = train_siamese_network(
            database_folders=args.base_folders,
            save_folder=args.save_folder,
            embedding_dim=args.embedding_dim,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            encoder_type=args.encoder_type,
            encoder_path=args.encoder_path,
            early_stopping_patience=args.es_patience,
            use_clip_loader=args.use_clip_loader,
            use_pair_csv_loader=args.use_pair_csv_loader
        )

        if args.test_folder:
            extract_embeddings(
                model=model,
                transform=transform,
                device=device,
                database_folders=args.test_folder,
                save_folder=args.save_folder,
                embedding_dim=args.embedding_dim,
                encoder_type=args.encoder_type
            )
        else:
            extract_embeddings(
                model=model,
                transform=transform,
                device=device,
                database_folders=args.base_folders,
                save_folder=args.save_folder,
                embedding_dim=args.embedding_dim,
                encoder_type=args.encoder_type
            )

    elif args.mode == "extract":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        encoder = build_encoder(args.encoder_type, args.embedding_dim, args.encoder_path, device)
        model = SiameseNetwork(encoder=encoder, embedding_dim=args.embedding_dim).to(device)
        model.load_state_dict(torch.load(args.encoder_path, map_location=device))

        extract_embeddings(
            model=model,
            transform=transform,
            device=device,
            database_folders=args.base_folders,
            save_folder=args.save_folder,
            embedding_dim=args.embedding_dim,
            encoder_type=args.encoder_type
        )
