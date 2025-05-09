import argparse
import os
import shutil
import faiss
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time


def ensure_subfolder_exists(folders):
    for database_folder in folders:
        default_class_folder = os.path.join(database_folder, "unlabeled")
        if not any(os.path.isdir(os.path.join(database_folder, d)) for d in os.listdir(database_folder)):
            os.makedirs(default_class_folder, exist_ok=True)
            for filename in os.listdir(database_folder):
                file_path = os.path.join(database_folder, filename)
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):
                    shutil.move(file_path, os.path.join(default_class_folder, filename))


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

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, embedding_dim=256, encoder_type="basic", freeze_encoder_epochs=5):
        super(Autoencoder, self).__init__()
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.encoder_frozen = False

        if encoder_type == "basic":
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128 * 28 * 28, embedding_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, 128 * 28 * 28),
                nn.ReLU(),
                nn.Unflatten(1, (128, 28, 28)),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )

        elif encoder_type == "better":
            self.encoder = BetterEncoder(embedding_dim=embedding_dim)
            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, 256 * 28 * 28),
                nn.ReLU(),
                nn.Unflatten(1, (256, 28, 28)),
                nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1),
                nn.Sigmoid()
            )

        elif encoder_type in ["resnet", "mobilenet", "efficientnet"]:
            if encoder_type == "resnet":
                model = models.resnet18(pretrained=True)
                model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
                encoder_output_size = model.fc.in_features

            elif encoder_type == "mobilenet":
                model = models.mobilenet_v2(pretrained=True)
                model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
                self.feature_extractor = model.features
                encoder_output_size = 1280

            elif encoder_type == "efficientnet":
                model = models.efficientnet_b0(pretrained=True)
                model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
                self.feature_extractor = model.features
                encoder_output_size = 1280

            self.encoder = nn.Sequential(
                self.feature_extractor,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )

            self.decoder = nn.Sequential(
                nn.Linear(encoder_output_size, 256 * 28 * 28),
                nn.ReLU(),
                nn.Unflatten(1, (256, 28, 28)),
                nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1),
                nn.Sigmoid()
            )

        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

    def maybe_freeze_encoder(self, current_epoch):
        if self.encoder_type in ["resnet", "mobilenet", "efficientnet"]:
            if current_epoch < self.freeze_encoder_epochs:
                if not self.encoder_frozen:
                    for param in self.encoder.parameters():
                        param.requires_grad = False
                    self.encoder_frozen = True
                    print(f"Encoder frozen (epoch {current_epoch})")
            elif self.encoder_frozen:
                for param in self.encoder.parameters():
                    param.requires_grad = True
                self.encoder_frozen = False
                print(f"Encoder unfrozen at epoch {current_epoch}")

    def forward(self, x, current_epoch=None):
        if current_epoch is not None:
            self.maybe_freeze_encoder(current_epoch)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_autoencoder(database_folders, save_folder, num_epochs=20, batch_size=16, embedding_dim=256,
                      encoder_type="basic", early_stopping_patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    writer = SummaryWriter(log_dir=os.path.join(save_folder, f"autoencoder_{encoder_type}_tensorboard"))
    autoencoder = Autoencoder(embedding_dim, encoder_type).to(device)
    print(f"Total trainable parameters: {sum(p.numel() for p in autoencoder.parameters() if p.requires_grad):,}")
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    datasets_list = [datasets.ImageFolder(root=folder, transform=transform) for folder in database_folders]
    dataset = ConcatDataset(datasets_list)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset loaded from {', '.join(database_folders)}: {len(dataset)} images found.")

    best_loss = float("inf")
    patience_counter = 0
    model_name = f"autoencoder_{encoder_type}.pth"
    model_path = os.path.join(save_folder, model_name)
    training_start_time = time.time()
    for epoch in range(num_epochs):
        autoencoder.train()
        train_loss = 0

        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            _, decoded = autoencoder(images, current_epoch=epoch)
            loss = criterion(decoded, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        autoencoder.eval()
        val_loss = 0
        correct_pixels = 0
        total_pixels = 0

        with torch.no_grad():
            for val_images, _ in val_loader:
                val_images = val_images.to(device)
                _, decoded = autoencoder(val_images, current_epoch=epoch)
                loss = criterion(decoded, val_images)
                val_loss += loss.item()

                pred = decoded > 0.5
                truth = val_images > 0.5
                correct_pixels += torch.sum(pred == truth).item()
                total_pixels += pred.numel()

        avg_val_loss = val_loss / len(val_loader)
        pixel_accuracy = correct_pixels / total_pixels

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Pixel Acc: {pixel_accuracy:.4f}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
        writer.add_scalar("Accuracy/val", pixel_accuracy, epoch + 1)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("LearningRate", current_lr, epoch + 1)
        print(f"Learning rate: {current_lr:.6f}")

        if (epoch + 1) % 5 == 0:
            save_reconstructions(val_images[:5], decoded[:5], save_folder, epoch + 1, encoder_type)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(autoencoder.state_dict(), model_path)
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
    return autoencoder, transform, device


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # In-place denormalization for visualization
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def save_reconstructions(originals, reconstructions, save_path, epoch, encoder_type):
    n = min(originals.shape[0], 5)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 4))

    for i in range(n):
        orig_tensor = denormalize(originals[i].cpu().clone())
        recon_tensor = denormalize(reconstructions[i].cpu().clone())

        axes[0, i].imshow(np.transpose(orig_tensor.numpy(), (1, 2, 0)))
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")

        axes[1, i].imshow(np.transpose(recon_tensor.numpy(), (1, 2, 0)))
        axes[1, i].axis("off")
        axes[1, i].set_title("Reconstructed")

    plt.tight_layout()
    out_path = os.path.join(save_path, f"reconstruction_{encoder_type}_epoch_{epoch}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved reconstruction preview to: {out_path}")

def extract_embeddings(autoencoder, transform, device, database_folders, save_folder, embedding_dim=256, encoder_type="basic"):
    start_time = time.time()

    autoencoder.to(device)
    autoencoder.eval()
    image_paths = []
    embeddings = []

    for database_folder in database_folders:
        for class_folder in os.listdir(database_folder):
            class_path = os.path.join(database_folder, class_folder)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    file_path = os.path.join(class_path, filename)
                    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):
                        try:
                            image = Image.open(file_path).convert("RGB")
                            image = transform(image).unsqueeze(0).to(device)
                            with torch.no_grad():
                                features, _ = autoencoder(image)
                            image_paths.append(file_path)
                            embeddings.append(features.squeeze().cpu().numpy())
                        except Exception as e:
                            print(f"Error processing {filename}: {e}")

    if len(embeddings) == 0:
        print("No valid embeddings found. Exiting.")
        return

    embeddings = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index.add(embeddings)

    index_file = os.path.join(save_folder, f"autoencoder_{encoder_type}_faiss.index")
    faiss.write_index(index, index_file)

    metadata_file = os.path.join(save_folder, f"autoencoder_{encoder_type}_metadata.csv")
    metadata_df = pd.DataFrame({"index": range(len(image_paths)), "image_path": image_paths})
    metadata_df.to_csv(metadata_file, index=False)

    print(f"Embeddings saved in {index_file}")
    print(f"Metadata saved in {metadata_file}")
    total_encoding_time = time.time() - start_time
    print(f"Total encoding time: {total_encoding_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an autoencoder and extract image embeddings.")
    parser.add_argument("--base_folders", nargs='+', help="List of folders containing images", required=True)
    parser.add_argument("--encoder_type", type=str, default="basic",
                        choices=["basic", "resnet", "mobilenet", "efficientnet", "better"],
                        help="Type of encoder to use in the autoencoder")
    parser.add_argument("--save_folder", help="Folder to save model and embeddings", required=True)
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Size of embedding vector")
    parser.add_argument("--es_patience", type=int, default=5, help="Number of epochs to wait before early stopping")
    parser.add_argument("--test_folder", nargs='+', help="Path to test images.")
    args = parser.parse_args()

    ensure_subfolder_exists(args.base_folders)

    autoencoder, transform, device = train_autoencoder(
        database_folders=args.base_folders,
        save_folder=args.save_folder,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        encoder_type=args.encoder_type,
        early_stopping_patience=args.es_patience
    )
    if args.test_folder:
        extract_embeddings(autoencoder,
                           transform,
                           device,
                           args.test_folder,
                           args.save_folder,
                           args.embedding_dim,
                           args.encoder_type)
    else:
        extract_embeddings(autoencoder,
                           transform,
                           device,
                           args.base_folders,
                           args.save_folder,
                           args.embedding_dim,
                           args.encoder_type)
