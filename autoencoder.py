import argparse
import os
import shutil
import faiss
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt


def ensure_subfolder_exists(database_folder, default_class_folder):
    if not any(os.path.isdir(os.path.join(database_folder, d)) for d in os.listdir(database_folder)):
        os.makedirs(default_class_folder, exist_ok=True)
        for filename in os.listdir(database_folder):
            file_path = os.path.join(database_folder, filename)
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):
                shutil.move(file_path, os.path.join(default_class_folder, filename))


class Autoencoder(nn.Module):
    def __init__(self, embedding_dim=256, encoder_type="basic"):
        super(Autoencoder, self).__init__()
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim

        if encoder_type == "basic":
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128 * 28 * 28, embedding_dim)
            )
            encoder_output_size = embedding_dim

        else:
            if encoder_type == "resnet":
                model = models.resnet18(pretrained=True)
                model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
                encoder_output_size = model.fc.in_features

            elif encoder_type == "mobilenet":
                model = models.mobilenet_v2(pretrained=True)
                model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
                self.feature_extractor = model.features
                encoder_output_size = 1280

            elif encoder_type == "efficientnet":
                model = models.efficientnet_b0(pretrained=True)
                model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
                self.feature_extractor = model.features
                encoder_output_size = 1280

            else:
                raise ValueError(f"Unsupported encoder type: {encoder_type}")

            self.encoder = nn.Sequential(
                self.feature_extractor,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(encoder_output_size, embedding_dim)
            )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128 * 28 * 28),
            nn.ReLU(),
            nn.Unflatten(1, (128, 28, 28)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_autoencoder(database_folder, save_folder, num_epochs=20, batch_size=16, embedding_dim=256,
                      encoder_type="basic", early_stopping_patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    writer = SummaryWriter(log_dir=os.path.join(save_folder, f"autoencoder_{encoder_type}_tensorboard"))
    autoencoder = Autoencoder(embedding_dim, encoder_type).to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    dataset = datasets.ImageFolder(root=database_folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset loaded from '{database_folder}': {len(dataset)} images found.")

    best_loss = float("inf")
    patience_counter = 0
    model_name = f"autoencoder_{encoder_type}.pth"
    model_path = os.path.join(save_folder, model_name)

    for epoch in range(num_epochs):
        autoencoder.train()
        epoch_loss = 0
        sample_images = None
        reconstructed_images = None

        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            optimizer.zero_grad()
            encoded, decoded = autoencoder(images)
            loss = criterion(decoded, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx == 0:
                sample_images = images[:5]
                reconstructed_images = decoded[:5]

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch + 1)

        if (epoch + 1) % 5 == 0:
            save_reconstructions(sample_images, reconstructed_images, save_folder, epoch + 1, encoder_type)
            img_grid = torch.cat([sample_images.cpu(), reconstructed_images.cpu()], dim=0)
            writer.add_images("Reconstruction", img_grid, global_step=epoch + 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(autoencoder.state_dict(), model_path)
            print(f"Saved new best model at epoch {epoch + 1}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    writer.close()
    return autoencoder, transform, device



def save_reconstructions(originals, reconstructions, save_path, epoch, encoder_type):
    originals = originals.cpu().numpy()
    reconstructions = reconstructions.cpu().detach().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(12, 4))
    for i in range(5):
        axes[0, i].imshow(originals[i][0], cmap='gray')
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")

        axes[1, i].imshow(reconstructions[i][0], cmap='gray')
        axes[1, i].axis("off")
        axes[1, i].set_title("Reconstructed")

    plt.tight_layout()
    out_path = os.path.join(save_path, f"reconstruction_{encoder_type}_epoch_{epoch}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved reconstruction preview to: {out_path}")


def extract_embeddings(autoencoder, transform, device, database_folder, save_folder, embedding_dim=256, encoder_type="basic"):
    autoencoder.to(device)
    autoencoder.eval()
    image_paths = []
    embeddings = []

    for class_folder in os.listdir(database_folder):
        class_path = os.path.join(database_folder, class_folder)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                file_path = os.path.join(class_path, filename)
                last_dir = os.path.basename(class_path)
                img_path = os.path.join(last_dir, filename)
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):
                    try:
                        image = Image.open(file_path).convert("L")
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

    d = embedding_dim
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    index_file = os.path.join(save_folder, f"autoencoder_{encoder_type}_faiss.index")
    faiss.write_index(index, index_file)

    metadata_file = os.path.join(save_folder, f"autoencoder_{encoder_type}_metadata.csv")
    metadata_df = pd.DataFrame({"index": range(len(image_paths)), "image_path": image_paths})
    metadata_df.to_csv(metadata_file, index=False)

    print(f"Embeddings saved in {index_file}")
    print(f"Metadata saved in {metadata_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an autoencoder and extract image embeddings.")
    parser.add_argument("--base_folder", help="Folder containing images", required=True)
    parser.add_argument("--encoder_type", type=str, default="basic",
                        choices=["basic", "resnet", "mobilenet", "efficientnet"],
                        help="Type of encoder to use in the autoencoder")
    parser.add_argument("--save_folder", help="Folder to save model and embeddings", required=True)
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Size of embedding vector")
    parser.add_argument("--es_patience", type=int, default=5, help="Number of epochs to wait before early stopping")

    args = parser.parse_args()

    default_class_folder = os.path.join(args.base_folder, "unlabeled")
    ensure_subfolder_exists(args.base_folder, default_class_folder)

    autoencoder, transform, device = train_autoencoder(
        database_folder=args.base_folder,
        save_folder=args.save_folder,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        encoder_type=args.encoder_type,
        early_stopping_patience=args.es_patience
    )

    extract_embeddings(autoencoder,
                       transform,
                       device,
                       args.base_folder,
                       args.save_folder,
                       args.embedding_dim)
