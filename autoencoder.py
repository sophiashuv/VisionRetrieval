import argparse
import os
import shutil
import faiss
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt


def ensure_subfolder_exists(database_folder, default_class_folder):
    """Ensures that images are inside a labeled folder structure."""
    if not any(os.path.isdir(os.path.join(database_folder, d)) for d in os.listdir(database_folder)):
        os.makedirs(default_class_folder, exist_ok=True)
        for filename in os.listdir(database_folder):
            file_path = os.path.join(database_folder, filename)
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):
                shutil.move(file_path, os.path.join(default_class_folder, filename))


class Autoencoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super(Autoencoder, self).__init__()
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




def train_autoencoder(database_folder, save_folder, num_epochs=20, batch_size=16, embedding_dim=256):
    """Trains the autoencoder and saves the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    autoencoder = Autoencoder(embedding_dim).to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    dataset = datasets.ImageFolder(root=database_folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
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

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

        if (epoch + 1) % 5 == 0:
            save_reconstructions(sample_images, reconstructed_images, save_folder, epoch + 1)

    torch.save(autoencoder.state_dict(), os.path.join(save_folder, "autoencoder.pth"))
    return autoencoder, transform, device

def save_reconstructions(originals, reconstructions, save_path, epoch):
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
    out_path = os.path.join(save_path, f"reconstruction_epoch_{epoch}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved reconstruction preview to: {out_path}")

def extract_embeddings(autoencoder, transform, device, database_folder, save_folder, embedding_dim=256):
    """Extracts image embeddings and stores them in a FAISS index."""
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

    # Convert list to NumPy array
    embeddings = np.array(embeddings, dtype=np.float32)

    # Create FAISS index
    d = embedding_dim
    index = faiss.IndexFlatL2(d)  # L2-based FAISS search
    index.add(embeddings)

    # Save FAISS index
    index_file = os.path.join(save_folder, "autoencoder_faiss.index")
    faiss.write_index(index, index_file)

    # Save metadata
    metadata_df = pd.DataFrame({"index": range(len(image_paths)), "image_path": image_paths})
    metadata_df.to_csv(os.path.join(save_folder, "autoencoder_metadata.csv"), index=False)

    print(f"Embeddings saved in {index_file}")
    print(f"Metadata saved in {save_folder}/autoencoder_metadata.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an autoencoder and extract image embeddings.")
    parser.add_argument("--base_folder", help="Folder containing images", required=True)
    parser.add_argument("--save_folder", help="Folder to save model and embeddings", required=True)
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Size of embedding vector")
    args = parser.parse_args()

    default_class_folder = os.path.join(args.base_folder, "unlabeled")
    ensure_subfolder_exists(args.base_folder, default_class_folder)

    autoencoder, transform, device = train_autoencoder(args.base_folder, args.save_folder, args.num_epochs,
                                                       args.batch_size, args.embedding_dim)

    extract_embeddings(autoencoder, transform, device, args.base_folder, args.save_folder, args.embedding_dim)
