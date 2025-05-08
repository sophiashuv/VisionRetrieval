import argparse
import cv2
import os
import faiss
import numpy as np
import pandas as pd
import scipy.fftpack
import time

def dhash(image, hash_size=8):
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def phash(image, hash_size=8):
    resized = cv2.resize(image, (32, 32))
    dct = scipy.fftpack.dct(scipy.fftpack.dct(resized.astype(float), axis=0), axis=1)
    dct_low_freq = dct[:hash_size, :hash_size]
    avg = dct_low_freq.mean()
    hash_array = dct_low_freq > avg
    return sum([2 ** i for (i, v) in enumerate(hash_array.flatten()) if v])


def hash_to_bitvector(hash_int, length=64):
    bit_list = [(hash_int >> i) & 1 for i in range(length)]
    bit_str = ''.join(str(b) for b in reversed(bit_list))
    byte_array = np.packbits(np.fromiter(bit_str, dtype=np.uint8))
    return byte_array.astype(np.uint8)


def compute_hash_faiss(base_folder, save_folder, method, hash_length=64):
    start_time = time.time()
    vectors = []
    image_paths = []

    for root, _, files in os.walk(base_folder):
        for filename in files:
            file_path = os.path.join(root, filename)
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):
                try:
                    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        print(f"Error loading image: {file_path}")
                        continue

                    if method == "dhash":
                        hash_val = dhash(image)
                    elif method == "phash":
                        hash_val = phash(image)
                    else:
                        continue

                    bit_vector = hash_to_bitvector(hash_val, length=hash_length)
                    vectors.append(bit_vector)
                    image_paths.append(file_path)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    if not vectors:
        print("No valid vectors found.")
        return

    vectors = np.stack(vectors).astype(np.uint8)  # Shape: [n_samples, hash_length // 8]
    index = faiss.IndexBinaryFlat(hash_length)    # Hamming index
    index.add(vectors)

    index_path = os.path.join(save_folder, f"{method}_faiss.index")
    faiss.write_index_binary(index, index_path)

    metadata_df = pd.DataFrame({"index": range(len(image_paths)), "image_path": image_paths})
    metadata_df.to_csv(os.path.join(save_folder, f"{method}_metadata.csv"), index=False)

    print(f"Hash vectors stored in: {index_path}")
    print(f"Metadata saved in: {save_folder}/{method}_metadata.csv")
    total_encoding_time = time.time() - start_time
    print(f"Total encoding time: {total_encoding_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute image hashes and store them in FAISS.")
    parser.add_argument("--base_folder", help="Base folder containing images", required=True)
    parser.add_argument("--save_folder", help="Where to save FAISS index and metadata", required=True)
    parser.add_argument("--method", choices=["dhash", "phash"], help="Hashing method to use", required=True)
    args = parser.parse_args()

    compute_hash_faiss(args.base_folder, args.save_folder, args.method)
