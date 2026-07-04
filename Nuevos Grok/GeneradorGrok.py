import os
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def save_to_bin(name, X, y):
    """
    Saves datasets for CUDA processing.
   
    Format:
    - Content: Transposed (Features x Samples).
    - Filename: Features x Samples.bin
   
    Example:
    If you have 2000 Samples and 2 Features.
    - Standard Shape: (2000, 2)
    - Saved Content: (2, 2000) -> [Feat0_all_samples, Feat1_all_samples]
    - Filename: "2x2000.bin"
    """
   
    # 1. Ensure y is (Samples, 1)
    y = y.reshape(-1, 1)
    # 2. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 3. Convert to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    # 4. Define Base Paths
    base_path = f"{name}/Bin"
    paths = {
        "train_x": f"{base_path}/Train/X",
        "train_y": f"{base_path}/Train/Y",
        "test_x": f"{base_path}/Test/X",
        "test_y": f"{base_path}/Test/Y",
        "Reg": f"{name}/Reg",
    }
    # 5. Create Directories
    for p in paths.values():
        ensure_dir(p)

    # 6. Save
    fn = f"{paths['train_x']}/{X_train.shape[1]}x{X_train.shape[0]}.bin"
    X_train.tofile(fn)
    fn = f"{paths['train_y']}/{y_train.shape[1]}x{y_train.shape[0]}.bin"
    y_train.tofile(fn)
    fn = f"{paths['test_x']}/{X_test.shape[1]}x{X_test.shape[0]}.bin"
    X_test.tofile(fn)
    fn = f"{paths['test_y']}/{y_test.shape[1]}x{y_test.shape[0]}.bin"
    y_test.tofile(fn)

    print(f"[{name}] Exported.")
    print(f" Train File: {X_train.shape[1]}x{X_train.shape[0]}.bin (Features x Samples)")


# --- Custom Dataset Generators ---
def make_xor(n_samples=1000):
    centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.3, random_state=42)
    y = np.array([0 if val < 2 else 1 for val in y])
    return X, y


def make_spiral(n_samples=1000):
    n = n_samples // 2
    theta = np.sqrt(np.random.rand(n)) * 720 * (2 * np.pi / 360)
    d1x = -np.cos(theta) * theta + np.random.rand(n) * 0.2
    d1y = np.sin(theta) * theta + np.random.rand(n) * 0.2
    d2x = np.cos(theta) * theta + np.random.rand(n) * 0.2
    d2y = -np.sin(theta) * theta + np.random.rand(n) * 0.2
    X = np.vstack((np.column_stack((d1x, d1y)), np.column_stack((d2x, d2y))))
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


# === NUEVOS GENERADORES (dificultad ≥ Problem6) ===
def make_spiral_3turns(n_samples=2000):
    n = n_samples // 2
    theta = np.sqrt(np.random.rand(n)) * 1080 * (2 * np.pi / 360)  # 3 vueltas
    d1x = -np.cos(theta) * theta + np.random.rand(n) * 0.2
    d1y = np.sin(theta) * theta + np.random.rand(n) * 0.2
    d2x = np.cos(theta) * theta + np.random.rand(n) * 0.2
    d2y = -np.sin(theta) * theta + np.random.rand(n) * 0.2
    X = np.vstack((np.column_stack((d1x, d1y)), np.column_stack((d2x, d2y))))
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


def make_spiral_4turns(n_samples=2000):
    n = n_samples // 2
    theta = np.sqrt(np.random.rand(n)) * 1440 * (2 * np.pi / 360)  # 4 vueltas
    d1x = -np.cos(theta) * theta + np.random.rand(n) * 0.2
    d1y = np.sin(theta) * theta + np.random.rand(n) * 0.2
    d2x = np.cos(theta) * theta + np.random.rand(n) * 0.2
    d2y = -np.sin(theta) * theta + np.random.rand(n) * 0.2
    X = np.vstack((np.column_stack((d1x, d1y)), np.column_stack((d2x, d2y))))
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


def make_spiral_high_noise(n_samples=2000):
    n = n_samples // 2
    theta = np.sqrt(np.random.rand(n)) * 720 * (2 * np.pi / 360)
    noise_std = 0.5
    d1x = -np.cos(theta) * theta + np.random.randn(n) * noise_std
    d1y = np.sin(theta) * theta + np.random.randn(n) * noise_std
    d2x = np.cos(theta) * theta + np.random.randn(n) * noise_std
    d2y = -np.sin(theta) * theta + np.random.randn(n) * noise_std
    X = np.vstack((np.column_stack((d1x, d1y)), np.column_stack((d2x, d2y))))
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Generating Datasets for CUDA (Features x Samples)...\n")

    X, y = make_blobs(n_samples=2000, centers=2, n_features=2, random_state=42, cluster_std=1.5)
    save_to_bin("Problem1_Blobs", X, y)

    data = load_breast_cancer()
    X, y = data.data, data.target
    save_to_bin("Problem2_Cancer", X, y)

    X, y = make_xor(n_samples=2000)
    save_to_bin("Problem3_XOR", X, y)

    X, y = make_circles(n_samples=2000, noise=0.05, factor=0.5, random_state=42)
    save_to_bin("Problem4_Circles", X, y)

    X, y = make_moons(n_samples=2000, noise=0.1, random_state=42)
    save_to_bin("Problem5_Moons", X, y)

    X, y = make_spiral(n_samples=2000)
    save_to_bin("Problem6_Spiral", X, y)

    try:
        print("Fetching Sonar dataset...")
        sonar = fetch_openml(name='sonar', version=1, as_frame=False)
        X_son, y_son = sonar.data, sonar.target
        y_son = (y_son == 'M').astype(int)
        save_to_bin("Problem7_Sonar", X_son, y_son)
    except Exception as e:
        print(f"Skipping Sonar: {e}")

    # === NUEVOS PROBLEMAS (muy difíciles) ===
    X, y = make_spiral_3turns(n_samples=2000)
    save_to_bin("Problem8_Spiral3Turns", X, y)

    X, y = make_spiral_4turns(n_samples=2000)
    save_to_bin("Problem9_Spiral4Turns", X, y)

    X, y = make_spiral_high_noise(n_samples=2000)
    save_to_bin("Problem10_SpiralHighNoise", X, y)

    print("\nAll datasets generated (10 problems).")