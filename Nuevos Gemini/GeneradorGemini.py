import os
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles, make_gaussian_quantiles, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def save_to_bin(name, X, y):
    """
    Saves datasets for CUDA processing.
    
    Format:
    - Content: Transposed (Features x Samples) -> For CUDA Coalesced Access.
    - Filename: Features x Samples.bin
    """
    
    # 1. Ensure y is (Samples, 1)
    y = y.reshape(-1, 1)

    # 2. Split Data (80% Train, 20% Test)
    # Shapes here are still (Samples, Features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Convert to float32 (Standard for CUDA/GPU processing)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # 4. Define Base Paths
    base_path = f"{name}/Bin"
    paths = {
        "train_x": f"{base_path}/Train/X",
        "train_y": f"{base_path}/Train/Y",
        "test_x":  f"{base_path}/Test/X",
        "test_y":  f"{base_path}/Test/Y",
        "Reg":     f"{name}/Reg",  # Folder for logs
    }

    # 5. Create Directories
    for p in paths.values():
        ensure_dir(p)

    # 6. Save Files
    # Formula: Filename = N_Features x M_Samples
    #          Content  = Transposed Matrix (.T)
    
    # --- Train X ---
    # Filename shows Transposed dimensions (Feat x Samp)
    fn = f"{paths['train_x']}/{X_train.shape[1]}x{X_train.shape[0]}.bin"
    # IMPORTANT: .T is applied here to align data with filename and CUDA requirements
    X_train.T.tofile(fn)
    
    # --- Train Y ---
    fn = f"{paths['train_y']}/{y_train.shape[1]}x{y_train.shape[0]}.bin"
    y_train.T.tofile(fn)
    
    # --- Test X ---
    fn = f"{paths['test_x']}/{X_test.shape[1]}x{X_test.shape[0]}.bin"
    X_test.T.tofile(fn)
    
    # --- Test Y ---
    fn = f"{paths['test_y']}/{y_test.shape[1]}x{y_test.shape[0]}.bin"
    y_test.T.tofile(fn)

    print(f"[{name}] Exported.")
    print(f"   Train File: {X_train.shape[1]}x{X_train.shape[0]}.bin (Features x Samples)")

# --- Custom Dataset Generators ---

def make_xor(n_samples=1000):
    centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.3, random_state=42)
    y = np.array([0 if val < 2 else 1 for val in y])
    return X, y

def make_spiral(n_samples=1000):
    n = n_samples // 2
    theta = np.sqrt(np.random.rand(n)) * 720 * (2 * np.pi / 360) 

    # Class 0
    d1x = -np.cos(theta) * theta + np.random.rand(n) * 0.2
    d1y = np.sin(theta) * theta + np.random.rand(n) * 0.2
    
    # Class 1
    d2x = np.cos(theta) * theta + np.random.rand(n) * 0.2
    d2y = -np.sin(theta) * theta + np.random.rand(n) * 0.2
    
    X = np.vstack((np.column_stack((d1x, d1y)), np.column_stack((d2x, d2y))))
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y

def make_checkerboard(n_samples=2000):
    """
    Hard Topology: 4x4 Grid.
    Very hard for shallow networks (requires many hyperplanes to cut).
    """
    X = np.random.uniform(-4, 4, size=(n_samples, 2))
    mask_x = (np.floor(X[:, 0]) % 2 == 0)
    mask_y = (np.floor(X[:, 1]) % 2 == 0)
    y = np.logical_xor(mask_x, mask_y).astype(int)
    return X, y

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("Generating Datasets for CUDA (Features x Samples)...\n")

    # 1. Linear (Easy): Blobs
    X, y = make_blobs(n_samples=2000, centers=2, n_features=2, random_state=42, cluster_std=1.5)
    save_to_bin("Problem1_Blobs", X, y)

    # 2. Real World (Easy High Dim): Breast Cancer
    data = load_breast_cancer()
    X, y = data.data, data.target
    save_to_bin("Problem2_Cancer", X, y)

    # 3. Non-Linear (Moderate): XOR (4 clusters mapped to 2 classes)
    X, y = make_xor(n_samples=2000)
    save_to_bin("Problem3_XOR", X, y)

    # 4. Topology (Hard): Circles (Bullseye)
    X, y = make_circles(n_samples=2000, noise=0.05, factor=0.5, random_state=42)
    save_to_bin("Problem4_Circles", X, y)

    # 5. Topology (Harder): Moons
    X, y = make_moons(n_samples=2000, noise=0.1, random_state=42)
    save_to_bin("Problem5_Moons", X, y)

    # 6. Path Following (Very Hard): Spiral
    X, y = make_spiral(n_samples=2000)
    save_to_bin("Problem6_Spiral", X, y)

    # 7. Real World (Hard/Noisy): Sonar
    try:
        print("Fetching Sonar dataset from OpenML...")
        sonar = fetch_openml(name='sonar', version=1, as_frame=False)
        X_son, y_son = sonar.data, sonar.target
        y_son = (y_son == 'M').astype(int) 
        save_to_bin("Problem7_Sonar", X_son, y_son)
    except Exception as e:
        print(f"Skipping Sonar: {e}")

    # --- NUEVOS DATASETS DIFÍCILES ---

    # 8. Topology (Very Hard): Checkerboard 4x4
    # Requiere que la red aprenda una función altamente no convexa.
    X, y = make_checkerboard(n_samples=2000)
    save_to_bin("Problem8_Checkerboard", X, y)

    # 9. Topology (Hard): Gaussian Quantiles
    # Múltiples nubes de una clase rodeadas por la otra. 
    # Es como el problema de Circles pero multimodal y más difuso.
    X, y = make_gaussian_quantiles(n_samples=2000, n_features=2, n_classes=2, random_state=42)
    save_to_bin("Problem9_Quantiles", X, y)

    # 10. Real World (Very Hard/Noisy): Magic Gamma Telescope
    # Detección de particulas físicas. Mucho ruido, clases superpuestas.
    # OpenML ID: 1120
    try:
        print("Fetching Magic Gamma Telescope dataset from OpenML...")
        # 'h' es clase 1, 'g' es clase 0
        magic = fetch_openml(data_id=1120, as_frame=False)
        X_mag, y_mag = magic.data, magic.target
        y_mag = (y_mag == 'h').astype(int)
        # Tomamos una muestra aleatoria de 2000-3000 si es muy grande, o todo si es manejable
        # Tiene ~19k ejemplos. Usaremos 3000 para mantener consistencia de tamaño.
        indices = np.random.choice(X_mag.shape[0], 3000, replace=False)
        save_to_bin("Problem10_MagicGamma", X_mag[indices], y_mag[indices])
    except Exception as e:
        print(f"Skipping Magic Gamma: {e}")

    print("\nAll datasets generated.")