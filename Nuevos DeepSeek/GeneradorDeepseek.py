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
    - Saved Content:  (2, 2000) -> [Feat0_all_samples, Feat1_all_samples]
    - Filename:       "2x2000.bin"
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
        "Reg": f"{name}/Reg",
    }

    # 5. Create Directories
    for p in paths.values():
        ensure_dir(p)

    # 6. Save Files
    # Formula: Filename = N_Features x M_Samples
    #          Content  = Transposed Matrix
    
    # --- Train X ---
    # shape[1] is Features, shape[0] is Samples
    fn = f"{paths['train_x']}/{X_train.shape[1]}x{X_train.shape[0]}.bin"
    X_train.tofile(fn)
    
    # --- Train Y ---
    # shape[1] is 1 (Label Dimension), shape[0] is Samples
    fn = f"{paths['train_y']}/{y_train.shape[1]}x{y_train.shape[0]}.bin"
    y_train.tofile(fn)
    
    # --- Test X ---
    fn = f"{paths['test_x']}/{X_test.shape[1]}x{X_test.shape[0]}.bin"
    X_test.tofile(fn)
    
    # --- Test Y ---
    fn = f"{paths['test_y']}/{y_test.shape[1]}x{y_test.shape[0]}.bin"
    y_test.tofile(fn)

    print(f"[{name}] Exported.")
    print(f"   Train File: {X_train.shape[1]}x{X_train.shape[0]}.bin (Features x Samples)")

# --- Custom Dataset Generators ---

def make_xor(n_samples=1000):
    # Generates a cloud of points that mimics the XOR problem
    # Centers 0,1 -> Class 0 | Centers 2,3 -> Class 1
    centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.3, random_state=42)
    
    # Map classes: 0&1 -> 0, 2&3 -> 1
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

# --- NEW: Complex Binary Classification Datasets ---

def make_swiss_roll_2d(n_samples=2000):
    """Modified Swiss Roll reduced to 2D with binary classification"""
    from sklearn.datasets import make_swiss_roll
    
    # Generate 3D Swiss Roll
    X_3d, _ = make_swiss_roll(n_samples=n_samples, noise=0.8, random_state=42)
    
    # Project to 2D and add complexity
    X = np.column_stack((
        X_3d[:, 0] * np.sin(X_3d[:, 0] * 0.3),
        X_3d[:, 1] * np.cos(X_3d[:, 1] * 0.3)
    ))
    
    # Create binary classification based on position in the roll
    angle = np.arctan2(X[:, 1], X[:, 0])
    y = ((angle + np.pi) * 3).astype(int) % 2
    y = np.where(X[:, 0] * X[:, 1] > 0, 1 - y, y)
    
    # Add some noise to labels
    flip_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    y[flip_indices] = 1 - y[flip_indices]
    
    return X, y

def make_nested_circles(n_samples=2000):
    """Three concentric circles with alternating classes (bullseye pattern)"""
    n = n_samples // 3
    radii = [1.0, 2.2, 3.5]
    
    points = []
    labels = []
    
    for i, r in enumerate(radii):
        theta = np.random.uniform(0, 2*np.pi, n)
        x = r * np.cos(theta) + np.random.randn(n) * 0.15
        y = r * np.sin(theta) + np.random.randn(n) * 0.15
        
        points.append(np.column_stack((x, y)))
        # Classes alternate: 0, 1, 0
        labels.append(np.full(n, i % 2))
    
    X = np.vstack(points)
    y = np.hstack(labels)
    
    return X, y

def make_sine_wave_clusters(n_samples=2000):
    """Clusters following sine wave patterns with noise"""
    n = n_samples // 2
    
    # Class 0: Points around sine wave
    x0 = np.random.uniform(-3*np.pi, 3*np.pi, n)
    y0 = np.sin(x0) + np.random.normal(0, 0.3, n)
    X0 = np.column_stack((x0, y0))
    
    # Class 1: Points around phase-shifted sine wave
    x1 = np.random.uniform(-3*np.pi, 3*np.pi, n)
    y1 = np.sin(x1 + np.pi) + np.random.normal(0, 0.3, n)
    X1 = np.column_stack((x1, y1))
    
    # Add perpendicular displacement to create separation
    X0[:, 1] += 0.5 * np.sin(X0[:, 0] * 0.5)
    X1[:, 1] -= 0.5 * np.sin(X1[:, 0] * 0.5)
    
    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(n), np.ones(n)))
    
    return X, y

def make_adaptive_mesh(n_samples=2000):
    """Adaptive mesh-like structure with classification based on distance to multiple centers"""
    # Create a grid of centers
    grid_x, grid_y = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
    centers = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    # Assign binary class to each center in checkerboard pattern
    center_classes = ((np.arange(25) + np.arange(25)//5) % 2)
    
    # Generate points around centers
    points_per_center = n_samples // 25
    points = []
    labels = []
    
    for i, (center, cls) in enumerate(zip(centers, center_classes)):
        # Vary cluster density and spread
        spread = 0.15 + (i % 3) * 0.1
        angle = np.random.uniform(0, 2*np.pi, points_per_center)
        radius = np.random.exponential(spread, points_per_center)
        
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        
        points.append(np.column_stack((x, y)))
        labels.append(np.full(points_per_center, cls))
    
    X = np.vstack(points)
    y = np.hstack(labels)
    
    return X, y

def make_checkerboard_2d(n_samples=2000):
    """Checkerboard pattern with Gaussian noise"""
    n = n_samples
    
    # Generate points in a grid pattern
    x = np.random.uniform(-4, 4, n)
    y = np.random.uniform(-4, 4, n)
    
    # Create checkerboard pattern (alternating squares)
    # Use sine of both coordinates to create the pattern
    scale = 1.5
    pattern = (np.sin(x * scale * np.pi) > 0) ^ (np.sin(y * scale * np.pi) > 0)
    y_labels = pattern.astype(int)
    
    # Add significant noise to make it challenging
    noise = np.random.normal(0, 0.4, (n, 2))
    X = np.column_stack((x, y)) + noise
    
    return X, y_labels

def make_concentric_polygons(n_samples=2000):
    """Concentric polygons (square, pentagon, hexagon) with alternating classes"""
    n = n_samples // 3
    polygons = [4, 5, 6]  # Number of sides for each polygon
    radii = [1.5, 3.0, 4.5]
    
    points = []
    labels = []
    
    for i, (sides, r) in enumerate(zip(polygons, radii)):
        angles = np.linspace(0, 2*np.pi, sides, endpoint=False)
        polygon_vertices = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
        
        # Generate points inside polygon with noise
        for _ in range(n):
            # Random convex combination of vertices
            weights = np.random.exponential(1, sides)
            weights /= weights.sum()
            
            point = np.dot(weights, polygon_vertices)
            noise = np.random.normal(0, 0.2, 2)
            
            points.append(point + noise)
            labels.append(i % 2)
    
    X = np.array(points)
    y = np.array(labels)
    
    return X, y

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("Generating Complex Binary Classification Datasets for CUDA (Features x Samples)...\n")

    # 1. Linear (Easy): Blobs - Keep as baseline
    X, y = make_blobs(n_samples=2000, centers=2, n_features=2, random_state=42, cluster_std=1.5)
    save_to_bin("Problem1_Blobs", X, y)

    # 2. Complex Pattern 1: Nested Circles (3 circles, alternating classes)
    X, y = make_nested_circles(n_samples=2000)
    save_to_bin("Problem2_NestedCircles", X, y)

    # 3. Complex Pattern 2: Sine Wave Clusters
    X, y = make_sine_wave_clusters(n_samples=2000)
    save_to_bin("Problem3_SineClusters", X, y)

    # 4. Complex Pattern 3: Adaptive Mesh
    X, y = make_adaptive_mesh(n_samples=2000)
    save_to_bin("Problem4_AdaptiveMesh", X, y)

    # 5. Complex Pattern 4: Checkerboard with Noise
    X, y = make_checkerboard_2d(n_samples=2000)
    save_to_bin("Problem5_Checkerboard", X, y)

    # 6. Complex Pattern 5: Swiss Roll 2D (Very Hard - similar to spiral)
    X, y = make_swiss_roll_2d(n_samples=2000)
    save_to_bin("Problem6_SwissRoll2D", X, y)

    # 7. Complex Pattern 6: Concentric Polygons
    X, y = make_concentric_polygons(n_samples=2000)
    save_to_bin("Problem7_ConcentricPolygons", X, y)

    print("\nAll complex binary classification datasets generated.")
    print("Difficulty levels are comparable to the original spiral problem (level 6).")