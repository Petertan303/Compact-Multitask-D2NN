import numpy as np
import torch
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt

class SyntheticShapesDataset(Dataset):

    def __init__(self, root='./data', train=True, n_samples_per_class=6000,
                 img_size=56, transform=None, download=True):
        self.root = root
        self.train = train
        self.img_size = img_size
        self.transform = transform

        split = 'train' if train else 'test'
        self.data_dir = os.path.join(root, f'synthetic_shapes_{split}')

        if download:
            self._generate_data(n_samples_per_class)

        self.data = []
        self.targets = []
        self._load_data()

    def _generate_data(self, n_samples_per_class):
        split = 'train' if self.train else 'test'
        data_dir = os.path.join(self.root, f'synthetic_shapes_{split}')
        os.makedirs(data_dir, exist_ok=True)

        print(f"Generating synthetic shapes dataset ({split})...")

        for label in range(10):
            class_dir = os.path.join(data_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)

            existing = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
            if existing >= n_samples_per_class:
                print(f"  Class {label}: {existing} samples already exist, skipping")
                continue
            if existing > 0:
                print(f"  Class {label}: resuming from {existing} (target {n_samples_per_class})")
            else:
                print(f"  Generating class {label}: ", end='', flush=True)
            for i in range(existing, n_samples_per_class):
                img = self._generate_shape(label)
                filepath = os.path.join(class_dir, f'img_{i:05d}.png')
                plt.imsave(filepath, img, cmap='gray')
                if (i + 1) % 1000 == 0:
                    print(f"{i+1}...", end='', flush=True)
            print(f"{n_samples_per_class} done")

    def _generate_shape(self, label):
        size = self.img_size

        img = np.zeros((size, size), dtype=np.float32)

        cx = cy = size // 2
        scale = 0.7
        line_width = 2

        if label == 0:
            img = self._draw_cross(img, cx, cy, scale, line_width)
        elif label == 1:
            img = self._draw_triangle(img, cx, cy, scale, line_width, filled=True)
        elif label == 2:
            img = self._draw_triangle(img, cx, cy, scale, line_width, filled=False)
        elif label == 3:
            img = self._draw_square(img, cx, cy, scale, line_width, filled=True)
        elif label == 4:
            img = self._draw_square(img, cx, cy, scale, line_width, filled=False)
        elif label == 5:
            img = self._draw_polygon(img, cx, cy, scale, 5, line_width, filled=True)
        elif label == 6:
            img = self._draw_polygon(img, cx, cy, scale, 5, line_width, filled=False)
        elif label == 7:
            img = self._draw_dumbbell(img, cx, cy, scale, line_width)
        elif label == 8:
            img = self._draw_star(img, cx, cy, scale, line_width, filled=True)
        elif label == 9:
            img = self._draw_star(img, cx, cy, scale, line_width, filled=False)

        return img

    def _draw_line(self, img, x1, y1, x2, y2, width=2):
        h, w = img.shape

        length = int(np.sqrt((x2-x1)**2 + (y2-y1)**2)) + 1
        xs = np.linspace(x1, x2, length).astype(int)
        ys = np.linspace(y1, y2, length).astype(int)

        mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs, ys = xs[mask], ys[mask]

        if len(xs) > 0:
            for w_offset in range(-width//2, width//2 + 1):
                for h_offset in range(-width//2, width//2 + 1):
                    ny = ys + h_offset
                    nx = xs + w_offset
                    valid = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
                    img[ny[valid], nx[valid]] = 1.0

    def _draw_cross(self, img, cx, cy, scale, line_width):
        half_size = int(20 * scale)
        self._draw_line(img, cx - half_size, cy, cx + half_size, cy, line_width)
        self._draw_line(img, cx, cy - half_size, cx, cy + half_size, line_width)
        return img

    def _draw_triangle(self, img, cx, cy, scale, line_width, filled):
        r = int(20 * scale)
        points = []
        for i in range(3):
            angle = np.radians(-90 + i * 120)
            px = cx + r * np.cos(angle)
            py = cy + r * np.sin(angle)
            points.append((px, py))

        if filled:
            self._fill_polygon(img, points)
        for i in range(3):
            self._draw_line(img, points[i][0], points[i][1],
                          points[(i+1)%3][0], points[(i+1)%3][1], line_width)
        return img

    def _draw_square(self, img, cx, cy, scale, line_width, filled):
        half = int(16 * scale)
        x1, y1 = cx - half, cy - half
        x2, y2 = cx + half, cy + half

        if filled:
            img[y1:y2, x1:x2] = 1.0
        else:
            self._draw_line(img, x1, y1, x2, y1, line_width)
            self._draw_line(img, x2, y1, x2, y2, line_width)
            self._draw_line(img, x2, y2, x1, y2, line_width)
            self._draw_line(img, x1, y2, x1, y1, line_width)
        return img

    def _draw_polygon(self, img, cx, cy, scale, n_sides, line_width, filled):
        r = int(20 * scale)
        points = []
        for i in range(n_sides):
            angle = np.radians(-90 + i * (360 / n_sides))
            px = cx + r * np.cos(angle)
            py = cy + r * np.sin(angle)
            points.append((px, py))

        if filled:
            self._fill_polygon(img, points)
        for i in range(n_sides):
            self._draw_line(img, points[i][0], points[i][1],
                          points[(i+1)%n_sides][0], points[(i+1)%n_sides][1], line_width)
        return img

    def _draw_dumbbell(self, img, cx, cy, scale, line_width):
        r = int(8 * scale)
        gap = int(14 * scale)
        self._draw_filled_circle(img, cx - gap, cy, r)
        self._draw_filled_circle(img, cx + gap, cy, r)
        self._draw_line(img, cx - gap + r, cy, cx + gap - r, cy, max(1, line_width-1))
        return img

    def _draw_star(self, img, cx, cy, scale, line_width, filled):
        r_outer = int(20 * scale)
        r_inner = int(8 * scale)

        points_outer = []
        points_inner = []
        for i in range(5):
            angle_outer = np.radians(-90 + i * 72)
            angle_inner = np.radians(-90 + i * 72 + 36)
            points_outer.append((cx + r_outer * np.cos(angle_outer),
                                cy + r_outer * np.sin(angle_outer)))
            points_inner.append((cx + r_inner * np.cos(angle_inner),
                                cy + r_inner * np.sin(angle_inner)))

        star_points = []
        for i in range(5):
            star_points.append(points_outer[i])
            star_points.append(points_inner[i])

        if filled:
            self._fill_polygon(img, star_points)

        for i in range(10):
            self._draw_line(img, star_points[i][0], star_points[i][1],
                          star_points[(i+1)%10][0], star_points[(i+1)%10][1], line_width)
        return img

    def _draw_filled_circle(self, img, cx, cy, r):
        h, w = img.shape
        y, x = np.ogrid[:h, :w]
        mask = (x - cx)**2 + (y - cy)**2 <= r**2
        img[mask] = 1.0

    def _fill_polygon(self, img, points):
        min_y = max(0, int(min(p[1] for p in points)))
        max_y = min(img.shape[0], int(max(p[1] for p in points)) + 1)

        for y in range(min_y, max_y):
            intersections = []
            n = len(points)
            for i in range(n):
                p1 = points[i]
                p2 = points[(i + 1) % n]

                if (p1[1] <= y < p2[1]) or (p2[1] <= y < p1[1]):
                    if p1[1] != p2[1]:
                        x = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                        intersections.append(x)

            intersections.sort()
            for i in range(0, len(intersections) - 1, 2):
                x1, x2 = int(intersections[i]), int(intersections[i+1])
                img[y, x1:x2+1] = 1.0

    def _load_data(self):
        split = 'train' if self.train else 'test'
        data_dir = os.path.join(self.root, f'synthetic_shapes_{split}')

        for label in range(10):
            class_dir = os.path.join(data_dir, str(label))
            if os.path.exists(class_dir):
                files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
                for f in files:
                    filepath = os.path.join(class_dir, f)
                    img = plt.imread(filepath)
                    if img.ndim == 3:
                        img = img[:, :, 0]
                    self.data.append(img.astype(np.float32))
                    self.targets.append(label)

        self.data = np.array(self.data, dtype=np.float32)
        self.targets = np.array(self.targets)

        self._normalize_energy()

    def _normalize_energy(self):
        target_energy = 300.0

        for label in range(10):
            indices = np.where(self.targets == label)[0]
            for idx in indices:
                current_energy = np.sum(self.data[idx] ** 2)
                if current_energy > 0:

                    scale = np.sqrt(target_energy / current_energy)
                    self.data[idx] = self.data[idx] * scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]

        img = torch.from_numpy(img).float().unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        return img, label

    def visualize_samples(self, save_path='synthetic_shapes_samples.png'):
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))

        class_names = [
            '0: Cross', '1: Triangle', '2: H-Triangle',
            '3: Square', '4: H-Square', '5: Pentagon',
            '6: H-Pentagon', '7: Dumbbell', '8: Star', '9: H-Star'
        ]

        for label in range(10):
            row, col = label // 5, label % 5
            ax = axes[row, col]

            idx = self.targets.tolist().index(label)
            ax.imshow(self.data[idx], cmap='gray')
            ax.set_title(class_names[label])
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved samples to: {save_path}")

if __name__ == "__main__":
    import shutil

    print("Removing old data...")
    for split in ['train', 'test']:
        path = f'./data/synthetic_shapes_{split}'
        if os.path.exists(path):
            shutil.rmtree(path)

    print("Generating training set...")
    train_dataset = SyntheticShapesDataset(root='./data', train=True, n_samples_per_class=6000, download=True)

    print("\nGenerating test set...")
    test_dataset = SyntheticShapesDataset(root='./data', train=False, n_samples_per_class=1000, download=True)

    print("\nVisualizing samples...")
    train_dataset.visualize_samples()

    print(f"\nDataset created successfully!")
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    print("\nPer-label energy (pixel sum):")
    for label in range(10):
        idx = test_dataset.targets.tolist().index(label)
        img = test_dataset.data[idx]
        print(f"  Label {label}: {img.sum():.2f} pixels")
