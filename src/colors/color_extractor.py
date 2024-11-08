import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import colorsys
import re
import os

class ColorBenchmark:
    def __init__(self):
        # Dictionary mapping color names to RGB values
        self.color_map = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'purple': (128, 0, 128),
            'orange': (255, 165, 0),
            'pink': (255, 192, 203),
            'brown': (165, 42, 42),
            'gray': (128, 128, 128),
            'black': (0, 0, 0),
            'white': (255, 255, 255)
        }
        
        # Color modifiers and their effects (multipliers)
        self.modifiers = {
            'light': 1.3,
            'dark': 0.7,
            'bright': 1.2,
            'deep': 0.6,
            'pale': 1.4,
            'vivid': 1.1
        }

    def extract_colors(self, image_path, n_colors=5):
        """Extract dominant colors from an image using K-means clustering"""
        img = Image.open(image_path)
        img = img.convert('RGB')
        # Convert image to numpy array and reshape
        pixels = np.array(img).reshape(-1, 3)
        
        # Use k-means to find dominant colors
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_
        
        # Calculate color proportions
        labels = kmeans.labels_
        proportions = np.bincount(labels) / len(labels)
        
        # Convert colors to RGB tuples with proportions
        dominant_colors = [
            (tuple(map(int, color)), float(prop))
            for color, prop in zip(colors, proportions)
        ]
        return sorted(dominant_colors, key=lambda x: x[1], reverse=True)

def main_color_extract(file_path, prompt):
    benchmark = ColorBenchmark()

    # Example usage
    image_path = os.path.join("static", file_path)

    # Run the benchmark
    results = benchmark.extract_colors(image_path)

    # Display results
    return results