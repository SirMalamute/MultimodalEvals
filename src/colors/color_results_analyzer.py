import numpy as np
from scipy.spatial import distance

class ColorNameMapper:
    def __init__(self):
        # Dictionary of basic colors with their RGB values
        self.color_map = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
            'gray': (128, 128, 128),
            'dark gray': (64, 64, 64),
            'light gray': (192, 192, 192),
            'brown': (165, 42, 42),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'pink': (255, 192, 203),
            'navy blue': (0, 0, 128),
            'sky blue': (135, 206, 235),
            'light blue': (173, 216, 230),
            'dark blue': (0, 0, 139),
            'forest green': (34, 139, 34),
            'olive': (128, 128, 0),
            'maroon': (128, 0, 0),
            'beige': (245, 245, 220),
            'tan': (210, 180, 140),
            'silver': (192, 192, 192)
        }

    def get_color_name(self, rgb, threshold=None):
        """
        Find the closest matching color name for an RGB value
        If threshold is provided, will return 'custom color' if no close match is found
        """
        rgb = np.array(rgb)
        min_distance = float('inf')
        closest_color = None
        
        for color_name, color_rgb in self.color_map.items():
            # Calculate Euclidean distance in RGB space
            dist = distance.euclidean(rgb, color_rgb)
            
            if dist < min_distance:
                min_distance = dist
                closest_color = color_name
        
        # If threshold is provided and minimum distance is too large,
        # return a custom color description
        if threshold and min_distance > threshold:
            # Create a custom description based on brightness and color components
            brightness = np.mean(rgb)
            max_component = np.argmax(rgb)
            
            if np.max(rgb) - np.min(rgb) < 30:  # Check if it's a grayscale color
                if brightness > 200:
                    return 'light gray'
                elif brightness > 100:
                    return 'gray'
                else:
                    return 'dark gray'
            
            brightness_prefix = 'light ' if brightness > 170 else 'dark ' if brightness < 85 else ''
            
            # Determine base color from maximum component
            if max_component == 0:
                base_color = 'red'
            elif max_component == 1:
                base_color = 'green'
            else:
                base_color = 'blue'
                
            return brightness_prefix + base_color
            
        return closest_color

    def analyze_color_distribution(self, color_proportions):
        """
        Analyze a list of RGB colors with their proportions
        Returns human-readable color names with their proportions
        """
        results = []
        for (r, g, b), proportion in color_proportions:
            color_name = self.get_color_name((r, g, b), threshold=100)
            results.append({
                'rgb': (r, g, b),
                'hex': '#{:02x}{:02x}{:02x}'.format(r, g, b),
                'proportion': proportion,
                'color_name': color_name
            })
        return results

# Example usage
def translate_colors(color_list):
    """
    Translate a list of RGB colors with proportions to human-readable names
    Input format: list of tuples ((R,G,B), proportion)
    """
    mapper = ColorNameMapper()
    
    # Convert the input format
    color_proportions = [
        (rgb, prop) for rgb, prop in color_list
    ]
    
    # Analyze colors
    results = mapper.analyze_color_distribution(color_proportions)
    
    return results

# Your example colors
if __name__ == "__main__":
    example_colors = [((254, 254, 254), 0.6013063632532659), ((6, 9, 20), 0.17071481247366202), ((19, 41, 90), 0.08128819005478298), ((33, 100, 198), 0.0788203223767383), ((119, 178, 222), 0.06787031184155078)]
    translate_colors(example_colors)