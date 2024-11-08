import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
from scipy.stats import entropy
import logging

class TechnicalImageQualityAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_image(self, image_path):
        """Load image in both BGR and grayscale formats"""
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise ValueError(f"Failed to load image: {image_path}")
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            return img_bgr, img_gray
        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            raise

    def analyze_blur(self, gray_img):
        """
        Detect blur using multiple methods:
        1. Laplacian variance
        2. FFT frequency analysis
        3. Edge sharpness
        """
        # Laplacian variance (higher = sharper)
        laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()
        
        # FFT frequency analysis
        fft = np.fft.fft2(gray_img)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        high_freq_content = np.mean(magnitude_spectrum[magnitude_spectrum > np.mean(magnitude_spectrum)])
        
        # Edge sharpness
        edges = cv2.Canny(gray_img, 100, 200)
        edge_intensity = np.mean(edges)
        
        return {
            'laplacian_variance': laplacian_var,
            'high_frequency_content': high_freq_content,
            'edge_sharpness': edge_intensity,
            'is_blurry': laplacian_var < 100  # Threshold can be adjusted
        }

    def analyze_noise(self, gray_img):
        """
        Analyze image noise using multiple approaches:
        1. Standard deviation in smooth areas
        2. Signal-to-noise ratio
        3. Noise pattern detection
        """
        # Smooth the image to identify noise
        smoothed = cv2.GaussianBlur(gray_img, (5, 5), 0)
        noise = gray_img.astype(np.float32) - smoothed.astype(np.float32)
        
        # Calculate noise metrics
        noise_std = np.std(noise)
        signal_mean = np.mean(gray_img)
        snr = signal_mean / (noise_std + 1e-10)  # Add small epsilon to avoid division by zero
        
        # Analyze noise patterns using local binary patterns
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
        pattern_entropy = entropy(np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3))[0])
        
        return {
            'noise_level': noise_std,
            'snr': snr,
            'noise_pattern_entropy': pattern_entropy,
            'is_noisy': noise_std > 30  # Threshold can be adjusted
        }

    def detect_artifacts(self, gray_img):
        """
        Detect various types of artifacts:
        1. Compression artifacts
        2. Banding in gradients
        3. Pixelation
        """
        # Compression artifacts (DCT analysis)
        dct = cv2.dct(np.float32(gray_img))
        dct_normalized = np.abs(dct) / np.max(np.abs(dct))
        high_freq_energy = np.sum(dct_normalized > 0.1) / dct.size
        
        # Banding detection
        gradient_mag = np.abs(np.gradient(gray_img.astype(float)))
        unique_gradients = len(np.unique(gradient_mag))
        banding_score = 1.0 - (unique_gradients / 255.0)
        
        # Pixelation detection
        edges = cv2.Canny(gray_img, 100, 200)
        # Fixed findContours call to work with all OpenCV versions
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        rect_score = 0
        if contours:
            for contour in contours:
                rect = cv2.minAreaRect(contour)
                _, (w, h), _ = rect
                if abs(w - h) < 2:  # Square-like shapes suggest pixelation
                    rect_score += 1
        pixelation_score = rect_score / (len(contours) + 1e-10)
        
        return {
            'compression_artifacts': high_freq_energy,
            'banding_score': banding_score,
            'pixelation_score': pixelation_score,
            'has_artifacts': high_freq_energy > 0.1 or banding_score > 0.5
        }
        
    def analyze_structural_coherence(self, gray_img):
        """
        Analyze structural coherence:
        1. Edge continuity
        2. Symmetry
        3. Perspective alignment
        """
        # Edge continuity
        edges = cv2.Canny(gray_img, 100, 200)
        edge_continuity = np.sum(edges > 0) / edges.size
        
        # Symmetry analysis
        height, width = gray_img.shape
        left_half = gray_img[:, :width//2]
        right_half = cv2.flip(gray_img[:, width//2:], 1)
        symmetry_score = ssim(left_half, right_half)
        
        # Perspective analysis (check for consistent lines)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        if lines is not None:
            angles = [line[0][1] for line in lines]
            angle_variance = np.var(angles)
        else:
            angle_variance = 0
            
        return {
            'edge_continuity': edge_continuity,
            'symmetry_score': symmetry_score,
            'perspective_consistency': 1.0 / (1.0 + angle_variance),
            'is_structurally_coherent': edge_continuity > 0.1 and symmetry_score > 0.7
        }

    def analyze_resolution(self, gray_img):
        """
        Analyze resolution quality:
        1. Detail preservation
        2. Texture clarity
        3. Fine feature detection
        """
        # Detail preservation (high-frequency content)
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        detail_score = np.std(laplacian)
        
        # Texture analysis
        glcm = self._calculate_glcm(gray_img)
        contrast = self._glcm_contrast(glcm)
        homogeneity = self._glcm_homogeneity(glcm)
        
        # Fine feature detection
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(gray_img, None)
        feature_density = len(keypoints) / gray_img.size
        
        return {
            'detail_score': detail_score,
            'texture_contrast': contrast,
            'texture_homogeneity': homogeneity,
            'feature_density': feature_density,
            'has_good_resolution': detail_score > 50 and feature_density > 0.001
        }

    def _calculate_glcm(self, img, distance=1, angles=[0]):
        """Calculate Gray-Level Co-occurrence Matrix"""
        glcm = np.zeros((256, 256))
        for angle in angles:
            dx = int(distance * np.cos(angle))
            dy = int(distance * np.sin(angle))
            for i in range(img.shape[0] - dx):
                for j in range(img.shape[1] - dy):
                    glcm[img[i,j], img[i+dx,j+dy]] += 1
        return glcm / glcm.sum()

    def _glcm_contrast(self, glcm):
        """Calculate contrast from GLCM"""
        contrast = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                contrast += glcm[i,j] * (i-j)**2
        return contrast

    def _glcm_homogeneity(self, glcm):
        """Calculate homogeneity from GLCM"""
        homogeneity = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                homogeneity += glcm[i,j] / (1 + abs(i-j))
        return homogeneity

    def analyze_all(self, image_path):
        """Run all technical quality analyses"""
        img_bgr, img_gray = self.load_image(image_path)
        
        results = {
            'blur_analysis': self.analyze_blur(img_gray),
            'noise_analysis': self.analyze_noise(img_gray),
            'artifact_detection': self.detect_artifacts(img_gray),
            'structural_coherence': self.analyze_structural_coherence(img_gray),
            'resolution_quality': self.analyze_resolution(img_gray)
        }
        
        # Overall quality score (simple weighted average)
        quality_score = (
            (1.0 - float(results['blur_analysis']['is_blurry'])) * 0.25 +
            (1.0 - float(results['noise_analysis']['is_noisy'])) * 0.20 +
            (1.0 - float(results['artifact_detection']['has_artifacts'])) * 0.20 +
            float(results['structural_coherence']['is_structurally_coherent']) * 0.20 +
            float(results['resolution_quality']['has_good_resolution']) * 0.15
        )
        
        results['overall_quality_score'] = quality_score
        return results

# Example usage:
# analyzer = TechnicalImageQualityAnalyzer()
# results = analyzer.analyze_all('bad3.png')
# print(results)
