import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json

class ImageQualityBenchmarker:
    def __init__(self):
        # Define quality thresholds for each metric
        self.thresholds = {
            'blur_analysis': {
                'laplacian_variance': {'min': 30, 'max': None, 'critical': 15},  # Reduced from 100/50
                'high_frequency_content': {'min': 0.1, 'max': None, 'critical': 0.05},  # Reduced from 0.3/0.1
                'edge_sharpness': {'min': 10, 'max': None, 'critical': 5}  # Reduced from 20/10
            },
            'noise_analysis': {
                'noise_level': {'min': None, 'max': 60, 'critical': 80},  # Increased from 30/50
                'snr': {'min': 2, 'max': None, 'critical': 1},  # Reduced from 5/3
                'noise_pattern_entropy': {'min': None, 'max': 5.5, 'critical': 6.5}  # Increased from 4.5/5.5
            },
            'artifact_detection': {
                'compression_artifacts': {'min': None, 'max': 0.3, 'critical': 0.5},  # Increased from 0.1/0.2
                'banding_score': {'min': None, 'max': 0.5, 'critical': 0.7},  # Increased from 0.3/0.5
                'pixelation_score': {'min': None, 'max': 0.5, 'critical': 0.7}  # Increased from 0.3/0.5
            },
            'structural_coherence': {
                'edge_continuity': {'min': 0.05, 'max': None, 'critical': 0.02},  # Reduced from 0.1/0.05
                'symmetry_score': {'min': 0.4, 'max': None, 'critical': 0.2},  # Reduced from 0.7/0.5
                'perspective_consistency': {'min': 0.3, 'max': None, 'critical': 0.15}  # Reduced from 0.6/0.4
            },
            'resolution_quality': {
                'detail_score': {'min': 25, 'max': None, 'critical': 15},  # Reduced from 50/30
                'texture_contrast': {'min': 0.15, 'max': None, 'critical': 0.05},  # Reduced from 0.3/0.1
                'texture_homogeneity': {'min': 0.25, 'max': None, 'critical': 0.15}  # Reduced from 0.5/0.3
            }
        }
        
    def analyze_results(self, results: Dict, image_name: str) -> Dict:
        """Analyze a single image's quality results against thresholds"""
        issues = []
        warnings = []
        metrics = {}
        
        for category, measurements in results.items():
            if category == 'overall_quality_score':
                continue
                
            if category in self.thresholds:
                for metric, value in measurements.items():
                    if metric.startswith('is_') or metric.startswith('has_'):
                        continue
                        
                    if metric in self.thresholds[category]:
                        threshold = self.thresholds[category][metric]
                        metrics[f"{category}_{metric}"] = value
                        
                        # Check against thresholds
                        if threshold['min'] and value < threshold['min']:
                            if value < threshold['critical']:
                                issues.append(f"Critical: {metric} is too low ({value:.2f} < {threshold['critical']})")
                            else:
                                warnings.append(f"Warning: {metric} is below threshold ({value:.2f} < {threshold['min']})")
                                
                        if threshold['max'] and value > threshold['max']:
                            if value > threshold['critical']:
                                issues.append(f"Critical: {metric} is too high ({value:.2f} > {threshold['critical']})")
                            else:
                                warnings.append(f"Warning: {metric} is above threshold ({value:.2f} > {threshold['max']})")
        
        return {
            'image_name': image_name,
            'overall_score': results['overall_quality_score'],
            'metrics': metrics,
            'issues': issues,
            'warnings': warnings,
            'critical_issues_count': len(issues),
            'warnings_count': len(warnings)
        }
    
    def generate_report(self, analysis_results: List[Dict]) -> pd.DataFrame:
        """Generate a DataFrame with all results for analysis"""
        df = pd.DataFrame(analysis_results)
        df['status'] = df.apply(self._determine_status, axis=1)
        return df
    
    def _determine_status(self, row) -> str:
        if row['critical_issues_count'] > 0:
            return 'FAILED'
        elif row['warnings_count'] > 0:
            return 'WARNING'
        return 'PASSED'
    

    def analyze_batch(self, results_list: List[Tuple[str, Dict]]):
        """Analyze a batch of image quality results and generate reports"""
        
        # Process all results
        analyses = [self.analyze_results(results, image_name) 
                   for image_name, results in results_list]
        
        # Generate DataFrame
        df = self.generate_report(analyses)
        
        # Calculate summary statistics
        summary = {
            'total_images': len(analyses),
            'passed': len(df[df['status'] == 'PASSED']),
            'warnings': len(df[df['status'] == 'WARNING']),
            'failed': len(df[df['status'] == 'FAILED']),
            'average_score': df['overall_score'].mean(),
            'worst_images': df.nsmallest(5, 'overall_score')[['image_name', 'overall_score', 'status']].to_dict('records'),
            'most_common_issues': self._get_most_common_issues(analyses)
        }
        
        return summary, df
    
    def _get_most_common_issues(self, analyses: List[Dict]) -> Dict:
        """Extract most common issues from analyses"""
        all_issues = []
        all_warnings = []
        
        for analysis in analyses:
            all_issues.extend(analysis['issues'])
            all_warnings.extend(analysis['warnings'])
        
        return {
            'critical_issues': self._count_occurrences(all_issues)[:5],
            'warnings': self._count_occurrences(all_warnings)[:5]
        }
    
    def _count_occurrences(self, items: List[str]) -> List[Tuple[str, int]]:
        """Count occurrences of items and return sorted list of (item, count) tuples"""
        counts = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)

# Example usage:
# Initialize the analyzer and benchmarker
# benchmarker = ImageQualityBenchmarker()

# # Process a batch of images
# results_list = []
# results_list.append(["apple",{'blur_analysis': {'laplacian_variance': np.float64(16.848120375393833), 'high_frequency_content': np.float64(13962.228013643078), 'edge_sharpness': np.float64(0.7074962397937258), 'is_blurry': np.True_}, 'noise_analysis': {'noise_level': np.float32(1.4610227), 'snr': np.float64(34.99649894897998), 'noise_pattern_entropy': np.float64(2.850971250972053), 'is_noisy': np.False_}, 'artifact_detection': {'compression_artifacts': np.float64(4.0287924366136656e-05), 'banding_score': 0.7098039215686274, 'pixelation_score': 0.04651162790686857, 'has_artifacts': True}, 'structural_coherence': {'edge_continuity': np.float64(0.002774495058014611), 'symmetry_score': np.float64(0.47724191836105784), 'perspective_consistency': 1.0, 'is_structurally_coherent': np.False_}, 'resolution_quality': {'detail_score': np.float64(4.10464619369244), 'texture_contrast': np.float64(8.902832167832145), 'texture_homogeneity': np.float64(0.5253902594642804), 'feature_density': 0.0038354103996562096, 'has_good_resolution': np.False_}, 'overall_quality_score': 0.2}])

# # Generate benchmark analysis
# summary, detailed_results = benchmarker.analyze_batch(
#     results_list
# )

# print("Analysis Summary:")
# print(json.dumps(summary, indent=2))
