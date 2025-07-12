"""
Analyzes ASL dataset structure/distribution
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASLDataAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.analysis_results = {}
        
    def analyze_dataset_structure(self):
        logger.info("Lemme see dat dataset structure rq...")
        
        class_info = {}
        total_images = 0

        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        for class_dir in class_dirs:
            class_name = class_dir.name

            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            num_images = len(image_files)
            
            class_info[class_name] = {
                'count': num_images,
                'percentage': 0 
            }
            total_images += num_images
            
            logger.info(f"Class '{class_name}': {num_images} images")
        
        for class_name in class_info:
            class_info[class_name]['percentage'] = (class_info[class_name]['count'] / total_images) * 100
        
        self.analysis_results['class_distribution'] = class_info
        self.analysis_results['total_images'] = total_images
        self.analysis_results['num_classes'] = len(class_info)
        
        logger.info(f"Total images: {total_images}")
        logger.info(f"Number of classes: {len(class_info)}")
        
        return class_info
    
    def analyze_image_quality(self, sample_size=100):
        """Analyze image quality and characteristics."""
        logger.info(f"Analyzing image quality (sample size: {sample_size})...")
        
        image_stats = {
            'widths': [],
            'heights': [],
            'channels': [],
            'file_sizes': [],
            'corrupted_count': 0
        }
        
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        samples_per_class = max(1, sample_size // len(class_dirs))
        
        for class_dir in class_dirs:
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            sampled_files = np.random.choice(image_files, 
                                           min(samples_per_class, len(image_files)), 
                                           replace=False)
            
            for img_file in sampled_files:
                try:
                    # Get file size
                    file_size = img_file.stat().st_size / 1024  # KB
                    image_stats['file_sizes'].append(file_size)
                    
                    # Load image and get dimensions
                    img = cv2.imread(str(img_file))
                    if img is None:
                        image_stats['corrupted_count'] += 1
                        continue
                    
                    h, w, c = img.shape
                    image_stats['heights'].append(h)
                    image_stats['widths'].append(w)
                    image_stats['channels'].append(c)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing {img_file}: {e}")
                    image_stats['corrupted_count'] += 1
        
        # Calculate statistics
        quality_stats = {
            'avg_width': np.mean(image_stats['widths']),
            'avg_height': np.mean(image_stats['heights']),
            'width_std': np.std(image_stats['widths']),
            'height_std': np.std(image_stats['heights']),
            'avg_file_size_kb': np.mean(image_stats['file_sizes']),
            'corrupted_count': image_stats['corrupted_count'],
            'total_analyzed': len(image_stats['widths'])
        }
        
        self.analysis_results['image_quality'] = quality_stats
        
        logger.info(f"Average image dimensions: {quality_stats['avg_width']:.1f} x {quality_stats['avg_height']:.1f}")
        logger.info(f"Average file size: {quality_stats['avg_file_size_kb']:.1f} KB")
        logger.info(f"Corrupted images: {quality_stats['corrupted_count']}")
        
        return quality_stats
    
    def generate_visualizations(self, output_dir="analysis_output"):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if 'class_distribution' in self.analysis_results:
            class_data = self.analysis_results['class_distribution']
            
            plt.figure(figsize=(15, 8))
            classes = list(class_data.keys())
            counts = [class_data[c]['count'] for c in classes]
            
            plt.subplot(2, 2, 1)
            bars = plt.bar(classes, counts)
            plt.title('ASL Class Distribution')
            plt.xlabel('ASL Classes')
            plt.ylabel('Number of Images')
            plt.xticks(rotation=45)
            
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                        str(count), ha='center', va='bottom', fontsize=8)

            plt.subplot(2, 2, 2)
            sorted_classes = sorted(class_data.items(), key=lambda x: x[1]['count'], reverse=True)
            top_classes = dict(sorted_classes[:10])
            
            plt.pie([c['count'] for c in top_classes.values()], 
                   labels=list(top_classes.keys()), 
                   autopct='%1.1f%%')
            plt.title('Top 10 Classes Distribution')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def generate_report(self, output_file="dataset_analysis_report.txt"):
        """Generate a comprehensive text report."""
        with open(output_file, 'w') as f:
            f.write("ASL Dataset Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Images: {self.analysis_results.get('total_images', 'N/A')}\n")
            f.write(f"Number of Classes: {self.analysis_results.get('num_classes', 'N/A')}\n\n")
            
            if 'class_distribution' in self.analysis_results:
                f.write("CLASS DISTRIBUTION\n")
                f.write("-" * 20 + "\n")
                class_data = self.analysis_results['class_distribution']
                for class_name, info in sorted(class_data.items()):
                    f.write(f"{class_name}: {info['count']} images ({info['percentage']:.2f}%)\n")
                f.write("\n")
            
            if 'image_quality' in self.analysis_results:
                f.write("IMAGE QUALITY ANALYSIS\n")
                f.write("-" * 20 + "\n")
                quality = self.analysis_results['image_quality']
                f.write(f"Average Dimensions: {quality['avg_width']:.1f} x {quality['avg_height']:.1f} pixels\n")
                f.write(f"Dimension Std Dev: {quality['width_std']:.1f} x {quality['height_std']:.1f} pixels\n")
                f.write(f"Average File Size: {quality['avg_file_size_kb']:.1f} KB\n")
                f.write(f"Corrupted Images: {quality['corrupted_count']}\n")
                f.write(f"Images Analyzed: {quality['total_analyzed']}\n")
        
        logger.info(f"Analysis report saved to {output_file}")
    
    def run_complete_analysis(self, output_dir="analysis_output"):
        """Run the complete analysis pipeline."""
        logger.info("Starting complete dataset analysis...")
        
        self.analyze_dataset_structure()
        self.analyze_image_quality()
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        self.generate_visualizations(output_dir)
        self.generate_report(output_path / "dataset_analysis_report.txt")
        
        logger.info("Dataset analysis complete!")
        return self.analysis_results

def main():
    """Main function to run the dataset analysis."""
    data_dir = '../ml_development/data/asl_train'
    output_dir = '../ml_development/analysis_output'
    
    analyzer = ASLDataAnalyzer(data_dir)
    results = analyzer.run_complete_analysis(output_dir)
    
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY YUH")
    print("="*50)
    print(f"Total images: {results['total_images']}")
    print(f"Number of classes: {results['num_classes']}")
    
    if 'image_quality' in results:
        quality = results['image_quality']
        print(f"Average image size: {quality['avg_width']:.0f}x{quality['avg_height']:.0f}")
        print(f"Corrupted images: {quality['corrupted_count']}")

if __name__ == "__main__":
    main()
