import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from collections import Counter
import os

class DataAnalyzer:
    def __init__(self, csv_path, output_dir="analysis_results"):
        """
        Initialize the DataAnalyzer with the path to the training CSV file.
        
        Args:
            csv_path (str): Path to the training CSV file
            output_dir (str): Directory to save analysis results
        """
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.text_results = []
        
    def analyze_basic_info(self):
        """Analyze basic information about the dataset."""
        result = []
        result.append("=== Basic Dataset Information ===")
        result.append(f"Total number of samples: {len(self.df)}")
        result.append(f"Number of unique images: {self.df['image_id'].nunique()}")
        result.append("\nColumns in the dataset:")
        for col in self.df.columns:
            result.append(f"- {col}")
        print("\n".join(result))
        self.text_results.extend(result)
            
    def analyze_class_distribution(self):
        """Analyze the distribution of classes in the dataset."""
        result = []
        result.append("\n=== Class Distribution Analysis ===")
        class_counts = {}
        for col in self.df.columns:
            if col != 'image_id':
                class_counts[col] = self.df[col].sum()
        result.append("\nClass Distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / len(self.df)) * 100
            result.append(f"{class_name}: {count} samples ({percentage:.2f}%)")
        print("\n".join(result))
        self.text_results.extend(result)
        # Plot class distribution
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
        plt.title('Class Distribution in Dataset')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png')
        plt.close()
        
    def analyze_multilabel_distribution(self):
        """Analyze the distribution of multilabel combinations."""
        result = []
        result.append("\n=== Multilabel Distribution Analysis ===")
        self.df['label_combination'] = self.df.apply(
            lambda row: '_'.join([col for col in self.df.columns if col != 'image_id' and row[col] == 1]),
            axis=1
        )
        combination_counts = self.df['label_combination'].value_counts()
        result.append("\nTop 10 most common label combinations:")
        for combo, count in combination_counts.head(10).items():
            percentage = (count / len(self.df)) * 100
            result.append(f"{combo}: {count} samples ({percentage:.2f}%)")
        print("\n".join(result))
        self.text_results.extend(result)
        # Plot top 10 combinations
        plt.figure(figsize=(12, 6))
        combination_counts.head(10).plot(kind='bar')
        plt.title('Top 10 Label Combinations')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'label_combinations.png')
        plt.close()
        
    def analyze_image_counts(self):
        """Analyze the number of labels per image."""
        result = []
        result.append("\n=== Image Label Count Analysis ===")
        label_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']
        label_counts = self.df[label_cols].astype(int).sum(axis=1)
        result.append(f"Average labels per image: {label_counts.mean():.2f}")
        result.append(f"Minimum labels per image: {label_counts.min()}")
        result.append(f"Maximum labels per image: {label_counts.max()}")
        print("\n".join(result))
        self.text_results.extend(result)
        # Plot distribution of label counts
        plt.figure(figsize=(10, 6))
        sns.histplot(label_counts, bins=range(int(label_counts.max()) + 2))
        plt.title('Distribution of Labels per Image')
        plt.xlabel('Number of Labels')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'labels_per_image.png')
        plt.close()
        
    def run_full_analysis(self):
        """Run all analysis functions and save results."""
        self.analyze_basic_info()
        self.analyze_class_distribution()
        self.analyze_multilabel_distribution()
        self.analyze_image_counts()
        # 保存文本分析结果
        with open(self.output_dir / 'analysis.txt', 'w', encoding='utf-8') as f:
            for line in self.text_results:
                f.write(line + '\n')
        print(f"\nAnalysis complete! Check the '{self.output_dir}' folder for results.")

if __name__ == "__main__":
    # Example usage
    analyzer = DataAnalyzer("data/plant_pathodolgy_data/train.csv")
    analyzer.run_full_analysis() 