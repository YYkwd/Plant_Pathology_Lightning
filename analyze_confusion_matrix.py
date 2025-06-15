import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os
import matplotlib as mpl

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_and_prepare_data():
    # 加载原始标签和混合标签
    train_df = pd.read_csv('data/plant_pathodolgy_data/train.csv')
    mixed_df = pd.read_csv('data/plant_pathodolgy_data/soft_labels.csv')
    
    # 获取标签列
    label_columns = ['healthy', 'multiple_diseases', 'rust', 'scab']
    
    # 将概率转换为预测标签（使用0.5作为阈值）
    original_labels = train_df[label_columns].values
    mixed_labels = mixed_df[label_columns].values
    
    # 将概率转换为二进制标签
    original_pred = (original_labels > 0.5).astype(int)
    mixed_pred = (mixed_labels > 0.5).astype(int)
    
    return original_pred, mixed_pred, label_columns

def analyze_label_distribution(original_pred, mixed_pred, label_columns):
    print("\n标签分布分析:")
    print("\n原始标签分布:")
    for i, label in enumerate(label_columns):
        count = np.sum(original_pred[:, i])
        percentage = (count / len(original_pred)) * 100
        print(f"{label}: {count} 样本 ({percentage:.2f}%)")
    
    print("\n混合标签分布:")
    for i, label in enumerate(label_columns):
        count = np.sum(mixed_pred[:, i])
        percentage = (count / len(mixed_pred)) * 100
        print(f"{label}: {count} 样本 ({percentage:.2f}%)")

def analyze_label_changes(original_pred, mixed_pred, label_columns):
    print("\n标签变化分析:")
    for i, label in enumerate(label_columns):
        # 计算标签变化
        changes = np.sum(original_pred[:, i] != mixed_pred[:, i])
        total_samples = len(original_pred)
        change_percentage = (changes / total_samples) * 100
        
        # 计算具体变化类型
        positive_to_negative = np.sum((original_pred[:, i] == 1) & (mixed_pred[:, i] == 0))
        negative_to_positive = np.sum((original_pred[:, i] == 0) & (mixed_pred[:, i] == 1))
        
        print(f"\n{label}:")
        print(f"总变化: {changes} 样本 ({change_percentage:.2f}%)")
        print(f"从正变负: {positive_to_negative} 样本")
        print(f"从负变正: {negative_to_positive} 样本")

def analyze_label_combinations(original_pred, mixed_pred, label_columns):
    print("\n标签组合分析:")
    
    def get_label_combinations(pred):
        combinations = []
        for i in range(len(pred)):
            combo = tuple(pred[i])
            combinations.append(combo)
        return combinations
    
    original_combos = get_label_combinations(original_pred)
    mixed_combos = get_label_combinations(mixed_pred)
    
    # 统计原始标签组合
    original_combo_counts = pd.Series(original_combos).value_counts()
    print("\n原始标签组合分布:")
    for combo, count in original_combo_counts.items():
        combo_str = " + ".join([label_columns[i] for i, val in enumerate(combo) if val == 1])
        print(f"{combo_str}: {count} 样本")
    
    # 统计混合标签组合
    mixed_combo_counts = pd.Series(mixed_combos).value_counts()
    print("\n混合标签组合分布:")
    for combo, count in mixed_combo_counts.items():
        combo_str = " + ".join([label_columns[i] for i, val in enumerate(combo) if val == 1])
        print(f"{combo_str}: {count} 样本")

def plot_correlation_matrix(original_pred, mixed_pred, label_columns, save_dir):
    # 创建标签相关性矩阵
    plt.figure(figsize=(15, 6))
    
    # 原始标签的相关性矩阵
    plt.subplot(1, 2, 1)
    original_corr = np.corrcoef(original_pred.T)
    sns.heatmap(original_corr, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm',
                xticklabels=label_columns,
                yticklabels=label_columns)
    plt.title('原始标签相关性矩阵')
    
    # 混合标签的相关性矩阵
    plt.subplot(1, 2, 2)
    mixed_corr = np.corrcoef(mixed_pred.T)
    sns.heatmap(mixed_corr, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm',
                xticklabels=label_columns,
                yticklabels=label_columns)
    plt.title('混合标签相关性矩阵')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'label_correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_label_distribution(original_pred, mixed_pred, label_columns, save_dir):
    # 创建标签分布条形图
    plt.figure(figsize=(15, 6))
    
    # 原始标签分布
    plt.subplot(1, 2, 1)
    original_counts = np.sum(original_pred, axis=0)
    sns.barplot(x=label_columns, y=original_counts)
    plt.title('原始标签分布')
    plt.xticks(rotation=45)
    plt.ylabel('样本数量')
    
    # 混合标签分布
    plt.subplot(1, 2, 2)
    mixed_counts = np.sum(mixed_pred, axis=0)
    sns.barplot(x=label_columns, y=mixed_counts)
    plt.title('混合标签分布')
    plt.xticks(rotation=45)
    plt.ylabel('样本数量')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'label_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_label_changes(original_pred, mixed_pred, label_columns, save_dir):
    # 创建标签变化热力图
    changes = np.zeros((len(label_columns), len(label_columns)))
    
    for i in range(len(label_columns)):
        for j in range(len(label_columns)):
            # 计算从i类别变到j类别的样本数
            changes[i, j] = np.sum((original_pred[:, i] == 1) & (mixed_pred[:, j] == 1))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(changes, 
                annot=True, 
                fmt='.0f', 
                cmap='YlOrRd',
                xticklabels=label_columns,
                yticklabels=label_columns)
    plt.title('标签变化热力图')
    plt.xlabel('混合标签')
    plt.ylabel('原始标签')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'label_changes_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 创建保存目录
    save_dir = 'analysis_results'
    ensure_dir(save_dir)
    
    # 加载数据
    original_pred, mixed_pred, label_columns = load_and_prepare_data()
    
    # 分析标签分布
    analyze_label_distribution(original_pred, mixed_pred, label_columns)
    
    # 分析标签变化
    analyze_label_changes(original_pred, mixed_pred, label_columns)
    
    # 分析标签组合
    analyze_label_combinations(original_pred, mixed_pred, label_columns)
    
    # 绘制相关性矩阵
    plot_correlation_matrix(original_pred, mixed_pred, label_columns, save_dir)
    
    # 绘制标签分布
    plot_label_distribution(original_pred, mixed_pred, label_columns, save_dir)
    
    # 绘制标签变化热力图
    plot_label_changes(original_pred, mixed_pred, label_columns, save_dir)
    
    print(f"可视化图表已保存到 {save_dir} 文件夹：")
    print("1. label_correlation_matrix.png - 标签相关性矩阵")
    print("2. label_distribution.png - 标签分布条形图")
    print("3. label_changes_heatmap.png - 标签变化热力图")

if __name__ == "__main__":
    main() 