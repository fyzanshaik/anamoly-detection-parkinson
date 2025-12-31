import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv('../datasets/combined_dataset.csv')

plt.style.use('bmh')

def plot_feature_clusters():
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(
        data=df,
        x='dominant_freq', 
        y='harmonic_ratio',
        hue='fault_type',
        palette={'normal': 'green', 'bearing_fault': 'red', 'rotor_imbalance': 'blue'},
        alpha=0.6,
        s=80
    )
    
    plt.title('Feature Space Separation (XAI Basis)', fontsize=14, fontweight='bold')
    plt.xlabel('Dominant Frequency (Hz)', fontsize=12)
    plt.ylabel('Harmonic Ratio', fontsize=12)
    plt.legend(title='Machine State', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('../datasets/feature_clusters.png', dpi=300)
    plt.close()

def plot_feature_distributions():
    features = ['rms', 'kurtosis', 'peak', 'energy']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.boxplot(
            data=df,
            x='fault_type',
            y=feature,
            ax=axes[i],
            palette={'normal': 'lightgreen', 'bearing_fault': 'salmon', 'rotor_imbalance': 'lightblue'}
        )
        axes[i].set_title(f'Distribution of {feature.upper()}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')
        
    plt.tight_layout()
    plt.savefig('../datasets/feature_distributions.png', dpi=300)
    plt.close()

def plot_pca_separation():
    features = ['mean', 'peak', 'rms', 'skewness', 'kurtosis', 'dominant_freq', 'harmonic_ratio', 'energy']
    x = df[features].values
    x = StandardScaler().fit_transform(x)
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    
    pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    pca_df['fault_type'] = df['fault_type']
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=pca_df,
        x='PC1',
        y='PC2',
        hue='fault_type',
        palette={'normal': 'green', 'bearing_fault': 'red', 'rotor_imbalance': 'blue'},
        alpha=0.7,
        s=60
    )
    
    plt.title('PCA Projection: Class Separability', fontsize=14, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} Variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} Variance)', fontsize=12)
    plt.legend(title='Fault Type')
    plt.tight_layout()
    plt.savefig('../datasets/pca_projection.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_feature_clusters()
    plot_feature_distributions()
    plot_pca_separation()
