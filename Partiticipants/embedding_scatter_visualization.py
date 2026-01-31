import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
class EmbeddingVisualizer:
    def __init__(self):
        self.scaler = StandardScaler()
    def extract_embeddings_resnet(self, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Extract embeddings using ResNet50 (2048-dim)"""
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1]) 
        model = model.to(device)
        model.eval()
        embeddings, labels = [], []
        with torch.no_grad():
            for images, lbls in dataloader:
                images = images.to(device)
                emb = model(images).squeeze().cpu().numpy()
                embeddings.append(emb)
                labels.extend(lbls.numpy())
        return np.vstack(embeddings), np.array(labels)
    def extract_embeddings_mobilenet(self, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Extract embeddings using MobileNetV2 (1280-dim)"""
        model = models.mobilenet_v2(pretrained=True)
        model.classifier = torch.nn.Identity()  
        model = model.to(device)
        model.eval()
        embeddings, labels = [], []
        with torch.no_grad():
            for images, lbls in dataloader:
                images = images.to(device)
                emb = model(images).cpu().numpy()
                embeddings.append(emb)
                labels.extend(lbls.numpy())
        return np.vstack(embeddings), np.array(labels)
    def plot_2d_scatter(self, embeddings, labels, title="2D PCA Scatter", class_names=None):
        scaled = self.scaler.fit_transform(embeddings)
        pca = PCA(n_components=2).fit(scaled)
        reduced = pca.transform(scaled)
        plt.figure(figsize=(12, 9))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, 
                            cmap='viridis', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        cbar = plt.colorbar(scatter, label='Class')
        if class_names:
            cbar.set_ticks(range(len(class_names)))
            cbar.set_ticklabels(class_names)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'{title}\nTotal variance explained: {sum(pca.explained_variance_ratio_):.2%}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt, pca.explained_variance_ratio_
    def plot_3d_scatter(self, embeddings, labels, title="3D PCA Scatter", class_names=None):
        scaled = self.scaler.fit_transform(embeddings)
        pca = PCA(n_components=3).fit(scaled)
        reduced = pca.transform(scaled)
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], 
                           c=labels, cmap='viridis', s=50, alpha=0.7, 
                           edgecolors='black', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax, label='Class', shrink=0.8)
        if class_names:
            cbar.set_ticks(range(len(class_names)))
            cbar.set_ticklabels(class_names)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
        total_var = sum(pca.explained_variance_ratio_)
        ax.set_title(f'{title}\nTotal variance explained: {total_var:.2%}')
        return fig, pca.explained_variance_ratio_
    def compare_dimensions(self, embeddings_list, labels, class_names=None):
        """Compare different embedding dimensions side by side"""
        n_models = len(embeddings_list)
        fig, axes = plt.subplots(n_models, 2, figsize=(16, 6*n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)
        variance_summary = []
        for idx, (emb, dim, model_name) in enumerate(embeddings_list):
            scaled = self.scaler.fit_transform(emb)
            pca_2d = PCA(n_components=2).fit(scaled)
            reduced_2d = pca_2d.transform(scaled)
            scatter = axes[idx, 0].scatter(reduced_2d[:, 0], reduced_2d[:, 1], 
                                          c=labels, cmap='viridis', s=40, alpha=0.7)
            axes[idx, 0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})')
            axes[idx, 0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})')
            axes[idx, 0].set_title(f'{model_name} - 2D (Dim: {dim})')
            axes[idx, 0].grid(True, alpha=0.3)
            pca_3d = PCA(n_components=3).fit(scaled)
            reduced_3d = pca_3d.transform(scaled)
            axes[idx, 1].scatter(reduced_3d[:, 0], reduced_3d[:, 2], 
                               c=labels, cmap='viridis', s=40, alpha=0.7)
            axes[idx, 1].set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
            axes[idx, 1].set_ylabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
            axes[idx, 1].set_title(f'{model_name} - PC1 vs PC3 (Dim: {dim})')
            axes[idx, 1].grid(True, alpha=0.3)
            variance_summary.append({
                'model': model_name,
                'dim': dim,
                '2d_variance': sum(pca_2d.explained_variance_ratio_),
                '3d_variance': sum(pca_3d.explained_variance_ratio_)
            })
        plt.tight_layout()
        return fig, variance_summary
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    viz = EmbeddingVisualizer()
    print("Extracting ResNet50 embeddings (2048-dim)...")
    print("Extracting MobileNetV2 embeddings (1280-dim)...")
    print("\nGenerating 2D scatter plots...")
    print("\nGenerating 3D scatter plots...")
    print("\nComparing different embedding dimensions...")
