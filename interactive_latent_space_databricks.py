#!/usr/bin/env python
# coding: utf-8

"""
Interactive Latent Space Exploration for DataBricks
Modified to use true labels instead of XGBoost predictions
DataBricks-compatible implementation using Plotly and widgets
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# For DataBricks compatibility
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("PySpark not available, using pandas only")

class InteractiveLatentSpace:
    """
    Interactive Latent Space Exploration using CVAE and true labels
    DataBricks-compatible implementation
    """
    
    def __init__(self, data, labels, features=None, latent_dim=2, hidden_layers=[26, 13]):
        """
        Initialize the Interactive Latent Space
        
        Parameters:
        -----------
        data : pandas.DataFrame or pyspark.sql.DataFrame
            Input features data
        labels : array-like
            True labels (ground truth) instead of model predictions
        features : list, optional
            Feature names, if None will use data columns
        latent_dim : int, default=2
            Dimension of latent space
        hidden_layers : list, default=[26, 13]
            Hidden layer dimensions for CVAE
        """
        self.data = data
        self.labels = np.array(labels)
        self.features = features if features is not None else list(data.columns)
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        
        # Convert to pandas if it's a Spark DataFrame
        if SPARK_AVAILABLE and hasattr(data, 'toPandas'):
            self.data_pd = data.toPandas()
        else:
            self.data_pd = data.copy()
            
        # Preprocess data
        self._preprocess_data()
        
        # Initialize CVAE
        self._initialize_cvae()
        
        # Compute latent space
        self._compute_latent_space()
        
        # Initialize SHAP explainer
        self._initialize_shap()
        
    def _preprocess_data(self):
        """Preprocess the data and prepare for CVAE"""
        # Standardize features
        self.scaler = StandardScaler()
        self.data_scaled = self.scaler.fit_transform(self.data_pd[self.features])
        
        # Combine features with true labels (not predictions)
        self.X_with_labels = np.hstack((self.data_scaled, self.labels.reshape(-1, 1)))
        
        # Split into train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_with_labels, self.labels, test_size=0.2, random_state=42
        )
        
        # Store original data for reference
        self.X_train_original = self.data_scaled[:len(self.X_train)]
        self.X_test_original = self.data_scaled[len(self.X_train):]
        
    def _initialize_cvae(self):
        """Initialize the Conditional Variational Autoencoder"""
        class FFNN_CVAE(nn.Module):
            def __init__(self, input_shape, hidden, latent_dim=2):
                super(FFNN_CVAE, self).__init__()
                
                # Encoding components
                self.fc1 = nn.Linear(input_shape, hidden[0])
                self.fc2 = nn.Linear(hidden[0], hidden[1])
                
                # Latent vectors mu and sigma
                self.fc3_mu = nn.Linear(hidden[1], latent_dim)
                self.fc3_logvar = nn.Linear(hidden[1], latent_dim)
                
                # Sampling vector
                self.fc4 = nn.Sequential(
                    nn.Linear(latent_dim + 1, hidden[1]),
                    nn.ReLU(inplace=True)
                )
                
                # Decoder
                self.fc5 = nn.Sequential(
                    nn.Linear(hidden[1], hidden[0]),
                    nn.ReLU(inplace=True)
                )
                
                self.fc6 = nn.Linear(hidden[0], input_shape)
                
            def encode(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
                return mu, logvar
                
            def reparameterize(self, mu, logvar):
                if self.training:
                    std = logvar.mul(0.5).exp_()
                    eps = Variable(std.data.new(std.size()).normal_())
                    return eps.mul(std).add_(mu)
                else:
                    return mu
                    
            def decode(self, z, y):
                x = torch.cat((z, y), dim=1)
                x = self.fc4(x)
                x = self.fc5(x)
                x = self.fc6(x)
                return x
                
            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return z, mu, logvar
        
        # Create CVAE model
        self.cvae = FFNN_CVAE(
            input_shape=len(self.features) + 1,  # +1 for labels
            hidden=self.hidden_layers,
            latent_dim=self.latent_dim
        )
        
        # Train the CVAE (you can load pre-trained weights here)
        self._train_cvae()
        
    def _train_cvae(self, epochs=100, lr=0.001):
        """Train the CVAE model"""
        # This is a simplified training - in practice you'd want more sophisticated training
        optimizer = torch.optim.Adam(self.cvae.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.cvae.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            z, mu, logvar = self.cvae(torch.tensor(self.X_train, dtype=torch.float32))
            
            # Reconstruction loss
            recon = self.cvae.decode(z, torch.tensor(self.y_train, dtype=torch.float32).reshape(-1, 1))
            recon_loss = criterion(recon, torch.tensor(self.X_train, dtype=torch.float32))
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Total loss
            loss = recon_loss + 0.1 * kl_loss
            
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.cvae.eval()
        
    def _compute_latent_space(self):
        """Compute the latent space representation"""
        with torch.no_grad():
            # Get latent representations for training data
            z_train, _, _ = self.cvae(torch.tensor(self.X_train, dtype=torch.float32))
            z_test, _, _ = self.cvae(torch.tensor(self.X_test, dtype=torch.float32))
            
            self.z_train = z_train.numpy()
            self.z_test = z_test.numpy()
            
            # Compute bounds for plotting
            self.x_min, self.x_max = self.z_test[:, 0].min(), self.z_test[:, 0].max()
            self.y_min, self.y_max = self.z_test[:, 1].min(), self.z_test[:, 1].max()
            
    def _initialize_shap(self):
        """Initialize SHAP explainer for feature importance"""
        # Create a simple surrogate model for SHAP explanations
        from sklearn.ensemble import RandomForestRegressor
        
        # Train surrogate model to predict latent space coordinates
        self.surrogate_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.surrogate_model.fit(self.X_train_original, self.z_train[:, 0])  # Predict first latent dimension
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.surrogate_model)
        
    def create_latent_space_plot(self, selected_point_idx=None):
        """
        Create the main latent space visualization
        
        Parameters:
        -----------
        selected_point_idx : int, optional
            Index of selected point to highlight
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        # Create main plot
        fig = go.Figure()
        
        # Add training points
        fig.add_trace(go.Scatter(
            x=self.z_train[:, 0],
            y=self.z_train[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=self.y_train,
                colorscale='RdBu',
                opacity=0.7,
                colorbar=dict(title="True Labels")
            ),
            name='Training Data',
            text=[f'Point {i}<br>Label: {self.y_train[i]}' for i in range(len(self.y_train))],
            hoverinfo='text'
        ))
        
        # Add test points
        fig.add_trace(go.Scatter(
            x=self.z_test[:, 0],
            y=self.z_test[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=self.y_test,
                colorscale='RdBu',
                opacity=0.7,
                symbol='diamond'
            ),
            name='Test Data',
            text=[f'Test Point {i}<br>Label: {self.y_test[i]}' for i in range(len(self.y_test))],
            hoverinfo='text'
        ))
        
        # Highlight selected point if provided
        if selected_point_idx is not None:
            if selected_point_idx < len(self.z_train):
                fig.add_trace(go.Scatter(
                    x=[self.z_train[selected_point_idx, 0]],
                    y=[self.z_train[selected_point_idx, 1]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='star',
                        line=dict(width=2, color='black')
                    ),
                    name='Selected Point'
                ))
        
        # Update layout
        fig.update_layout(
            title="Interactive Latent Space Exploration (True Labels)",
            xaxis_title="Latent Dimension 1",
            yaxis_title="Latent Dimension 2",
            template='plotly_white',
            hovermode='closest'
        )
        
        return fig
    
    def create_feature_importance_plot(self, selected_point_idx=0):
        """
        Create feature importance visualization for a selected point
        
        Parameters:
        -----------
        selected_point_idx : int
            Index of the point to analyze
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        if selected_point_idx >= len(self.X_train_original):
            raise ValueError("Selected point index out of range")
            
        # Get SHAP values for the selected point
        shap_values = self.explainer.shap_values(
            self.X_train_original[selected_point_idx:selected_point_idx+1]
        )
        
        # Create feature importance plot
        fig = go.Figure()
        
        # Sort features by importance
        feature_importance = np.abs(shap_values[0])
        sorted_indices = np.argsort(feature_importance)[::-1]
        
        # Top 15 most important features
        top_features = sorted_indices[:15]
        
        fig.add_trace(go.Bar(
            x=[self.features[i] for i in top_features],
            y=feature_importance[top_features],
            marker_color='lightblue',
            name='Feature Importance'
        ))
        
        fig.update_layout(
            title=f"Feature Importance for Point {selected_point_idx}",
            xaxis_title="Features",
            yaxis_title="SHAP Value (Absolute)",
            template='plotly_white',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_clustering_analysis(self, cluster1_indices=None, cluster2_indices=None):
        """
        Create clustering analysis visualization
        
        Parameters:
        -----------
        cluster1_indices : array-like, optional
            Indices of first cluster
        cluster2_indices : array-like, optional
            Indices of second cluster
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Latent Space Clusters', 'Feature Distribution Comparison'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "violin", "colspan": 2}, None]]
        )
        
        # Latent space with clusters
        if cluster1_indices is not None:
            fig.add_trace(go.Scatter(
                x=self.z_train[cluster1_indices, 0],
                y=self.z_train[cluster1_indices, 1],
                mode='markers',
                marker=dict(size=10, color='red', opacity=0.8),
                name='Cluster 1'
            ), row=1, col=1)
            
        if cluster2_indices is not None:
            fig.add_trace(go.Scatter(
                x=self.z_train[cluster2_indices, 0],
                y=self.z_train[cluster2_indices, 1],
                mode='markers',
                marker=dict(size=10, color='blue', opacity=0.8),
                name='Cluster 2'
            ), row=1, col=1)
            
        # Add all points
        fig.add_trace(go.Scatter(
            x=self.z_train[:, 0],
            y=self.z_train[:, 1],
            mode='markers',
            marker=dict(size=6, color='lightgray', opacity=0.5),
            name='All Points'
        ), row=1, col=1)
        
        # Feature distribution comparison
        if cluster1_indices is not None and cluster2_indices is not None:
            # Compare distributions of top features
            top_features = self._get_top_separating_features(cluster1_indices, cluster2_indices, n_features=5)
            
            for i, feature_idx in enumerate(top_features):
                cluster1_values = self.X_train_original[cluster1_indices, feature_idx]
                cluster2_values = self.X_train_original[cluster2_indices, feature_idx]
                
                fig.add_trace(go.Violin(
                    y=cluster1_values,
                    name=f'{self.features[feature_idx]} - Cluster 1',
                    side='negative',
                    line_color='red'
                ), row=2, col=1)
                
                fig.add_trace(go.Violin(
                    y=cluster2_values,
                    name=f'{self.features[feature_idx]} - Cluster 2',
                    side='positive',
                    line_color='blue'
                ), row=2, col=1)
        
        fig.update_layout(
            title="Clustering Analysis",
            height=800,
            template='plotly_white'
        )
        
        return fig
    
    def _get_top_separating_features(self, cluster1_indices, cluster2_indices, n_features=5):
        """Get features that best separate two clusters"""
        cluster1_mean = np.mean(self.X_train_original[cluster1_indices], axis=0)
        cluster2_mean = np.mean(self.X_train_original[cluster2_indices], axis=0)
        
        # Calculate separation score
        separation_scores = np.abs(cluster1_mean - cluster2_mean)
        
        # Return top n features
        return np.argsort(separation_scores)[-n_features:][::-1]
    
    def explore_latent_space(self, n_points=5):
        """
        Interactive exploration of the latent space
        
        Parameters:
        -----------
        n_points : int
            Number of points to explore
            
        Returns:
        --------
        dict
            Dictionary containing all visualizations
        """
        # Create main latent space plot
        main_plot = self.create_latent_space_plot()
        
        # Create feature importance plots for sample points
        sample_indices = np.random.choice(len(self.X_train_original), n_points, replace=False)
        feature_plots = {}
        
        for idx in sample_indices:
            feature_plots[f'point_{idx}'] = self.create_feature_importance_plot(idx)
        
        # Create clustering analysis
        # Randomly create two clusters for demonstration
        n_samples = len(self.X_train_original)
        cluster1_size = n_samples // 3
        cluster1_indices = np.random.choice(n_samples, cluster1_size, replace=False)
        remaining_indices = np.setdiff1d(np.arange(n_samples), cluster1_indices)
        cluster2_indices = np.random.choice(remaining_indices, cluster1_size, replace=False)
        
        clustering_plot = self.create_clustering_analysis(cluster1_indices, cluster2_indices)
        
        return {
            'main_latent_space': main_plot,
            'feature_importance_plots': feature_plots,
            'clustering_analysis': clustering_plot,
            'sample_points': sample_indices,
            'cluster1_indices': cluster1_indices,
            'cluster2_indices': cluster2_indices
        }
    
    def save_model(self, filepath):
        """Save the trained CVAE model"""
        torch.save(self.cvae.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained CVAE model"""
        self.cvae.load_state_dict(torch.load(filepath))
        self.cvae.eval()
        print(f"Model loaded from {filepath}")


# Example usage for DataBricks
def create_databricks_demo():
    """
    Create a demo for DataBricks usage
    """
    # Generate sample data (replace with your actual data)
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create synthetic features
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic labels (binary classification)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Initialize Interactive Latent Space
    ils = InteractiveLatentSpace(
        data=df,
        labels=y,
        features=feature_names,
        latent_dim=2,
        hidden_layers=[20, 10]
    )
    
    # Explore the latent space
    results = ils.explore_latent_space(n_points=3)
    
    # Display results in DataBricks
    print("=== Interactive Latent Space Exploration ===")
    print(f"Data shape: {df.shape}")
    print(f"Labels distribution: {np.bincount(y)}")
    print(f"Latent space shape: {ils.z_train.shape}")
    
    # In DataBricks, you would display these plots using:
    # display(results['main_latent_space'])
    # display(results['clustering_analysis'])
    
    return ils, results


if __name__ == "__main__":
    # Run demo
    ils, results = create_databricks_demo()
    
    # Save model
    ils.save_model('./models/interactive_latent_space_model.pt')
    
    print("Demo completed successfully!")
    print("To use in DataBricks:")
    print("1. Upload this script to your DataBricks workspace")
    print("2. Run the create_databricks_demo() function")
    print("3. Use display() to show the Plotly figures")
    print("4. Use DataBricks widgets for interactive parameter tuning")