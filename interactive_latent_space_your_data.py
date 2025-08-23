#!/usr/bin/env python
# coding: utf-8

"""
Interactive Latent Space Exploration for Your Data
Optimized for 356 features with imbalanced labels (0.5% anomalies)
DataBricks-compatible implementation
"""

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.class_weight import compute_class_weight

class InteractiveLatentSpace:
    """
    Interactive Latent Space Exploration using CVAE and true labels
    Optimized for your specific data: 356 features, imbalanced labels
    """
    
    def __init__(self, X_train_scaled, X_test_scaled, y_train, y_test, feature_names, 
                 latent_dim=2, hidden_layers=[200, 100, 50]):
        """
        Initialize the Interactive Latent Space
        
        Parameters:
        -----------
        X_train_scaled : numpy.ndarray
            Pre-scaled training features (2089, 356)
        X_test_scaled : numpy.ndarray
            Pre-scaled test features (523, 356)
        y_train : numpy.ndarray
            Training labels (2089,)
        y_test : numpy.ndarray
            Test labels (523,)
        feature_names : list
            List of feature names (356 features)
        latent_dim : int, default=2
            Dimension of latent space
        hidden_layers : list, default=[200, 100, 50]
            Hidden layer dimensions for CVAE (optimized for 356 features)
        """
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        self.feature_names = feature_names
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        
        # Data shapes
        self.n_train = X_train_scaled.shape[0]
        self.n_test = X_test_scaled.shape[0]
        self.n_features = X_train_scaled.shape[1]
        
        print(f"=== Data Overview ===")
        print(f"Training samples: {self.n_train}")
        print(f"Test samples: {self.n_test}")
        print(f"Features: {self.n_features}")
        print(f"Anomaly rate (train): {self.y_train.mean():.4f}")
        print(f"Anomaly rate (test): {self.y_test.mean():.4f}")
        
        # Prepare data for CVAE (combine features with labels)
        self.X_train_cvae = np.hstack((X_train_scaled, y_train.reshape(-1, 1)))
        self.X_test_cvae = np.hstack((X_test_scaled, y_test.reshape(-1, 1)))
        
        # Initialize CVAE
        self._initialize_cvae()
        
        # Compute latent space
        self._compute_latent_space()
        
        # Initialize SHAP explainer
        self._initialize_shap()
        
    def _initialize_cvae(self):
        """Initialize the Conditional Variational Autoencoder"""
        class FFNN_CVAE(nn.Module):
            def __init__(self, input_shape, hidden, latent_dim=2):
                super(FFNN_CVAE, self).__init__()
                
                # Encoding components - optimized for 356 features
                layers = []
                prev_dim = input_shape
                
                for hidden_dim in hidden:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2)
                    ])
                    prev_dim = hidden_dim
                
                self.encoder = nn.Sequential(*layers)
                
                # Latent vectors mu and sigma
                self.fc_mu = nn.Linear(hidden[-1], latent_dim)
                self.fc_logvar = nn.Linear(hidden[-1], latent_dim)
                
                # Decoder - reverse architecture
                decoder_layers = []
                prev_dim = latent_dim + 1  # +1 for labels
                
                for hidden_dim in reversed(hidden):
                    decoder_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2)
                    ])
                    prev_dim = hidden_dim
                
                decoder_layers.append(nn.Linear(hidden[0], input_shape))
                self.decoder = nn.Sequential(*decoder_layers)
                
            def encode(self, x):
                x = self.encoder(x)
                mu, logvar = self.fc_mu(x), self.fc_logvar(x)
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
                return self.decoder(x)
                
            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return z, mu, logvar
        
        # Create CVAE model
        self.cvae = FFNN_CVAE(
            input_shape=self.n_features + 1,  # +1 for labels
            hidden=self.hidden_layers,
            latent_dim=self.latent_dim
        )
        
        # Train the CVAE
        self._train_cvae()
        
    def _train_cvae(self, epochs=200, lr=0.001, batch_size=64):
        """Train the CVAE model with handling for imbalanced data"""
        print("=== Training CVAE Model ===")
        
        # Handle class imbalance
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(self.y_train), 
            y=self.y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # Convert to tensors
        X_train_tensor = torch.tensor(self.X_train_cvae, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        
        # Data loader for batch training
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.cvae.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        self.cvae.train()
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                z, mu, logvar = self.cvae(batch_X)
                
                # Reconstruction loss
                recon = self.cvae.decode(z, batch_y.reshape(-1, 1))
                recon_loss = criterion(recon, batch_X)
                
                # KL divergence loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                # Total loss with KL annealing
                kl_weight = min(1.0, epoch / 50)  # Gradually increase KL weight
                loss = recon_loss + kl_weight * 0.1 * kl_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cvae.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
            
            avg_loss = total_loss / len(dataloader)
            avg_recon_loss = total_recon_loss / len(dataloader)
            avg_kl_loss = total_kl_loss / len(dataloader)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}, Loss: {avg_loss:.6f}, "
                      f"Recon: {avg_recon_loss:.6f}, KL: {avg_kl_loss:.6f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        self.cvae.eval()
        print("=== Training Complete ===")
        
    def _compute_latent_space(self):
        """Compute the latent space representation"""
        print("=== Computing Latent Space ===")
        
        with torch.no_grad():
            # Get latent representations
            z_train, _, _ = self.cvae(torch.tensor(self.X_train_cvae, dtype=torch.float32))
            z_test, _, _ = self.cvae(torch.tensor(self.X_test_cvae, dtype=torch.float32))
            
            self.z_train = z_train.numpy()
            self.z_test = z_test.numpy()
            
            # Compute bounds for plotting
            self.x_min, self.x_max = self.z_test[:, 0].min(), self.z_test[:, 0].max()
            self.y_min, self.y_max = self.z_test[:, 1].min(), self.z_test[:, 1].max()
            
            print(f"Latent space computed:")
            print(f"  Training shape: {self.z_train.shape}")
            print(f"  Test shape: {self.z_test.shape}")
            print(f"  Bounds: X[{self.x_min:.3f}, {self.x_max:.3f}], Y[{self.y_min:.3f}, {self.y_max:.3f}]")
            
    def _initialize_shap(self):
        """Initialize SHAP explainer for feature importance"""
        print("=== Initializing SHAP Explainer ===")
        
        # Create surrogate model to predict latent space coordinates
        self.surrogate_model = RandomForestRegressor(
            n_estimators=200, 
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Train on first latent dimension
        self.surrogate_model.fit(self.X_train_scaled, self.z_train[:, 0])
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.surrogate_model)
        print("SHAP explainer initialized")
        
    def create_latent_space_plot(self, selected_point_idx=None, highlight_anomalies=True):
        """
        Create the main latent space visualization
        
        Parameters:
        -----------
        selected_point_idx : int, optional
            Index of selected point to highlight
        highlight_anomalies : bool, default=True
            Whether to highlight anomaly points differently
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        fig = go.Figure()
        
        # Training points - separate normal and anomaly
        normal_mask = self.y_train == 0
        anomaly_mask = self.y_train == 1
        
        # Normal training points
        fig.add_trace(go.Scatter(
            x=self.z_train[normal_mask, 0],
            y=self.z_train[normal_mask, 1],
            mode='markers',
            marker=dict(
                size=6,
                color='blue',
                opacity=0.6,
                symbol='circle'
            ),
            name='Normal (Train)',
            text=[f'Point {i}<br>Label: Normal' for i in np.where(normal_mask)[0]],
            hoverinfo='text'
        ))
        
        # Anomaly training points
        if anomaly_mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=self.z_train[anomaly_mask, 0],
                y=self.z_train[anomaly_mask, 1],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    opacity=0.8,
                    symbol='x'
                ),
                name='Anomaly (Train)',
                text=[f'Point {i}<br>Label: Anomaly' for i in np.where(anomaly_mask)[0]],
                hoverinfo='text'
            ))
        
        # Test points - separate normal and anomaly
        normal_test_mask = self.y_test == 0
        anomaly_test_mask = self.y_test == 1
        
        # Normal test points
        fig.add_trace(go.Scatter(
            x=self.z_test[normal_test_mask, 0],
            y=self.z_test[normal_test_mask, 1],
            mode='markers',
            marker=dict(
                size=6,
                color='lightblue',
                opacity=0.6,
                symbol='diamond'
            ),
            name='Normal (Test)',
            text=[f'Test Point {i}<br>Label: Normal' for i in np.where(normal_test_mask)[0]],
            hoverinfo='text'
        ))
        
        # Anomaly test points
        if anomaly_test_mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=self.z_test[anomaly_test_mask, 0],
                y=self.z_test[anomaly_test_mask, 1],
                mode='markers',
                marker=dict(
                    size=10,
                    color='orange',
                    opacity=0.8,
                    symbol='diamond'
                ),
                name='Anomaly (Test)',
                text=[f'Test Point {i}<br>Label: Anomaly' for i in np.where(anomaly_test_mask)[0]],
                hoverinfo='text'
            ))
        
        # Highlight selected point if provided
        if selected_point_idx is not None:
            if selected_point_idx < len(self.z_train):
                point_type = "Anomaly" if self.y_train[selected_point_idx] == 1 else "Normal"
                fig.add_trace(go.Scatter(
                    x=[self.z_train[selected_point_idx, 0]],
                    y=[self.z_train[selected_point_idx, 1]],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='yellow',
                        symbol='star',
                        line=dict(width=3, color='black')
                    ),
                    name=f'Selected Point ({point_type})'
                ))
        
        # Update layout
        fig.update_layout(
            title="Interactive Latent Space Exploration - Your Data (356 Features)",
            xaxis_title="Latent Dimension 1",
            yaxis_title="Latent Dimension 2",
            template='plotly_white',
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig
    
    def create_feature_importance_plot(self, selected_point_idx=0, top_n=20):
        """
        Create feature importance visualization for a selected point
        
        Parameters:
        -----------
        selected_point_idx : int
            Index of the point to analyze
        top_n : int, default=20
            Number of top features to show
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        if selected_point_idx >= len(self.X_train_scaled):
            raise ValueError("Selected point index out of range")
            
        # Get SHAP values for the selected point
        shap_values = self.explainer.shap_values(
            self.X_train_scaled[selected_point_idx:selected_point_idx+1]
        )
        
        # Create feature importance plot
        feature_importance = np.abs(shap_values[0])
        sorted_indices = np.argsort(feature_importance)[::-1]
        top_features = sorted_indices[:top_n]
        
        fig = go.Figure()
        
        # Color code based on importance
        colors = ['red' if i < 10 else 'orange' if i < 20 else 'blue' for i in range(len(top_features))]
        
        fig.add_trace(go.Bar(
            x=[self.feature_names[i] for i in top_features],
            y=feature_importance[top_features],
            marker_color=colors,
            name='Feature Importance'
        ))
        
        point_type = "Anomaly" if self.y_train[selected_point_idx] == 1 else "Normal"
        fig.update_layout(
            title=f"Top {top_n} Feature Importance for Point {selected_point_idx} ({point_type})",
            xaxis_title="Features",
            yaxis_title="SHAP Value (Absolute)",
            template='plotly_white',
            xaxis_tickangle=-45,
            height=600
        )
        
        return fig
    
    def create_anomaly_analysis_plot(self):
        """
        Create specialized plot for anomaly analysis
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Latent Space - Training Data',
                'Latent Space - Test Data', 
                'Feature Distribution: Normal vs Anomaly',
                'Anomaly Detection Performance'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "violin"}, {"type": "bar"}]]
        )
        
        # Training data
        normal_mask = self.y_train == 0
        anomaly_mask = self.y_train == 1
        
        fig.add_trace(go.Scatter(
            x=self.z_train[normal_mask, 0],
            y=self.z_train[normal_mask, 1],
            mode='markers',
            marker=dict(size=6, color='blue', opacity=0.6),
            name='Normal (Train)',
            showlegend=False
        ), row=1, col=1)
        
        if anomaly_mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=self.z_train[anomaly_mask, 0],
                y=self.z_train[anomaly_mask, 1],
                mode='markers',
                marker=dict(size=8, color='red', opacity=0.8),
                name='Anomaly (Train)',
                showlegend=False
            ), row=1, col=1)
        
        # Test data
        normal_test_mask = self.y_test == 0
        anomaly_test_mask = self.y_test == 1
        
        fig.add_trace(go.Scatter(
            x=self.z_test[normal_test_mask, 0],
            y=self.z_test[normal_test_mask, 1],
            mode='markers',
            marker=dict(size=6, color='lightblue', opacity=0.6),
            name='Normal (Test)',
            showlegend=False
        ), row=1, col=2)
        
        if anomaly_test_mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=self.z_test[anomaly_test_mask, 0],
                y=self.z_test[anomaly_test_mask, 1],
                mode='markers',
                marker=dict(size=8, color='orange', opacity=0.8),
                name='Anomaly (Test)',
                showlegend=False
            ), row=1, col=2)
        
        # Feature distribution comparison (top separating feature)
        if anomaly_mask.sum() > 0:
            # Find feature that best separates normal vs anomaly
            normal_mean = np.mean(self.X_train_scaled[normal_mask], axis=0)
            anomaly_mean = np.mean(self.X_train_scaled[anomaly_mask], axis=0)
            separation_scores = np.abs(normal_mean - anomaly_mean)
            best_feature_idx = np.argmax(separation_scores)
            
            normal_values = self.X_train_scaled[normal_mask, best_feature_idx]
            anomaly_values = self.X_train_scaled[anomaly_mask, best_feature_idx]
            
            fig.add_trace(go.Violin(
                y=normal_values,
                name=f'{self.feature_names[best_feature_idx]} - Normal',
                side='negative',
                line_color='blue'
            ), row=2, col=1)
            
            fig.add_trace(go.Violin(
                y=anomaly_values,
                name=f'{self.feature_names[best_feature_idx]} - Anomaly',
                side='positive',
                line_color='red'
            ), row=2, col=1)
        
        # Anomaly statistics
        train_anomaly_rate = self.y_train.mean()
        test_anomaly_rate = self.y_test.mean()
        
        fig.add_trace(go.Bar(
            x=['Training', 'Test'],
            y=[train_anomaly_rate, test_anomaly_rate],
            marker_color=['red', 'orange'],
            name='Anomaly Rate'
        ), row=2, col=2)
        
        fig.update_layout(
            title="Anomaly Analysis Dashboard",
            height=800,
            template='plotly_white'
        )
        
        return fig
    
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
        print("=== Creating Interactive Visualizations ===")
        
        # Create main latent space plot
        main_plot = self.create_latent_space_plot()
        
        # Create anomaly analysis plot
        anomaly_plot = self.create_anomaly_analysis_plot()
        
        # Create feature importance plots for sample points
        # Include both normal and anomaly points
        normal_indices = np.where(self.y_train == 0)[0]
        anomaly_indices = np.where(self.y_train == 1)[0]
        
        sample_indices = []
        if len(normal_indices) > 0:
            sample_indices.extend(np.random.choice(normal_indices, min(n_points//2, len(normal_indices)), replace=False))
        if len(anomaly_indices) > 0:
            sample_indices.extend(np.random.choice(anomaly_indices, min(n_points//2, len(anomaly_indices)), replace=False))
        
        # Fill remaining slots with random points
        remaining_slots = n_points - len(sample_indices)
        if remaining_slots > 0:
            all_indices = np.arange(len(self.y_train))
            used_indices = set(sample_indices)
            available_indices = [i for i in all_indices if i not in used_indices]
            if len(available_indices) > 0:
                sample_indices.extend(np.random.choice(available_indices, min(remaining_slots, len(available_indices)), replace=False))
        
        feature_plots = {}
        for idx in sample_indices:
            feature_plots[f'point_{idx}'] = self.create_feature_importance_plot(idx)
        
        return {
            'main_latent_space': main_plot,
            'anomaly_analysis': anomaly_plot,
            'feature_importance_plots': feature_plots,
            'sample_points': sample_indices,
            'data_summary': {
                'n_features': self.n_features,
                'n_train': self.n_train,
                'n_test': self.n_test,
                'train_anomaly_rate': self.y_train.mean(),
                'test_anomaly_rate': self.y_test.mean()
            }
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


# Example usage with your data
def create_interactive_latent_space_for_your_data(X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
    """
    Create Interactive Latent Space for your specific data
    
    Parameters:
    -----------
    X_train_scaled : numpy.ndarray (2089, 356)
        Your pre-scaled training features
    X_test_scaled : numpy.ndarray (523, 356)
        Your pre-scaled test features  
    y_train : numpy.ndarray (2089,)
        Your training labels
    y_test : numpy.ndarray (523,)
        Your test labels
    feature_names : list
        Your feature names (356 features)
    """
    
    # Initialize Interactive Latent Space
    ils = InteractiveLatentSpace(
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        latent_dim=2,
        hidden_layers=[200, 100, 50]  # Optimized for 356 features
    )
    
    # Explore the latent space
    results = ils.explore_latent_space(n_points=5)
    
    # Display results
    print("\n=== Interactive Latent Space Ready ===")
    print(f"Data Summary: {results['data_summary']}")
    print(f"Sample points for analysis: {results['sample_points']}")
    
    # In DataBricks, you would display these plots using:
    # display(results['main_latent_space'])
    # display(results['anomaly_analysis'])
    
    return ils, results


if __name__ == "__main__":
    print("Interactive Latent Space for Your Data (356 Features)")
    print("To use:")
    print("1. Import this module in your DataBricks notebook")
    print("2. Call create_interactive_latent_space_for_your_data() with your data")
    print("3. Use display() to show the Plotly figures")
    print("4. Use DataBricks widgets for interactive exploration")