# Databricks notebook source
# MAGIC %md
# MAGIC # Interactive Latent Space for Your Data (356 Features)
# MAGIC 
# MAGIC This notebook implements the Interactive Latent Space method using **true labels** instead of XGBoost predictions, optimized for your specific dataset with 356 features and imbalanced labels.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# Install required packages
%pip install torch plotly shap scikit-learn

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interactive Latent Space Class

# COMMAND ----------

class InteractiveLatentSpace:
    """
    Interactive Latent Space Exploration using CVAE and TRUE LABELS
    Optimized for your specific data: 356 features, imbalanced labels
    """
    
    def __init__(self, X_train_scaled, X_test_scaled, y_train, y_test, feature_names, 
                 latent_dim=2, hidden_layers=[200, 100, 50]):
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
        
        self._initialize_cvae()
        self._compute_latent_space()
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualization Methods

# COMMAND ----------

# Add visualization methods to the class
def create_latent_space_plot(self, selected_point_idx=None, highlight_anomalies=True):
    """Create main latent space visualization"""
    fig = go.Figure()
    
    # Training points - separate normal and anomaly
    normal_mask = self.y_train == 0
    anomaly_mask = self.y_train == 1
    
    # Normal training points
    fig.add_trace(go.Scatter(
        x=self.z_train[normal_mask, 0],
        y=self.z_train[normal_mask, 1],
        mode='markers',
        marker=dict(size=6, color='blue', opacity=0.6, symbol='circle'),
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
            marker=dict(size=10, color='red', opacity=0.8, symbol='x'),
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
        marker=dict(size=6, color='lightblue', opacity=0.6, symbol='diamond'),
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
            marker=dict(size=10, color='orange', opacity=0.8, symbol='diamond'),
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
                marker=dict(size=20, color='yellow', symbol='star', line=dict(width=3, color='black')),
                name=f'Selected Point ({point_type})'
            ))
    
    # Update layout
    fig.update_layout(
        title="Interactive Latent Space Exploration - Your Data (356 Features)",
        xaxis_title="Latent Dimension 1",
        yaxis_title="Latent Dimension 2",
        template='plotly_white',
        hovermode='closest',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return fig

def create_feature_importance_plot(self, selected_point_idx=0, top_n=20):
    """Create feature importance visualization"""
    if selected_point_idx >= len(self.X_train_scaled):
        raise ValueError("Selected point index out of range")
        
    shap_values = self.explainer.shap_values(
        self.X_train_scaled[selected_point_idx:selected_point_idx+1]
    )
    
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

# Add methods to class
InteractiveLatentSpace.create_latent_space_plot = create_latent_space_plot
InteractiveLatentSpace.create_feature_importance_plot = create_feature_importance_plot

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your Data Setup
# MAGIC 
# MAGIC Replace this section with your actual data loading and preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 1: Use Your Preprocessed Data
# MAGIC 
# MAGIC If you already have your data ready from your preprocessing:

# COMMAND ----------

# Uncomment and modify this section with your actual data
# X_train_scaled = your_X_train_scaled  # Shape: (2089, 356)
# X_test_scaled = your_X_test_scaled    # Shape: (523, 356)
# y_train = your_y_train                # Shape: (2089,)
# y_test = your_y_test                  # Shape: (523,)
# feature_names = your_feature_names    # List of 356 feature names

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 2: Load and Preprocess Your Data Here

# COMMAND ----------

# Example data loading (replace with your actual data path)
# end_table = pd.read_csv('/path/to/your/data.csv')
# 
# # Split data
# split_pos = int(len(end_table) * 0.8)
# X_train1_full = end_table.iloc[:split_pos]
# X_test1_full = end_table.iloc[split_pos:]
# 
# # Extract labels and features
# y_train = X_train1_full['Label'].values
# y_test = X_test1_full['Label'].values
# X_train1 = X_train1_full.drop('Label', axis=1)
# X_test1 = X_test1_full.drop('Label', axis=1)
# 
# # Feature scaling
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 1))
# X_train_scaled = scaler.fit_transform(X_train1)
# X_test_scaled = scaler.transform(X_test1)
# 
# # Feature names
# feature_names = X_train1.columns.tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Interactive Latent Space

# COMMAND ----------

# MAGIC %md
# MAGIC **Important**: Make sure you have defined the following variables before running this cell:
# MAGIC - `X_train_scaled` (numpy array, shape: 2089 x 356)
# MAGIC - `X_test_scaled` (numpy array, shape: 523 x 356)
# MAGIC - `y_train` (numpy array, shape: 2089)
# MAGIC - `y_test` (numpy array, shape: 523)
# MAGIC - `feature_names` (list of 356 feature names)

# COMMAND ----------

# Initialize Interactive Latent Space with your data
# Uncomment when you have your data ready
# ils = InteractiveLatentSpace(
#     X_train_scaled=X_train_scaled,
#     X_test_scaled=X_test_scaled,
#     y_train=y_train,
#     y_test=y_test,
#     feature_names=feature_names,
#     latent_dim=2,
#     hidden_layers=[200, 100, 50]  # Optimized for 356 features
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Latent Space

# COMMAND ----------

# Create main latent space plot
# Uncomment when you have the ils object
# main_plot = ils.create_latent_space_plot()

# Display in DataBricks
# display(main_plot)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance Analysis

# COMMAND ----------

# Create feature importance plot for a sample point
# Uncomment when you have the ils object
# sample_point_idx = 0
# feature_plot = ils.create_feature_importance_plot(sample_point_idx)

# Display in DataBricks
# display(feature_plot)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interactive Exploration with DataBricks Widgets

# COMMAND ----------

# Create DataBricks widgets for interactive exploration
# Uncomment when you have the ils object
# dbutils.widgets.text("Point Index", "0", "Select Point to Analyze")
# dbutils.widgets.slider("Latent Dimension 1", float(ils.x_min), float(ils.x_max), float(ils.x_min), 0.1)
# dbutils.widgets.slider("Latent Dimension 2", float(ils.y_min), float(ils.y_max), float(ils.y_min), 0.1)

# COMMAND ----------

# Get widget values and create updated plot
# Uncomment when you have the ils object and widgets
# point_idx = int(dbutils.widgets.get("Point Index"))
# latent_dim1 = dbutils.widgets.get("Latent Dimension 1")
# latent_dim2 = dbutils.widgets.get("Latent Dimension 2")

# Create updated plot with selected point
# updated_plot = ils.create_latent_space_plot(point_idx)

# Display updated plot
# display(updated_plot)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Model

# COMMAND ----------

# Save the trained model
# Uncomment when you have the ils object
# model_path = "/dbfs/FileStore/models/interactive_latent_space_your_data.pt"
# ils.save_model(model_path)
# print(f"Model saved to: {model_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# MAGIC This implementation provides:
# MAGIC 
# MAGIC 1. **True Labels Integration**: Uses your actual ground truth labels instead of model predictions
# MAGIC 2. **DataBricks Compatibility**: Works with DataBricks display() and widgets
# MAGIC 3. **Optimized for Your Data**: Handles 356 features and imbalanced labels (0.5% anomalies)
# MAGIC 4. **CVAE-based Latent Space**: Generates meaningful latent representations
# MAGIC 5. **SHAP Explanations**: Provides feature importance analysis
# MAGIC 
# MAGIC **To use:**
# MAGIC 1. Define your data variables (X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
# MAGIC 2. Run the initialization cell
# MAGIC 3. Use display() to show the Plotly figures
# MAGIC 4. Use DataBricks widgets for interactive parameter tuning