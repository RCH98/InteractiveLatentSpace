# Databricks notebook source
# MAGIC %md
# MAGIC # Interactive Latent Space Exploration
# MAGIC 
# MAGIC This notebook implements the Interactive Latent Space method using **true labels** instead of XGBoost predictions, making it compatible with DataBricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Dependencies

# COMMAND ----------

# Install required packages if not already available
# MAGIC %pip install torch plotly shap scikit-learn

# COMMAND ----------

# Import required libraries
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
from sklearn.ensemble import RandomForestRegressor

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interactive Latent Space Class

# COMMAND ----------

class InteractiveLatentSpace:
    """
    Interactive Latent Space Exploration using CVAE and TRUE LABELS
    DataBricks-compatible implementation
    """
    
    def __init__(self, data, labels, features=None, latent_dim=2, hidden_layers=[26, 13]):
        self.data = data
        self.labels = np.array(labels)
        self.features = features if features is not None else list(data.columns)
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        
        # Convert to pandas if it's a Spark DataFrame
        if hasattr(data, 'toPandas'):
            self.data_pd = data.toPandas()
        else:
            self.data_pd = data.copy()
            
        self._preprocess_data()
        self._initialize_cvae()
        self._compute_latent_space()
        self._initialize_shap()
        
    def _preprocess_data(self):
        """Preprocess data and prepare for CVAE"""
        # Standardize features
        self.scaler = StandardScaler()
        self.data_scaled = self.scaler.fit_transform(self.data_pd[self.features])
        
        # Combine features with TRUE LABELS (not predictions)
        self.X_with_labels = np.hstack((self.data_scaled, self.labels.reshape(-1, 1)))
        
        # Split into train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_with_labels, self.labels, test_size=0.2, random_state=42
        )
        
        self.X_train_original = self.data_scaled[:len(self.X_train)]
        self.X_test_original = self.data_scaled[len(self.X_train):]
        
    def _initialize_cvae(self):
        """Initialize Conditional Variational Autoencoder"""
        class FFNN_CVAE(nn.Module):
            def __init__(self, input_shape, hidden, latent_dim=2):
                super(FFNN_CVAE, self).__init__()
                
                # Encoding
                self.fc1 = nn.Linear(input_shape, hidden[0])
                self.fc2 = nn.Linear(hidden[0], hidden[1])
                self.fc3_mu = nn.Linear(hidden[1], latent_dim)
                self.fc3_logvar = nn.Linear(hidden[1], latent_dim)
                
                # Decoding
                self.fc4 = nn.Sequential(
                    nn.Linear(latent_dim + 1, hidden[1]),  # +1 for labels
                    nn.ReLU(inplace=True)
                )
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
        
        self._train_cvae()
        
    def _train_cvae(self, epochs=100, lr=0.001):
        """Train the CVAE model"""
        optimizer = torch.optim.Adam(self.cvae.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.cvae.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            z, mu, logvar = self.cvae(torch.tensor(self.X_train, dtype=torch.float32))
            recon = self.cvae.decode(z, torch.tensor(self.y_train, dtype=torch.float32).reshape(-1, 1))
            
            recon_loss = criterion(recon, torch.tensor(self.X_train, dtype=torch.float32))
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + 0.1 * kl_loss
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.cvae.eval()
        
    def _compute_latent_space(self):
        """Compute latent space representation"""
        with torch.no_grad():
            z_train, _, _ = self.cvae(torch.tensor(self.X_train, dtype=torch.float32))
            z_test, _, _ = self.cvae(torch.tensor(self.X_test, dtype=torch.float32))
            
            self.z_train = z_train.numpy()
            self.z_test = z_test.numpy()
            
            # Compute bounds for plotting
            self.x_min, self.x_max = self.z_test[:, 0].min(), self.z_test[:, 0].max()
            self.y_min, self.y_max = self.z_test[:, 1].min(), self.z_test[:, 1].max()
            
    def _initialize_shap(self):
        """Initialize SHAP explainer"""
        # Train surrogate model to predict latent space coordinates
        self.surrogate_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.surrogate_model.fit(self.X_train_original, self.z_train[:, 0])
        
        self.explainer = shap.TreeExplainer(self.surrogate_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualization Methods

# COMMAND ----------

# Add visualization methods to the class
def create_latent_space_plot(self, selected_point_idx=None):
    """Create main latent space visualization"""
    fig = go.Figure()
    
    # Training points
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
    
    # Test points
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
    
    # Highlight selected point
    if selected_point_idx is not None and selected_point_idx < len(self.z_train):
        fig.add_trace(go.Scatter(
            x=[self.z_train[selected_point_idx, 0]],
            y=[self.z_train[selected_point_idx, 1]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star', line=dict(width=2, color='black')),
            name='Selected Point'
        ))
    
    fig.update_layout(
        title="Interactive Latent Space Exploration (True Labels)",
        xaxis_title="Latent Dimension 1",
        yaxis_title="Latent Dimension 2",
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig

def create_feature_importance_plot(self, selected_point_idx=0):
    """Create feature importance visualization"""
    if selected_point_idx >= len(self.X_train_original):
        raise ValueError("Selected point index out of range")
        
    shap_values = self.explainer.shap_values(
        self.X_train_original[selected_point_idx:selected_point_idx+1]
    )
    
    feature_importance = np.abs(shap_values[0])
    sorted_indices = np.argsort(feature_importance)[::-1]
    top_features = sorted_indices[:15]
    
    fig = go.Figure()
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

# Add methods to class
InteractiveLatentSpace.create_latent_space_plot = create_latent_space_plot
InteractiveLatentSpace.create_feature_importance_plot = create_feature_importance_plot

# COMMAND ----------

# MAGIC %md
# MAGIC ## DataBricks Demo

# COMMAND ----------

# Generate sample data (replace with your actual data)
np.random.seed(42)
n_samples = 1000
n_features = 20

# Create synthetic features
X = np.random.randn(n_samples, n_features)

# Create synthetic labels (binary classification) - these are your TRUE LABELS
y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)

# Create DataFrame
feature_names = [f'feature_{i}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)

print("=== Data Overview ===")
print(f"Data shape: {df.shape}")
print(f"Labels distribution: {np.bincount(y)}")
print(f"Features: {feature_names[:5]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Interactive Latent Space

# COMMAND ----------

# Initialize Interactive Latent Space with TRUE LABELS
ils = InteractiveLatentSpace(
    data=df,
    labels=y,  # Using true labels instead of XGBoost predictions
    features=feature_names,
    latent_dim=2,
    hidden_layers=[20, 10]
)

print("=== Model Training Complete ===")
print(f"Latent space shape: {ils.z_train.shape}")
print(f"Training samples: {len(ils.y_train)}")
print(f"Test samples: {len(ils.y_test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Latent Space

# COMMAND ----------

# Create main latent space plot
main_plot = ils.create_latent_space_plot()

# Display in DataBricks
display(main_plot)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance Analysis

# COMMAND ----------

# Create feature importance plot for a sample point
sample_point_idx = 0
feature_plot = ils.create_feature_importance_plot(sample_point_idx)

# Display in DataBricks
display(feature_plot)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interactive Exploration with DataBricks Widgets

# COMMAND ----------

# Create DataBricks widgets for interactive exploration
dbutils.widgets.text("Point Index", "0", "Select Point to Analyze")
dbutils.widgets.slider("Latent Dimension 1", float(ils.x_min), float(ils.x_max), float(ils.x_min), 0.1)
dbutils.widgets.slider("Latent Dimension 2", float(ils.y_min), float(ils.y_max), float(ils.y_min), 0.1)

# COMMAND ----------

# Get widget values
point_idx = int(dbutils.widgets.get("Point Index"))
latent_dim1 = dbutils.widgets.get("Latent Dimension 1")
latent_dim2 = dbutils.widgets.get("Latent Dimension 2")

# Create updated plot with selected point
updated_plot = ils.create_latent_space_plot(point_idx)

# Display updated plot
display(updated_plot)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Model

# COMMAND ----------

# Save the trained model
model_path = "/dbfs/FileStore/models/interactive_latent_space_model.pt"
ils.save_model(model_path)
print(f"Model saved to: {model_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# MAGIC This implementation provides:
# MAGIC 
# MAGIC 1. **True Labels Integration**: Uses actual ground truth labels instead of XGBoost predictions
# MAGIC 2. **DataBricks Compatibility**: Works with DataBricks display() and widgets
# MAGIC 3. **Interactive Exploration**: Allows point selection and parameter tuning
# MAGIC 4. **CVAE-based Latent Space**: Generates meaningful latent representations
# MAGIC 5. **SHAP Explanations**: Provides feature importance analysis
# MAGIC 
# MAGIC To use with your data:
# MAGIC 1. Replace the synthetic data generation with your actual data
# MAGIC 2. Adjust the CVAE architecture if needed
# MAGIC 3. Use DataBricks widgets for interactive parameter tuning
# MAGIC 4. Save and load models as needed