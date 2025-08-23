#!/usr/bin/env python
# coding: utf-8

"""
Usage Example for Interactive Latent Space with Your Data
This shows exactly how to use the class with your 356-feature dataset
"""

# Import the class
from interactive_latent_space_your_data import InteractiveLatentSpace

# Assuming you have your data ready from your preprocessing:
# X_train_scaled, X_test_scaled, y_train, y_test, feature_names

def run_interactive_latent_space():
    """
    Run the Interactive Latent Space analysis with your data
    """
    
    # Your data (from your preprocessing)
    # X_train_scaled: (2089, 356) - MinMaxScaler scaled training features
    # X_test_scaled: (523, 356) - MinMaxScaler scaled test features  
    # y_train: (2089,) - Training labels (0=normal, 1=anomaly)
    # y_test: (523,) - Test labels (0=normal, 1=anomaly)
    # feature_names: list of 356 feature names
    
    print("=== Starting Interactive Latent Space Analysis ===")
    
    # Initialize the Interactive Latent Space
    ils = InteractiveLatentSpace(
        X_train_scaled=X_train_scaled,      # Your scaled training data
        X_test_scaled=X_test_scaled,        # Your scaled test data
        y_train=y_train,                    # Your training labels
        y_test=y_test,                      # Your test labels
        feature_names=feature_names,        # Your feature names
        latent_dim=2,                       # 2D latent space for visualization
        hidden_layers=[200, 100, 50]       # Optimized for 356 features
    )
    
    # Create all visualizations
    results = ils.explore_latent_space(n_points=5)
    
    # Display results in DataBricks
    print("\n=== Results Ready ===")
    print(f"Data Summary: {results['data_summary']}")
    
    # In DataBricks, display the plots:
    # display(results['main_latent_space'])           # Main latent space plot
    # display(results['anomaly_analysis'])            # Anomaly analysis dashboard
    # display(results['feature_importance_plots']['point_0'])  # Feature importance for point 0
    
    return ils, results

# Alternative: Use the convenience function
def run_with_convenience_function():
    """
    Alternative way using the convenience function
    """
    from interactive_latent_space_your_data import create_interactive_latent_space_for_your_data
    
    ils, results = create_interactive_latent_space_for_your_data(
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled, 
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names
    )
    
    return ils, results

# Example of interactive exploration
def explore_specific_points(ils, point_indices=[0, 100, 500]):
    """
    Explore specific points in detail
    """
    print("=== Exploring Specific Points ===")
    
    for idx in point_indices:
        if idx < len(ils.y_train):
            point_type = "Anomaly" if ils.y_train[idx] == 1 else "Normal"
            print(f"Point {idx}: {point_type}")
            
            # Create feature importance plot for this point
            feature_plot = ils.create_feature_importance_plot(idx, top_n=15)
            
            # In DataBricks:
            # display(feature_plot)
            
            # Create latent space plot highlighting this point
            latent_plot = ils.create_latent_space_plot(selected_point_idx=idx)
            
            # In DataBricks:
            # display(latent_plot)

# Example of anomaly analysis
def analyze_anomalies(ils):
    """
    Focus on anomaly analysis
    """
    print("=== Anomaly Analysis ===")
    
    # Get anomaly indices
    train_anomaly_indices = np.where(ils.y_train == 1)[0]
    test_anomaly_indices = np.where(ils.y_test == 1)[0]
    
    print(f"Training anomalies: {len(train_anomaly_indices)}")
    print(f"Test anomalies: {len(test_anomaly_indices)}")
    
    # Analyze a few anomalies
    if len(train_anomaly_indices) > 0:
        anomaly_idx = train_anomaly_indices[0]
        print(f"Analyzing training anomaly at index {anomaly_idx}")
        
        # Feature importance for this anomaly
        anomaly_feature_plot = ils.create_feature_importance_plot(anomaly_idx, top_n=20)
        
        # In DataBricks:
        # display(anomaly_feature_plot)
    
    # Create anomaly analysis dashboard
    anomaly_dashboard = ils.create_anomaly_analysis_plot()
    
    # In DataBricks:
    # display(anomaly_dashboard)

# Main execution
if __name__ == "__main__":
    print("Interactive Latent Space Usage Example")
    print("To use this:")
    print("1. Make sure your data variables are defined:")
    print("   - X_train_scaled, X_test_scaled, y_train, y_test, feature_names")
    print("2. Run the functions above")
    print("3. In DataBricks, use display() to show the plots")
    
    # Example of how to run (uncomment when you have your data):
    # ils, results = run_interactive_latent_space()
    # explore_specific_points(ils, [0, 100, 500])
    # analyze_anomalies(ils)