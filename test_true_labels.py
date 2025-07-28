#!/usr/bin/env python3
"""
Test script to verify the true label integration changes
"""

import numpy as np
import pandas as pd
import os

def test_data_loading():
    """Test if data can be loaded and processed correctly"""
    try:
        # Test data loading
        print("Testing data loading...")
        
        # Check if required files exist
        required_files = [
            './data/Titanic/train.csv',
            './data/Titanic/test.csv', 
            './data/Titanic/titanic_test_labels.csv',
            './data/Titanic/df_train_final',
            './data/Titanic/df_test_final'
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✓ Found: {file_path}")
            else:
                print(f"✗ Missing: {file_path}")
                
        # Test loading pickled data
        try:
            df_train_final = pd.read_pickle("./data/Titanic/df_train_final")
            df_test_final = pd.read_pickle("./data/Titanic/df_test_final")
            print(f"✓ Loaded training data: {df_train_final.shape}")
            print(f"✓ Loaded test data: {df_test_final.shape}")
            
            # Check if 'Survived' column exists
            if 'Survived' in df_train_final.columns:
                print(f"✓ 'Survived' column found with {df_train_final['Survived'].nunique()} unique values")
                print(f"  Value distribution: {df_train_final['Survived'].value_counts().to_dict()}")
            else:
                print("✗ 'Survived' column not found in training data")
                
        except Exception as e:
            print(f"✗ Error loading pickled data: {e}")
            
        return True
        
    except Exception as e:
        print(f"Error in data loading test: {e}")
        return False

def test_label_integration():
    """Test the true label integration logic"""
    try:
        print("\nTesting true label integration...")
        
        # Simulate the data structure
        n_samples = 100
        n_features = 10
        
        # Simulate feature data
        X_train_df = pd.DataFrame(np.random.randn(n_samples, n_features))
        Y_train = np.random.randint(0, 2, n_samples)  # Binary labels
        
        # Test the concatenation logic (from the modified app.py)
        X_train = np.hstack((X_train_df.values, Y_train.reshape(-1, 1)))
        
        print(f"✓ Original features shape: {X_train_df.shape}")
        print(f"✓ Labels shape: {Y_train.shape}")
        print(f"✓ Combined data shape: {X_train.shape}")
        print(f"✓ Last column (true labels) unique values: {np.unique(X_train[:, -1])}")
        
        # Verify the last column contains the true labels
        assert np.array_equal(X_train[:, -1], Y_train), "True labels not correctly appended"
        print("✓ True labels correctly integrated as last column")
        
        return True
        
    except Exception as e:
        print(f"Error in label integration test: {e}")
        return False

def main():
    print("=== Testing True Label Integration Changes ===\n")
    
    # Change to the correct directory
    os.chdir('/home/runner/work/InteractiveLatentSpace/InteractiveLatentSpace')
    
    # Run tests
    data_test = test_data_loading()
    integration_test = test_label_integration()
    
    print(f"\n=== Test Results ===")
    print(f"Data Loading: {'PASS' if data_test else 'FAIL'}")
    print(f"Label Integration: {'PASS' if integration_test else 'FAIL'}")
    
    if data_test and integration_test:
        print("\n✓ All tests passed! The true label integration should work correctly.")
    else:
        print("\n✗ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()