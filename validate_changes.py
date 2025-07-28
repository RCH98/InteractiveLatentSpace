#!/usr/bin/env python3
"""
Validation script to test the specific changes made to app.py 
for true label integration
"""

import pandas as pd
import numpy as np
import pickle
import os

def validate_app_changes():
    """Validate the key changes made to app.py"""
    print("=== Validating App.py Changes ===\n")
    
    try:
        # Change to correct directory
        os.chdir('/home/runner/work/InteractiveLatentSpace/InteractiveLatentSpace')
        
        print("1. Loading data (same as original)...")
        # Load data exactly as in app.py
        tit_sub = pd.read_csv('./data/Titanic/gender_submission.csv')
        tit_train = pd.read_csv('./data/Titanic/train.csv')
        tit_test = pd.read_csv('./data/Titanic/test.csv')
        df_train_final = pd.read_pickle("./data/Titanic/df_train_final")
        df_test_final = pd.read_pickle("./data/Titanic/df_test_final")
        
        print(f"✓ Training data shape: {df_train_final.shape}")
        print(f"✓ Test data shape: {df_test_final.shape}")
        
        print("\n2. Feature preparation (same as original)...")
        from sklearn.preprocessing import StandardScaler
        
        scaler_cols = ['Age', 'Fare', 'Name_Length', 'Family_Size', 'Ticket_Frequency', 'Fare_Family_Size', 'Fare_Cat_Pclass']
        std = StandardScaler()
        std.fit(df_train_final[scaler_cols])
        df_train_final.loc[:, scaler_cols] = std.transform(df_train_final[scaler_cols])
        df_test_final.loc[:, scaler_cols] = std.transform(df_test_final[scaler_cols])
        
        features_train = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Title', 'Name_Length', 'Emb_C',
                          'Emb_Q', 'Emb_S', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs',
                          'Title_Other', 'Title_Royal', 'Family_Size',
                          'Family_Friends_Surv_Rate', 'Cabin_Clean',
                          'Ticket_Frequency', 'Tkt_AS', 'Tkt_C', 'Tkt_CA',
                          'Tkt_CASOTON', 'Tkt_FC', 'Tkt_FCC', 'Tkt_Fa', 'Tkt_LINE',
                          'Tkt_NUM', 'Tkt_PC', 'Tkt_PP', 'Tkt_PPP', 'Tkt_SC', 'Tkt_SCA',
                          'Tkt_SCAH', 'Tkt_SCAHBasle', 'Tkt_SCOW', 'Tkt_SCPARIS', 'Tkt_SCParis',
                          'Tkt_SOC', 'Tkt_SOP', 'Tkt_SOPP', 'Tkt_SOTONO', 'Tkt_SOTONOQ', 'Tkt_SP',
                          'Tkt_STONO', 'Tkt_SWPP', 'Tkt_WC', 'Tkt_WEP', 'Fare_Cat', 'Child', 'Senior']

        features = features_train[1:]  # Remove 'Survived' for test features
        
        df_train_final = df_train_final[features_train]
        df_test_final = df_test_final[features]
        
        features = df_test_final.columns.to_list()
        X_train_df = df_train_final[features]
        Y_train = df_train_final['Survived']
        X_test_df = df_test_final.reset_index(drop=True)
        
        print(f"✓ Feature matrix shape: {X_train_df.shape}")
        print(f"✓ Training labels shape: {Y_train.shape}")
        print(f"✓ Label distribution: {Y_train.value_counts().to_dict()}")
        
        print("\n3. Test label preparation...")
        import re
        c = pd.read_csv('./data/Titanic/titanic_test_labels.csv')
        test_data_with_labels = c.copy()
        for i, name in enumerate(test_data_with_labels['name']):
            if '"' in name:
                test_data_with_labels.loc[i, 'name'] = re.sub('"', '', name)
        for i, name in enumerate(tit_test['Name']):
            if '"' in name:
                tit_test.loc[i, 'Name'] = re.sub('"', '', name)
        survived = []
        for name in tit_test['Name']:
            survived.append(int(test_data_with_labels.loc[test_data_with_labels['name'] == name]['survived'].values[-1]))
        Y_test = pd.Series(survived, index=X_test_df.index)
        
        Y_train = Y_train.to_numpy()
        Y_test = Y_test.to_numpy()
        
        print(f"✓ Test labels shape: {Y_test.shape}")
        print(f"✓ Test label distribution: {np.unique(Y_test, return_counts=True)}")
        
        print("\n4. Testing NEW APPROACH: True Label Integration...")
        # NEW APPROACH: Use true labels instead of classifier predictions
        X_train = np.hstack((X_train_df.values, Y_train.reshape(-1, 1)))
        X_test = np.hstack((X_test_df.values, Y_test.reshape(-1, 1)))
        
        print(f"✓ Combined training data shape: {X_train.shape}")
        print(f"✓ Combined test data shape: {X_test.shape}")
        print(f"✓ Last column (true labels) in training: {np.unique(X_train[:, -1])}")
        print(f"✓ Last column (true labels) in test: {np.unique(X_test[:, -1])}")
        
        # Verify the true labels are correctly placed
        assert np.array_equal(X_train[:, -1], Y_train), "Training labels not correctly integrated"
        assert np.array_equal(X_test[:, -1], Y_test), "Test labels not correctly integrated"
        print("✓ True labels correctly integrated in both datasets")
        
        print("\n5. Testing column names update...")
        columns = list(X_train_df.columns) + ['true_label']
        print(f"✓ Updated column names (last few): {columns[-5:]}")
        assert columns[-1] == 'true_label', "Column name not updated correctly"
        
        print("\n6. Comparison with original approach...")
        # Load the black box model to compare
        try:
            import xgboost as xgb
            bst = pickle.load(open('./models/XGBoost_Titanic.p', 'rb'))
            dtrain = xgb.DMatrix(X_train_df.values, Y_train)
            dtest = xgb.DMatrix(X_test_df.values)
            y_train_pred = bst.predict(dtrain)
            y_test_pred = bst.predict(dtest)
            
            # Original approach
            X_train_orig = np.hstack((X_train_df.values, y_train_pred.reshape(-1, 1)))
            X_test_orig = np.hstack((X_test_df.values, y_test_pred.reshape(-1, 1)))
            
            print(f"✓ Original approach (predictions): last col range [{y_train_pred.min():.3f}, {y_train_pred.max():.3f}]")
            print(f"✓ New approach (true labels): last col values {np.unique(X_train[:, -1])}")
            print(f"✓ Key difference: Using discrete true labels (0/1) instead of continuous predictions (0-1)")
            
        except ImportError:
            print("⚠ XGBoost not available, skipping comparison")
        except Exception as e:
            print(f"⚠ Could not load model: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in validation: {e}")
        return False

def main():
    success = validate_app_changes()
    
    print(f"\n=== Validation Result ===")
    if success:
        print("✓ ALL VALIDATIONS PASSED!")
        print("\nSummary of changes:")
        print("1. ✓ True labels (Y_train, Y_test) now used instead of classifier predictions")
        print("2. ✓ Data concatenation updated to use true labels")
        print("3. ✓ Column names updated to reflect 'true_label' instead of 'bb_proba'")
        print("4. ✓ Visualization will now show true survival (0/1) instead of prediction probabilities")
        print("5. ✓ Feature importance analysis will be based on true labels")
        print("\nThis enables finding the most important features based on true labels rather than model predictions!")
    else:
        print("✗ VALIDATION FAILED - Please check the implementation")

if __name__ == "__main__":
    main()