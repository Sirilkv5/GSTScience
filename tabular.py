from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute  import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder as OrdinalEncoder_Sk
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score

from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer

from feature_engine.encoding import RareLabelEncoder,OrdinalEncoder
from feature_engine.selection import DropFeatures
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, ConfusionMatrixDisplay

# to visualise all the columns and upto 100 rows in the dataframe
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", 100)

# for supressing warnings
import warnings
warnings.filterwarnings('ignore')

df_X_train = pd.read_csv('data/X_Train_Data_Input_Inf.csv')
df_X_test = pd.read_csv('data/X_Test_Data_Input.csv')
df_Y_train = pd.read_csv('data/Y_Train_Data_Target_Inf.csv')
df_Y_test = pd.read_csv('data/Y_Test_Data_Target.csv')
print(df_X_train.shape,df_X_test.shape, df_Y_train.shape,df_Y_test.shape)

vars_num = [var for var in df_X_train.columns if df_X_train[var].dtypes !='O' and var not in ['Id']]

# Missing values in our numerical variables
Num_V = df_X_train[vars_num].isnull().mean().sort_values(ascending=False)
#print(Num_V)
len(Num_V)

# Imputate numerical variables
imputer = SimpleImputer(strategy='constant', fill_value=-1) ##
df_X_train['Column9'] = imputer.fit_transform(df_X_train['Column9'].to_frame())
df_X_test['Column9'] = imputer.transform(df_X_test['Column9'].to_frame())


imputer = SimpleImputer(strategy='most_frequent')
df_X_train[vars_num] = imputer.fit_transform(df_X_train[vars_num])
df_X_test[vars_num] = imputer.transform(df_X_test[vars_num])

df_X_train1 =  df_X_train.drop(columns=['ID'])
df_Y_train1 =  df_Y_train.drop(columns=['ID'])
df_X_test1 =  df_X_test.drop(columns=['ID'])
df_Y_test1 =  df_Y_test.drop(columns=['ID'])

# Create XGBoost classifier model
xgb_model1 = XGBClassifier(
    learning_rate = 0.14972574734435318,
    n_estimators = 200,
    max_depth = 1,
    min_child_weight = 6,
    gamma = 0.5,
    subsample = 0.55,
    colsample_bytree = 1,
    scale_pos_weight = 6,
    objective = 'binary:logistic'
)

xgb_model1.fit(df_X_train1, df_Y_train1)

# Save the trained model to a file
xgb_model1.save_model('xgboost_model.json')

# Load the saved model from file
saved_model = xgb.XGBClassifier()
saved_model.load_model('xgboost_model.json')

df_X_inference = pd.read_csv('data/X_Train_Data_Inference.csv')
df_Y_inference = pd.read_csv('data/Y_Train_Data_Inference.csv')
df_X_inference1 =  df_X_inference.drop(columns=['ID'])
df_Y_inference1 =  df_Y_inference.drop(columns=['ID'])


# Make predictions on new data
y_pred = saved_model.predict(df_X_inference1)

# Assuming 'actual_data' contains the actual values corresponding to 'new_data'
df_preds = pd.DataFrame({
    'Actual': df_Y_inference1.squeeze(),  # Ensure actual data is in correct format
    'Predicted': y_pred
})

df_preds.to_csv('predictions_comparison.csv', index=False)


def compare_columns(csv_file_path):
    """
    This function compares 'Actual' and 'Predict' columns in a given CSV file 
    and calculates the percentage of true (matches) and false (mismatches).

    Parameters:
        csv_file_path (str): Path to the input CSV file.
    
    Returns:
        None
    """
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Ensure both 'Actual' and 'Predict' columns exist
    if 'Actual' not in df.columns or 'Predicted' not in df.columns:
        print("Error: The required columns ('Actual', 'Predict') are not present in the provided CSV.")
        return
    
    # Create a new column 'Match' that stores True if Actual == Predict, else False
    df['Match'] = df['Actual'] == df['Predicted']
    
    # Calculate total number of rows
    total_rows = len(df)
    
    # Calculate number of matches (True)
    true_count = df['Match'].sum()
    
    # Calculate number of mismatches (False)
    false_count = total_rows - true_count
    
    # Calculate percentages
    true_percentage = (true_count / total_rows) * 100
    false_percentage = (false_count / total_rows) * 100
    
    print(f'True Percentage: {true_percentage:.2f}%')
    print(f'False Percentage: {false_percentage:.2f}%')


# Example usage:
compare_columns('predictions_comparison.csv')


def evaluate_model(model, df_X_train1, df_Y_train1, df_X_test1, df_Y_test1):
    """
    Evaluates the model using various performance metrics and displays results.
    
    Parameters:
        cboost (model): The trained model to be evaluated.
        df_X_train1 (DataFrame): Training features.
        df_Y_train1 (DataFrame): Training targets.
        df_X_test1 (DataFrame): Testing features.
        df_Y_test1 (DataFrame): Testing targets.

    Returns:
        None
    """
    
    # Evaluate performance on training set using MSE and RMSE
    train_pred = model.predict(df_X_train1)
    train_mse = mean_squared_error(df_Y_train1.squeeze(), train_pred)
    train_rmse = sqrt(train_mse)
    
    print(f'Linear Train MSE: {train_mse:.4f}')
    print(f'Linear Train RMSE: {train_rmse:.4f}\n')
     
    # Evaluate performance on test set using MSE and RMSE 
    test_pred=model.predict(df_X_test1)
    test_mse=mean_squared_error(df_Y_test1.squeeze(),test_pred)
    testrmse=sqrt(test_mse)

    print(f"linear Test mse:{test_mse:.4f}")
    print(f"linear Test rmse:{testrmse:.4f}\n")

    # Model Performance On All Sets 
    predictions=model.predict(df_X_test1)

    test_preds=model.predict_proba(df_X_test1)[:,[0]]
    train_preds=model.predict_proba(df_X_train1)[:,[0]]

    train_auc=roc_auc_score(df_Y_train1.squeeze(),train_preds)
    test_auc=roc_auc_score(df_Y_test1.squeeze(),test_preds)

    accuracy=accuracy_score(df_Y_test1,predictions)

    F1=f1_score(df_Y_test1,predictions)
    recall=recall_score(df_Y_test1,predictions)
    precision=precision_score(df_Y_test1,predictions)
    print(f"F1:{F1:.6f}")
    print(f"Recall:{recall:.6f}")
    print(f"Precision:{precision:.6f}\n\n")
    print(f"train_auc:{train_auc:.6f}")
    print(f"test_auc:{test_auc:.6f}")
    print(f"accuracy:{accuracy:.6f}")

    print(F1,recall,precision)

    cm=confusion_matrix(df_Y_test1,predictions,labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=model.classes_)
   
    disp.plot()
    plt.show()

model = saved_model
evaluate_model(model, df_X_train1, df_Y_train1, df_X_test1, df_Y_test1)