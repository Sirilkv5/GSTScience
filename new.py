import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder as OrdinalEncoder_Sk
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, metrics
import joblib
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, ConfusionMatrixDisplay

from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer

from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder
# Set display options for Pandas DataFrame to visualize all columns and up to 100 rows.
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", 100)

# Suppress warnings for cleaner output.
import warnings 
warnings.filterwarnings('ignore')


def load_data():
    """
    Load training and testing data from CSV files.
    
    Returns:
        tuple: DataFrames containing training features (X), testing features (X), 
               training targets (Y), and testing targets (Y).
    """
    df_X_train = pd.read_csv('data/X_Train_Data_Input.csv')
    df_X_test = pd.read_csv('data/X_Test_Data_Input.csv')
    df_Y_train = pd.read_csv('data/Y_Train_Data_Target.csv')
    df_Y_test = pd.read_csv('data/Y_Test_Data_Target.csv')
    print(df_X_train.shape,df_X_test.shape, df_Y_train.shape,df_Y_test.shape)


    
    return df_X_train, df_X_test, df_Y_train, df_Y_test

def preprocess_data(df_X_train, df_X_test,df_Y_train,df_Y_test):
    """
    Preprocess the data by imputing missing values and dropping unnecessary columns.
    
    Parameters:
        df_X_train (DataFrame): Training features.
        df_X_test (DataFrame): Testing features.
        
    Returns:
        tuple: Preprocessed training features (X) and testing features (X).
    """
    
     # Identify numerical variables excluding 'Id'
    vars_num = [var for var in df_X_train.columns if var != 'ID'and  not isinstance(df_X_train[var].dtype , object)]
     
     # Impute numerical variables with constant value (-1)
    imputer_constant = SimpleImputer(strategy='constant', fill_value=-1)
    imputer_frequent = SimpleImputer(strategy='most_frequent')

     # Apply imputations on specified columns 
    for col in ['Column9'] + vars_num:
         if col == 'Column9':
             imputed_values=imputer_constant.fit_transform(df_X_train[col].to_frame())
             test_imputed_values=imputer_constant.transform(df_X_test[col].to_frame())
         else:
             imputed_values=imputer_frequent.fit_transform(df_X_train[vars_num])
             test_imputed_values=imputer_frequent.transform(df_X_test[vars_num])

         # Update original dataframe with transformed values  
         if col == 'Column9':
            df_X_train['Column9']=imputed_values.squeeze()
            df_X_test['Column9']=test_imputed_values.squeeze()
         else :
            df_X_train.update(imputed_values)
            df_X_test.update(test_imputed_values)
    
    df_X_train =  df_X_train.drop(columns=['ID'])
    df_X_test =  df_X_test.drop(columns=['ID'])
    df_Y_train =  df_Y_train.drop(columns=['ID'])    
    df_Y_test =  df_Y_test.drop(columns=['ID'])

    return df_X_train, df_X_test,df_Y_train,df_Y_test

def train_model_cat(Xtrain,Ytrain):
    """
    This is catboost model
    """
    # Create CatBoost model
    cboost = CatBoostClassifier(learning_rate = 1,
                            depth = 1,
                            scale_pos_weight = 6,
                            l2_leaf_reg = 8,
                            border_count = 65)
    cboost.fit(Xtrain,Ytrain.values.ravel())
    return cboost


def train_model(Xtrain,Ytrain):
   """"
   Train an xgboost classifier model using provided dataset 

   Parameters :
       -df_xtrains : preprocessed training set of input variables/features 
       -df_ytrains : preprocessed target variable/labels corresponding to above inputs 

   Return :
       trained model instance after fitting it on given datasets 

   """
## Create XGBoost classifier model
   xgb_model1= XGBClassifier(
      learning_rate=0.14972574734435318,
      n_estimators=200,
      max_depth=1,
      min_child_weight=6,
      gamma=.5,
      subsample=.55,
      colsample_bytree=1.,
      scale_pos_weight=6.,
      objective='binary:logistic'
      )
   xgb_model1.fit(Xtrain,Ytrain.values.ravel())
   return xgb_model1

def load_combined_models(cat_path):
    # Load previously saved models 
    save_model = CatBoostClassifier()
    cat_boost_loaded = save_model.load_model(cat_path)
    

    return cat_boost_loaded 


def save_model(model,path):
    """ Save trained machine learning models into file at specified location/path 
    Parameters :
        - models : fitted/trained instance of any ML algorithm/classifier/regressor etc .
        - paths : string representing full path where serialized version should be stored/saved .

    Return :
    None  
"""
    model.save_model(path)


def load_models(path):
    """
    Load previously saved serialized versions back into memory from disk storage locations .
    Parameters :
        -paths : strings representing full paths where serialized versions were stored/saved earlier .
    Return :
        loaded instances ready-to-use directly without needing retraining again . 
"""
    loaded=xgb.XGBClassifier()
    loaded.load_model(path)
    return loaded



def make_predictions(models,Xinference):
    """
        Generate predictions based upon new unseen examples using already trained/fitted classifiers/regressors etc .
        Parameters :
            -models : fitted/trained instance of any ML algorithm/classifier/regressor etc .
            -Xinferences: dataframe containing fresh/unseen examples whose outcomes need predicting now . 
        Return :
            predicted labels/classes/values corresponding exactly one-for-one against each row present inside Xinferences itself . 
    """
    return models.predict(Xinference)



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
        print("Error: The required columns ('Actual', 'Predicted') are not present in the provided CSV.")
        return
    
    # Create a new column Match that stores True if Actual == Predict, else False 
    df['Match']=df['Actual'].eq(df['Predicted'])

    # Calculate total number of rows 
    total_rows=len(df)

    # Calculate number of matches (True)
    true_count=df["Match"].sum()

    # Calculate number of mismatches(False)
    false_count=total_rows-true_count

    # Calculate percentages 
    true_percentage=(true_count/total_rows)*100
    false_percentage=(false_count/total_rows)*100

    print(f'True Percentage:{true_percentage:.2f}%')
    print(f"False Percentage:{false_percentage:.2f}%")

def evaluate_model(model, df_X_train, df_Y_train, df_X_test, df_Y_test):
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
    train_pred = model.predict(df_X_train)
    train_mse = mean_squared_error(df_Y_train.squeeze(), train_pred)
    train_rmse = sqrt(train_mse)
    
    #print(f'Linear Train MSE: {train_mse:.4f}')
    #print(f'Linear Train RMSE: {train_rmse:.4f}\n')
     
    # Evaluate performance on test set using MSE and RMSE 
    test_pred=model.predict(df_X_test)
    test_mse=mean_squared_error(df_Y_test.squeeze(),test_pred)
    testrmse=sqrt(test_mse)

    #print(f"Linear Test mse:{test_mse:.4f}")
    #print(f"Linear Test rmse:{testrmse:.4f}\n")

    # Model Performance On All Sets 
    predictions=model.predict(df_X_test)

    test_preds=model.predict_proba(df_X_test)[:,[0]]
    train_preds=model.predict_proba(df_X_train)[:,[0]]

    train_auc=roc_auc_score(df_Y_train.squeeze(),train_preds)
    test_auc=roc_auc_score(df_Y_test.squeeze(),test_preds)

    accuracy=accuracy_score(df_Y_test,predictions)

    F1=f1_score(df_Y_test,predictions)
    recall=recall_score(df_Y_test,predictions)
    precision=precision_score(df_Y_test,predictions)
    print(f"F1 Score:{F1:.6f}")
    print(f"Recall:{recall:.6f}")
    print(f"Precision:{precision:.6f}\n\n")
    print(f"train_auc:{train_auc:.6f}")
    print(f"test_auc:{test_auc:.6f}")
    print(f"accuracy:{accuracy:.6f}")

    #print(F1,recall,precision)

    cm=confusion_matrix(df_Y_test,predictions,labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=model.classes_)
   
    disp.plot()
    plt.show()



def main():
    # Load data
    df_X_train, df_X_test, df_Y_train, df_Y_test = load_data()
    
    # Preprocess data
    Xtrain, Xtest, Ytrain, Ytest = preprocess_data(df_X_train, df_X_test,df_Y_train,df_Y_test)
    
    # Train model
    #model = train_model(Xtrain,Ytrain)
    model = train_model_cat(Xtrain,Ytrain)
    
    # Save trained model
    #save_model(model,'xgboost_model.json')
    save_model(model,'cboost_model.json')
    
    # Load saved model for inference
    #loaded_model = load_models('xgboost_model.json')
    loaded_model = load_combined_models('cboost_model.json')

       
    # Make predictions on new data - a separate inference dataset
    df_X_inference = pd.read_csv('data/X_Train_Data_Input_Inference.csv')
    df_Y_inference = pd.read_csv('data/Y_Train_Data_Target_Inference.csv')
    
    Xinference=  df_X_inference.drop(columns=['ID'])
    Yinference=  df_Y_inference.drop(columns=['ID'])

    y_pred=make_predictions(loaded_model,Xinference)

    # Assuming actual_data contains the actual values corresponding to new_data
    df_preds=pd.DataFrame({
         "Actual":Yinference.squeeze(),  # Ensure actual data is in correct format 
         "Predicted":y_pred})

    df_preds.to_csv('predictions_comparison.csv',index=False)
    
    compare_columns('predictions_comparison.csv')

    evaluate_model(loaded_model, Xtrain, Ytrain, Xtest, Ytest)

if __name__ == '__main__':
  main()