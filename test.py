def preprocess_data(df_X_train, df_Y_train, df_X_test, df_Y_test):
    
    # Identify numerical variables excluding 'Id'
    vars_num = [var for var in df_X_train.columns if df_X_train[var].dtypes != 'O' and var not in ['Id']]

    # Check for missing values in numerical variables
    num_v = df_X_train[vars_num].isnull().mean().sort_values(ascending=False)
    
    # Impute Column9 with constant value -1
    imputer_constant = SimpleImputer(strategy='constant', fill_value=-1)
    
   # Fit on train data then transform both train and test data 
   # to ensure consistency between them
    
   # Transform Column9 of train data 
   
   ##df_X_train['Column9'] = imputer_constant.fit_transform(df_X_train[['Column9']])
   
   ##df_X_test['Column9'] = imputer_constant.transform(df_X_test[['Column9']])

     ##Transform Column9 of train data 
     
     column_name='Column9'
     
     if column_name in vars_num:
         print(f"Processing {column_name}")
         
         ##Fit on train data then transform both train and test data to ensure consistency between them
        
         ###Transform Column9 of train data 
        
         ###df_xtrain[column_name]=imputer_constant.fit_transform(df_xtrain[[column_name]])
        
          ###Transform Column9 of test data 
        
          ###df_xtest[column_name]=imputer_constant.transform(df_xtest[[column_name]])
          
          

# Impute remaining numerical variables with most frequent value
imputer_frequent = SimpleImputer(strategy='most_frequent')
df_xtrain[vars_num] = imputer_frequent.fit_transform(df_xtrain[vars_num])
df_xtest[vars_num] = imputer_frequent.transform(df_xtest[vars_num])

# Drop 'ID' column from all datasets
columns_to_drop=['ID']
if set(columns_to_drop).issubset(set(dfxtrain.columns)):
print("Dropping ID column")
##Drop ID column from all datasets 

###Train Data 

####Features 

#####X_Train=dfxtrain.drop(columns=columns_to_drop)

####Target 

#####Y_Train=dfytrain.drop(columns=columns_to_drop)

###Test Data 

####Features 

#####X_Test=dfxtest.drop(columns=columns_to_drop)

####Target 

#####Y_Test=dfytest.drop(columns=columns_to_drop)


return X_Train,Y_Train,X_Test,Y_Test