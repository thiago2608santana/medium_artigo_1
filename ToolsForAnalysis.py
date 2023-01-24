import os
import re
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

class ToolsForAnalysis:
    
    def __init__(self):
        pass
    
    def convert_to_int(self, value):
        try:
            return int(value)
        except:
            return 3
        
    def get_data_by_name(self, file_name, path):
        
        """
        Parameters:

        file_name: (str) Name of the file containing data
        path: (str) Path of data
        
        Return:
        data: (str) Pandas dataframe with data
        """

        files_in_dir = os.listdir(path)
        
        for file in files_in_dir:
            if re.search(file_name, file):
                data = pd.read_csv(f'{path}{file}')
                
        return data
    
    def split_train_test_and_transform_data(self, data, type_of_transformation, non_transform_columns, target_variable, test_size):
        
        #Split train/test data
        X_train, X_test, y_train, y_test = train_test_split(data.drop(target_variable, axis=1),
                                                            data[target_variable], test_size=test_size, random_state=42)
        
        #Reset index
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)
        
        #Transform data
        if type_of_transformation == 'min_max':
            min_max_scaler = MinMaxScaler()
            data_normalized = min_max_scaler.fit_transform(X_train.drop(non_transform_columns, axis=1))
            X_train_transformed = pd.DataFrame(data=data_normalized, columns=list(X_train.drop(non_transform_columns, axis=1)))
            X_train_transformed = pd.concat([X_train_transformed, X_train[non_transform_columns]], axis=1)
            
            min_max_scaler = MinMaxScaler()
            data_normalized = min_max_scaler.fit_transform(X_test.drop(non_transform_columns, axis=1))
            X_test_transformed = pd.DataFrame(data=data_normalized, columns=list(X_test.drop(non_transform_columns, axis=1)))
            X_test_transformed = pd.concat([X_test_transformed, X_test[non_transform_columns]], axis=1)
        else:
            standard_scaler = StandardScaler()
            data_standardized = standard_scaler.fit_transform(X_train.drop(non_transform_columns, axis=1))
            X_train_transformed = pd.DataFrame(data=data_standardized, columns=list(X_train.drop(non_transform_columns, axis=1)))
            X_train_transformed = pd.concat([X_train_transformed, X_train[non_transform_columns]], axis=1)
            
            standard_scaler = StandardScaler()
            data_standardized = standard_scaler.fit_transform(X_test.drop(non_transform_columns, axis=1))
            X_test_transformed = pd.DataFrame(data=data_standardized, columns=list(X_test.drop(non_transform_columns, axis=1)))
            X_test_transformed = pd.concat([X_test_transformed, X_test[non_transform_columns]], axis=1)
        
        X_train_transformed = X_train_transformed.drop(non_transform_columns, axis=1).round(2).copy()
        X_train_transformed = pd.concat([X_train_transformed, X_train[non_transform_columns]], axis=1)
        
        X_test_transformed = X_test_transformed.drop(non_transform_columns, axis=1).round(2).copy()
        X_test_transformed = pd.concat([X_test_transformed, X_test[non_transform_columns]], axis=1)
        
        return X_train_transformed, X_test_transformed, y_train, y_test
    
    def get_model_name(self, model):
        
        model_name = type(model).__name__
        
        return model_name
    
    def seek_best_params_for_models(self, X_train, y_train, model, parameters, score_type):
        
        grid_model = GridSearchCV(estimator=model, param_grid=parameters, scoring=score_type, n_jobs=-1, verbose=0)
        
        grid_model.fit(X_train, y_train)
        
        best_model = grid_model.best_estimator_
        best_parameters = grid_model.best_params_
        best_score = grid_model.best_score_
        all_results = grid_model.cv_results_
        
        return best_model, best_parameters, best_score, all_results
    
    def read_best_params_for_model(self, model_name, best_params_dict):
        
        try:
            best_parameters = best_params_dict[model_name]
        except:
            best_parameters = {}
            
        return best_parameters
    
    def save_summary_view_of_results(self, df_model_results, target_metric, type_of_transformation, start_time_folder_name, file_name):
        
        """
        Parameters:
        
        df_model_results: (pandas Dataframe) Results of models (accuracy, f1, precision, recall ...)
        target_metric: (str) Metric that will be use for select the best model
        type_of_transformation: (str) Indicate standardization or min_max transform
        start_time_folder_name: (str) Date and time as text format for tracking the experiments
        file_name: (str) Name of the file containing the summary of results
        """
    
        best_metric = df_model_results[df_model_results[target_metric] == df_model_results[target_metric].max()][target_metric].iloc[0]
        best_model = df_model_results[df_model_results[target_metric] == df_model_results[target_metric].max()]['Model'].iloc[0]
        
        if os.path.isfile(f'./{file_name}.xlsx') == False:
            df_monitoring = pd.DataFrame()
            create_new_monitoring = True
        else:
            df_monitoring = pd.read_excel(f'./{file_name}.xlsx')
            current_index_monitoring = df_monitoring.index[-1]+1
            create_new_monitoring = False
            
        if create_new_monitoring == True:
            df_monitoring[target_metric] = [best_metric]
            df_monitoring['Best model'] = [best_model]
            df_monitoring['Type of transformation'] = [type_of_transformation]
            df_monitoring['Execution date'] = [start_time_folder_name]
            df_monitoring.to_excel(f'./{file_name}.xlsx', index=False)
        else:
            df_monitoring.at[current_index_monitoring, target_metric] = best_metric
            df_monitoring.at[current_index_monitoring, 'Best model'] = best_model
            df_monitoring.at[current_index_monitoring, 'Type of transformation'] = type_of_transformation
            df_monitoring.at[current_index_monitoring, 'Execution date'] = start_time_folder_name
            df_monitoring.to_excel(f'./{file_name}.xlsx', index=False)
            