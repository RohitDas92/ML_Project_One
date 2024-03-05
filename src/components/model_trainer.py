import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (AdaBoostRegressor,
                              GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomExeption
from src.logger import logging


from src.utils import (save_object,evaluate_model)

@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting training and test data')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1])
            
            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree' : DecisionTreeRegressor(),
                'Liner Regrssion' : LinearRegression(), 
                'K Neighbour Regressor' : KNeighborsRegressor(),
                'AdaBoost Regressor' : AdaBoostRegressor(),
                'GradientBoost Regressor' : GradientBoostingRegressor()
            }

            model_report:dict= evaluate_model(X_train=X_train, y_train = y_train,X_test = X_test, 
                                              y_test = y_test, models=models)
            
            best_model_score = max(sorted(model_report.values())) #to get best model score

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score) # to get best model name
            ]
            
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomExeption('No Best Model Found')
            
            logging.info(f'Best model founded on both training and test data')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2 = r2_score(y_test,predicted)

            return r2


        except Exception as e:
            raise CustomExeption(e,sys)