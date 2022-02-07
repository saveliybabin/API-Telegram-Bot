from flask_restx import Resource
from app import api, models_dao
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import numpy as np

log = logging.getLogger(__name__)

models = {1:LogisticRegression,
          2:DecisionTreeClassifier}


@api.route('/api/ml_models')
class MLModels(Resource):

    def get(self):
        return models_dao._ml_models

@api.route('/api/ml_models/trained_models')
class MLTrainedModel(Resource):
    """Класс для отображения обученных моделей
       и их гиперпараметров
    """
    def get(self):
        return models_dao._models_backlog_params

@api.route('/api/ml_models/results')
class MLModelResults(Resource):
    """Класс для отображения 
    """
    def get(self):
        # log.info(f'INFO: Trained models = {models_dao._models_backlog}')
        return models_dao.predictions
        
@api.route('/api/ml_models/<int:id>/train')
class MLModelTrain(Resource): 
    """Класс для обучения
    """
    def put(self, id):
        model = self.train(id)
        self.save_model(model, id)
        return models_dao.get(id)

    def train(self, id):
        log.info("INFO: Preparing to train...",)
        try: 
            df = api.payload
            try:
                if df is None:
                    model = models[id]()
                else:
                    model = models[id](**df['hyperparameters'])  
            except KeyError or AttributeError as e:
                log.error("Something wrong with parameters")
                api.abort(404, e)
            df_train = load_iris()    
            x = df_train['data']
            y = df_train['target']
            log.info("INFO: Start training",)
            model.fit(x, y)
            log.info("INFO: Training finished!")
            return model
        except KeyError or AttributeError as e:
            log.error("ERROR: Looks like you make mistake in hyperparameters")
            api.abort(404, e)

    def save_model(self, model, id):
        log.info(f"INFO: Saving the model...")
        models_dao._models_backlog_[id][models_dao.n] = model
        params = model.get_params()
        upd_params = {}
        for k, v in params.items():
            if k in models_dao._ml_models[id-1]['hyperparameters'].keys():
                upd_params[k] = v
        models_dao._models_backlog_params[id][models_dao.n] = upd_params
        models_dao.n += 1

@api.route('/api/ml_models/<int:id>/predict')
class MLModelPredict(Resource): 
    """Класс для предсказания
    """

    def post(self, id): 
        id_ = int(id)
        prediction = self._predict(id_)
        log.info(f"INFO: Here is the prediction: {prediction}")
        return prediction


    def _predict(self, id):
        try:
            df = api.payload
            number = int(df['num'])
            model = models_dao._models_backlog_[id][number]
            if model is not None:
                X_upd = np.array(df['X'].split()).astype(float)            
                prediction = str(model.predict([X_upd]))
                dict_pred_temp = {}
                dict_pred_temp['y_pred'] = prediction
                dict_pred_temp['X_test'] = df['X']
                models_dao.predictions[id][number] = dict_pred_temp
                return prediction
            else:
                return None
        except KeyError as e:
            log.error("ERROR: Invalid model number or No trained models")
            api.abort(404, e)


@api.route('/api/ml_models/<int:id>/delete')
class MLModelDelete(Resource): 
    """Класс для удаления
    """

    def delete(self, id):
        try: 
            df = api.payload
            num = df['num']
            if num in models_dao._models_backlog_[id].keys():
                models_dao.delete(id, num)
                log.info(f"INFO: model {num} deleted")
                return ''
            else:
                api.abort(404, "WARNING: No model with this number")

        except KeyError as e:
            log.error("ERROR: Invalid model number")
            api.abort(404, e)
