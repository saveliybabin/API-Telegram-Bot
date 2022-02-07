class MLModelsDAO:

    def __init__(self):
        self._ml_models = [{'id':1, 'name':'LR', 'hyperparameters':{"max_iter":"int",
                                                                    "penalty":["l1", "l2"],
                                                                    "C":"float"
                                                                    }}, 
                            {'id':2, 'name':'DT', 'hyperparameters':{"criterion":["gini", "entropy"],
                                                                    "max_depth":"int",
                                                                    "min_samples_split":"int or float",
                                                                    "min_samples_leaf":"int or float"
                                                                    }}]
        self._models_backlog_params = {1: {}, 2: {}}
        self._models_backlog_ = {1: {}, 2: {}}
        self.predictions = {1: {}, 2: {}}
        self.n = 1
                                

    def get(self, id):
        """Получение информации о моделях

        Args:
            id (int): ID модели
        Returns:
            dict: Информация о модели
        """
        try:
            for model in self._ml_models:
                if model['id'] == id:
                    return model
        except NotImplementedError as e:
            raise e('ml_model {} doesnot exist'.format(id))

    def update(self, id, data):
        """Обновление информации о моделях

        Args:
            id (int): ID модели
            data (dict): Словарь для изменения данных

        Returns:
            dict: Измененная модель
        """
        model = self.get(id)
        model.update(data)
        return model

    def delete(self, id, n):
        """Удаление обученной модели

        Args:
            id (int): ID модели
            n (str): Номер обученной модели
        """

        try:
            self._models_backlog_params[id][n] = None
            self._models_backlog_[id][n] = None
            self.predictions[id][n] = None
        except NotImplementedError as e:
            raise e('Model {} with number {} does not exist'.format(id, n))

