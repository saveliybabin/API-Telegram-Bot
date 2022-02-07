import telebot
import requests
from telebot import types
import numpy as np
from app import views
import ast
import config
import time
import json

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

if __name__ == "__main__":
    token = config.token
    bot = telebot.TeleBot(token)

    @bot.message_handler(commands=['help'])
    def send_welcome(message):
        help_message = '''
        Информационная справка по методам бота:

        1. /start - приветствие
        
        2. /help - информация по методам
        
        3. /models - классы возможных моделей для обучения
        
        4. /train - обучение новой модели

        5. /trained_models - каталог обученных модели
        
        5. /predict - прогноз на обученных моделях
        
        6. /delete - удаление модели
        
        7. /show_results - результаты прогнозов 
        '''
        bot.reply_to(message, text=help_message)

    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        """
        Приветствие
        """

        help_message = '''Я бот, который предсказывает тип ирисов по вашим данным. Чтобы узнать, что я умею нажмите /help'''
        bot.reply_to(message,
                     text='Привет, ' + message.from_user.first_name + '! ' + help_message)

    @bot.message_handler(commands=['models'])
    def available_models(message):  
        """Сообщение с описанием классов моделей
        """
        res = requests.get(config.URL+'/api/ml_models').json()
        lr_id = res[0]['id']
        lr_hypers = res[0]['hyperparameters']
        dt_id = res[1]['id']
        dt_hypers = res[1]['hyperparameters']
        text = f"""Доступные для обучения модели:

        id: {lr_id}
        Класс: Logistic Regression
        Гиперпараметры:{list(lr_hypers.keys())}

        id: {dt_id}
        Класс: Decision Tree
        Гиперпараметры:{list(dt_hypers.keys())}
        """

        bot.send_message(message.chat.id,
                         text = text)    
    
    @bot.message_handler(commands=['trained_models'])
    def available_models(message):  
        """Сообщение с описанием обученных моделей и их гиперпараметров
        """
        res = requests.get(config.URL+'/api/ml_models/trained_models').json()
        text = 'Модели LR' + ':\n'
        for key, value in res["1"].items():
            text += '\t\t\t№' + key + ':  ' + str(value) + '\n'
        text += '\n\nМодели DT' + ':\n'
        for key, value in res["2"].items():
            text += '\t\t\t№' + key + ':  ' + str(value) + '\n'
        bot.send_message(message.chat.id,
                         text)    

    @bot.message_handler(commands=['train'])
    def train_model(message):
        """
        Модуль начала обучения новых моделей
        """
        chat_id = message.chat.id
        reply1 = '''
        Для начала выберите модель, которую вы хотите обучить. Ниже напишите одно число: 1 - LogReg, 2 - Decision Tree
        '''
        msg = bot.send_message(chat_id = chat_id,
                                text = reply1)
        bot.register_next_step_handler(msg, process_id_step)

    def process_id_step(message):
        """
        Получение гиперпараметров для обучения новой модели
        """
        chat_id = message.chat.id
        model_id = message.text
        if model_id == str(1):
            hypers = views.models_dao._ml_models[0]['hyperparameters']
            model_name = views.models_dao._ml_models[0]['name']
            text = """
            Вы выбрали модель {0}.\nТеперь введите гиперпараметры модели в формате:\n\n{1}
            """.format(model_name, hypers)
            msg = bot.send_message(chat_id = chat_id,
                                    text = text)
            bot.register_next_step_handler(msg, process_train_model1)
        elif model_id == str(2):
            hypers = views.models_dao._ml_models[1]['hyperparameters']
            model_name = views.models_dao._ml_models[1]['name']
            text = """
            Вы выбрали модель {0}.\nТеперь введите гиперпараметры модели в формате:\n\n{1}
            """.format(model_name, hypers)
            msg = bot.send_message(chat_id = chat_id,
                                    text = text)
            bot.register_next_step_handler(msg, process_train_model2)
        else:
            text = '''К сожалению, вы выбрали неправильный номер модели:( Попробуйте ещё раз /train'''
            msg = bot.send_message(chat_id = chat_id,
                                    text = text)

    def process_train_model1(message):
        """
        Обучение модели первого типа
        """
        try:
            chat_id = message.chat.id
            hypers = ast.literal_eval(message.text)
            params  = {}
            params['hyperparameters'] = hypers
            requests.put(config.URL + '/api/ml_models/1/train', data=json.dumps(params, indent=4), headers=headers)
            res = requests.get(config.URL+'/api/ml_models/trained_models').json()
            num = list(res["1"].keys())[-1]
            text = f'''
            Для модели LR были выбраны следующие гиперпараметры: {params['hyperparameters']}\nМодель с ID = 1 обучена и сохранена под номером {num}'''
            bot.send_message(chat_id = chat_id, 
                             text = text)
        except:
            text = '''Во время обучения что-то пошло не так. Проверьте, правильно ли вы ввели гиперпараметры и попробуйте ещё раз /train'''
            bot.send_message(chat_id = chat_id,
                             text = text)



    def process_train_model2(message):
        """
        Обучение модели второго типа
        """
        try:
            chat_id = message.chat.id        
            hypers = ast.literal_eval(message.text)
            params  = {}
            params['hyperparameters'] = hypers
            requests.put(config.URL + '/api/ml_models/2/train', data=json.dumps(params, indent=4), headers=headers)
            res = requests.get(config.URL+'/api/ml_models/trained_models').json()
            num = list(res["2"].keys())[-1]
            bot.send_message(chat_id = chat_id, text = f'''Для модели LR были выбраны следующие гиперпараметры: {params['hyperparameters']}\nМодель с ID = 2 обучена и сохранена под номером {num}''')
        except:
            text = '''Во время обучения что-то пошло не так. Проверьте, правильно ли вы ввели гиперпараметры и попробуйте ещё раз /train'''
            bot.send_message(chat_id = chat_id,
                             text = text)

    @bot.message_handler(commands=['predict'])
    def chose_model_for_prediction(message):  
        """
        Начало работы с прогнозирование
        """
        chat_id = message.chat.id
        msg = bot.send_message(chat_id = chat_id, text = '''Выберите обученную модель и отправьте данные для прогноза.\nФормат (через пробел):\nID (номер класса) num (номер модели) X (1 2 3 4 (данные для прогноза, 4 положительных рациональных числа)\n\nПример:\n1 1 0.5 0.5 0.5 0.5''')
        bot.register_next_step_handler(msg, process_predict)

    def process_predict(message):  
        """
        Обработка прогнозирования
        """
        chat_id = message.chat.id
        input = message.text
        try:
            input_s =  input.split()
            params = {}
            params['id'] = int(input_s[0])
            params['num'] = int(input_s[1])
            if len(input_s[2:]) == 4:
                params['X'] = input[len(input_s[0]) + len(input_s[1]) + 1:]
                try:
                    prediction = requests.post(config.URL + '/api/ml_models/{}/predict'.format(params['id']), data=json.dumps(params, indent=4), headers=headers).json()
                    text = 'Результат прогноза: ' + prediction
                    bot.send_message(chat_id = chat_id, text = text)
                except:
                    text = f'''Модели класса {params['id']} под номером {params['num']} не существует. Попробуйте ещё раз /predict'''
                    bot.send_message(chat_id = chat_id,
                                    text = text)
            else:
                text = '''Вы ввели неправильное количество данных для обучения. Попробуйте ещё раз /predict'''
                bot.send_message(chat_id = chat_id,
                                text = text)
        except:
             text = '''Вы неправильно ввели данные. Попробуйте ещё раз /predict'''
             bot.send_message(chat_id = chat_id,
                                text = text)
    
    @bot.message_handler(commands=['delete'])
    def chose_model_for_elimination(message):  
        """
        Начало работы с удалением моделей
        """
        chat_id = message.chat.id
        msg = bot.send_message(chat_id = chat_id, text = '''Выберите модель для удаления.\nФормат:\nID номер модели \n\nПример:\n1 1''')
        bot.register_next_step_handler(msg, process_eliminate)

    def process_eliminate(message): 
        """
        Обработка удаления
        """ 
        chat_id = message.chat.id
        input = message.text
        input_s =  input.split()
        params = {}
        try:
            params['id'] = int(input_s[0])
            params['num'] = int(input_s[1])
            try:
                requests.delete(config.URL + '/api/ml_models/{}/delete'.format(params['id']), data=json.dumps(params, indent=4), headers=headers).json()
                text = f'''Данные модели класса {params['id']} номера {params['num']} были удалены'''
                bot.send_message(chat_id = chat_id, text = text)
            except:
                text = '''Такой модели не существует. Попробуйте ещё раз /delete'''
                bot.send_message(chat_id = chat_id,
                                 text = text)
        except:
            text = '''Вы неправильно ввели данные. Попробуйте ещё раз /delete'''
            bot.send_message(chat_id = chat_id,
                             text = text)


    @bot.message_handler(commands=['show_results'])
    def chose_model_for_elimination(message):  
        """
        Каталог прогнозов

        """
        res = requests.get(config.URL+'/api/ml_models/results').json()
        text = 'Модели LR' + ':\n'
        for key, value in res["1"].items():
            if value is None:
                text += '\t\t\t№' + key + ':  Данные ' + 'None' + ', Прогноз ' + 'None' + '\n'
            else:    
                text += '\t\t\t№' + key + ':  Данные ' + str(value["X_test"]) + ', Прогноз ' + str(value["y_pred"]) + '\n'
        text += '\n\nМодели DT' + ':\n'
        for key, value in res["2"].items():
            if value is None:
                text += '\t\t\t№' + key + ':  Данные ' + 'None' + ', Прогноз ' + 'None' + '\n'
            else:    
                text += '\t\t\t№' + key + ':  Данные ' + str(value["X_test"]) + ', Прогноз ' + str(value["y_pred"]) + '\n'
        bot.send_message(message.chat.id, text) 

if __name__ == '__main__':
    try:
       bot.polling(none_stop=True)
    except Exception as e:
       print(e)
       time.sleep(15)