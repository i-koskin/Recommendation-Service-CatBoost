from fastapi import FastAPI
import pandas as pd
from sqlalchemy import create_engine
import os
from catboost import CatBoostClassifier
from typing import List
from datetime import datetime
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker
import hashlib
import logging
from loguru import logger

# Создаем экземпляр FastAPI
app = FastAPI()

# Константы для разделения на A/B группы
SALT = "salt_value"
CONTROL_PERCENT = 0.5  # 50/50 разделение

# Создаем подключение к базе данных PostgreSQL
engine = create_engine(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
)
# Создаем локальную сессию для работы с базой данных
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    # Функция для получения локальной сессии базы данных
    with SessionLocal() as db: # Используем контекстный менеджер для автоматического закрытия сессии
        return db

# Модель для представления постов
class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True # Включаем режим ORM для поддержки работы с SQLAlchemy

# Модель для ответа с рекомендациями
class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]


# Функция для получения пути к модели в зависимости от окружения
def get_model_path(model_name: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        model_paths = {
            'control': '/workdir/user_input/catboost_model_W2V',  # путь к контрольной модели
            'test': '/workdir/user_input/catboost_model_PCA'  # путь к тестовой модели
        }
    else:  # Если код выполняется не в LMS
        model_paths = {
            'control': '/Users/user/Downloads/HW_/catboost_model_W2V',  # локальный путь к контрольной модели
            'test': '/Users/user/Downloads/HW_/mcatboost_model_PCA'  # локальный путь к тестовой модели
        }

    if model_name not in model_paths:
        raise ValueError(f"Unknown model: {model_name}")

    return model_paths[model_name]

# Функция для загрузки модели
def load_models(model_name: str):
    model_path = get_model_path(model_name)
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    return loaded_model

# Функция для пакетной загрузки данных из SQL в DataFrame
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000 # Размер чанка для загрузки данных

    conn = engine.connect().execution_options(stream_results=True) # Устанавливаем соединение с БД
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

# Функция для загрузки признаков постов в зависимости от модели
def load_posts_features(model_name: str) -> pd.DataFrame:
    if model_name == 'control':
        return batch_load_sql('SELECT * FROM i_koskin_posts_features_lesson_22')
    elif model_name == 'test':
        return batch_load_sql('SELECT * FROM i_koskin_posts_features_lesson_25')
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
# Функция для загрузки признаков пользователей
def load_users_features() -> pd.DataFrame:
    return batch_load_sql('SELECT * FROM i_koskin_users_features_lesson_22')

# Функция для загрузки текстов постов
def load_post_text() -> pd.DataFrame:
    return batch_load_sql('SELECT * FROM public.post_text_df')

# Функция для загрузки постов, которые пользователи лайкнули
def load_liked_posts() -> pd.DataFrame:
    return batch_load_sql("""
                          SELECT distinct post_id, user_id
                          FROM public.feed_data
                          WHERE action='like'
                          """)

# Загружаем модели и данные
model_control = load_models('control')  # Загружаем контрольную модель
model_test = load_models('test')  # Загружаем тестовую модель
df_user = load_users_features()
df_post_control = load_posts_features('control')
df_post_test = load_posts_features('test')
post_table = load_post_text()
liked_posts = load_liked_posts()

# Функция для разбиения пользователей на группы
def get_exp_group(user_id: int) -> str:
    user_hash = int(hashlib.md5(f"{user_id}{SALT}".encode()).hexdigest(), 16)
    return "control" if user_hash % 100 < CONTROL_PERCENT * 100 else "test"

# Функции для рекомендаций, привязанные к моделям
def recommend_with_control_model(id: int, time: datetime, limit: int, exp_group: str) -> List[PostGet]:
    # Функция для получения рекомендаций с использованием контрольной модели
    return get_recommended_feed(model_control, id, time, limit, exp_group)

def recommend_with_test_model(id: int, time: datetime, limit: int, exp_group: str) -> List[PostGet]:
    # Функция для получения рекомендаций с использованием тестовой модели
    return get_recommended_feed(model_test, id, time, limit, exp_group)

def extract_user_features(user_id: int) -> pd.Series:
    # Функция для получения характеристик пользователя
    logger.info(f'Извлечение характеристик для user_id: {user_id}')
    user_features = df_user.loc[df_user['user_id'] == user_id]
    if user_features.empty:
        raise ValueError(f'Пользователь с ID {user_id} не найден.')
    return user_features.drop(['user_id'], axis=1).iloc[0]

def add_time_features(df: pd.DataFrame, current_time: datetime) -> pd.DataFrame:
    # Функция для добавления временных характеристик
    logger.info('Добавление временных характеристик')
    
    # Преобразуем текущее время в datetime
    current_time = pd.to_datetime(current_time)
    
    # Добавляем временные характеристики
    df['hour'] = pd.to_datetime(current_time).hour
    df['weekday'] = pd.to_datetime(current_time).day_of_week # 0 - понедельник, 6 - воскресенье
    
    # Определяем время суток
    df['time_of_day'] = pd.cut(df['hour'],
                               bins=[0, 6, 12, 18, 24],
                               labels=['ночь', 'утро', 'день', 'вечер'],
                               right=False)
    
    # Определяем тип дня
    df['day_of_week'] = pd.cut(df['weekday'],
                               bins=[-1, 4, 6],
                               labels=['будний', 'выходной'])
    return df.drop(['hour', 'weekday'], axis=1)

def get_recommended_feed(model, user_id: int, current_time: datetime, limit: int, exp_group: str):
    # Функция для получения списка рекоммендованных постов
    # Получаем характеристики пользователя
    user_features = extract_user_features(user_id)
    
    # Получаем DataFrame с постами (df_post_control или df_post_test)
    posts = df_post_control if user_is_in_control_group(user_id) else df_post_test
    
    # Добавляем временные характеристики к DataFrame постов
    posts_with_time_features = add_time_features(posts, current_time)
    
    # Объединяем характеристики пользователя с постами
    user_posts_features = user_features.append(posts_with_time_features, ignore_index=True)
    
    # Закрепление порядка колонок
    if exp_group == 'control':
        user_posts_features = user_posts_features[['post_id', 'time_of_day', 'day_of_week', 'topic',
                                               'pca_1', 'pca_2', 'gender', 'city','exp_group',
                                               'os', 'source', 'age_group']]
    elif exp_group == 'test':
        user_posts_features = user_posts_features[['post_id', 'time_of_day', 'day_of_week', 'topic', 'vector_0',
                                                   'vector_1', 'vector_2', 'vector_3', 'vector_4', 'vector_5',
                                                   'vector_6', 'vector_7', 'vector_8', 'vector_9', 'vector_10',
                                                   'vector_11', 'vector_12', 'vector_13', 'vector_14', 'vector_15',
                                                   'vector_16', 'vector_17', 'vector_18', 'vector_19', 'vector_20',
                                                   'vector_21', 'vector_22', 'vector_23', 'vector_24', 'vector_25',
                                                   'vector_26', 'vector_27', 'vector_28', 'vector_29', 'vector_30',
                                                   'vector_31', 'vector_32', 'vector_33', 'vector_34', 'vector_35',
                                                   'vector_36', 'vector_37', 'vector_38', 'vector_39', 'vector_40',
                                                   'vector_41', 'vector_42', 'vector_43', 'vector_44', 'vector_45',
                                                   'vector_46', 'vector_47', 'vector_48', 'vector_49', 'vector_50',
                                                   'vector_51', 'vector_52', 'vector_53', 'vector_54', 'vector_55',
                                                   'vector_56', 'vector_57', 'vector_58', 'vector_59', 'vector_60',
                                                   'vector_61', 'vector_62', 'vector_63', 'vector_64', 'vector_65',
                                                   'vector_66', 'vector_67', 'vector_68', 'vector_69', 'vector_70',
                                                   'vector_71', 'vector_72', 'vector_73', 'vector_74', 'vector_75',
                                                   'vector_76', 'vector_77', 'vector_78', 'vector_79', 'vector_80',
                                                   'vector_81', 'vector_82', 'vector_83', 'vector_84', 'vector_85',
                                                   'vector_86', 'vector_87', 'vector_88', 'vector_89', 'vector_90',
                                                   'vector_91', 'vector_92', 'vector_93', 'vector_94', 'vector_95',
                                                   'vector_96', 'vector_97', 'vector_98', 'vector_99',
                                                   'gender', 'city', 'exp_group', 'os', 'source', 'age_group']]
    else:
        raise ValueError('Unknown group')
    

    # Формируем вероятности лайкнуть пост для всех постов
    logger.info('predicting')
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features['predicts'] = predicts

    # Удаляем посты, лайкнутые пользователем
    logger.info('deleting liked posts')
    like_posts = liked_posts
    like_posts = list(like_posts[like_posts['user_id'] == id])
    filtered_ = user_posts_features[~user_posts_features.post_id.isin(like_posts)]

    # Формируем список рекомендованных постов
    recommended_posts = filtered_.sort_values('predicts')[-limit:].post_id

    return [
        PostGet(**{
            'id': i,
            'text': post_table[post_table['post_id'] == i].text.values[0],
            'topic': post_table[post_table['post_id'] == i].topic.values[0]
        }) for i in recommended_posts
    ]

# Эндпоинт для получения списка рекомендованных постов
@app.get('/post/recommendations/', response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int=10) -> Response:
    exp_group = get_exp_group(id)  # Определяем группу пользователя

    if exp_group == 'control':
        recommendations = recommend_with_control_model(id, time, limit, exp_group)
    elif exp_group == 'test':
        recommendations = recommend_with_test_model(id, time, limit, exp_group)
    else:
        raise ValueError('Unknown group')

        # Если recommendations пустой, заменяем на пустой список
    if recommendations is None:
        recommendations = []

    # Возвращаем объект Response с информацией о группе и рекомендациями
    return Response(exp_group=exp_group, recommendations=recommendations)  
