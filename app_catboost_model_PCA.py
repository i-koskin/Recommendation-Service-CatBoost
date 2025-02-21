from fastapi import FastAPI
import pandas as pd
from sqlalchemy import create_engine
import os
from catboost import CatBoostClassifier
from typing import List
from datetime import datetime
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker
import logging

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем экземпляр FastAPI
app = FastAPI()

# Создаем подключение к базе данных PostgreSQL
engine = create_engine(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
)

# Создаем локальную сессию для работы с базой данных
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    # Функция для получения локальной сессии базы данных
    with SessionLocal() as db:
        return db

# Модель для представления данных поста
class PostGet(BaseModel):
    id: int
    text: str
    topic: str

def get_model_path(path: str) -> str:
    # Функция для получения пути к модели
    MODEL_PATH = path
    return MODEL_PATH

def load_models():
    # Функция для загрузки модели CatBoost
    model_path = get_model_path("./catboost_model_PCA")
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    return loaded_model

def batch_load_sql(query: str) -> pd.DataFrame:
    # Функция для загрузки данных из SQL базы по частям
    CHUNKSIZE = 200000  # Размер чанка для загрузки данных
    
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def load_posts_features() -> pd.DataFrame:
    # Функция для загрузки характеристик постов
    return batch_load_sql('SELECT * FROM i_koskin_posts_features_lesson_22')

def load_features() -> pd.DataFrame:
    # Функция для загрузки характеристик пользователей
    return batch_load_sql('SELECT * FROM i_koskin_users_features_lesson_22')

def load_post_text() -> pd.DataFrame:
    # Функция для загрузки текстов постов
    return batch_load_sql('SELECT * FROM public.post_text_df')

def load_liked_posts() -> pd.DataFrame:
    # Функция для загрузки постов, которые пользователи лайкнули
    return batch_load_sql("""
                          SELECT distinct post_id, user_id
                          FROM public.feed_data
                          WHERE action='like'
                          """)

# Загрузка модели и данных
model = load_models()
df_user = load_features()
df_post = load_posts_features()
post_table = load_post_text()
liked_posts = load_liked_posts()

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

def get_recommended_feed(user_id: int, current_time: datetime, limit: int):
    # Функция для получения списка рекоммендованных постов
    # Получаем характеристики пользователя
    user_features = extract_user_features(user_id)
    
    # Получаем DataFrame с постами
    posts = df_post.copy()
    
    # Добавляем временные характеристики к DataFrame постов
    posts_with_time_features = add_time_features(posts, current_time)
    
    # Объединяем характеристики пользователя с постами
    user_posts_features = user_features.append(posts_with_time_features, ignore_index=True)

    # Закрепление порядка колонок
    user_posts_features = user_posts_features[['post_id', 'time_of_day', 'day_of_week', 'topic',
                                               'pca_1', 'pca_2', 'gender', 'city', 'exp_group',
                                               'os', 'source', 'age_group']]

    # Формируем вероятности лайкнуть пост для всех постов
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features['predicts'] = predicts

    # Удаляем посты, лайкнутые пользователем
    like_posts = liked_posts
    like_posts = list(like_posts[like_posts['user_id'] == id]['post_id'])
    filtered_ = user_posts_features[~user_posts_features.post_id.isin(like_posts)]

    # Формируем список рекомендованных постов
    recommended_posts = filtered_.sort_values('predicts')[-limit:].post_id

    return [
        PostGet({
            'id': i,
            'text': post_table[post_table['post_id'] == i].text.values[0],
            'topic': post_table[post_table['post_id'] == i].topic.values[0]
        }) for i in recommended_posts
    ]

# Эндпоинт для получения рекомендованных постов
@app.get('/post/recommendations/', response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 10) -> List[PostGet]:
    return get_recommended_feed(id, time, limit) 
