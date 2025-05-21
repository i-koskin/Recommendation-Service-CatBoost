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

    class Config:
        from_attributes = True # Включаем режим ORM для поддержки работы с SQLAlchemy

def get_model_path(path: str) -> str:
    # Функция для получения пути к модели
    MODEL_PATH = path
    return MODEL_PATH

def load_models():
    # Функция для загрузки модели CatBoost
    model_path = get_model_path("catboost_model_PCA")
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

def get_recommended_feed(id: int, time: datetime, limit: int = 10):
    # Функция для получения списка рекоммендованных постов
    # Получение фич пользователя по его ID
    user_features = df_user.loc[df_user['user_id'] == id]
    user_features = user_features.drop(['user_id'], axis=1)

    # Загрузка фич по постам
    posts_features = df_post.copy()
    
    # Объединение фич
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.reset_index(drop=True)
    
    # Добавление фич о текущей дате рекомендаций
    time = datetime.now()
    time = time.strftime("%Y-%m-%dT%H:%M:%S")
    user_posts_features['hour'] = pd.to_datetime(time).hour
    user_posts_features['weekday'] = pd.to_datetime(time).day_of_week
    user_posts_features['time_of_day'] = pd.cut(
        user_posts_features['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening'],
        right=False
        )
    user_posts_features['day_of_week'] = pd.cut(
        user_posts_features['weekday'],
        bins=[-1, 4, 6],
        labels=['weekday', 'weekend']
        )

    user_posts_features = user_posts_features.drop(['hour', 'weekday'], axis=1)

    # Закрепление порядка колонок
    user_posts_features = user_posts_features[['post_id', 'time_of_day', 'day_of_week', 'topic',
                                               'pca_1', 'pca_2', 'gender', 'city','exp_group',
                                               'os', 'source', 'age_group']]

    # Формируем вероятности лайкнуть пост для всех постов
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features['predicts'] = predicts

    # Удаление постов, лайкнутых пользователем
    like_posts = liked_posts
    like_posts = list(like_posts[like_posts['user_id'] == id])
    filtered_ = user_posts_features[~user_posts_features.post_id.isin(like_posts)]

    # Формирование списка рекомендованных постов
    top_post_ids = filtered_.nlargest(limit, 'predicts')['post_id'].to_list()
    post_lookup = post_table.set_index('post_id').loc[top_post_ids]

    return [
        PostGet(
            id=i,
            text=post_lookup.loc[i]['text'],
            topic=post_lookup.loc[i]['topic']
        ) for i in top_post_ids
    ]

# Эндпоинт для получения рекомендованных постов
@app.get('/post/recommendations', response_model=List[PostGet])
def recommended_posts(id: int, limit: int = 10) -> List[PostGet]:
    return get_recommended_feed(id, limit)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 

