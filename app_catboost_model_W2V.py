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
    # Функция для получения пути к модели в зависимости от окружения
    MODEL_PATH = path
    return MODEL_PATH

def load_models():
    # Функция для загрузки модели CatBoost
    model_path = get_model_path("./catboost_model_W2V")
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
    return batch_load_sql('SELECT * FROM i_koskin_posts_features_lesson_25')

def load_features() -> pd.DataFrame:
    # Функция для загрузки характеристик пользователей
    return batch_load_sql('SELECT * FROM i_koskin_users_features_lesson_25')


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

def get_recommended_feed(id: int, limit: int = 10):
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
    user_posts_features = user_posts_features[['post_id', 'time_of_day', 'day_of_week', 'topic', 'vector_0', 'vector_1', 'vector_2', 'vector_3', 'vector_4', 'vector_5',
                                               'vector_6', 'vector_7', 'vector_8', 'vector_9', 'vector_10', 'vector_11', 'vector_12', 'vector_13', 'vector_14', 'vector_15',
                                               'vector_16', 'vector_17', 'vector_18', 'vector_19', 'vector_20', 'vector_21', 'vector_22', 'vector_23', 'vector_24', 'vector_25',
                                               'vector_26', 'vector_27', 'vector_28', 'vector_29', 'vector_30', 'vector_31', 'vector_32', 'vector_33', 'vector_34', 'vector_35',
                                               'vector_36', 'vector_37', 'vector_38', 'vector_39', 'vector_40', 'vector_41', 'vector_42', 'vector_43', 'vector_44', 'vector_45',
                                               'vector_46', 'vector_47', 'vector_48', 'vector_49', 'vector_50', 'vector_51', 'vector_52', 'vector_53', 'vector_54', 'vector_55',
                                               'vector_56', 'vector_57', 'vector_58', 'vector_59', 'vector_60', 'vector_61', 'vector_62', 'vector_63', 'vector_64', 'vector_65',
                                               'vector_66', 'vector_67', 'vector_68', 'vector_69', 'vector_70', 'vector_71', 'vector_72', 'vector_73', 'vector_74', 'vector_75',
                                               'vector_76', 'vector_77', 'vector_78', 'vector_79', 'vector_80', 'vector_81', 'vector_82', 'vector_83', 'vector_84', 'vector_85',
                                               'vector_86', 'vector_87', 'vector_88', 'vector_89', 'vector_90', 'vector_91', 'vector_92', 'vector_93', 'vector_94', 'vector_95',
                                               'vector_96', 'vector_97', 'vector_98', 'vector_99', 'gender', 'city', 'exp_group', 'os', 'source', 'age_group']]

    # Формируем вероятности лайкнуть пост для всех постов
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features['predicts'] = predicts

    # Удаляем посты, лайкнутых пользователем
    like_posts = liked_posts
    like_posts = list(like_posts[like_posts['user_id'] == id])
    filtered_ = user_posts_features[~user_posts_features.post_id.isin(like_posts)]

    # Формирование списка рекоммендованных постов
    recommended_posts = filtered_.sort_values('predicts')[-limit:].post_id

    return [
        PostGet(**{
            'id': i,
            'text': post_table[post_table['post_id'] == i].text.values[0],
            'topic': post_table[post_table['post_id'] == i].topic.values[0]
        }) for i in recommended_posts
    ]

# Эндпоинт для получения рекомендованных постов
@app.get('/post/recommendations', response_model=List[PostGet])
def recommended_posts(id: int, limit: int = 10) -> List[PostGet]:
    return get_recommended_feed(id, limit)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)    
