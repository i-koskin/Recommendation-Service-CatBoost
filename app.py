from fastapi import FastAPI, Depends, HTTPException
import pandas as pd
from sqlalchemy import create_engine, text
import os
from catboost import CatBoostClassifier
from typing import List
from datetime import time, datetime
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker


app = FastAPI()

engine = create_engine(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    
    with SessionLocal() as db:
        return db


class PostGet(BaseModel):
    id: int
    text: str
    topic: str


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("/Users/user/Downloads/HW_/catboost_model")
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    return loaded_model


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_posts_features() -> pd.DataFrame:
    return batch_load_sql('SELECT * FROM i_koskin_posts_features_lesson_22')

def load_features() -> pd.DataFrame:
    return batch_load_sql('SELECT * FROM i_koskin_users_features_lesson_22')


def load_post_text() -> pd.DataFrame:
    return batch_load_sql('SELECT * FROM public.post_text_df')

def load_liked_posts() -> pd.DataFrame:
    return batch_load_sql("""
                          SELECT distinct post_id, user_id
                          FROM public.feed_data
                          WHERE action='like'
                          """)

model = load_models()
df_user = load_features()
df_post = load_posts_features()
post_table = load_post_text()
liked_posts = load_liked_posts()

def get_recommended_feed(id: int, time: datetime, limit: int):

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
    user_posts_features = user_posts_features[['post_id','time_of_day', 'day_of_week', 'topic',
                                               'pca_1', 'pca_2', 'gender', 'city','exp_group',
                                               'os', 'source', 'age_group']]

    # Формировка вероятности лайкнуть пост для всех постов
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features['predicts'] = predicts

    # Удаление постов, лайкнутых пользователем
    like_posts = liked_posts
    like_posts = list(like_posts[like_posts['user_id'] == id])
    filtered_ = user_posts_features[~user_posts_features.post_id.isin(like_posts)]

    # Формирование списка постов для рекомендий
    recommended_posts = filtered_.sort_values('predicts')[-limit:].post_id

    return [
        PostGet(**{
            'id': i,
            'text': post_table[post_table['post_id'] == i].text.values[0],
            'topic': post_table[post_table['post_id'] == i].topic.values[0]
        }) for i in recommended_posts
    ]


@app.get('/post/recommendations/', response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int=10) -> List[PostGet]:
    return get_recommended_feed(id, time, limit)    