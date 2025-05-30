from fastapi import FastAPI
import pandas as pd
from sqlalchemy import create_engine
from catboost import CatBoostClassifier
from typing import List
from datetime import datetime
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker
import hashlib
import logging

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        from_attributes = True # Включаем режим ORM для поддержки работы с SQLAlchemy

# Модель для ответа с рекомендациями
class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]

# Функция для получения пути к модели в зависимости от окружения
def get_model_path(model_name: str) -> str:
    model_paths = {
        'control': './catboost_model_PCA',  # локальный путь к контрольной модели
        'test': './catboost_model_W2V'  # локальный путь к тестовой модели
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
def recommend_with_control_model(id: int, exp_group: str, limit: int) -> List[PostGet]:
    # Функция для получения рекомендаций с использованием контрольной модели
    return get_recommended_feed(model_control, id, exp_group, limit)

def recommend_with_test_model(id: int, exp_group: str, limit: int) -> List[PostGet]:
    # Функция для получения рекомендаций с использованием тестовой модели
    return get_recommended_feed(model_test, id, exp_group, limit)

def get_recommended_feed(model, id: int, exp_group: str, limit: int = 10):
    # Функция для получения списка рекоммендованных постов
    # Получение фич пользователя по его ID
    logger.info(f'user_id: {id}')
    logger.info('reading user features')
    user_features = df_user.loc[df_user['user_id'] == id]
    user_features = user_features.drop(['user_id'], axis=1)

    # Загрузка фич по постам
    logger.info('reading posts features')
    if exp_group == 'control':
        posts_features = df_post_control.copy()
    elif exp_group == 'test':
        posts_features = df_post_test.copy()
    else:
        raise ValueError('Unknown group')
    
    
    # Объединение фич
    logger.info('zipping features')
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    logger.info('assigning features')
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.reset_index(drop=True)
    
    # Добавление фич о текущей дате рекомендаций
    logger.info('add time info')
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
    if exp_group == 'control':
        user_posts_features = user_posts_features[['post_id', 'time_of_day', 'day_of_week', 'topic',
                                               'pca_1', 'pca_2', 'gender', 'city','exp_group',
                                               'os', 'source', 'age_group']]
    elif exp_group == 'test':
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
    else:
        raise ValueError('Unknown group')
    

    # Формируем вероятности лайкнуть пост для всех постов
    logger.info('predicting')
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features['predicts'] = predicts

    # Удаление постов, лайкнутых пользователем
    logger.info('deleting liked posts')
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

@app.get('/post/recommendations', response_model=Response)
def recommended_posts(id: int, limit: int=10) -> Response:
    exp_group = get_exp_group(id)  # Определяем группу пользователя

    if exp_group == 'control':
        recommendations = recommend_with_control_model(id, exp_group, limit)
    elif exp_group == 'test':
        recommendations = recommend_with_test_model(id, exp_group, limit)
    else:
        raise ValueError('Unknown group')

        # Если recommendations пустой, заменяем на пустой список
    if recommendations is None:
        recommendations = []

    # Возвращаем объект Response с информацией о группе и рекомендациями
    return Response(exp_group=exp_group, recommendations=recommendations)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 
