import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
#import statsmodels.api as sm
#import altair as alt

#from urllib.error import URLError


def get_UN_data():
    AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
    df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
    return df.set_index("Region")

# Кэшируем, что бы не обновлялись значение при каждом рендеринге страницы
@st.cache

def get_data_csv():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
       dataframe = pd.read_csv(uploaded_file)
    return  dataframe

# Функция для датафрема наполненного случайными числами
def get_data():
    df = pd.DataFrame(np.random.rand(100, 5))
    df.columns = ['n1','n2','n3','n4','n5']
    return df



#df = get_UN_data()
st.title('Исходные данные')
#Выводим созданный датафрейм
data = get_data()
st.write(data)
st.write(data.describe().round(2))

st.title('Информация о данных')
col1, col2 = st.columns(2)

st.title('Моделирование')

with col1:
    st.subheader("Переменные факторы")
    # Выбираем факторы из дата фрейма (колонки), преобразуем в датафрейми для вывода
    factors = st.multiselect(
        "Выберете факторы (переменные)", list(data.columns)
    )
    if not factors:
        st.error("Выберете факторы")
        st.stop()
    else:
        st.subheader("Выбранные факторы")
        factors = data[factors]
        st.write(factors)
        st.subheader("Корреляция")
        corr = factors.corr()
        st.write(corr)
        if st.button('Теповая матрица без целевого значения'):
            ax = sns.heatmap(corr)
            plt.show()

with col2:
    st.subheader("Целевое значение")
    targValue = st.selectbox(
        "Выберете целевое значение", list(data.columns)
    )
    if not targValue:
        st.error("Выберете целевое значение")
        st.stop()
    elif targValue in factors:
        st.warning("Целевое значение не должно быть выбрано в факторах")
        st.stop()
    elif targValue != [factors]:
        st.subheader("Выбранное значение")
        targValue = data[targValue]
        st.write(targValue)
        st.subheader("Корреляция")
        corrCel = factors.join(targValue)
        corrCel = corrCel.corr()
        st.write(corrCel)
        if st.button('Тепловая матрица с целевым значением'):
            ax1 = sns.heatmap(corrCel)
            plt.show()

def data_selection(factors,targValue, value_test_size):
    X_train, X_test, y_train, y_test = train_test_split(factors, targValue, test_size=value_test_size)
    return X_train, X_test, y_train, y_test


option = st.selectbox(
     'Модель предсказания',
     ('-','Простое экспотенциальное сглаживание', 'Множественная линейная регрессия'))


if option == 'Простое экспотенциальное сглаживание':
    smoothing_level = st.slider('Уровень размытия', min_value=0.1, max_value=1.0, value=0.6)
    #if st.button('Предсказать'):
    model = SimpleExpSmoothing(targValue)
    model_fit = model.fit(smoothing_level=smoothing_level, optimized=False)
  #  st.write(model_fit.params)
  #  st.write(model_fit.sse)
    st.write(model_fit.k)
# make prediction
    yhat = model_fit.predict()
    st.write(yhat)

if option == 'Множественная линейная регрессия':
    value_test_size = st.slider('Размер тестовой выборки', min_value=10, max_value=70, value=30, step=10)
    X_train, X_test, y_train, y_test = data_selection(factors, targValue, value_test_size / 100)
    # st.write(X_train, X_test, y_train, y_test)
    #if st.button('Предсказать линейная'):
    #factNP = factors.to_numpy().reshape((-1, 1))
    factNP_train = X_train.to_numpy()
    targValueNP_train = y_train.to_numpy()
    model1 = LinearRegression().fit(factNP_train, targValueNP_train)
    #r_sq = model1.score(factNP, targValueNP)
    #st.write(model1)
    st.write('Коэффценты:', model1.coef_)
    st.write('Интерсепт', model1.intercept_)
    y_pred = model1.predict(X_test)
    st.write(y_pred)
    st.write('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    st.write('R2:', np.round(metrics.r2_score(y_test, y_pred), 2))

    val = [1]*(len(factors.columns))
    length = len(factors.columns)
    for i in range(0, length):
       val[i] = st.number_input(str(factors.columns[i]), key=[i], step=1)


x_new = np.array(val).reshape(1,-1)
st.write(x_new)
#x_new = val.reshape(-1, 1)
y_new = model1.predict(x_new)
st.write(y_new)





