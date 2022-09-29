# streamlit 실습
# 220929, by wygo
# code: https://github.com/airobotlab/lecture_1_advanced_cnn/blob/main/stramlit_practice_220929.py
# ref: https://zzsza.github.io/mlops/2021/02/07/python-streamlit-dashboard/

# install: pip install streamlit
# run: streamlit run stramlit_practice_220929.py

import streamlit as st
import pandas as pd
import numpy as np



# ############################################
# ## 제목 설정하기 
# st.title("KIRD")
# st.header("이미지프로세싱")
# st.subheader("3주차")
# st.write("물체검출에 대해 배워볼까요??")

# ############################################
# ## 위젯 만들기 
# ## 버튼 만들기
# if st.button("click button"):
#   st.write("Data Loading..")


# ## 체크 박스 만들기
# checkbox_btn = st.checkbox('Checktbox Button', value=True)
# if checkbox_btn:
# 	st.write('Great!')



# ## 라디오 버튼 만들기
# selected_item = st.radio("Radio Part", ("A", "B", "C"))

# if selected_item == "A":
# 	st.write("A!!")
# elif selected_item == "B":
# 	st.write("B!")
# elif selected_item == "C":
# 	st.write("C!")



# ## 선택 박스 만들기
# option = st.selectbox('Please select in selectbox!',
# 					 ('Object detection', 'Segmentation', 'Anomaly detection'))
# st.write('You selected:', option)


# ## 다중 선택 박스 만들기
# # 결과가 배열로 나옴
# # 100개 이상될 경우 유저 경험이 떨어질 수 있음
# multi_select = st.multiselect('Please select somethings in multi selectbox!',
# 							  ['Object detection', 'Segmentation', 'Anomaly detection'])
# st.write('You selected:', multi_select)


# ## 슬라이더 만들기
# values = st.slider('Select a range of values', 0.0, 100.0, (25.0, 75.0))
# st.write('Values:', values)




############################################
st.title('Uber pickups in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
          'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache
def load_data(nrows):
  data = pd.read_csv(DATA_URL, nrows=nrows)
  lowercase = lambda x: str(x).lower()
  data.rename(lowercase, axis='columns', inplace=True)
  data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
  return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
  st.subheader('Raw data')
  st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)


# map_data = pd.DataFrame(
#     np.random.randn(1000, 2) / [50, 50] + [36.321655, 127.378953],
#     columns=['lat', 'lon'])
# st.map(map_data)


