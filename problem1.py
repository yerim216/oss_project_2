# import pandas as pd
# from pandas import Series, DataFrame

# data = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

# print(data)

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def filter_data_by_year(data, start_year, end_year):
    """
    'p_year' 열의 값이 start_year부터 end_year까지인 행들을 뽑아내는 함수

    Parameters:
    - data: 데이터가 담긴 numpy 배열
    - start_year: 뽑아내고자 하는 범위의 시작 연도
    - end_year: 뽑아내고자 하는 범위의 끝 연도

    Returns:
    - filtered_data: 'p_year' 열의 값이 start_year부터 end_year까지인 행들로 이루어진 배열
    """
    # 'p_year' 열의 값이 start_year부터 end_year까지인 행을 필터링
    mask = (data[:, data_columns.tolist().index('p_year')] >= start_year) & (data[:, data_columns.tolist().index('p_year')] <= end_year)
    filtered_data = data[mask]

    return filtered_data

# CSV 파일을 pandas를 사용하여 읽어옴
df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

# 데이터를 numpy 배열로 변환
data = df.values

# CSV 파일의 열 이름들을 가져옴
data_columns = df.columns

# 'p_year' 열의 값이 2015부터 2018까지인 행을 뽑아냄
filtered_data = filter_data_by_year(data, 2015, 2018)

# 최대 H 찾기
column_h_index = data_columns.tolist().index('H')


top_10_H_max_values = filtered_data[np.argsort(filtered_data[:, column_h_index])[-10:][::-1]]


top_10_H_batter_names = top_10_H_max_values[:, data_columns.tolist().index('batter_name')]

print("project-2-1-1")
print("최대 hits(안타, H) player 10명:")
print(top_10_H_batter_names)

# 최대 avg 찾기
column_avg_index = data_columns.tolist().index('avg')


top_10_avg_max_values = filtered_data[np.argsort(filtered_data[:, column_avg_index])[-10:][::-1]]


top_10_avg_batter_names = top_10_avg_max_values[:, data_columns.tolist().index('batter_name')]

# 결과 출력
print("최대 avg(타율, avg) player 10명:")
print(top_10_avg_batter_names)


# 최대 HR 찾기
column_HR_index = data_columns.tolist().index('HR')


top_10_HR_max_values = filtered_data[np.argsort(filtered_data[:, column_HR_index])[-10:][::-1]]


top_10_HR_batter_names = top_10_HR_max_values[:, data_columns.tolist().index('batter_name')]

# 결과 출력
print("최대 HR(홈런, HR) player 10명:")
print(top_10_HR_batter_names)

# 최대 OBP 찾기
column_OBP_index = data_columns.tolist().index('OBP')


top_10_OBP_max_values = filtered_data[np.argsort(filtered_data[:, column_OBP_index])[-10:][::-1]]


top_10_OBP_batter_names = top_10_OBP_max_values[:, data_columns.tolist().index('batter_name')]

# 결과 출력
print("최대 OBP(출루육, OBP) player 10명:")
print(top_10_OBP_batter_names)

