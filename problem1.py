import numpy as np
import pandas as pd
from pandas import Series, DataFrame


def filter_data_by_year(data, start_year, end_year):
    mask = (data[:, data_columns.tolist().index('p_year')] >= start_year) & (data[:, data_columns.tolist().index('p_year')] <= end_year)
    filtered_data = data[mask]

    return filtered_data

df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

data = df.values

data_columns = df.columns

filtered_data = filter_data_by_year(data, 2015, 2018)

column_h_index = data_columns.tolist().index('H')

top_10_H_max_values = filtered_data[np.argsort(filtered_data[:, column_h_index])[-10:][::-1]]

top_10_H_batter_names = top_10_H_max_values[:, data_columns.tolist().index('batter_name')]

print("project-2-1-1")
print("최대 hits(안타, H) player 10명:")
print(top_10_H_batter_names)

column_avg_index = data_columns.tolist().index('avg')

top_10_avg_max_values = filtered_data[np.argsort(filtered_data[:, column_avg_index])[-10:][::-1]]

top_10_avg_batter_names = top_10_avg_max_values[:, data_columns.tolist().index('batter_name')]

print("최대 avg(타율, avg) player 10명:")
print(top_10_avg_batter_names)

column_HR_index = data_columns.tolist().index('HR')

top_10_HR_max_values = filtered_data[np.argsort(filtered_data[:, column_HR_index])[-10:][::-1]]

top_10_HR_batter_names = top_10_HR_max_values[:, data_columns.tolist().index('batter_name')]

print("최대 HR(홈런, HR) player 10명:")
print(top_10_HR_batter_names)

column_OBP_index = data_columns.tolist().index('OBP')

top_10_OBP_max_values = filtered_data[np.argsort(filtered_data[:, column_OBP_index])[-10:][::-1]]

top_10_OBP_batter_names = top_10_OBP_max_values[:, data_columns.tolist().index('batter_name')]

print("최대 OBP(출루육, OBP) player 10명:")
print(top_10_OBP_batter_names)

# problem 2
def find_player_with_highest_war_by_position(filtered_data, position_column, war_column, player_name_column):
    unique_positions = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']

    highest_war_by_position = {}

    for position in unique_positions:
        position_data = filtered_data[filtered_data[position_column] == position]
        if not position_data.empty:
            max_war_index = position_data[war_column].idxmax()
            highest_war_batter_name = position_data.loc[max_war_index, player_name_column]
            highest_war_by_position[position] = highest_war_batter_name

    return highest_war_by_position

mask = (data[:, data_columns.tolist().index('p_year')] == 2018)
filtered_data = pd.DataFrame(data[mask], columns=data_columns)

filtered_data['war'] = pd.to_numeric(filtered_data['war'], errors='coerce')

highest_war_by_position = find_player_with_highest_war_by_position(filtered_data, 'tp', 'war', 'batter_name')

print("project-2-1-2")
print("각 포지션별로 가장 높은 WAR 값을 가지는 batter_name:")
for position, batter_name in highest_war_by_position.items():
    print(f"{position}: {batter_name}")

# problem 3
selected_columns = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']

selected_df = df[selected_columns]

correlation_matrix = selected_df.corr()

salary_correlation = correlation_matrix['salary']

most_correlated_column = salary_correlation.drop('salary').idxmax()

print("project-2-1-3")
print(f"{most_correlated_column} , {salary_correlation[most_correlated_column]}")

