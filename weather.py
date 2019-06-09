import pandas as pd
import datetime
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os


def get_near_grid(station_name, station_data, knn, grid_data):
    '''get the near 6 grid stations' id'''
    coordinate = np.array(station_data.loc[station_name]).reshape(1, -1)
    points = knn.kneighbors(coordinate, return_distance=False).flatten()
    grid_list = grid_data.iloc[points].index.tolist()

    return grid_list


def get_weather_information(station, near_grid_list, station_name_list, grid_weather):
    '''generate weather information from the grid station's weather information'''
    station_grid = near_grid_list[station_name_list.index(station)]
    weather_df = grid_weather[grid_weather['station_id'].isin(station_grid)].groupby('time').mean()
    return weather_df


def main():
    begin = datetime.datetime(2017, 1, 1, 19, 0, 0)
    end = datetime.datetime(2018, 4, 30, 23, 0, 0)
    gap = datetime.timedelta(hours=1)
    weather_time_period = pd.date_range(begin - 5 * gap, end + 48 * gap, freq='H')

    station_data = pd.read_excel('Beijing_AirQuality_Stations_en.xlsx', header=10).dropna()
    grid_data = pd.read_csv('Beijing_grid_weather_station.csv', header=None)
    grid_data.columns = ['Station ID', 'latitude', 'longitude']
    # station_data.index=station_data['Station ID']
    station_data = station_data.set_index(['Station ID'])
    # grid_data.index=grid_data['Station ID']
    grid_data = grid_data.set_index(['Station ID'])[['longitude', 'latitude']]
    knn = NearestNeighbors(n_neighbors=6)
    knn.fit(grid_data)

    station_name_list = list(station_data.index)
    near_grid_list = []
    for station in station_name_list:
        near_grid_list.append(get_near_grid(station, station_data, knn, grid_data))

    column_name = ['station_id', 'time', 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed']
    weather1 = pd.read_csv('gridWeather_201701-201803.csv').drop(['longitude', 'latitude'], axis=1)
    weather2 = pd.read_csv('gridWeather_201804.csv')
    weather3 = pd.read_csv( 'gridWeather_20180501-20180502.csv')
    weather1.columns = column_name

    grid_weather = pd.concat([weather1[column_name], weather2[column_name], weather3[column_name], ], sort=False)
    grid_weather.index = list(range(len(grid_weather)))
    grid_weather = grid_weather.drop_duplicates(keep='first')
    grid_weather = grid_weather.drop(['wind_direction'], axis=1)

    station_weather = []
    for station in station_name_list:
        new_weather_df = get_weather_information(station, near_grid_list, station_name_list, grid_weather)
        new_weather_df.index = pd.to_datetime(new_weather_df.index)
        new_weather_df = new_weather_df.reindex(weather_time_period, method='bfill')
        new_weather_df['time'] = pd.to_datetime(new_weather_df.index)
        new_weather_df = new_weather_df.set_index('time')
        new_weather_df.loc[end + 49 * gap] = np.array(new_weather_df.iloc[-1])
        new_weather_df.loc[end + 50 * gap] = np.array(new_weather_df.iloc[-1])
        new_weather_df.loc[end + 51 * gap] = np.array(new_weather_df.iloc[-1])
        #     print(new_weather_df.index)
        station_weather.append(new_weather_df)

    # save
    for name, df in zip(station_name_list, station_weather):  #
        df.to_csv( 'weather_' + name + '.csv')


if __name__ == '__main__':
    main()
