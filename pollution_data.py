import pandas as pd
import numpy as np
import copy
import datetime
import os


def fill_data(data_df, pollution_name_list, time_mean_df):
    '''fill in missing data'''
    for item in pollution_name_list:
        index_list = data_df[data_df[item].isna()].index
        for timeindex in index_list:
            data_df.loc[timeindex][item] = time_mean_df.loc[timeindex][item].copy()
    return data_df


def main():
    begin = datetime.datetime(2017, 1, 1, 19, 0, 0)
    end = datetime.datetime(2018, 4, 30, 23, 0, 0)
    gap = datetime.timedelta(hours=1)
    time = pd.date_range(begin - 5 * gap, end, freq='H')


    data3 = pd.read_csv( 'aiqQuality_201804.csv')
    data2 = pd.read_csv( 'airQuality_201802-201803.csv')
    data1 = pd.read_csv( 'airQuality_201701-201801.csv')
    column_name = list(data2.columns)
    data3 = data3.drop(['id'], axis=1)
    data3.columns = column_name
    data = pd.concat([data1, data2, data3], sort=False)
    data = data.drop_duplicates(keep='first')
    data.index = range(len(data))

    station_list = data['stationId'].dropna().unique()
    # station_list
    station_group_df = data.groupby('stationId')
    grouped_data_list = [station_group_df.get_group(x) for x in station_list]

    time_mean_df = data.groupby('utc_time').mean()
    time_mean_df.index = pd.to_datetime(time_mean_df.index)
    time_mean_df = time_mean_df.reindex(time, fill_value=np.nan)
    time_mean_df = time_mean_df.fillna(method='bfill')
    # time_mean_df.isna().any()

    pollution_name_list = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']

    pollution_data = []
    for station_df in grouped_data_list:
        new_station_df = station_df.drop(['stationId'], axis=1)
        new_station_df['time'] = pd.to_datetime(new_station_df['utc_time'])
        new_station_df = new_station_df.set_index(['time']).drop(['utc_time'], axis=1)
        new_station_df = new_station_df.reindex(time, fill_value=np.nan)
        new_station_df = fill_data(new_station_df, pollution_name_list, time_mean_df)
        # print(len(new_station_df))
        pollution_data.append(new_station_df)

    # save
    for name, df in zip(station_list, pollution_data):
        df.to_csv( 'pollution_' + name + '.csv')


if __name__ == '__main__':
    main()
