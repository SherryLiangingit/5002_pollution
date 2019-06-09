import pandas as pd
import datetime
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import copy
from joblib import dump, load
import os


def get_log(df):
    '''change the value of a dataframe to its log(x+100)'''
    for col in df.columns:
        df[col]=np.log(df[col]+100)
    return df


def get_statistic_pollution(pollution_data):
    '''get statistic data of pollution: calculate according to date, hour, get the mean,max, min and change ratio'''
    grouped_pollution = pollution_data.copy()
    grouped_pollution['date'] = pollution_data.index.map(lambda x: x.date())
    #     grouped_pollution['month']=pollution_data.index.map(lambda x: x.date().month)
    grouped_pollution['hour'] = pollution_data.index.map(lambda x: x.time().hour)
    #     grouped_pollution['week']=pollution_data.index.map(lambda x: x.date().weekday())
    #     grouped_pollution['day']=pollution_data.index.map(lambda x: x.date().day)

    date_pollution = grouped_pollution.groupby('date').mean()[pollution_name_list]
    date_pollution = date_pollution.merge(grouped_pollution.groupby('date').min()[pollution_name_list], on='date')
    date_pollution = date_pollution.merge(grouped_pollution.groupby('date').max()[pollution_name_list], on='date')
    date_pollution.columns = ['PM2.5_MEAN', 'PM10_MEAN', 'O3_MEAN', 'PM2.5_MIN', 'PM10_MIN', 'O3_MIN', 'PM2.5_MAX',
                              'PM10_MAX', 'O3_MAX']

    #     month_pollution=grouped_pollution.groupby('month').mean()[pollution_name_list]
    #     month_pollution=month_pollution.merge(grouped_pollution.groupby('month').max()[pollution_name_list],on='month')
    #     month_pollution.columns=['PM2.5_MEAN','PM10_MEAN','O3_MEAN','PM2.5_MAX','PM10_MAX','O3_MAX']

    hour_pollution = grouped_pollution.groupby('hour').mean()[pollution_name_list]
    for name in pollution_name_list:
        hour_pollution[name + '_ratio'] = pd.DataFrame(
            np.array(hour_pollution[name][1:]) / np.array(hour_pollution[name][:-1]) - 1,
            index=hour_pollution.index[1:])
        #         date_pollution[name+'_ratio']=pd.DataFrame(np.array(date_pollution[name+'_MEAN'][1:])/np.array(date_pollution[name+'_MEAN'][:-1])-1,
        #                                                    index=date_pollution.index[1:])
        hour_pollution.loc[0][name + '_ratio'] = hour_pollution.loc[0][name] / hour_pollution.loc[23][name] - 1

    #     week_pollution=grouped_pollution.groupby('week').mean()[pollution_name_list]
    #     day_pollution=grouped_pollution.groupby('day').mean()[pollution_name_list]

    return date_pollution, hour_pollution


def add_pollution_features(df, name, pollution_data):
    '''generate pollution feature, first get statistic data and then select'''
    df_begin = df.index[0]
    df_end = df.index[-1]

    date_pollution, hour_pollution = get_statistic_pollution(pollution_data)

    df['before_day_mean'] = df.index.map(lambda x: date_pollution.loc[x.date() - 24 * gap][name + '_MEAN'])
    df['before_day_max'] = df.index.map(lambda x: date_pollution.loc[x.date() - 24 * gap][name + '_MAX'])
    df['before_day_min'] = df.index.map(lambda x: date_pollution.loc[x.date() - 24 * gap][name + '_MIN'])

    #     df['monthly_mean']=df.index.map(lambda x: month_pollution.loc[x.date().month][name+'_MEAN'])
    #     df['monthly_max']=df.index.map(lambda x: month_pollution.loc[x.date().month][name+'_MAX'])

    df['hourly_ratio'] = df.index.map(lambda x: hour_pollution.loc[x.time().hour][name + '_ratio'])
    df['hourly_mean'] = df.index.map(lambda x: hour_pollution.loc[x.time().hour][name])

    #     df['weekly_mean']=df.index.map(lambda x:week_pollution.loc[x.date().weekday()][name])
    #     df['day_mean']=df.index.map(lambda x:day_pollution.loc[x.date().day][name])

    return df


def add_weather_features(df, weather_data):
    '''generate weather feature, 3 hours before to 3 hours later'''
    df_begin = df.index[0]
    df_end = df.index[-1]

    for i in range(1, 4):
        for weather in weather_feature_name:
            df[weather + '_0'] = np.array(weather_data[weather][df_begin:df_end])
            df[weather + '_b' + str(i)] = np.array(weather_data[weather][df_begin - i * gap:df_end - i * gap])
            df[weather + '_a' + str(i)] = np.array(weather_data[weather][df_begin + i * gap:df_end + i * gap])
    return df


def check_date(timestamp):
    '''check the special date and time and set a label'''
    holiday = ['2017-01-01', '2017-01-02', '2017-01-27', '2017-01-28', '2017-01-29',
               '2017-01-30', '2017-01-31', '2017-02-01', '2017-02-02', '2017-04-02',
               '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
               '2017-05-28', '2017-05-29', '2017-05-30', '2017-10-01', '2017-10-02',
               '2017-10-03', '2017-10-04', '2017-10-05', '2017-10-06', '2017-10-07',
               '2017-10-08', '2017-12-30', '2017-12-31',
               '2018-01-01', '2018-02-15', '2018-02-16', '2018-02-17', '2018-02-18',
               '2018-02-19', '2018-02-20', '2018-02-21', '2018-04-05', '2018-04-06',
               '2018-04-07', '2018-04-29', '2018-04-30', '2018-05-01', '2018-06-16',
               '2018-06-17', '2018-06-18']

    weekend_work = ['2017-01-22', '2017-02-04', '2017-04-01', '2017-05-27', '2017-09-30',
                    '2018-02-11', '2018-02-24', '2018-04-08', '2018-04-28']

    rest_first_day = ['2017-01-27', '2017-02-05', '2017-04-02', '2017-05-28', '2017-10-01', '2018-02-15', '2018-02-25',
                      '2018-04-05', '2018-04-29']
    rest_last_day = ['2017-01-02', '2017-01-21', '2017-02-02', '2017-02-05', '2017-04-04', '2017-05-01', '2017-05-30',
                     '2018-01-01', '2018-02-21', '2018-04-07', '2018-05-01']

    work_firt_day = ['2017-01-03', '2017-01-22', '2017-02-03', '2017-04-05', '2017-05-02', '2017-05-31', '2018-01-02',
                     '2018-02-11', '2018-02-22', '2018-04-08', '2018-05-02']
    work_last_day = ['2017-01-26', '2017-02-04', '2017-04-01', '2017-05-27', '2017-09-30', '2018-02-14', '2018-02-24',
                     '2018-04-04', '2018-04-28']

    not_rest_first_day = ['2017-01-28', '2017-02-04', '2017-04-01', '2017-05-27', '2017-09-30', '2017-10-07',
                          '2018-02-17',
                          '2018-02-24', '2018-04-07', '2018-04-28']
    not_rest_last_day = ['2017-01-01', '2017-01-22', '2017-02-29', '2017-04-02', '2017-04-30', '2017-05-28',
                         '2017-10-01'
                         '2017-12-31', '2018-02-11', '2018-02-18', '2018-04-08', '2018-04-29']
    not_work_firt_day = ['2017-01-02', '2017-01-23', '2017-01-30', '2017-04-03', '2017-05-01', '2017-05-29',
                         '2017-10-02',
                         '2018-01-01', '2018-02-12', '2018-02-19', '2018-04-09', '2018-04-30']
    not_work_last_day = ['2017-01-27', '2017-02-03', '2017-03-31', '2017-05-26', '2017-09-29', '2017-10-06',
                         '2018-02-16',
                         '2018-02-23', '2018-04-04', '2018-04-06', '2018-04-27']

    holiday_flag = dict()
    for day in holiday:
        holiday_flag[day] = 1

    work_flag = dict()
    for day in weekend_work:
        work_flag[day] = 1

    #     timestamp=timestamp+gap
    day = timestamp.date()

    month_num = timestamp.date().month
    hour_num = timestamp.time().hour

    date_label = ''

    day_str = str(timestamp.date())
    time_str = str(timestamp.time())

    week = day.weekday()
    # weekend?
    if week >= 5:
        date_label = date_label + 'A_'
    else:
        date_label = date_label + 'B_'
    # holiday?
    if day_str in holiday_flag:
        date_label = date_label + 'A_'
    else:
        date_label = date_label + 'B_'
    # work in holiday?
    if (week >= 5 and (day_str not in work_flag)) or day_str in holiday_flag:
        date_label = date_label + 'A_'
    else:
        date_label = date_label + 'B_'
    # rest last day?
    if (week == 6 and (not day_str in not_rest_last_day)) or (day_str in rest_last_day):
        date_label = date_label + 'A_'
    else:
        date_label = date_label + 'B_'
    # work first day?
    if (week == 0 and (not day_str in not_work_firt_day)) or (day_str in work_firt_day):
        date_label = date_label + 'A_'
    else:
        date_label = date_label + 'B_'

    if (week == 4 and (not day_str in not_work_last_day)) or (day_str in work_last_day):
        date_label = date_label + 'A_'
    else:
        date_label = date_label + 'B_'

    if (week == 5 and (not day_str in not_rest_first_day)) or (day_str in rest_first_day):
        date_label = date_label + 'A_'
    else:
        date_label = date_label + 'B_'

    date_label = date_label + str(month_num) + '_' + str(hour_num)

    return date_label


def add_date_features(train):
    '''generate date features according to date label '''
    total_date_features = train.index
    total_date_features = pd.Series(total_date_features.map(lambda x: check_date(x)))

    split_date_df = total_date_features.str.split('_', expand=True)
    #     split_date_df=split_date_df.applymap(lambda x: int(x))
    split_date_df.columns = ['date_feature_' + str(i) for i in range(len(split_date_df.columns))]
    split_date_df.index = train.index

    for col in split_date_df.columns:
        split_date_df[col] = split_date_df[col].astype('category')

    train = pd.concat([train, split_date_df], axis=1, sort=False)

    return train


def get_train_df(name, pollution_data, weather_data):
    '''get all the train data'''
    train = pd.DataFrame(pollution_data[begin - gap:end - gap][name])
    train.index = train.index + gap

    train = add_pollution_features(train, name, pollution_data)

    train = add_weather_features(train, weather_data)

    train = add_date_features(train)

    #     train=train.drop([name],axis=1)

    return train





if __name__ == '__main__':

    begin = datetime.datetime(2017, 1, 2, 0, 0, 0)
    end = datetime.datetime(2018, 4, 30, 23, 0, 0)
    gap = datetime.timedelta(hours=1)

    # print(path)
    station_data = pd.read_excel('Beijing_AirQuality_Stations_en.xlsx', header=10).dropna()
    station_data = station_data.set_index(['Station ID'])
    station_name_list = list(station_data.index)
    pollution_name_list = ['PM2.5', 'PM10', 'O3']
    weather_feature_name = ['temperature', 'pressure', 'humidity', 'wind_speed']

    for station in station_name_list:
        pollution_data = pd.read_csv('pollution_' + station + '.csv', index_col=0)
        weather_data = pd.read_csv('weather_' + station + '.csv', index_col=0)
        pollution_data.index = pollution_data.index.map(lambda x: pd.to_datetime(x))
        weather_data.index = weather_data.index.map(lambda x: pd.to_datetime(x))

        pollution_data = get_log(pollution_data)

        for name in pollution_name_list:
            train_x = get_train_df(name, pollution_data, weather_data)
            train_y = pollution_data[name][begin:end]
            #         para_dict[name]=train_x.columns.tolist()
            train_x.to_csv(station + '_' + name + 'train_x.csv')  # save
            train_y.to_csv(station + '_' + name + 'train_y.csv')

        print(station, 'files saved.')