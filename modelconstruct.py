import pandas as pd
import datetime
import numpy as np

import lightgbm as lgbm


import copy
from joblib import dump, load
import os
import warnings

# warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')


def generate_model(station):
    '''for evey station and pollutant, build the model'''
    for name in pollution_name_list:
        print('> start ', name, ' model construction...')
        tx_name =  station + '_' + name + 'train_x.csv'
        ty_name =  station + '_' + name + 'train_y.csv'
        train_x = pd.read_csv(tx_name, index_col=0)

        for col in train_x.columns[-9:]:
            train_x[col] = train_x[col].astype('category')

        train_y = pd.read_csv(ty_name, index_col=0, header=None)
        #     para_dict[name]=train_x.columns.tolist()
        #     train_x.to_csv(path+'/feature/'+name+'train_x.csv') # save
        #     train_y.to_csv(path+'/feature/'+name+'train_y.csv')
        params = dict(learning_rate=0.01, reg_alpha=0.1, reg_lambda=0.01, max_depth=56, num_leaves=100,
                      n_estimators=500, feature_fraction=0.9, bagging_freq=5, bagging_fraction=0.9)

        model = lgbm.LGBMRegressor(**params)
        model.fit(train_x, train_y)
        print('> complete ', name, ' model construction.')

        save_name =  station + name + '_model.m'
        dump(model, filename=save_name, compress=True)






if __name__ == '__main__':
    begin = datetime.datetime(2017, 1, 2, 0, 0, 0)
    end = datetime.datetime(2018, 4, 30, 23, 0, 0)
    gap = datetime.timedelta(hours=1)
    # time = pd.date_range(begin - 5 * gap, end, freq='H')
    # weather_time_period = pd.date_range(begin - 5 * gap, end + 51 * gap, freq='H')

    path = os.getcwd()
    station_data = pd.read_excel('Beijing_AirQuality_Stations_en.xlsx', header=10).dropna()
    station_data = station_data.set_index(['Station ID'])
    station_name_list = list(station_data.index)
    pollution_name_list = ['PM2.5', 'PM10', 'O3']
    weather_feature_name = ['temperature', 'pressure', 'humidity', 'wind_speed']
    # columns_name_of_all_station=[]

    for station in station_name_list:
        print('now start ', station, '...')
        generate_model(station)
        # columns_name_of_all_station.append(generate_model(station))
    print('All stations done.')
    # pd.DataFrame(columns_name_of_all_station,index=station_name_list)
