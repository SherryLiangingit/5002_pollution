import os
import pandas as pd
import numpy as np


def main():

    station_data = pd.read_excel( 'Beijing_AirQuality_Stations_en.xlsx', header=10).dropna()
    station_data = station_data.set_index(['Station ID'])
    station_name_list = list(station_data.index)

    results = pd.DataFrame()
    for station in station_name_list:
        station_result = pd.read_csv( 'prediction_' + station + '.csv', index_col=0)
        station_result['test_id'] = [station + '#' + str(i) for i in range(48)]
        station_result = station_result.set_index('test_id')
        #     print(station_result)
        results = pd.concat([results, station_result], axis=0, sort=False)

    results.to_csv('submission.csv')
    print('ALL DONE! THANK YOU!')

if __name__ == '__main__':
    main()

