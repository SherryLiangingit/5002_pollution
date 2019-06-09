import os

def main():
    print('Dealing with pollution data...')
    os.system('python pollution_data.py')
    print('Dealing with weather data...')
    os.system('python weather.py')
    print('Feature Engineering...')
    os.system('python feature.py')
    os.system('python modelconstruct.py')
    os.system('python prediction.py')
    os.system('python getresult.py')

if __name__ == '__main__':
    main()
