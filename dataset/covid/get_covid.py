import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

feature_key = [
    'retail_and_recreation_percent_change_from_baseline', 
    'grocery_and_pharmacy_percent_change_from_baseline',
    'parks_percent_change_from_baseline', 
    'transit_stations_percent_change_from_baseline', 
    'workplaces_percent_change_from_baseline', 
    'residential_percent_change_from_baseline', 
    # "covidnet",
    "Observed Number",
    "Excess Estimate",
    "positiveIncr_cumulative",
    "positiveIncr",
    "death_jhu_cumulative",
    "death_jhu_incidence",
]


all_hhs_regions = ['AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
    'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
    'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'NE',
    'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
    'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
    'VA', 'WA', 'WV', 'WI', 'X'
]


path = './dataset/covid/covid-hospitalization-all-state-merged_vEW202239.csv'


def get_delay(x, y, k):
    tau = 0
    sim = 0
    for i in range(k+1):
        temp_sim = np.dot(x[i:len(x)-k+i], y[k:])/(np.linalg.norm(x[i:len(x)-k+i])*np.linalg.norm(y[k:]))
        if abs(temp_sim) >= sim:
            sim = abs(temp_sim)
            tau = i
    return tau  


def get_data(path, region):
    raw_data = pd.read_csv(path)
    raw_data = raw_data.loc[raw_data['region'] == region][10:140]
    raw_data = raw_data.fillna(method='ffill')

    for i, key in enumerate(feature_key):
        if i <= 5:
            data = gaussian_filter1d(raw_data[key].to_numpy(), sigma=3)
        else:
            data = raw_data[key].to_numpy()
        try:
            # x_data = np.concatenate((x_data, raw_data[key].to_numpy().reshape(-1,1)), axis=1)
            x_data = np.concatenate((x_data, data.reshape(-1,1)), axis=1)
        except:
            # x_data = raw_data[key].to_numpy().reshape(-1,1)
            x_data = data.reshape(-1,1)

    x_data = x_data
    y_data = raw_data['death_jhu_incidence'].to_numpy()

    return x_data, y_data


def create_window_region(region, current_time, window_size=16, k=8):
    x, y = get_data(path, region)

    x_train, x_shift, y_train = [], [], []
    x_test, y_test = [], []
    train_mask, shift_mask = [], []

    delay_list = []
    for j in range(6):
        tau = get_delay(x[:,j], y, k)
        delay_list.append(tau)

    delay_list = delay_list + [0 for i in range(6)]

    for i in range(window_size+k, current_time+k):
        temp_x = []
        temp_mask = []

        x_train.append(x[i-k-window_size:i-k])

        for j,s in enumerate(delay_list):
            temp_x.append(x[i-k-window_size+s:i-k+s,j])
            if i <= current_time+k-s:
                temp_mask.append(np.ones_like(x_train[-1][:,j]))
            else:
                temp_mask.append(np.concatenate((np.ones_like(x_train[-1][:-(i-(current_time+k-s)),j]), np.zeros_like(x_train[-1][-(i-(current_time+k-s)):,j]))))

        x_shift.append(np.array(temp_x).T)
        shift_mask.append(np.array(temp_mask).T)

        y_train.append(y[i-window_size:i])

        if i <= current_time:
            train_mask.append(np.ones_like(y_train[-1]))
        else:
            train_mask.append(np.concatenate((np.ones_like(y_train[-1][:-(i-current_time)]), np.zeros_like(y_train[-1][-(i-current_time):]))))

    x_test = x[current_time-window_size:current_time]
    x_test_shift = []
    for j,s in enumerate(delay_list):
        x_test_shift.append(x[current_time-window_size+s:current_time+s,j])
    x_test_shift = np.array(x_test_shift).T
    y_test = y[current_time-window_size+k:current_time+k]

    return np.array(x_train), np.array(x_shift), np.array(y_train), np.array(x_test), np.array(x_test_shift), np.array(y_test), np.array(train_mask), np.array(shift_mask)


def create_covid_norm_all(current_time, window_size=16, k=8):
    x_train_scalers = dict(zip(all_hhs_regions, [StandardScaler() for _ in range(len(all_hhs_regions))]))
    y_train_scalers = dict(zip(all_hhs_regions, [StandardScaler() for _ in range(len(all_hhs_regions))]))

    for i, region in enumerate(all_hhs_regions):
        x_train_region, x_shift_region, y_train_region, x_test_region, x_test_shift_region, y_test_region, mask_train_region, mask_shift_region\
            = create_window_region(region, current_time=current_time, window_size=window_size, k=k)
        
        batch_size, seq_len, feature_dim = x_train_region.shape[0], x_train_region.shape[1], x_train_region.shape[2]

        x_train_scalers[region].fit_transform(x_train_region.reshape(-1, feature_dim))
        x_train_region_norm = x_train_scalers[region].transform(x_train_region.reshape(-1, feature_dim)).reshape(batch_size, seq_len, feature_dim)
        x_shift_region_norm = x_train_scalers[region].transform(x_shift_region.reshape(-1, feature_dim)).reshape(batch_size, seq_len, feature_dim)
        x_test_region_norm = x_train_scalers[region].transform(x_test_region.reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim)
        x_test_shift_region_norm = x_train_scalers[region].transform(x_test_shift_region.reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim)

        y_train_scalers[region].fit_transform(y_train_region.reshape(-1, 1))
        y_train_region_norm = y_train_scalers[region].transform(y_train_region.reshape(-1, 1)).reshape(batch_size, window_size)
        y_test_region_norm = y_train_scalers[region].transform(y_test_region.reshape(-1, 1)).reshape(-1, window_size)

        try:
            x_train = np.concatenate((x_train, x_train_region_norm), axis=0)
            x_shift = np.concatenate((x_shift, x_shift_region_norm), axis=0)
            y_train = np.concatenate((y_train, y_train_region_norm), axis=0)

            x_test = np.concatenate((x_test, x_test_region_norm), axis=0)
            x_test_shift = np.concatenate((x_test_shift, x_test_shift_region_norm), axis=0)
            y_test = np.concatenate((y_test, y_test_region_norm), axis=0)

            meta_data = np.concatenate((meta_data, np.ones((batch_size, seq_len, 1)) * i), axis=0)
            mask_train = np.concatenate((mask_train, mask_train_region), axis=0)
            mask_shift = np.concatenate((mask_shift, mask_shift_region), axis=0)

        except:
            x_train, x_shift, y_train = x_train_region_norm, x_shift_region_norm, y_train_region_norm
            x_test, x_test_shift, y_test = x_test_region_norm, x_test_shift_region_norm, y_test_region_norm
            
            meta_data = np.ones((batch_size, seq_len, 1)) * i
            mask_train, mask_shift = mask_train_region, mask_shift_region

    return [x_train, x_shift, y_train, x_test, x_test_shift, y_test], [x_train_scalers, y_train_scalers], [mask_train, mask_shift], meta_data


if __name__ == '__main__':
    [x_train, x_shift, y_train, x_test, x_test_shift, y_test], [x_train_scalers, y_train_scalers], [mask_train, mask_shift], meta_data = create_covid_norm_all(86)
    
    print(mask_train.shape)
    print(mask_shift.shape)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(x_test_shift.shape)

    print(meta_data.shape)

    for i in range(86-24):
        for j in range(5,10):
            plt.plot(range(i,i+16), x_train[i,:,j])
    plt.show()