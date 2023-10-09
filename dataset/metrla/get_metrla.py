import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

path = './dataset/metrla/METR-LA.csv'

def get_delay(x, y, k):
    tau = 0
    sim = 0
    for i in range(k+1):
        temp_sim = np.dot(x[i:len(x)-k+i], y[k:])/(np.linalg.norm(x[i:len(x)-k+i])*np.linalg.norm(y[k:]))
        if abs(temp_sim) >= sim:
            sim = abs(temp_sim)
            tau = i
    return tau  

def get_data(current_time=2000, window_size=48, k=24, use_col=24):
    raw_data = pd.read_csv(path, skiprows=20000, usecols=range(1,use_col+1)).replace(to_replace=0, method='ffill').to_numpy()


    for i in range(use_col):
        if i <= use_col-2:
            data = gaussian_filter1d(raw_data[:,i], sigma=3)
        else:
            data = raw_data[:,i]
        try:
            # x_data = np.concatenate((x_data, raw_data[key].to_numpy().reshape(-1,1)), axis=1)
            x_data = np.concatenate((x_data, data.reshape(-1,1)), axis=1)
        except:
            # x_data = raw_data[key].to_numpy().reshape(-1,1)
            x_data = data.reshape(-1,1)
    # x_data = raw_data

    x = x_data
    y = x_data[:,-1:]

    x_train, x_shift, y_train = [], [], []
    x_test, y_test = [], []
    train_mask, shift_mask = [], []

    delay_list = []
    for j in range(use_col-1):
        tau = get_delay(x[:,j], y, k)
        delay_list.append(tau)

    delay_list = delay_list + [0]

    # print(delay_list)

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

    return np.array(x_train), np.array(x_shift), np.array(y_train).squeeze(), \
        np.expand_dims(np.array(x_test),axis=0), np.expand_dims(np.array(x_test_shift),axis=0), np.array(y_test).reshape(1,-1), \
        np.array(train_mask).squeeze(), np.array(shift_mask)

if __name__ == '__main__':
    use_col = 16
    l = 1500
    # x_train, x_shift, y_train, x_test, x_test_shift, y_test, train_mask, shift_mask = get_data(l, use_col=use_col)
    x, x_shift, y, _, _, _, _, _ = get_data(l, k=24, use_col=use_col)


    # print(x.shape)
    # raise Exception('stop')
    x_train, x_shift_train, y_train = x[:2000], x_shift[:2000], y[:2000]
    x_valid, x_shift_valid, y_valid = x[2000:2452], x_shift[2000:2452], y[2000:2452]
    x_test, x_shift_test, y_test = x[-452:], x_shift[-452:], y[-452:]


    # print(y_test.shape)
    # print(x_test_shift.shape)


    for i in range(452):
        # for j in range(use_col):
        plt.plot(range(i,i+48), y_test[i,-48:])
    plt.show()