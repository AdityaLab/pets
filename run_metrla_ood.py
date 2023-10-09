import numpy as np
import random
import os
import argparse
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import wget
import argparse
from sklearn.preprocessing import StandardScaler

from model.rnn import vanilla_rnn, perform_rnn
from model.transformer import vanilla_transformer, perform_transformer
from model.lstnet import vanilla_lstnet, perform_lstnet
from model.informer import vanilla_informer, perform_informer

from dataset.metrla.get_metrla import get_data


def train(model, optim, x, x_shift, y, x_valid, x_shift_valid, y_valid, device, fps=False, k=8, epoch=8000):
    model.to(device)

    scheduler = MultiStepLR(optim, milestones=[20000], gamma=0.1)

    x = torch.tensor(x, dtype=torch.float32).to(device)
    x_shift = torch.tensor(x_shift, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    loss_track = []

    for i in range(epoch):
        model.train()

        if fps:
            pred_trend, pred = model(x)
            loss_trend = torch.mean((pred_trend[:,:,:6] - x_shift[:,:,:6])**2)
            loss_data = torch.mean((pred[:,-k:] - y[:,-k:])**2)
            loss = loss_trend + loss_data
        else:
            pred = model(x)
            loss_data = torch.mean((pred[:,-k:] - y[:,-k:])**2)
            loss = loss_data

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        loss_track.append(loss_data.item())

        if i%1000 == 0:
            print("epoch: {:d}, loss: {:4f}".format(i, loss_track[-1]))
            pred_valid = test(model, x_valid, y_valid, device, k=k, fps=fps)

    return model, np.array(loss_track)



def test(model, x, y, device, k=8, fps=False, mask=None):
    model.to(device)
    model.eval()

    x = torch.tensor(x, dtype=torch.float32).to(device)

    with torch.no_grad():
        if fps:
            pred_trend, pred = model(x)
        else:
            pred = model(x)

    pred = pred.cpu().detach().numpy()
    if mask is not None:
        loss = np.mean((pred[:,-k:] - y[:,-k:])**2 * mask[:,-k:])
    else:
        loss = np.mean((pred[:,-k:] - y[:,-k:])**2)

    print("Valid/Test Loss: {:4f}".format(loss))
    return pred



def main():
    url = "https://zenodo.org/record/5146275/files/METR-LA.csv?download=1"
    wget.download(url, out='./dataset/metrla/')

    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    parser.add_argument('--model', type=str, default='lstnet', help='select model by name') 
    parser.add_argument('--fps', type=int, default=0, help='select model by name') 
    parser.add_argument('--k', type=int, default=24, help='forecasting time steps')          
    parser.add_argument('--seed', type=int, default=0, help='random seed')     
    parser.add_argument('--dev', type=str, default='cpu', help='device name')    
    args = parser.parse_args()

    device = args.dev

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model_name = args.model 
    fps = args.fps != 0
    k = args.k
    in_dim = 16
    perf_dim = in_dim-1

    print(seed, device, model_name, fps)

    if fps:
        if model_name == 'rnn':
            model = perform_rnn(in_dim=in_dim, window=48, meta_dim=0, perf_dim=perf_dim)
        if model_name == 'transformer':
            model = perform_transformer(in_dim=in_dim, window=48, meta_dim=0, perf_dim=perf_dim)
        if model_name == 'lstnet':
            model = perform_lstnet(in_dim=in_dim, window=48, meta_dim=0, perf_dim=perf_dim)
        if model_name == 'informer':
            model = perform_informer(enc_in=in_dim, dec_in=in_dim, meta_dim=0, perf_dim=perf_dim, device=device)
    else:
        if model_name == 'rnn':
            model = vanilla_rnn(in_dim=in_dim, window=48, meta_dim=0)
        if model_name == 'transformer':
            model = vanilla_transformer(in_dim=in_dim, window=48, meta_dim=0)
        if model_name == 'lstnet':
            model = vanilla_lstnet(in_dim=in_dim, window=48, meta_dim=0)
        if model_name == 'informer':
            model = vanilla_informer(enc_in=in_dim, dec_in=in_dim, meta_dim=0, device=device)
    
    optim = Adam(model.parameters(), lr=1e-4)

    x, x_shift, y, _, _, _, _, _ = get_data(2000, k=k, use_col=in_dim)

    x_train, x_shift_train, y_train = x[:1500], x_shift[:1500], y[:1500]
    x_test, x_shift_test, y_test = x[-452:], x_shift[-452:], y[-452:]

    window, features = x_train.shape[1], x_train.shape[2]

    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_scaler.fit_transform(x_train.reshape(-1,features))
    y_scaler.fit_transform(y_train.reshape(-1,1))

    x_train = x_scaler.transform(x_train.reshape(-1,features)).reshape(-1,window,features)
    x_shift_train = x_scaler.transform(x_shift_train.reshape(-1,features)).reshape(-1,window,features)
    x_test = x_scaler.transform(x_test.reshape(-1,features)).reshape(-1,window,features)
    x_shift_test = x_scaler.transform(x_shift_test.reshape(-1,features)).reshape(-1,window,features)
    
    y_train = y_scaler.transform(y_train.reshape(-1,1)).reshape(-1,window)
    y_test = y_scaler.transform(y_test.reshape(-1,1)).reshape(-1,window)

    model, loss_track = train(model, optim, x_train, x_shift_train, y_train, x_test, x_shift_test, y_test, device, fps=fps, k=k, epoch=30000)

    pred = test(model, x_test, y_test, device, k=k, fps=fps)

    pred = y_scaler.inverse_transform(pred.reshape(-1,1)).reshape(-1,window)


    if fps:
        if not os.path.exists('./metrla_ood/res_fps/{}/seed_{}'.format(model_name, seed)):
            os.makedirs('./metrla_ood/res_fps/{}/seed_{}'.format(model_name, seed))
        np.save('./metrla_ood/res_fps/{}/seed_{}/pred_{}.npy'.format(model_name, seed, 6000), pred)
    else:
        if not os.path.exists('./metrla_ood/res_base/{}/seed_{}'.format(model_name, seed)):
            os.makedirs('./metrla_ood/res_base/{}/seed_{}'.format(model_name, seed))
        np.save('./metrla_ood/res_base/{}/seed_{}/pred_{}.npy'.format(model_name, seed, 6000), pred)


if __name__ == '__main__':
    main()