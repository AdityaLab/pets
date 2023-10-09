import numpy as np
import random
import os
import argparse
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import argparse

from dataset.covid.get_covid import create_covid_norm_all
from model.rnn import vanilla_rnn, perform_rnn
from model.transformer import vanilla_transformer, perform_transformer
from model.lstnet import vanilla_lstnet, perform_lstnet
from model.informer import vanilla_informer, perform_informer

# start epiweek 202142, end epiweek 202212
start, end = 80, 108 

all_hhs_regions = ['AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
    'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
    'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'NE',
    'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
    'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
    'VA', 'WA', 'WV', 'WI', 'X']


def to_one_hot(a):
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b


def train(model, optim, x, x_shift, y, x_valid, x_shift_valid, y_valid, meta_train, meta_valid, mask_train, mask_shift, device, fps=False, k=8, epoch=8000):
    model.to(device)

    scheduler = MultiStepLR(optim, milestones=[10000], gamma=0.1)

    x = torch.tensor(x, dtype=torch.float32).to(device)
    x_shift = torch.tensor(x_shift, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    meta_train = torch.tensor(meta_train, dtype=torch.float32).to(device)
    mask_train = torch.tensor(mask_train, dtype=torch.float32).to(device)
    mask_shift = torch.tensor(mask_shift, dtype=torch.float32).to(device)

    loss_track = []

    for i in range(epoch):
        model.train()

        if fps:
            pred_trend, pred = model(x, meta_train)
            loss_trend = torch.mean((pred_trend[:,:,:6] - x_shift[:,:,:6])**2 * mask_shift[:,:,:6])
            loss_data = torch.mean((pred[:,-k:] - y[:,-k:])**2 * mask_train[:,-k:])
            loss = loss_trend + loss_data
        else:
            pred = model(x, meta_train)
            loss_data = torch.mean((pred[:,-k:] - y[:,-k:])**2 * mask_train[:,-k:])
            loss = loss_data

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        loss_track.append(loss_data.item())

        if i%1000 == 0:
            print("epoch: {:d}, loss: {:4f}".format(i, loss_track[-1]))
            pred_valid = test(model, x_valid, y_valid, meta_valid, device, k=k, fps=fps)

    return model, np.array(loss_track)


def test(model, x, y, meta, device, k=8, fps=False, mask=None):
    model.to(device)
    model.eval()

    x = torch.tensor(x, dtype=torch.float32).to(device)
    meta = torch.tensor(meta, dtype=torch.float32).to(device)

    with torch.no_grad():
        if fps:
            pred_trend, pred = model(x,meta)
        else:
            pred = model(x,meta)

    pred = pred.cpu().detach().numpy()
    if mask is not None:
        loss = np.mean((pred[:,-k:] - y[:,-k:])**2 * mask[:,-k:])
    else:
        loss = np.mean((pred[:,-k:] - y[:,-k:])**2)

    print("Valid/Test Loss: {:4f}".format(loss))
    return pred


def main():
    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    parser.add_argument('--model', type=str, default='patchtst', help='select model by name') 
    parser.add_argument('--fps', type=int, default=1, help='select model by name') 
    parser.add_argument('--k', type=int, default=8, help='forecasting time steps')          
    parser.add_argument('--seed', type=int, default=0, help='random seed')     
    parser.add_argument('--dev', type=str, default='cuda:0', help='device name')   
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

    print(seed, device, model_name, fps)

    if fps:
        if model_name == 'rnn':
            model = perform_rnn()
        if model_name == 'transformer':
            model = perform_transformer()
        if model_name == 'lstnet':
            model = perform_lstnet()
        if model_name == 'informer':
            model = perform_informer(meta_dim=48, device=device)
    else:
        if model_name == 'rnn':
            model = vanilla_rnn()
        if model_name == 'transformer':
            model = vanilla_transformer()
        if model_name == 'lstnet':
            model = vanilla_lstnet()
        if model_name == 'informer':
            model = vanilla_informer(meta_dim=48, device=device)
    
    optim = Adam(model.parameters(), lr=1e-3)

    loss_tracker = []
    
    for i in tqdm(range(start,end)):
        [x_train, x_shift, y_train, x_test, x_test_shift, y_test], [x_scalers, y_scalers], [mask_train, mask_shift], meta_data = create_covid_norm_all(i)

        meta_test = np.array([meta_data[i] for i in range(0,len(meta_data), int(len(meta_data)/48))])

        num_samples, seq_len, num_features = x_train.shape[0], x_train.shape[-2], x_train.shape[-1]
        
        meta_data = to_one_hot(meta_data.astype(int).reshape(-1)).reshape(num_samples, seq_len, -1)
        meta_test = to_one_hot(meta_test.astype(int).reshape(-1)).reshape(48, seq_len, -1)


        if i == start:
            model, loss_track = train(model, optim, x_train, x_shift, y_train, x_test, x_test_shift, y_test, meta_data, \
                                    meta_test, mask_train, mask_shift, device, fps=fps, k=k, epoch=20000)
        else:
            model, loss_track = train(model, optim, x_train, x_shift, y_train, x_test, x_test_shift, y_test, meta_data, \
                                    meta_test, mask_train, mask_shift, device, fps=fps, k=k, epoch=3000)

        loss_tracker.append(loss_track)

        pred = test(model, x_test, y_test, meta_test, device, k=k, fps=fps)

        for j, region in enumerate(all_hhs_regions):
            scaler = y_scalers[region]
            region_pred = pred[j:j+1]
            true_pred = scaler.inverse_transform(region_pred)

            if fps:
                if not os.path.exists('./covid/res_fps/{}/seed_{}/{}'.format(model_name, seed, region)):
                    os.makedirs('./covid/res_fps/{}/seed_{}/{}'.format(model_name, seed, region))
                np.save('./covid/res_fps/{}/seed_{}/{}/pred_{}.npy'.format(model_name, seed, region, i), true_pred)
            else:
                if not os.path.exists('./covid/res_base/{}/seed_{}/{}'.format(model_name, seed, region)):
                    os.makedirs('./covid/res_base/{}/seed_{}/{}'.format(model_name, seed, region))
                np.save('./covid/res_base/{}/seed_{}/{}/pred_{}.npy'.format(model_name, seed, region, i), true_pred)

        # if fps:
        #     np.save('./covid/res_fps/{}/seed_{}/loss_tracker.npy'.format(model_name, seed), loss_tracker)
        #     torch.save(model.state_dict(), './covid/res_fps/{}/seed_{}/model_{}.pt'.format(model_name, seed, i))
        # else:
        #     np.save('./covid/res_base/{}/seed_{}/loss_tracker.npy'.format(model_name, seed), loss_tracker)
        #     torch.save(model.state_dict(), './covid/res_base/{}/seed_{}/model_{}.pt'.format(model_name, seed, i))
        #     # np.save('./res_shift/{}/seed_{}/loss_tracker.npy'.format(model_name, seed), loss_tracker)
        #     # torch.save(model.state_dict(), './res_shift/{}/seed_{}/model_{}.pt'.format(model_name, seed, i))


if __name__ == '__main__':
    main()