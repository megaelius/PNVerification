import os
import sys
import torch
import argparse
import pandas as pd
from torch import nn, optim
import torch.nn.functional as F

import utils_PN as PN
from utils_NN import NeuralNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Complete and sound verification through branch and bound and interval propagation')
    parser.add_argument('--data_root', type=str, default='./data', help='Root data folder')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Root data folder')
    parser.add_argument('--output', default='./weights_softplus', type=str, help='Root weights folder')
    parser.add_argument('--hidden_size', default = 100, help='Number of hidden dimensions', type = int)
    parser.add_argument('--hidden_layers', default=0, type=int, help='number of hidden layers')
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=200, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--augment', type = int, default =0, help = 'Use data augmentation durin training')
    parser.add_argument('--gamma', default=0.1, type=float, help='Decaying factor for the learning rate')
    parser.add_argument('--optim', default='Adam', type=str, help='Optimizer')
    parser.add_argument('--attack', default=0, type = int, help='Adversarial training with PGD')
    parser.add_argument('--resize', default=0, type = int, help='Size to resize the image in the STL10 case.')
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--batch_norm', default=1, type=int)
    parser.add_argument('--activation', default='softplus', type = str)
    parser.add_argument('--gpu', default=1, type=int, help='use gpu?')
    parser.add_argument('--cr', default=0.01, type=float, help='Curvature regularization parameter')
    args = parser.parse_args()

    train_loader, valid_loader, test_loader, image_size, n_classes, channels_in = PN.load_db(root = args.data_root, name = args.dataset, batch_size=args.batch_size, augment = args.augment, resize = args.resize)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda and args.gpu else 'cpu')
    print('Using GPU?',device)
    # create the model.
    net = NeuralNet(image_size*image_size*channels_in, args.hidden_size, n_classes, args.hidden_layers, BN = args.batch_norm, beta = args.beta, activation = args.activation, device = device)
    # # define the optimizer.
    if args.optim == 'Adam':
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay = args.wd)
    elif args.optim == 'SGD':
        opt = optim.SGD(net.parameters(), lr=args.lr, momentum = 0.9, weight_decay = args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 60, 80],
                                           gamma=args.gamma)

    #define minimization objective (for classification we use cross entropy loss)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.to(device)

    # # aggregate losses and accuracy.
    train_losses, acc_list = [], []
    df = {'epoch':[], 'train_loss':[], 'valid_acc':[],'test_acc':[], 'alpha':[]}
    acc = 0.
    best_acc = 0.
    for epoch in range(args.epochs):
        df['epoch'].append(epoch)
        print('Epoch {} (previous validation accuracy: {:.03f})'.format(epoch, acc))
        if args.augment:
            loss_tr = PN.train(train_loader, net, opt, criterion, epoch, device, attack = args.attack, curvature_reg=args.cr)
        else:
            loss_tr = PN.train(train_loader, net, opt, criterion, epoch, device, cutmix_prob = 0, attack = args.attack, curvature_reg=args.cr)
        acc = PN.test(net, valid_loader, device=device)
        scheduler.step()
        # Save weights if accuracy is improved
        if acc > best_acc:
            os.makedirs(args.output,exist_ok = True)
            torch.save(net,os.path.join(args.output, f'NN_{args.activation}_{args.augment}_{net.n_hidden_layers}x{net.hidden_size}_'+ args.dataset.lower()+ f'_cr_{args.cr}'+ ('_PGD' if args.attack else '') + f'_resize_{args.resize}' + '_best.pt'))
            best_acc = acc
            test_acc = PN.test(net, test_loader, device=device)
            df['test_acc'].append(test_acc)
        else:
            os.makedirs(args.output,exist_ok = True)
            torch.save(net,os.path.join(args.output, f'NN_{args.activation}_{args.augment}_{net.n_hidden_layers}x{net.hidden_size}_'+ args.dataset.lower()+ f'_cr_{args.cr}'+ ('_PGD' if args.attack else '') + f'_resize_{args.resize}' + '_last.pt'))
            df['test_acc'].append(None)

        train_losses.append(loss_tr)
        acc_list.append(acc)
        df['train_loss'].append(loss_tr)
        df['valid_acc'].append(acc)
        df['alpha'].append(float(net.singla_2020(device,0,1).cpu())/2)

        pd_df = pd.DataFrame.from_dict(df,orient='columns')
        pd_df.to_csv(os.path.join(args.output, f'NN_{args.activation}_{args.augment}_{net.n_hidden_layers}x{net.hidden_size}_'+ args.dataset.lower()+ f'_cr_{args.cr}' + ('_PGD' if args.attack else '') + f'_resize_{args.resize}' + '_best.csv'),index=False)
