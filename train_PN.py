import os
import sys
import torch
import argparse
import pandas as pd
from torch import nn, optim
import torch.nn.functional as F

import utils_PN as PN


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Complete and sound verification through branch and bound and interval propagation')
    parser.add_argument('--data_root', type=str, default='./data', help='Root data folder')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Root data folder')
    parser.add_argument('--output', default='./weights', type=str, help='Root weights folder')
    parser.add_argument('--architecture', default='Prod_Poly', type=str, help='Type of PN architecture')
    parser.add_argument('--arch_in', default='CCP', type=str, help='Type of PN architecture inside the Prod_Poly')
    parser.add_argument('--weight_sharing', default=0, type=int, help='Do weight sharing (1) or not (0)')
    parser.add_argument('--hidden_sizes', nargs="+", default=[100], help='Number of hidden dimensions')
    parser.add_argument('--degree', default=2, type=int, help='Polynomial Network degree')
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--bn', default=0, type=int, help='Batch Normalization')
    parser.add_argument('--degree_list', nargs="+", default=[4], help='List of degrees for Prod_Poly architecture')
    parser.add_argument('--stride_list', nargs="+", default=[1], help='List of convolution strides for Prod_Poly architecture')
    parser.add_argument('--channel_list', nargs="+", default=[1], help='List of convolution channels for Prod_Poly architecture')
    parser.add_argument('--kernel_size', type = int, default = 1, help = 'kernel size for convolutions')
    parser.add_argument('--down_sample', type = int, default = 0, help = 'Down sample after each layer in NCP/CCP_Conv or after each block in Prod_poly')
    parser.add_argument('--augment', type = int, default =0, help = 'Use data augmentation durin training')
    parser.add_argument('--gamma', default=0.1, type=float, help='Decaying factor for the learning rate')
    parser.add_argument('--step_size', default=20, type=int, help='number of epochs after which the learning rate is decayed')
    parser.add_argument('--optim', default='SGD', type=str, help='Optimizer')
    parser.add_argument('--attack', default=0, type = int, help='Adversarial training with PGD')
    parser.add_argument('--resize', default=0, type = int, help='Size to resize the image in the STL10 case.')
    parser.add_argument('--use_w', default=0, type = int, help='use weight in the fourth layer.')
    args = parser.parse_args()

    args.degree_list = [int(a) for a in args.degree_list]
    args.stride_list = [int(a) for a in args.stride_list]
    args.channel_list = [int(a) for a in args.channel_list]
    args.hidden_sizes = [int(a) for a in args.hidden_sizes]

    train_loader, valid_loader, test_loader, image_size, n_classes, channels_in = PN.load_db(root = args.data_root, name = args.dataset, batch_size=args.batch_size, augment = args.augment, resize = args.resize)
    # create the model.
    if args.architecture == 'CCP':
        net = PN.CCP(args.hidden_sizes[0], weight_sharing = args.weight_sharing, image_size=image_size, channels_in = channels_in, n_classes=n_classes, n_degree = args.degree)
        net.apply(net.weights_init)
    elif args.architecture == 'NCP':
        net = PN.NCP(args.hidden_sizes[0], bn = args.bn, down_sample = args.down_sample, image_size=image_size, channels_in = channels_in, n_classes=n_classes, n_degree = args.degree)
    elif args.architecture == 'CCP_Conv':
        net = PN.CCP_Conv(n_channels = args.hidden_sizes[0], downsample_degs = [], n_degree=args.degree, use_w = args.use_w, BN = args.bn, kernel_size=args.kernel_size, n_classes=n_classes, channels_in=channels_in, image_size=image_size)
    elif args.architecture == 'NCP_Conv':
        net = PN.NCP_Conv(channels_in, args.hidden_sizes[0], stride=1, use_alpha=True, kernel_sz=args.kernel_size,norm =True, kernel_size_S=1, n_classes = n_classes, image_size = image_size)
    else:
        net = PN.Prod_Poly(args.hidden_sizes, args.degree_list, args.stride_list, args.channel_list, use_w = args.use_w, kernel_size = args.kernel_size ,BN = args.bn, architecture = args.arch_in, image_size=image_size, channels_in=channels_in, bias=False, n_classes=n_classes)

    # # define the optimizer.
    if args.optim == 'Adam':
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay = args.wd)
    elif args.optim == 'SGD':
        opt = optim.SGD(net.parameters(), lr=args.lr, momentum = 0.9, weight_decay = args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 60, 80],
                                           gamma=args.gamma)
    # define device (cuda if possible)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print('Using GPU?',cuda)
    #define minimization objective (for classification we use cross entropy loss)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.to(device)

    # # aggregate losses and accuracy.
    train_losses, acc_list = [], []
    df = {'epoch':[], 'train_loss':[], 'valid_acc':[],'test_acc':[]}
    acc = 0.
    best_acc = 0.
    for epoch in range(args.epochs):
        df['epoch'].append(epoch)
        print('Epoch {} (previous validation accuracy: {:.03f})'.format(epoch, acc))
        if args.augment:
            loss_tr = PN.train(train_loader, net, opt, criterion, epoch, device, attack = args.attack)
        else:
            loss_tr = PN.train(train_loader, net, opt, criterion, epoch, device, cutmix_prob = 0, attack = args.attack)
        acc = PN.test(net, valid_loader, device=device)
        scheduler.step()
        # Save weights if accuracy is improved
        if acc > best_acc:
            os.makedirs(args.output,exist_ok = True)
            if args.architecture != 'Prod_Poly':
                if not args.weight_sharing:
                    torch.save(net,os.path.join(args.output, f'{args.architecture}_{args.augment}_order_{net.n_degree}_hidden_{args.hidden_sizes[0]}_'+ args.dataset.lower()+ ('_PGD' if args.attack else '')+ f'_resize_{args.resize}' + '_best.pt'))
                else:
                    torch.save(net,os.path.join(args.output, f'{args.architecture}_{args.augment}_order_{net.n_degree}_shared_weights_hidden_{args.hidden_sizes[0]}_'+ args.dataset.lower()+ ('_PGD' if args.attack else '')+ f'_resize_{args.resize}' + '_best.pt'))
            else:
                torch.save(net,os.path.join(args.output, f'{args.architecture}_{args.arch_in}_{args.augment}_{args.bn}_order_{net.degree_list}_hidden_{net.hidden_sizes}_'+ args.dataset.lower()+ ('_PGD' if args.attack else '') + f'_resize_{args.resize}' + '_best.pt'))
            best_acc = acc
            test_acc = PN.test(net, test_loader, device=device)
            df['test_acc'].append(test_acc)
        else:
            df['test_acc'].append(None)

        train_losses.append(loss_tr)
        acc_list.append(acc)
        df['train_loss'].append(loss_tr)
        df['valid_acc'].append(acc)
        pd_df = pd.DataFrame.from_dict(df,orient='columns')
        if args.architecture != 'Prod_Poly':
            if not args.weight_sharing:
                pd_df.to_csv(os.path.join(args.output, f'{args.architecture}_{args.augment}_order_{net.n_degree}_hidden_{net.hidden_size}_'+ args.dataset.lower()+ ('_PGD' if args.attack else '')+ f'_resize_{args.resize}' + '_best.csv'),index=False)
            else:
                pd_df.to_csv(os.path.join(args.output,f'{args.architecture}_{args.augment}_order_{net.n_degree}_shared_weights_hidden_{net.hidden_size}_'+ args.dataset.lower()+ ('_PGD' if args.attack else '')+ f'_resize_{args.resize}' + '_best.csv'),index=False)
        else:
            pd_df.to_csv(os.path.join(args.output, f'{args.architecture}_{args.arch_in}_{args.augment}_{args.bn}_order_{net.degree_list}_hidden_{net.hidden_sizes}_'+ args.dataset.lower()+ ('_PGD' if args.attack else '')+ f'_resize_{args.resize}' + '_best.csv'),index=False)
