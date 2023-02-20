import os
import numpy as np
import pandas as pd
import torch
import argparse
import gurobipy as gp
from gurobipy import GRB
import time
from queue import PriorityQueue
import tqdm
import pickle
import subprocess
import onnx
from onnx2pytorch import ConvertModel

import utils_PN as PN
import utils_BaB as UBAB
import utils_NN as NN

def verify(net,test_loader,args, device):
    start_job = time.time()
    if 'cr' in args.weights:
        cr = float(args.weights.split('cr')[2].split('_')[1])
    else:
        cr = 0
    csv_path = os.path.join(args.output,'test_results_' + args.solver +'_' + args.bounds + '-' + args.alpha_method +'_' + args.dataset.lower()+ f'_cr_{cr}' + f'_{net.act_name}_{net.n_hidden_layers}x{net.hidden_size}_eps{args.eps}.csv')
    if args.resume:
        pd_df = pd.read_csv(csv_path)
        df = pd_df.to_dict(orient="list")
    else:
        df = {'Id':[], 'True_class':[], 'Predicted_class':[], 'Logit':[], 'Epsilon':[], 'Time':[], 'Finish_code':[], 'Adversary_class':[], 'Adversary_logit':[], 'L':[], 'U':[], 'Adversary_order':[]}
    #for i, (z, tc) in tqdm.tqdm(enumerate(test_loader.dataset)):
    for i in tqdm.tqdm(range(1000)):
        if i not in df['Id']:
            z, tc = test_loader.dataset[i]
            z = z.to(device)
            if i==1000:break
            df['Id'].append(i)
            df['True_class'].append(tc)
            df['Epsilon'].append(args.eps)

            with torch.no_grad():
                pred = net(z)
            logit, predicted = pred.max(1)
            df['Predicted_class'].append(predicted.item())
            df['Logit'].append(logit.item())
            if predicted == tc:
                z_flat = z.flatten().cpu().numpy()
                l = torch.zeros((1,net.input_size),device = device)
                u = torch.zeros((1,net.input_size),device = device)
                for i in range(net.input_size):
                    l[0,i] = max(0, z_flat[i]-args.eps)
                    u[0,i] = min(1, z_flat[i]+args.eps)

                pred2 = [(-p,i) for i, p in enumerate(pred[0])]
                start = time.time()
                for order,(p, ac) in enumerate(sorted(pred2)[1:]):
                    #net.compute_min_eig(tc,ac,l,u,device)
                    z,L,U,sol = UBAB.solve_BaB(net,tc,ac,l,u,z_flat,time.time(),device,maxtime=args.maxtime,maxit=args.maxit,go_for_global_minima = False, picking_rule = args.picking_rule, n_branches = args.n_branches, bounds = args.bounds, bounds_upper = args.bounds_upper, axis_rule = args.axis, optim = args.optim, alpha_method = args.alpha_method, alpha_update_freq = args.alpha_update_freq, debug = args.debug)
                    if sol == 0 or sol == 2:
                        finish = time.time()
                        df['L'].append(float(L))
                        df['U'].append(float(U))
                        df['Finish_code'].append(sol)
                        df['Adversary_class'].append(ac)
                        df['Adversary_logit'].append(-p.item())
                        df['Adversary_order'].append(order+2)
                        df['Time'].append(finish-start)
                        break
                if sol == 1:
                    finish = time.time()
                    df['L'].append(None)
                    df['U'].append(None)
                    df['Finish_code'].append(1)
                    df['Adversary_class'].append(None)
                    df['Adversary_logit'].append(None)
                    df['Adversary_order'].append(None)
                    df['Time'].append(finish-start)
            else:
                df['L'].append(None)
                df['U'].append(None)
                df['Finish_code'].append(None)
                df['Adversary_class'].append(None)
                df['Adversary_logit'].append(None)
                df['Adversary_order'].append(None)
                df['Time'].append(None)

            pd_df = pd.DataFrame.from_dict(df,orient='columns')
            pd_df.to_csv(csv_path,index=False)

def from_conv_to_lin(conv,channels,input_size,device):
    with torch.no_grad():
        U4 = []
        bias = conv.bias.data.clone()
        conv.bias.data = torch.zeros(conv.bias.data.shape,device='cpu')
        for l in range(channels):
            for j in range(input_size):
                for k in range(input_size):
                    U2 = torch.zeros((1,channels, input_size, input_size), device = 'cpu')
                    U2[:,l,j,k] = 1
                    U3 = conv(U2)
                    if l==0 and j==0 and k==0:
                        channels_out = U3.shape[1]
                        imsize_out = U3.shape[2]
                    U4.append(U3.view(-1,1).clone())
        U4 = torch.cat(U4,dim=-1)
        b = bias.unsqueeze(1).repeat((1,U4.shape[0]//bias.shape[0])).view(U4.shape[0])
    return U4.to(device), b.to(device), channels_out, imsize_out

def from_onnx_to_ours(onnx_net,weights,device):
    '''
    Only valid for ERAN nets
    '''
    if weights[:3]=='con':
        if weights[9:12] == 'SIG':
            activation = 'sigmoid'
        elif weights[9:12] == 'TAN':
            activation = 'tanh'
        #print(activation)
        #print(onnx_net.__dict__.keys())
        #print(onnx_net.__dict__['_modules'])
        net = NN.NeuralNet(784, 100, 10, 2, activation = activation, device = device)
        U1,b1,channels_out,imsize_out  = from_conv_to_lin(onnx_net.__dict__['_modules']['Conv_13'],1,28,device)
        U1 /= onnx_net.__dict__['_modules'][f'Constant_11'].constant.squeeze()
        b1 -= torch.sum(U1*onnx_net.__dict__['_modules'][f'Constant_9'].constant.squeeze(),dim=1)
        U2,b2,channels_out,imsize_out  = from_conv_to_lin(onnx_net.__dict__['_modules']['Conv_15'],channels_out,imsize_out,device)
        net.l1 = torch.nn.Linear(U1.shape[1], U1.shape[0])
        net.l1.weight.data = U1
        net.l1.bias.data = b1
        net.lhidden1 = torch.nn.Linear(U2.shape[1], U2.shape[0])
        net.lhidden1.weight.data = U2
        net.lhidden1.bias.data = b2
        net.lhidden2 = onnx_net.__dict__['_modules'][f'Gemm_18'].to(device)
        net.lout = onnx_net.__dict__['_modules'][f'Gemm_output'].to(device)
        return net

    else:
        if weights[4:7] == 'SIG':
            activation = 'sigmoid'
        elif weights[4:7] == 'TAN':
            activation = 'tanh'
        hidden_size = int(weights[-8:-5])
        hidden_layers = int(weights[-10])-1
        net = NN.NeuralNet(784, hidden_size, 10, hidden_layers, activation = activation, device = device)
        if hidden_layers == 5:
            for i in range(0,12,2):
                if i == 0:
                    net.l1 = onnx_net.__dict__['_modules'][f'Gemm_{20+i}'].to(device)
                    print(onnx_net.__dict__['_modules'][f'Constant_17'].constant)
                    net.l1.weight.data /= onnx_net.__dict__['_modules'][f'Constant_17'].constant.squeeze()
                    net.l1.bias.data -= torch.sum(net.l1.weight.data*onnx_net.__dict__['_modules'][f'Constant_15'].constant.squeeze(),dim=1)
                else:
                    setattr(net,f'lhidden{i//2}',onnx_net.__dict__['_modules'][f'Gemm_{20+i}'].to(device))
        elif hidden_layers == 8:
            for i in range(0,18,2):
                if i == 0:
                    net.l1 = onnx_net.__dict__['_modules'][f'Gemm_{26+i}'].to(device)
                    print(onnx_net.__dict__['_modules'][f'Constant_23'].constant)
                    net.l1.weight.data /= onnx_net.__dict__['_modules'][f'Constant_23'].constant.squeeze()
                    net.l1.bias.data -= torch.sum(net.l1.weight.data*onnx_net.__dict__['_modules'][f'Constant_21'].constant.squeeze(),dim=1)
                else:
                    setattr(net,f'lhidden{i//2}',onnx_net.__dict__['_modules'][f'Gemm_{26+i}'].to(device))
        setattr(net,f'lout',onnx_net.__dict__['_modules'][f'Gemm_output'].to(device))
        return net

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Complete and sound verification through branch and bound and interval propagation')
    #parser.add_argument('--weights', default='./weights_sigmoid/ffnnSIGMOID__Point_6x100.onnx', type=str, help='Path of the PN weights')
    parser.add_argument('--weights', default='/weights_sigmoid_BN/NN_sigmoid_0_0x100_mnist_resize_0_best.pt', type=str, help='Path of the PN weights')
    parser.add_argument('--data_root', type=str, default='./data', help='Root data folder')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Root data folder')
    parser.add_argument('--eps', default=0.1, type=float, help='Maximum allowed perturbation in l_inf norm')
    parser.add_argument('--maxtime', default=60, type=int, help='Maximum time for the verifier')
    parser.add_argument('--maxit', default=None, type=int, help='Maximum number of iterations for the BaB verifier, set to 1 for incomplete verification')
    parser.add_argument('--n_branches', default=2, type=int, help='Number of generated subproblems')
    parser.add_argument('--picking_rule', default='lower', type=str, help='Subproblem picking rule (the subproblem with the smallest of this criterion will be picked first)')
    parser.add_argument('--global_minima', default=0, type=int, help='Go for global minima (True) or cut earlier (False)')
    parser.add_argument('--bounds', default='alpha-conv', type=str, help='Lowerbounding method')
    parser.add_argument('--bounds_upper', default='PGD', type=str, help='Upperbounding method')
    parser.add_argument('--axis', default='widest', type=str, help='Axis picking rule')
    parser.add_argument('--output', default='./verif_results', type=str, help='Path of the output verification file')
    parser.add_argument('--optim', default='PGD', type=str, help='Optimization method for alpha-conv bounds')
    parser.add_argument('--solver', default='bab', type=str, help='Global solver to use (our BaB or Gurobi)')
    parser.add_argument('--resume', default=0, type=int, help='Resume unfinished verification procedure')
    parser.add_argument('--alpha_update_freq', default=1000000, type=int, help='Depth level after which we recompute the minimum eigenvalue')
    parser.add_argument('--resize', default=0, type = int, help='Size to resize the image in the STL10 case.')
    parser.add_argument('--alpha_method', default='L', type=str, help='alpha stimation method')
    parser.add_argument('--gpu', default=1, type=int, help='use gpu?')
    parser.add_argument('--debug', default=0, type=int, help='debug?')
    args = parser.parse_args()

    train_loader, valid_loader, test_loader, image_size, n_classes, channels_in = PN.load_db(root = args.data_root, name = args.dataset ,batch_size=64, resize = args.resize)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda and args.gpu else 'cpu')
    #f = open(args.weights, "r+b")
    #f.seek(0)
    print(args.weights)
    if args.weights[-4:] == 'onnx':
        onnx_net = onnx.load(args.weights)
        #print(onnx_net.__dict__)
        net = ConvertModel(onnx_net)
        z,t = train_loader.dataset[0]
        #print(net(z),t)
        net = from_onnx_to_ours(net,args.weights.split('/')[-1],device)
        #print(net(z))
    else:
        net = torch.load(args.weights, map_location = device)
        net.device = device

    if hasattr(net,'BN') and net.BN:
        net = NN.fuse_BN_with_linear(net)


    net.compute_signs(device)
    os.makedirs(args.output,exist_ok = True)
    verify(net,test_loader,args,device)
