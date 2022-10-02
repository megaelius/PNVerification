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

import utils_PN as PN
import utils_BaB as UBAB

def solve_gurobi(net,tc,ac,l,u,debug = False):
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', int(debug))
        env.start()
        with gp.Model(env=env) as m:
            # Create decision variables
            z = m.addMVar(len(l),lb = -1.0, ub = 1.0)

            # Declare constraints
            for j in range(len(l)):
                m.addConstr(z[j] >= l[j])
                m.addConstr(z[j] <= u[j])

            # Set objective
            m.setObjective(z @ net.Q @ z + net.q.T @ z + net.beta[tc] - net.beta[ac], GRB.MINIMIZE)
            m.params.NonConvex = 2
            #m.Params.IterationLimit = 15000
            #m.params.OutputFlag = 0

            # Run optimization engine
            m.optimize()
            L = m.getObjective().getValue()
            U = L
            bestz = torch.Tensor(m.getAttr('X')).unsqueeze(0)
    if U < 0:
        sol = 0
    if L > 0:
        sol = 1
    return bestz, L, U, sol

def verify(net,test_loader,args, device):
    start_job = time.time()
    if not hasattr(net,'hidden_sizes'):
        net.hidden_sizes = net.hidden_size

    arch = args.weights.split('/')[1].split('_')[0]
    if args.solver == 'bab':
        if arch == 'Prod':
            if args.resize:
                csv_path = os.path.join(args.output,'test_results_' + args.solver +'_' + args.bounds+ '-' + ('single' if args.alpha_method == 'L' else 'multi') +'_' + args.dataset.lower() + f'_{arch}_{net.architecture}_{net.degree_list}x{net.hidden_sizes}_eps{args.eps}_resize_{args.resize}.csv')
            else:
                csv_path = os.path.join(args.output,'test_results_' + args.solver +'_' + args.bounds+ '-' + ('single' if args.alpha_method == 'L' else 'multi') +'_' + args.dataset.lower() + f'_{arch}_{net.architecture}_{net.degree_list}x{net.hidden_sizes}_eps{args.eps}.csv')
        else:
            if args.resize:
                csv_path = os.path.join(args.output,'test_results_' + args.solver +'_' + args.bounds+ '-' + ('single' if args.alpha_method == 'L' else 'multi') +'_' + args.dataset.lower() + f'_{arch}_{net.n_degree}x{net.hidden_sizes}_eps{args.eps}_resize_{args.resize}.csv')
            else:
                csv_path = os.path.join(args.output,'test_results_' + args.solver +'_' + args.bounds+ '-' + ('single' if args.alpha_method == 'L' else 'multi') +'_' + args.dataset.lower() + f'_{arch}_{net.n_degree}x{net.hidden_sizes}_eps{args.eps}.csv')
    if args.solver == 'gurobi':
        net.hidden_sizes = net.hidden_size
        csv_path = os.path.join(args.output,'test_results_' + args.solver +'_' + args.dataset.lower() + f'_{arch}_{net.n_degree}x{net.hidden_sizes}_eps{args.eps}.csv')

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
                l = torch.zeros((1,net.total_image_size),device = device)
                u = torch.zeros((1,net.total_image_size),device = device)
                for i in range(net.total_image_size):
                    l[0,i] = max(0, z_flat[i]-args.eps)
                    u[0,i] = min(1, z_flat[i]+args.eps)

                pred2 = [(-p,i) for i, p in enumerate(pred[0])]
                start = time.time()
                for order,(p, ac) in enumerate(sorted(pred2)[1:]):
                    #net.compute_min_eig(tc,ac,l,u,device)
                    if args.solver == 'bab':
                        z,L,U,sol = UBAB.solve_BaB(net,tc,ac,l,u,z_flat,time.time(),device,maxtime=args.maxtime,go_for_global_minima = False, picking_rule = args.picking_rule, n_branches = args.n_branches, bounds = args.bounds, bounds_upper = args.bounds_upper, axis_rule = args.axis, optim = args.optim, alpha_method = args.alpha_method, alpha_update_freq = args.alpha_update_freq)
                    elif args.solver == 'gurobi':
                        net.compute_Qq(tc,ac,cpu = args.dataset == 'STL10')
                        z,L,U,sol = solve_gurobi(net,tc,ac,l.squeeze().cpu().numpy(),u.squeeze().cpu().numpy())
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Complete and sound verification through branch and bound and interval propagation')
    parser.add_argument('--weights', default='weights/Prod_Poly_CCP_0_0_order_[4]_hidden_[100]_mnist_resize_0_best.pt', type=str, help='Path of the PN weights')
    parser.add_argument('--data_root', type=str, default='./data', help='Root data folder')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Root data folder')
    parser.add_argument('--eps', default=0.005, type=float, help='Maximum allowed perturbation in l_inf norm')
    parser.add_argument('--maxtime', default=60, type=int, help='Maximum time for the verifier')
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
    args = parser.parse_args()

    #Load network and MNIST dataloader
    weights_data = args.weights.split('_')[-4]
    weights_data2 = args.weights.split('_')[-2]
    os.makedirs(args.output,exist_ok = True)

    if args.dataset.lower() == weights_data or args.dataset.lower() == weights_data2:
        train_loader, valid_loader, test_loader, image_size, n_classes, channels_in = PN.load_db(root = args.data_root, name = args.dataset ,batch_size=64, resize = args.resize)
        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')
        #f = open(args.weights, "r+b")
        #f.seek(0)
        net = torch.load(args.weights, map_location = device)
        #net = pickle.load(f)
        #f.close()

        net.bias = False
        if args.weights.split('/')[1][:4] == 'Prod':
            if net.architecture == 'CCP_Conv':
                for i in range(1,len(net.degree_list)+1):
                    setattr(net,f'Poly{i}' , PN.from_CCP_Conv_to_CCP(getattr(net,f'Poly{i}'), device))
        net.compute_signs(device)
        verify(net,test_loader,args,device)
    else:
        print('ERROR: weights and dataset don\'t match.')
