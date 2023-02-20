import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB
from queue import PriorityQueue
import time
import utils_zonotopes as UZ

'''
--------------------------------------------------------------------------------------------------------
'''

def check_feasible(l,u,z):
    for i in range(len(z)):
        if l[i] - z[i] > 1e-12:
            return False, l[i] - z[i]
        elif z[i] - u[i] > 1e-12 :
            return False, z[i] - u[i]
    return True, None

class PQItem:
    def __init__(self, L, U, l, u, mm, c, axis, depth, alpha, rule = 'upper'):

        self.L = L
        self.U = U
        self.l = l
        self.u = u
        self.mm = mm
        self.c = c
        self.axis = axis
        self.depth = depth
        self.alpha = alpha

        if rule == 'upper':
            self.crit = U
        elif rule == 'width':
            self.crit = (l[0,axis]-u[0,axis])
        elif rule == 'lower':
            self.crit = L

    def __lt__(self, other):
        return self.crit < other.crit

def subproblems(l,u,axis,n_branches):
    ls = []
    us = []
    la, ua = l[0,axis], u[0,axis]
    for i in range(1,n_branches+1):
        ls.append(l.clone())
        us.append(u.clone())
        #s[i-1][0][0,axis] = l[0,axis] + ((i-1)/n_branches)*(u[0,axis]-l[0,axis])
        ls[i-1][0,axis] = ((i-1)/n_branches)*ua + (1 - ((i-1)/n_branches))*la
        us[i-1][0,axis] = ((i)/n_branches)*ua + (1 - ((i)/n_branches))*la
    return torch.cat(ls), torch.cat(us)

def power_method(M,device,shift=0,debug=False):
    if debug:
        print('Computing min_eig:')
    with torch.no_grad():
        d = M.shape[0]
        z = torch.rand((d,1), device = device)
        z = z/(torch.sqrt((z*z).sum()))
        prev_z = torch.ones((d,1), device = device)
        diff = 1
        while not diff < 1e-3:
            #Each iteration we do two steps because if the dominant eigenvalue is negative,
            #with a single step prev_z = -z and never converges.
            prev_z = z.clone()
            z = torch.matmul(M,z) - shift*z
            min_eig = torch.sqrt((z*z).sum())
            sign_eig = 1-2*(((z/prev_z)[0,0]) < 0)
            z = z/min_eig
            z = torch.matmul(M,z) - shift*z
            min_eig = torch.sqrt((z*z).sum())
            z = z/min_eig
            if debug:
                print(min_eig,torch.sum(torch.abs(z-prev_z)))
            diff = torch.sum(torch.abs(z-prev_z))
        return min_eig, sign_eig*min_eig


def PGD(net,device,l,u,tc,ac,alpha, lr = 1, gamma = 0.95, debug = False, start = None):
    prev_z = l.clone()
    if start is None:
        z = ((l+u)/2)
    else:
        z = start
    z.requires_grad = True
    grad = [1 for i in range(z.shape[1])]
    first = True
    while first or not torch.sum(torch.abs(z-prev_z)) < (1e-6):
        z.requires_grad = True
        output = net(z)
        #print(hasattr(alpha, 'shape'))
        #print(((z-l)*(z-u)).shape)
        fz = output[:,tc] - output[:,ac] + torch.sum(alpha*(z-l).squeeze()*(z-u).squeeze())
        net.zero_grad()
        for i in range(fz.shape[0]):
            fz[i].backward(retain_graph = True)
        #sum(fz).backward()
        grad = z.grad.data
        prev_z = z.clone()
        z = torch.where(z - lr*grad > u, u, (z - lr*grad).detach())
        z = torch.where(z < l, l, z)
        lr *= gamma
        if debug:
            print(fz)
        first = False
    return z, fz


def get_bounds(net,tc,ac,l,u,bounds,bounds_upper,device,debug = False, optim = 'PGD'):
    if debug:
        print('Getting bounds:')
    mm = None
    c = None
    if bounds == 'intervals':
        with torch.no_grad():
            lout, uout = net.forward_bounds(l,u)
            L = (lout[0,tc] - uout[0,ac]).item()
            bestz = (l + u)/2

    elif bounds == 'linear':
        mm, c = lin_coefs_global_lowerbound(net.Q,net.q,net.beta,tc,ac,l.squeeze().numpy(),u.squeeze().numpy())
        bestz = torch.zeros(l.shape)
        for i in range(l.shape[1]):
            if mm[i] > 0:
                bestz[0,i] = l[0,i]
            else:
                bestz[0,i] = u[0,i]
        L = bestz.squeeze().numpy() @ mm + c

    elif bounds == 'alpha-conv':
        if optim == 'PGD':
            bestz, L = PGD(net,device,l,u,tc,ac,-net.min_eig,debug = False)
            #bestz, L = PGD(net,device,l,u,tc,ac,-net.min_eig,debug = debug)
        elif optim == 'gurobi':
            lnp = l.squeeze().numpy()
            unp = u.squeeze().numpy()
            alpha = -net.min_eig

            Qp = (net.Q + net.Q.T)/2 + alpha*np.identity(l.shape[1])
            #w,v = np.linalg.eig(Qp)
            #print(w)
            qp = net.q - alpha*(lnp + unp)
            cp = net.beta[tc] - net.beta[ac] + alpha*(lnp @ unp)
            with gp.Env(empty=True) as env:
                    env.setParam('OutputFlag', int(debug))
                    env.start()
                    with gp.Model(env=env) as m:
                        # Create decision variables
                        z = m.addMVar(len(lnp),lb = -1.0, ub = 1.0)

                        # Declare constraints
                        for i in range(len(lnp)):
                            m.addConstr(z[i] >= lnp[i])
                            m.addConstr(z[i] <= unp[i])

                        # Set objective
                        m.setObjective(z @ Qp @ z + qp.T @ z + cp, GRB.MINIMIZE)
                        #m.params.NonConvex = 2
                        #m.Params.IterationLimit = 15000

                        # Run optimization engine
                        m.optimize()
                        L = m.getObjective().getValue()
                        bestz = torch.tensor(m.getAttr('X')).unsqueeze(0)
    elif bounds == 'zonotopes':
        if hasattr(net, 'Poly1'):
            w,c = UZ.abstract_PN(net.Poly1,l,u,device,precise = False)
        else:
            w,c = UZ.abstract_PN(net,l,u,device,precise = False)
        bestz = (l + u)/2
        L = -torch.sum(torch.abs(w[tc] - w[ac])) + c[tc] - c[ac]

    if bounds_upper == 'eval':
        with torch.no_grad():
            output = net(bestz.to(device))
            U = (output[0,tc] - output[0,ac]).item()
    elif bounds_upper == 'PGD':
        if hasattr(net,'activation'):
            _, U = PGD(net,device,l,u,tc,ac,0,lr = 100,debug=False)
        else:
            _, U = PGD(net,device,l,u,tc,ac,0,debug=False)
    return L, U, bestz, mm, c

def get_bounds_batch(net,tc,ac,lb,ub,bounds,bounds_upper,device,mm_old,c_old,l_old,u_old,debug = False, optim = 'PGD'):
    if debug:
        print('Getting bounds:')
    mmb = [None for i in range(lb.shape[0])]
    cb = [None for i in range(lb.shape[0])]
    if bounds == 'intervals':
        with torch.no_grad():
            lout, uout = net.forward_bounds(lb,ub)
            Lb = lout[:,tc] - uout[:,ac]
            zb = (lb + ub)/2

    elif bounds == 'linear':
        Lb = []
        zb = []
        mmb = []
        cb = []
        for i in range(lb.shape[0]):
            mm, c = update_lin_coef(net.Q,l_old.squeeze().numpy(),u_old.squeeze().numpy(),lb[i,:].numpy(),ub[i,:].numpy(),mm_old.copy(),c_old)
            mmb.append(mm)
            cb.append(c)
            #mm2, c2 = UBAB.lin_coefs_global_lowerbound(net.Q,net.q,net.beta,tc,ac,lb[i,:].numpy(),ub[i,:].numpy())
            #print(c_old,c,c2)
            bestz = torch.zeros((1,lb.shape[1]))
            for j in range(lb.shape[1]):
                if mm[j] > 0:
                    bestz[0,j] = lb[i,j]
                else:
                    bestz[0,j] = ub[i,j]
            zb.append(bestz)
            L = bestz.squeeze().numpy() @ mm + c
            Lb.append(L)
        Lb = torch.tensor(Lb)
        zb = torch.cat(zb)

    elif bounds == 'alpha-conv':
        if optim == 'PGD':
            zb = []
            Lb = []
            for l,u in zip(lb,ub):
                #net.compute_min_eig(tc,ac,l.unsqueeze(0),u.unsqueeze(0),device, debug = debug)
                bestz, L = PGD(net,device,l.unsqueeze(0),u.unsqueeze(0),tc,ac,-net.min_eig,debug = debug)
                zb.append(bestz)
                Lb.append(L)
            zb = torch.cat(zb)
            Lb = torch.cat(Lb)
        elif optim == 'gurobi':
            Lb = []
            zb = []
            alpha = -net.min_eig
            for i in range(lb.shape[0]):
                lnp = lb[i,:].squeeze().numpy()
                unp = ub[i,:].squeeze().numpy()

                Qp = (net.Q + net.Q.T)/2 + alpha*np.identity(lb.shape[1])
                #w,v = np.linalg.eig(Qp)
                #print(w)
                qp = net.q - alpha*(lnp + unp)
                cp = net.beta[tc] - net.beta[ac] + alpha*(lnp @ unp)
                with gp.Env(empty=True) as env:
                    env.setParam('OutputFlag', int(debug))
                    env.start()
                    with gp.Model(env=env) as m:
                        # Create decision variables
                        z = m.addMVar(len(lnp),lb = -1.0, ub = 1.0)

                        # Declare constraints
                        for j in range(len(lnp)):
                            m.addConstr(z[j] >= lnp[j])
                            m.addConstr(z[j] <= unp[j])

                        # Set objective
                        m.setObjective(z @ Qp @ z + qp.T @ z + cp, GRB.MINIMIZE)
                        #m.params.NonConvex = 2
                        #m.Params.IterationLimit = 15000
                        #m.params.OutputFlag = 0

                        # Run optimization engine
                        m.optimize()
                        L = m.getObjective().getValue()
                        bestz = torch.tensor(m.getAttr('X')).unsqueeze(0)
                        Lb.append(L)
                        zb.append(bestz)
            Lb = torch.tensor(Lb)
            zb = torch.cat(zb)
    elif bounds == 'zonotopes':
        zb = []
        Lb = []
        for l,u in zip(lb,ub):
            if hasattr(net, 'Poly1'):
                w,c = UZ.abstract_PN(net.Poly1,l,u,device,precise = False)
            else:
                w,c = UZ.abstract_PN(net,l,u,device,precise = False)
            bestz = (l + u)/2
            L = (-torch.sum(torch.abs(w[tc] - w[ac])) + c[tc] - c[ac]).unsqueeze(0)
            zb.append(bestz)
            Lb.append(L)
        zb = torch.cat(zb)
        Lb = torch.cat(Lb)

    if bounds_upper == 'eval':
        with torch.no_grad():
            output = net(zb.to(device))
            Ub = output[:,tc] - output[:,ac]
    elif bounds_upper == 'PGD':
        if hasattr(net,'activation'):
            zb, Ub = PGD(net,device,lb,ub,tc,ac,0,lr = 100,debug=False)
        else:
            zb, Ub = PGD(net,device,lb,ub,tc,ac,0,debug=False)
    return Lb, Ub, zb, mmb, cb

def get_best_axis(net,device,tc,ac,l,u,alpha,l_diag_h,u_diag_h,rule='widest'):
    if rule == 'widest':
        return torch.argmax((u-l).squeeze())
    elif rule == 'intervals':
        widths = []
        for i in range(l.shape[1]):
            ls, us = subproblems(l,u,i,2)
            L,U,_,_,_ = get_bounds_batch(net,tc,ac,ls,us,'intervals','eval',device,None,None,None,None)
            #print(sum(U-L))
            widths.append((U-L).sum().item())
        return np.argmin(widths)
    elif rule == 'alphas' and hasattr(alpha, 'shape'):
        #return torch.argmax(-(u-l).squeeze()*alpha.squeeze())
        return torch.argmax(-(u-l).squeeze()*alpha.squeeze())
    elif rule == 'hess_diag':
        return torch.argmax((u-l).squeeze()*(u_diag_h-l_diag_h).squeeze())


def solve_BaB(net,tc,ac,l,u,z0,start_time,device,maxtime = None,maxit = None,go_for_global_minima = False, picking_rule = 'upper', n_branches = 4, bounds = 'intervals', bounds_upper = 'PGD', axis_rule = 'intervals', optim = 'PGD', alpha_update_freq = 100, alpha_method = 'L', debug = False, track_improvement = False):
    d = u.shape[0]
    tol = 1e-6
    #l = torch.Tensor(l).unsqueeze(0).to(device)
    #u = torch.Tensor(u).unsqueeze(0).to(device)
    #print(u.shape)
    if axis_rule == 'hess_diag':
        l_diag_h, u_diag_h = net.bounds_diag_hess(tc,ac,l,u,device)
    else:
        l_diag_h, u_diag_h = None, None
    if bounds == 'alpha-conv':
        net.compute_min_eig(tc,ac,l,u,device, method = alpha_method, debug = debug)
    L,U,bestz,mm,c = get_bounds(net,tc,ac,l,u,bounds,bounds_upper,device,debug = debug, optim = optim)
    L = float(L)
    U = float(U)
    if bounds == 'alpha-conv':
        alpha = net.min_eig
    else: alpha = None
    P = PriorityQueue()
    axis = get_best_axis(net,device,tc,ac,l,u,alpha,l_diag_h,u_diag_h,rule=axis_rule)
    P.put(PQItem(L,U,l,u,mm,c,axis,1,alpha,rule = picking_rule))
    Ls = {L:1}
    if debug:
        print(L,U)
    if maxit is not None:
        track_L = [L]
        track_times = [time.time()-start_time]
    if track_improvement:
        track_L = [L]
        track_times = [time.time()-start_time]
    i = 0
    while not P.empty() and not U-L < tol and (maxit is None or i < maxit) and(go_for_global_minima or (L < 0 and U > 0)) and (maxtime is None or time.time() - start_time < maxtime):
        item = P.get()
        Lp, Up, lp, up, mmp, cp, axis, depth , alpha = item.L, item.U, item.l, item.u, item.mm, item.c, item.axis, item.depth, item.alpha
        if Ls[Lp] == 1:
            del Ls[Lp]
        else:
            Ls[Lp] -= 1
        if bounds == 'alpha-conv':
            if not depth%alpha_update_freq:
                net.compute_min_eig(tc,ac,lp,up,device, method = alpha_method,debug = debug)
                alpha = net.min_eig
            else:
                net.min_eig = alpha
        if debug and not i%1:
            print(L,U,Lp,Up)
            print(depth)
        if Lp < U and (go_for_global_minima or (Lp < 0 and Up > 0)):
            '''
            branch
            '''
            ls, us = subproblems(lp,up,axis,n_branches)
            '''
            bound
            '''
            Lb,Ub,zb,mmb,cb = get_bounds_batch(net,tc,ac,ls,us,bounds,bounds_upper,device,mmp,cp,lp,up,debug = debug,optim = optim)
            for j in range(n_branches):
                Lpp = Lb[j].item()
                Upp = Ub[j].item()
                zp = zb[j]
                if debug or track_improvement:
                    print(i,j,Lpp,Upp,axis)
                if Lpp < U:
                    if Lpp < 0 or go_for_global_minima:
                        if Upp < U :
                            '''
                            update upperbound
                            '''
                            U = float(Upp)
                            bestz = zp

                        if (not go_for_global_minima) and Upp < 0:
                            #An adversarial example exists in [lp,up]
                            if track_improvement:
                                return bestz, L, U, 0, track_L, track_times
                            else:
                                return bestz, L, U, 0
                        axisp = get_best_axis(net,device,tc,ac,lp,up,alpha,l_diag_h,u_diag_h,rule=axis_rule)
                        P.put(PQItem(float(Lpp),float(Upp),ls[j].unsqueeze(0),us[j].unsqueeze(0),mmb[j],cb[j],axisp,depth+1,alpha,rule = picking_rule))
                        if Lpp in Ls:
                            Ls[Lpp] += 1
                        else:
                            Ls[Lpp] = 1
        i+=1
        if maxit is not None or track_improvement:
            print(i)
            track_L.append(L)
            track_times.append(time.time()-start_time)
        #If Ls is empty, it means that there was a L<0 that lead to two
        # subproblems with LPP > 0
        if len(Ls) == 0:
            if track_improvement:
                return bestz, Lpp, U, 1, track_L, track_times
            else:
                return bestz, Lpp, U, 1
        else:
            L = min(Ls.keys())
        #print(sorted(Ls.keys()))
    if L > 0:
        '''
        Property is verified
        '''
        if track_improvement:
            return bestz, L, U, 1, track_L, track_times
        else:
            return bestz, L, U, 1
    elif U < 0:
        '''
        Property is falsified
        '''
        if track_improvement:
            return bestz, L, U, 0, track_L, track_times
        else:
            return bestz, L, U, 0
    else:
        '''
        Ran out of time
        '''
        if track_improvement:
            return bestz, L, U, 2, track_L, track_times
        else:
            return bestz, L, U, 2
    return None,None,None,None
