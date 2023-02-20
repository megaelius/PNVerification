import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

import utils_BaB as UBAB

def fuse_BN_with_linear(net):
    for i in range(1,net.n_hidden_layers+1):
        k,d = getattr(net,f'lhidden{i}').weight.data.shape
        li2 = torch.nn.Linear(d,k)
        li2.weight.data = getattr(net,f'lhidden{i}').weight.data.clone()
        li2.bias.data = getattr(net,f'lhidden{i}').bias.data.clone()
        li2.weight.data *= getattr(net,f'BN{i}').weight.data.unsqueeze(1)*(1/torch.sqrt(getattr(net,f'BN{i}').running_var.data.unsqueeze(1) + getattr(net,f'BN{i}').eps))
        li2.bias.data *= getattr(net,f'BN{i}').weight.data*(1/torch.sqrt(getattr(net,f'BN{i}').running_var.data + getattr(net,f'BN{i}').eps))
        li2.bias.data += -  getattr(net,f'BN{i}').weight.data*getattr(net,f'BN{i}').running_mean.data*(1/torch.sqrt(getattr(net,f'BN{i}').running_var.data + getattr(net,f'BN{i}').eps)) + getattr(net,f'BN{i}').bias.data
        setattr(net,f'lhidden{i}',li2)
    net.BN = False
    return net

'''
Softplus:
'''
class grad_softplus(nn.Module):
    def __init__(self,beta):
        super(grad_softplus, self).__init__()
        self.beta = beta
        self.sig = nn.Sigmoid()
    def forward(self,x):
        return self.sig(self.beta*x)
class hess_softplus(nn.Module):
    def __init__(self,beta):
        super(hess_softplus, self).__init__()
        self.beta = beta
        self.sig = nn.Sigmoid()
        self.grad = grad_softplus(beta)
    def forward(self,x):
        return self.beta*self.grad(x)*(1-self.grad(x))

'''
Sigmoid
'''
class grad_sigmoid(nn.Module):
    def __init__(self):
        super(grad_sigmoid, self).__init__()
        self.sig = nn.Sigmoid()
    def forward(self,x):
        return self.sig(x)*(1-self.sig(x))

class hess_sigmoid(nn.Module):
    def __init__(self):
        super(hess_sigmoid, self).__init__()
        self.sig = nn.Sigmoid()
    def forward(self,x):
        return self.sig(x)*(1-self.sig(x))*(1-self.sig(x)) - self.sig(x)*self.sig(x)*(1-self.sig(x))

'''
Tanh
'''
class grad_tanh(nn.Module):
    def __init__(self):
        super(grad_tanh, self).__init__()
        self.tanh = nn.Tanh()
    def forward(self,x):
        return 1-self.tanh(x)**2

class hess_tanh(nn.Module):
    def __init__(self):
        super(hess_tanh, self).__init__()
        self.tanh = nn.Tanh()
    def forward(self,x):
        return -2*self.tanh(x)*(1-self.tanh(x)**2)

'''
GeLU
'''
class grad_gelu(nn.Module):
    def __init__(self,device):
        super(grad_gelu, self).__init__()
        self.device=device
    def forward(self,x):
        return torch.tensor(norm.cdf(x.cpu().numpy()),device=self.device) + x*torch.tensor(norm.pdf(x.cpu().numpy()),device=self.device)

class hess_gelu(nn.Module):
    def __init__(self,device):
        super(hess_gelu, self).__init__()
        self.device=device
    def forward(self,x):
        return 2*torch.tensor(norm.pdf(x.cpu().numpy()),device=self.device) - 2*x*x*torch.tensor(norm.pdf(x.cpu().numpy()),device=self.device)
'''
SiLU
'''
class grad_silu(nn.Module):
    def __init__(self):
        super(grad_silu, self).__init__()
        self.sig = torch.nn.Sigmoid()
    def forward(self,x):
        return self.sig(x) + x*self.sig(x)*(1-self.sig(x))
class hess_silu(nn.Module):
    def __init__(self):
        super(hess_silu, self).__init__()
        self.sig = torch.nn.Sigmoid()
    def forward(self,x):
        return 2*self.sig(x)*(1-self.sig(x)) + x*self.sig(x)*(1-self.sig(x))*(1-self.sig(x)) - x*self.sig(x)*self.sig(x)*(1-self.sig(x))

'''
Sin
'''
class act_sin(nn.Module):
    def __init__(self):
        super(act_sin, self).__init__()
    def forward(self,x):
        return torch.sin(x)
class grad_sin(nn.Module):
    def __init__(self):
        super(grad_sin, self).__init__()
    def forward(self,x):
        return torch.cos(x)
class hess_sin(nn.Module):
    def __init__(self):
        super(hess_sin, self).__init__()
    def forward(self,x):
        return -torch.sin(x)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes, n_hidden_layers, BN = False, activation = "softplus", beta = 10, device = 'cpu'):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.l1 = nn.Linear(input_size, hidden_size)
        self.BN = BN
        self.act_name = activation
        self.device = device
        if activation == 'softplus':
            self.activation = nn.Softplus(beta = beta)#nn.Sigmoid() #nn.ReLU()
            self.sig = nn.Sigmoid()
            #self.grad_act = lambda x: self.sig(beta*x)
            self.grad_act = grad_softplus(beta)
            #self.hess_act = lambda x: beta*self.grad_act(x)*(1-self.grad_act(x))
            self.hess_act = hess_softplus(beta)
            self.argmax_grad_act = None
            self.argmin_grad_act = None
            self.max_grad = 1
            self.argmax_hess_act = torch.zeros(1, device = device)
            self.argmin_hess_act = None
            self.max_hess = self.hess_act(self.argmax_hess_act).item()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
            #self.grad_act = lambda x: self.sig(beta*x)
            self.grad_act = grad_sigmoid()
            #self.hess_act = lambda x: beta*self.grad_act(x)*(1-self.grad_act(x))
            self.hess_act = hess_sigmoid()
            self.argmax_grad_act = torch.zeros(1, device = device)
            self.argmin_grad_act = None
            self.argmax_hess_act = torch.log(torch.full([1],float(2.0-np.sqrt(3.0)), device = device))
            self.argmin_hess_act = torch.log(torch.full([1],float(2.0+np.sqrt(3.0)), device = device))
            self.max_hess = self.hess_act(self.argmax_hess_act).item()
            self.min_hess = self.hess_act(self.argmin_hess_act).item()
            self.max_grad = self.grad_act(self.argmax_grad_act).item()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
            #self.grad_act = lambda x: self.sig(beta*x)
            self.grad_act = grad_tanh()
            #self.hess_act = lambda x: beta*self.grad_act(x)*(1-self.grad_act(x))
            self.hess_act = hess_tanh()
            self.argmax_grad_act = torch.zeros(1, device = device)
            self.argmin_grad_act = None
            self.argmax_hess_act = 0.5*torch.log(torch.full([1],float(2.0-np.sqrt(3.0)), device = device))
            self.argmin_hess_act = 0.5*torch.log(torch.full([1],float(2.0+np.sqrt(3.0)), device = device))
            self.max_hess = self.hess_act(self.argmax_hess_act).item()
            self.min_hess = self.hess_act(self.argmin_hess_act).item()
            self.max_grad = self.grad_act(self.argmax_grad_act).item()
        elif activation == 'gelu':
            self.activation = torch.nn.GELU()
            self.grad_act = grad_gelu(device)
            self.hess_act = hess_gelu(device)
            self.argmax_grad_act = torch.full([1],1, device = device)
            self.argmin_grad_act = torch.full([1],-1, device = device)
            self.argmax_hess_act = torch.zeros(1, device = device)
            self.argmin_hess_act = None
            self.max_hess = self.hess_act(self.argmax_hess_act).item()
            self.min_hess = torch.zeros(1, device = device)
            self.max_grad = self.grad_act(self.argmax_grad_act).item()
        elif activation == 'silu':
            self.activation = torch.nn.SiLU()
            self.grad_act = grad_silu()
            self.hess_act = hess_silu()
            self.argmax_grad_act = None
            self.argmin_grad_act = None
            self.argmax_hess_act = torch.zeros(1, device = device)
            self.argmin_hess_act = None
            self.max_hess = self.hess_act(self.argmax_hess_act).item()
            self.min_hess = torch.zeros(1, device = device)
            self.max_grad = 1.1
        elif activation == 'sin':
            self.activation = act_sin()
            self.grad_act = grad_sin()
            self.hess_act = hess_sin()
            self.argmax_grad_act = torch.zeros(1, device = device)
            self.argmin_grad_act = None
            self.argmax_hess_act = torch.full([1],-np.pi/2, device = device)
            self.argmin_hess_act = None
            self.max_hess = self.hess_act(self.argmax_hess_act).item()
            self.min_hess = -self.hess_act(self.argmax_hess_act).item()
            self.max_grad = self.grad_act(self.argmax_grad_act).item()
        elif activation == 'relu':
            self.activation = nn.ReLU()
            '''
            Gradients and hessians are not defined everywhere
            '''

        self.n_hidden_layers = n_hidden_layers
        for i in range(1,n_hidden_layers+1):
            setattr(self, f'lhidden{i}', nn.Linear(hidden_size, hidden_size))
            if BN:
                setattr(self, f'BN{i}', torch.nn.BatchNorm1d(hidden_size))
        self.lout = nn.Linear(hidden_size, n_classes)

    def build_model(self,init_type):
        #torch.manual_seed(self.seed)
        self.apply(getattr(self,f'{init_type}_init'))

    def NTK_init(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, np.sqrt(2.0 / m.out_features))
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, np.sqrt(2.0 / m.out_channels))

    def LeCun_init(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, np.sqrt(1.0 / m.in_features))
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, np.sqrt(1.0 / m.in_channels))

    def He_init(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, np.sqrt(2.0 / m.in_features))
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, np.sqrt(2.0 / m.in_channels))

    def forward(self, z):
        z = z.view((-1,self.input_size))
        out = self.l1(z)
        out = self.activation(out)
        for i in range(1,self.n_hidden_layers+1):
            out = getattr(self, f'lhidden{i}')(out)
            if self.BN:
                out = getattr(self, f'BN{i}')(out)
            out = self.activation(out)
        out = self.lout(out)
        # no activation and no softmax at the end
        return out

    def compute_signs(self,device):
        self.l1p = nn.Linear(self.input_size, self.l1.weight.data.shape[0], bias = False,device = device)
        self.l1n = nn.Linear(self.input_size, self.l1.weight.data.shape[0], bias = False,device = device)
        with torch.no_grad():
            self.l1p.weight.copy_(self.l1.weight*(self.l1.weight > 0))
            self.l1n.weight.copy_(self.l1.weight*(self.l1.weight < 0))
        for i in range(1,self.n_hidden_layers+1):
            setattr(self, f'lhidden{i}p',nn.Linear(getattr(self, f'lhidden{i}').weight.data.shape[1], getattr(self, f'lhidden{i}').weight.data.shape[0], bias = False,device = device))
            setattr(self, f'lhidden{i}n',nn.Linear(getattr(self, f'lhidden{i}').weight.data.shape[1], getattr(self, f'lhidden{i}').weight.data.shape[0], bias = False,device = device))
            with torch.no_grad():
                getattr(self, f'lhidden{i}p').weight.copy_(getattr(self, f'lhidden{i}').weight*(getattr(self, f'lhidden{i}').weight > 0))
                getattr(self, f'lhidden{i}n').weight.copy_(getattr(self, f'lhidden{i}').weight*(getattr(self, f'lhidden{i}').weight < 0))
        self.loutp = nn.Linear(self.lout.weight.data.shape[1], self.n_classes, bias = False,device = device)
        self.loutn = nn.Linear(self.lout.weight.data.shape[1], self.n_classes, bias = False,device = device)
        with torch.no_grad():
            self.loutp.weight.copy_(self.lout.weight*(self.lout.weight > 0))
            self.loutn.weight.copy_(self.lout.weight*(self.lout.weight < 0))

    def forward_bounds(self, l, u):
        lout = self.l1p(l) + self.l1n(u) + self.l1.bias.data
        uout = self.l1p(u) + self.l1n(l) + self.l1.bias.data
        lout = self.activation(lout)
        uout = self.activation(uout)
        for i in range(1,self.n_hidden_layers+1):
            lout2 = getattr(self, f'lhidden{i}p')(lout) + getattr(self, f'lhidden{i}n')(uout) + getattr(self, f'lhidden{i}').bias.data
            uout2 = getattr(self, f'lhidden{i}p')(uout) + getattr(self, f'lhidden{i}n')(lout) + getattr(self, f'lhidden{i}').bias.data
            lout = self.activation(lout2)
            uout = self.activation(uout2)
        lout2 = self.loutp(lout) + self.loutn(uout) + self.lout.bias.data
        uout2 = self.loutp(uout) + self.loutn(lout) + self.lout.bias.data
        return lout2, uout2

    def verif_objective(self, z, tc, ac):
        out = self.forward(z)
        return out[:,tc] - out[:,ac]

    def hessian_vector(self,tc,ac,z,v):
        xhat = self.l1(z)
        x = self.activation(xhat)
        h_act = self.hess_act(xhat)
        g_act = self.grad_act(xhat)
        Hv = h_act*self.l1.weight.data.transpose(0,1)*torch.matmul(v,self.l1.weight.data.transpose(0,1))
        J = self.l1.weight.data
        J = (J.transpose(0,1)*g_act).transpose(0,1)
        for i in range(1,self.n_hidden_layers+1):

            xhat = getattr(self,f'lhidden{i}')(x)
            x = getattr(self,f'lhidden{i}')(xhat)
            g_act = self.grad_act(xhat)
            h_act = self.hess_act(xhat)

            JW = torch.matmul(getattr(self,f'lhidden{i}').weight.data,J)
            Hv_right = h_act*JW.transpose(0,1)*torch.matmul(v,JW.transpose(0,1))
            #print(JW.shape, g_act.shape)
            #J = (JW.transpose(0,1)*g_act).transpose(0,1)

            Hv = g_act*torch.matmul(Hv, getattr(self,f'lhidden{i}').weight.data.transpose(0,1)) + Hv_right
            J = (JW.transpose(0,1)*g_act).transpose(0,1)

        return torch.matmul(Hv,self.lout.weight.data[tc,:] - self.lout.weight.data[ac,:])

    def bounds_grad_activation(self,l,u):
        if self.act_name == 'softplus':
            lg_act = self.grad_act(l)
            ug_act = self.grad_act(u)
            return lg_act, ug_act
        if self.act_name == 'sigmoid' or self.act_name == 'tanh':
            hl = self.grad_act(l)
            hu = self.grad_act(u)
            return (l>=self.argmax_grad_act)*hu + (l<self.argmax_grad_act)*(u<self.argmax_grad_act)*hl + (l<self.argmax_grad_act)*(u>self.argmax_grad_act)*torch.min(hl,hu), (l>=self.argmax_grad_act)*hl + (l<self.argmax_grad_act)*(u<self.argmax_grad_act)*hu + (l<self.argmax_grad_act)*(u>self.argmax_grad_act)*torch.full((hu.shape[0],hu.shape[1]), self.max_grad, device = self.device)

    def bounds_hess_activation(self,l,u):
        if self.act_name == 'softplus':
            hl = self.hess_act(l)
            hu = self.hess_act(u)
            return (l>=self.argmax_hess_act)*hu + (l<self.argmax_hess_act)*(u<self.argmax_hess_act)*hl + (l<self.argmax_hess_act)*(u>self.argmax_hess_act)*torch.min(hl,hu), (l>=self.argmax_hess_act)*hl + (l<self.argmax_hess_act)*(u<self.argmax_hess_act)*hu + (l<self.argmax_hess_act)*(u>self.argmax_hess_act)*torch.full((hu.shape[0],hu.shape[1]), self.max_hess, device = self.device)
        if self.act_name == 'sigmoid' or self.act_name == 'tanh':
            hl = self.hess_act(l)
            hu = self.hess_act(u)
            lh = torch.zeros((hu.shape[0],hu.shape[1]), device = self.device)
            uh = torch.zeros((hu.shape[0],hu.shape[1]), device = self.device)
            for i in range(hu.shape[1]):
                if l[0,i] <= self.argmax_hess_act and u[0,i] <= self.argmax_hess_act:
                    '''
                    l-u-max-min
                    '''
                    lh[0,i] = hl[0,i]
                    uh[0,i] = hu[0,i]
                elif l[0,i] <= self.argmax_hess_act and u[0,i] >= self.argmax_hess_act and u[0,i] <= self.argmin_hess_act:
                    '''
                    l-max-u-min
                    '''
                    lh[0,i] = torch.min(hl[0,i],hu[0,i])
                    uh[0,i] = self.max_hess
                elif l[0,i] <= self.argmax_hess_act and u[0,i] >= self.argmin_hess_act:
                    '''
                    l-max-min-u
                    '''
                    lh[0,i] = self.min_hess
                    uh[0,i] = self.max_hess
                elif l[0,i] >= self.argmax_hess_act and l[0,i] <= self.argmin_hess_act and u[0,i] >= self.argmax_hess_act and u[0,i] <= self.argmin_hess_act:
                    '''
                    max-l-u-min
                    '''
                    lh[0,i] = hu[0,i]
                    uh[0,i] = hl[0,i]
                elif l[0,i] >= self.argmax_hess_act and l[0,i] <= self.argmin_hess_act and u[0,i] >= self.argmin_hess_act:
                    '''
                    max-l-min-u
                    '''
                    lh[0,i] = self.min_hess
                    uh[0,i] = torch.max(hl[0,i],hu[0,i])
                elif l[0,i] >= self.argmin_hess_act and u[0,i] >= self.argmin_hess_act:
                    '''
                    max-min-l-u
                    '''
                    lh[0,i] = hl[0,i]
                    uh[0,i] = hu[0,i]
            #lh = torch.min(torch.min(hl, (u<=self.argmin_hess_act)*hu), (u>=self.argmin_hess_act)*torch.full((hu.shape[0],hu.shape[1]), self.min_hess, device = self.device))
            #uh = torch.max((l<=self.argmax_hess_act)*torch.full((hu.shape[0],hu.shape[1]), self.min_hess, device = self.device), torch.max((l>=self.argmax_hess_act)*hl, hu))
            return lh,uh

    def lowerbounding_hessian_vector(self,tc,ac,l,u,v,device):
        lxhat = self.l1p(l) + self.l1n(u) + self.l1.bias.data
        uxhat = self.l1p(u) + self.l1n(l) + self.l1.bias.data

        lx = self.activation(lxhat)
        ux = self.activation(uxhat)
        lh_act,uh_act = self.bounds_hess_activation(lxhat,uxhat)
        lg_act,ug_act = self.bounds_grad_activation(lxhat,uxhat)

        lh_act_p = lh_act*(lh_act>0)
        lh_act_n = lh_act*(lh_act<0)
        uh_act_p = uh_act*(uh_act>0)
        uh_act_n = uh_act*(uh_act<0)

        lg_act_p = lg_act*(lg_act>0)
        lg_act_n = lg_act*(lg_act<0)
        ug_act_p = ug_act*(ug_act>0)
        ug_act_n = ug_act*(ug_act<0)

        wp = self.l1.weight.data.transpose(0,1)*(self.l1.weight.data.transpose(0,1) > 0)
        wn = self.l1.weight.data.transpose(0,1)*(self.l1.weight.data.transpose(0,1) < 0)

        Lnv = lh_act_n*(wp*torch.matmul(v,wp) + wn*torch.matmul(v,wn)) + uh_act_p*(wp*torch.matmul(v,wn) + wn*torch.matmul(v,wp))
        Lpv = lh_act_p*(wp*torch.matmul(v,wp) + wn*torch.matmul(v,wn)) + uh_act_n*(wp*torch.matmul(v,wn) + wn*torch.matmul(v,wp))
        Unv = lh_act_p*(wp*torch.matmul(v,wn) + wn*torch.matmul(v,wp)) + uh_act_n*(wp*torch.matmul(v,wp) + wn*torch.matmul(v,wn))
        Upv = lh_act_n*(wp*torch.matmul(v,wn) + wn*torch.matmul(v,wp)) + uh_act_p*(wp*torch.matmul(v,wp) + wn*torch.matmul(v,wn))

        Ln1 = lh_act_n*(wp*torch.sum(wp,0) + wn*torch.sum(wn,0)) + uh_act_p*(wp*torch.sum(wn,0) + wn*torch.sum(wp,0))
        Lp1 = lh_act_p*(wp*torch.sum(wp,0) + wn*torch.sum(wn,0)) + uh_act_n*(wp*torch.sum(wn,0) + wn*torch.sum(wp,0))
        Un1 = lh_act_p*(wp*torch.sum(wn,0) + wn*torch.sum(wp,0)) + uh_act_n*(wp*torch.sum(wp,0) + wn*torch.sum(wn,0))
        Up1 = lh_act_n*(wp*torch.sum(wn,0) + wn*torch.sum(wp,0)) + uh_act_p*(wp*torch.sum(wp,0) + wn*torch.sum(wn,0))

        lg = torch.min(ug_act*self.l1.weight.data.transpose(0,1), lg_act*self.l1.weight.data.transpose(0,1)).transpose(0,1)
        ug = torch.max(ug_act*self.l1.weight.data.transpose(0,1), lg_act*self.l1.weight.data.transpose(0,1)).transpose(0,1)

        for i in range(1,self.n_hidden_layers+1):
            #print(getattr(self,f'lhidden{i}p').weight.data.shape,lg.shape)
            lJW = torch.matmul(getattr(self,f'lhidden{i}p').weight.data,lg) + torch.matmul(getattr(self,f'lhidden{i}n').weight.data,ug)
            uJW = torch.matmul(getattr(self,f'lhidden{i}p').weight.data,ug) + torch.matmul(getattr(self,f'lhidden{i}n').weight.data,lg)
            lxhat = getattr(self,f'lhidden{i}p')(lx) + getattr(self,f'lhidden{i}n')(ux) + getattr(self,f'lhidden{i}').bias.data
            uxhat = getattr(self,f'lhidden{i}p')(ux) + getattr(self,f'lhidden{i}n')(lx) + getattr(self,f'lhidden{i}').bias.data
            lx = self.activation(lxhat)
            ux = self.activation(uxhat)
            lh_act,uh_act = self.bounds_hess_activation(lxhat,uxhat)
            lg_act,ug_act = self.bounds_grad_activation(lxhat,uxhat)
            S = torch.cat(((lJW.transpose(0,1)*lg_act).unsqueeze(-1),(lJW.transpose(0,1)*ug_act).unsqueeze(-1),(uJW.transpose(0,1)*lg_act).unsqueeze(-1),(uJW.transpose(0,1)*ug_act).unsqueeze(-1)),dim = -1)
            lg,_ = torch.min(S, dim = -1)
            ug,_ = torch.max(S, dim = -1)
            lg = lg.transpose(0,1)
            ug = ug.transpose(0,1)

            lgp = lJW*(lJW>0)
            lgn = lJW*(lJW<0)
            ugp = uJW*(uJW>0)
            ugn = uJW*(uJW<0)

            lh_act_p = lh_act*(lh_act>0)
            lh_act_n = lh_act*(lh_act<0)
            uh_act_p = uh_act*(uh_act>0)
            uh_act_n = uh_act*(uh_act<0)

            lg_act_p = lg_act*(lg_act>0)
            lg_act_n = lg_act*(lg_act<0)
            ug_act_p = ug_act*(ug_act>0)
            ug_act_n = ug_act*(ug_act<0)
            #print(self.l1.weight.data.shape,getattr(self,f'lhidden{i}p').weight.data.shape, Upv.shape)

            Lnv_ = lg_act_n*(torch.matmul(Upv,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + torch.matmul(Lnv,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1))) + ug_act_p*(torch.matmul(Lnv,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + torch.matmul(Upv,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1)))
            Lpv_ = lg_act_p*(torch.matmul(Lpv,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + torch.matmul(Unv,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1))) + ug_act_n*(torch.matmul(Unv,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + torch.matmul(Lpv,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1)))
            Unv_ = lg_act_p*(torch.matmul(Unv,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + torch.matmul(Lpv,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1))) + ug_act_n*(torch.matmul(Lpv,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + torch.matmul(Unv,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1)))
            Upv_ = lg_act_n*(torch.matmul(Lnv,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + torch.matmul(Upv,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1))) + ug_act_p*(torch.matmul(Upv,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + torch.matmul(Lnv,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1)))

            Lnv = Lnv_
            Lpv = Lpv_
            Unv = Unv_
            Upv = Upv_

            Ln1_ = lg_act_n*(torch.matmul(Up1,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + torch.matmul(Ln1,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1))) + ug_act_p*(torch.matmul(Ln1,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + torch.matmul(Up1,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1)))
            Lp1_ = lg_act_p*(torch.matmul(Lp1,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + torch.matmul(Un1,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1))) + ug_act_n*(torch.matmul(Un1,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + torch.matmul(Lp1,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1)))
            Un1_ = lg_act_p*(torch.matmul(Un1,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + torch.matmul(Lp1,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1))) + ug_act_n*(torch.matmul(Lp1,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + torch.matmul(Un1,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1)))
            Up1_ = lg_act_n*(torch.matmul(Ln1,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + torch.matmul(Up1,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1))) + ug_act_p*(torch.matmul(Up1,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + torch.matmul(Ln1,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1)))

            Ln1 = Ln1_
            Lp1 = Lp1_
            Un1 = Un1_
            Up1 = Up1_

            #print(ugp.shape, v.shape, lh_act_n.shape,Lnv.shape)

            Lnv += lh_act_n*(ugp*torch.matmul(ugp,v.transpose(0,1)) + lgn*torch.matmul(lgn,v.transpose(0,1))).transpose(0,1) + uh_act_p*(lgn*torch.matmul(ugp,v.transpose(0,1)) + ugp*torch.matmul(lgn,v.transpose(0,1))).transpose(0,1)
            Lpv += lh_act_p*(lgp*torch.matmul(lgp,v.transpose(0,1)) + ugn*torch.matmul(ugn,v.transpose(0,1))).transpose(0,1) + uh_act_n*(ugn*torch.matmul(lgp,v.transpose(0,1)) + lgp*torch.matmul(ugn,v.transpose(0,1))).transpose(0,1)
            Unv += lh_act_p*(ugn*torch.matmul(lgp,v.transpose(0,1)) + lgp*torch.matmul(ugn,v.transpose(0,1))).transpose(0,1) + uh_act_n*(lgp*torch.matmul(lgp,v.transpose(0,1)) + ugn*torch.matmul(ugn,v.transpose(0,1))).transpose(0,1)
            Upv += lh_act_n*(lgn*torch.matmul(ugp,v.transpose(0,1)) + ugp*torch.matmul(lgn,v.transpose(0,1))).transpose(0,1) + uh_act_p*(lgn*torch.matmul(lgn,v.transpose(0,1)) + ugp*torch.matmul(ugp,v.transpose(0,1))).transpose(0,1)

            #print(Ln1.shape,ugp.shape, lh_act_n.shape,(ugp*torch.sum(ugp,1).unsqueeze(1)).shape )
            Ln1 += (lh_act_n.transpose(0,1)*(ugp*torch.sum(ugp,1).unsqueeze(1) + lgn*torch.sum(lgn,1).unsqueeze(1)) + uh_act_p.transpose(0,1)*(lgn*torch.sum(ugp,1).unsqueeze(1) + ugp*torch.sum(lgn,1).unsqueeze(1))).transpose(0,1)
            Lp1 += (lh_act_p.transpose(0,1)*(lgp*torch.sum(lgp,1).unsqueeze(1) + ugn*torch.sum(ugn,1).unsqueeze(1)) + uh_act_n.transpose(0,1)*(ugn*torch.sum(lgp,1).unsqueeze(1) + lgp*torch.sum(ugn,1).unsqueeze(1))).transpose(0,1)
            Un1 += (lh_act_p.transpose(0,1)*(ugn*torch.sum(lgp,1).unsqueeze(1) + lgp*torch.sum(ugn,1).unsqueeze(1)) + uh_act_n.transpose(0,1)*(lgp*torch.sum(lgp,1).unsqueeze(1) + ugn*torch.sum(ugn,1).unsqueeze(1))).transpose(0,1)
            Up1 += (lh_act_n.transpose(0,1)*(lgn*torch.sum(ugp,1).unsqueeze(1) + ugp*torch.sum(lgn,1).unsqueeze(1)) + uh_act_p.transpose(0,1)*(lgn*torch.sum(lgn,1).unsqueeze(1) + ugp*torch.sum(ugp,1).unsqueeze(1))).transpose(0,1)

        C = self.lout.weight.data[tc,:] - self.lout.weight.data[ac,:]
        Ca = torch.abs(C)
        return (torch.matmul(Lnv+Lpv+Unv+Upv,C).unsqueeze(1) + torch.matmul(Ln1+Lp1-Un1-Up1,Ca).unsqueeze(1)*v.transpose(0,1))/2

    '''
    VECTOR ALPHA
    '''

    def lowerbound_diag(self,tc,ac,l,u,device):
        termsv = []
        terms1 = []
        lxhat = self.l1p(l) + self.l1n(u) + self.l1.bias.data
        uxhat = self.l1p(u) + self.l1n(l) + self.l1.bias.data

        lx = self.activation(lxhat)
        ux = self.activation(uxhat)
        lh_act,uh_act = self.bounds_hess_activation(lxhat,uxhat)
        lg_act,ug_act = self.bounds_grad_activation(lxhat,uxhat)


        lsqh = torch.sqrt(lh_act)
        usqh = torch.sqrt(uh_act)
        #print(lh_act[0,:5], lsqh[0,:5])

        L_ = torch.min(lsqh*self.l1.weight.data.transpose(0,1), usqh*self.l1.weight.data.transpose(0,1))
        U_ = torch.max(lsqh*self.l1.weight.data.transpose(0,1), usqh*self.l1.weight.data.transpose(0,1))

        L = torch.min(L_*L_,U_*U_)
        U = torch.max(L_*L_,U_*U_)

        lg = torch.min(ug_act*self.l1.weight.data.transpose(0,1), lg_act*self.l1.weight.data.transpose(0,1)).transpose(0,1)
        ug = torch.max(ug_act*self.l1.weight.data.transpose(0,1), lg_act*self.l1.weight.data.transpose(0,1)).transpose(0,1)

        for i in range(1,self.n_hidden_layers+1):
            lJW = torch.matmul(getattr(self,f'lhidden{i}p').weight.data,lg) + torch.matmul(getattr(self,f'lhidden{i}n').weight.data,ug)
            uJW = torch.matmul(getattr(self,f'lhidden{i}p').weight.data,ug) + torch.matmul(getattr(self,f'lhidden{i}n').weight.data,lg)
            lxhat = getattr(self,f'lhidden{i}p')(lx) + getattr(self,f'lhidden{i}n')(ux) + getattr(self,f'lhidden{i}').bias.data
            uxhat = getattr(self,f'lhidden{i}p')(ux) + getattr(self,f'lhidden{i}n')(lx) + getattr(self,f'lhidden{i}').bias.data
            lx = self.activation(lxhat)
            ux = self.activation(uxhat)
            lh_act,uh_act = self.bounds_hess_activation(lxhat,uxhat)
            lg_act,ug_act = self.bounds_grad_activation(lxhat,uxhat)
            lsqh = torch.sqrt(lh_act).transpose(0,1)
            usqh = torch.sqrt(uh_act).transpose(0,1)
            lsqg = torch.sqrt(lg_act)
            usqg = torch.sqrt(ug_act)
            S = torch.cat(((lJW.transpose(0,1)*lg_act).unsqueeze(-1),(lJW.transpose(0,1)*ug_act).unsqueeze(-1),(uJW.transpose(0,1)*lg_act).unsqueeze(-1),(uJW.transpose(0,1)*ug_act).unsqueeze(-1)),dim = -1)
            lg,_ = torch.min(S, dim = -1)
            ug,_ = torch.max(S, dim = -1)

            S = torch.cat(((lsqh*lJW).unsqueeze(-1),(lsqh*uJW).unsqueeze(-1),(usqh*lJW).unsqueeze(-1),(usqh*uJW).unsqueeze(-1)),dim = -1)
            LL_,_ = torch.min(S, dim = -1)
            UU_,_ = torch.max(S, dim = -1)

            L_ = lg_act*torch.matmul(L,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + ug_act*torch.matmul(U,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1))
            U_ = ug_act*torch.matmul(U,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + lg_act*torch.matmul(L,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1))

            L = L_ + torch.min(LL_*LL_,UU_*UU_).transpose(0,1)
            U = U_ + torch.max(LL_*LL_,UU_*UU_).transpose(0,1)

        C = self.lout.weight.data[tc,:] - self.lout.weight.data[ac,:]
        Cp = C*(C>0)
        Cn = C*(C<0)
        return torch.matmul(L,Cp) + torch.matmul(U,Cn)

    def lowerbound_diag2(self,tc,ac,l,u,device):
        termsv = []
        terms1 = []
        lxhat = self.l1p(l) + self.l1n(u) + self.l1.bias.data
        uxhat = self.l1p(u) + self.l1n(l) + self.l1.bias.data

        lx = self.activation(lxhat)
        ux = self.activation(uxhat)
        lh_act,uh_act = self.bounds_hess_activation(lxhat,uxhat)
        lg_act,ug_act = self.bounds_grad_activation(lxhat,uxhat)

        lh_act_p = lh_act*(lh_act>0)
        lh_act_n = lh_act*(lh_act<0)
        uh_act_p = uh_act*(uh_act>0)
        uh_act_n = uh_act*(uh_act<0)

        lg_act_p = lg_act*(lg_act>0)
        lg_act_n = lg_act*(lg_act<0)
        ug_act_p = ug_act*(ug_act>0)
        ug_act_n = ug_act*(ug_act<0)

        wp = self.l1.weight.data.transpose(0,1)*(self.l1.weight.data.transpose(0,1) > 0)
        wn = self.l1.weight.data.transpose(0,1)*(self.l1.weight.data.transpose(0,1) < 0)

        Ln = lh_act_n*(wp*wp + wn*wn) + uh_act_p*(wp*wn + wn*wp)
        Lp = lh_act_p*(wp*wp + wn*wn) + uh_act_n*(wp*wn + wn*wp)
        Un = lh_act_p*(wp*wn + wn*wp) + uh_act_n*(wp*wp + wn*wn)
        Up = lh_act_n*(wp*wn + wn*wp) + uh_act_p*(wp*wp + wn*wn)

        lg = torch.min(ug_act*self.l1.weight.data.transpose(0,1), lg_act*self.l1.weight.data.transpose(0,1)).transpose(0,1)
        ug = torch.max(ug_act*self.l1.weight.data.transpose(0,1), lg_act*self.l1.weight.data.transpose(0,1)).transpose(0,1)

        for i in range(1,self.n_hidden_layers+1):
            lJW = torch.matmul(getattr(self,f'lhidden{i}p').weight.data,lg) + torch.matmul(getattr(self,f'lhidden{i}n').weight.data,ug)
            uJW = torch.matmul(getattr(self,f'lhidden{i}p').weight.data,ug) + torch.matmul(getattr(self,f'lhidden{i}n').weight.data,lg)
            lxhat = getattr(self,f'lhidden{i}p')(lx) + getattr(self,f'lhidden{i}n')(ux) + getattr(self,f'lhidden{i}').bias.data
            uxhat = getattr(self,f'lhidden{i}p')(ux) + getattr(self,f'lhidden{i}n')(lx) + getattr(self,f'lhidden{i}').bias.data
            lx = self.activation(lxhat)
            ux = self.activation(uxhat)
            lh_act,uh_act = self.bounds_hess_activation(lxhat,uxhat)
            lg_act,ug_act = self.bounds_grad_activation(lxhat,uxhat)
            lsqh = torch.sqrt(lh_act).transpose(0,1)
            usqh = torch.sqrt(uh_act).transpose(0,1)
            lsqg = torch.sqrt(lg_act)
            usqg = torch.sqrt(ug_act)
            lg = torch.min(lJW.transpose(0,1)*lg_act, lJW.transpose(0,1)*ug_act, uJW.transpose(0,1)*lg_act, uJW.transpose(0,1)*ug_act).transpose(0,1)
            ug = torch.max(lJW.transpose(0,1)*lg_act, lJW.transpose(0,1)*ug_act, uJW.transpose(0,1)*lg_act, uJW.transpose(0,1)*ug_act).transpose(0,1)

            S = torch.cat(((lsqh*lJW).unsqueeze(-1),(lsqh*uJW).unsqueeze(-1),(usqh*lJW).unsqueeze(-1),(usqh*uJW).unsqueeze(-1)),dim = -1)
            LL_,_ = torch.min(S, dim = -1)
            UU_,_ = torch.max(S, dim = -1)

            L_ = lg_act*torch.matmul(L,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + ug_act*torch.matmul(U,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1))
            U_ = ug_act*torch.matmul(U,getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)) + lg_act*torch.matmul(L,getattr(self,f'lhidden{i}n').weight.data.transpose(0,1))

            L = L_ + torch.min(LL_*LL_,UU_*UU_).transpose(0,1)
            U = U_ + torch.max(LL_*LL_,UU_*UU_).transpose(0,1)



        C = self.lout.weight.data[tc,:] - self.lout.weight.data[ac,:]
        Cp = C*(C>0)
        Cn = C*(C<0)
        return torch.matmul(L,Cp) + torch.matmul(U,Cn)

    def sum_abs_out_diag(self,tc,ac,l,u,v,device):
        lxhat = self.l1p(l) + self.l1n(u) + self.l1.bias.data
        uxhat = self.l1p(u) + self.l1n(l) + self.l1.bias.data

        lx = self.activation(lxhat)
        ux = self.activation(uxhat)
        lh_act,uh_act = self.bounds_hess_activation(lxhat,uxhat)
        lg_act,ug_act = self.bounds_grad_activation(lxhat,uxhat)


        lsqh = torch.sqrt(lh_act)
        usqh = torch.sqrt(uh_act)
        #print(lh_act[0,:5], lsqh[0,:5])

        L = torch.min(lsqh*self.l1.weight.data.transpose(0,1), usqh*self.l1.weight.data.transpose(0,1))
        U = torch.max(lsqh*self.l1.weight.data.transpose(0,1), usqh*self.l1.weight.data.transpose(0,1))

        maxnorm = torch.max(torch.abs(L), torch.abs(U))
        result = []
        for j in range(maxnorm.shape[0]):
            v_aux = v.clone()
            v_aux/=v[0,j]
            v_aux[0,j] = 0
            result.append((maxnorm*torch.matmul(v_aux,maxnorm))[j,:].unsqueeze(0))

        result = torch.cat(result,dim = 0)

        lg = torch.min(ug_act*self.l1.weight.data.transpose(0,1), lg_act*self.l1.weight.data.transpose(0,1)).transpose(0,1)
        ug = torch.max(ug_act*self.l1.weight.data.transpose(0,1), lg_act*self.l1.weight.data.transpose(0,1)).transpose(0,1)

        for i in range(1,self.n_hidden_layers+1):
            lJW = torch.matmul(getattr(self,f'lhidden{i}p').weight.data,lg) + torch.matmul(getattr(self,f'lhidden{i}n').weight.data,ug)
            uJW = torch.matmul(getattr(self,f'lhidden{i}p').weight.data,ug) + torch.matmul(getattr(self,f'lhidden{i}n').weight.data,lg)
            lxhat = getattr(self,f'lhidden{i}p')(lx) + getattr(self,f'lhidden{i}n')(ux) + getattr(self,f'lhidden{i}').bias.data
            uxhat = getattr(self,f'lhidden{i}p')(ux) + getattr(self,f'lhidden{i}n')(lx) + getattr(self,f'lhidden{i}').bias.data
            lx = self.activation(lxhat)
            ux = self.activation(uxhat)
            lh_act,uh_act = self.bounds_hess_activation(lxhat,uxhat)
            lg_act,ug_act = self.bounds_grad_activation(lxhat,uxhat)
            lsqh = torch.sqrt(lh_act).transpose(0,1)
            usqh = torch.sqrt(uh_act).transpose(0,1)
            lsqg = torch.sqrt(lg_act)
            usqg = torch.sqrt(ug_act)
            S = torch.cat(((lJW.transpose(0,1)*lg_act).unsqueeze(-1),(lJW.transpose(0,1)*ug_act).unsqueeze(-1),(uJW.transpose(0,1)*lg_act).unsqueeze(-1),(uJW.transpose(0,1)*ug_act).unsqueeze(-1)),dim = -1)
            lg,_ = torch.min(S, dim = -1)
            ug,_ = torch.max(S, dim = -1)

            S = torch.cat(((lsqh*lJW).unsqueeze(-1),(lsqh*uJW).unsqueeze(-1),(usqh*lJW).unsqueeze(-1),(usqh*uJW).unsqueeze(-1)),dim = -1)
            L,_ = torch.min(S, dim = -1)
            U,_ = torch.max(S, dim = -1)

            maxnorm = torch.max(torch.abs(L), torch.abs(U)).transpose(0,1)
            result_ = []
            for j in range(maxnorm.shape[0]):
                v_aux = v.clone()
                v_aux/=v[0,j]
                v_aux[0,j] = 0
                result_.append((maxnorm*torch.matmul(v_aux,maxnorm))[j,:].unsqueeze(0))

            result_ = torch.cat(result_,dim = 0)

            result = torch.max(torch.abs(ug_act)*torch.matmul(result,torch.abs(getattr(self,f'lhidden{i}').weight.data.transpose(0,1))),\
                               torch.abs(lg_act)*torch.matmul(result,torch.abs(getattr(self,f'lhidden{i}').weight.data.transpose(0,1)))) + result_


        C = self.lout.weight.data[tc,:] - self.lout.weight.data[ac,:]
        return torch.matmul(result,torch.abs(C))

    def alphas(self,tc,ac,l,u,device,v = None):
        with torch.no_grad():
            if v is None:
                v = torch.ones(l.shape,device = device)
            ldiag = self.lowerbound_diag(tc,ac,l,u,device)
            sum_abs = self.sum_abs_out_diag(tc,ac,l,u,v,device)
            return -(ldiag-sum_abs)/2

    def dom_eig_L(self,tc,ac,l,u,device,momentum = 0.5,shift = 0,debug = False):
        if debug:
            print('Computing min_eig:')
        with torch.no_grad():
            z = torch.rand((1,self.input_size), device = device)
            z = z/(torch.sqrt((z*z).sum()))
            prev_z = torch.ones(self.input_size, device = device)
            min_eig = 1
            diff = 1
            i = 0
            while not diff < 1e-3:
                #Each iteration we do two steps because if the dominant eigenvalue is negative,
                #with a single step prev_z = -z and never converges.
                prev_prev_z = prev_z.clone()
                prev_z = z.clone()
                z = self.lowerbounding_hessian_vector(tc,ac,l,u,z,device).transpose(0,1) - (i>=1)*momentum*prev_prev_z - shift*z
                sign_eig = 1-2*((z/prev_z)[0,0] < 0)
                min_eig = torch.sqrt((z*z).sum())

                #print(z.shape,min_eig.shape)
                z = z/min_eig
                if momentum !=0:
                    prev_z/=min_eig
                z = self.lowerbounding_hessian_vector(tc,ac,l,u,z,device).transpose(0,1) - momentum*prev_z - shift*z
                min_eig = torch.sqrt((z*z).sum())
                z = z/min_eig
                if momentum !=0:
                    prev_z/=min_eig
                if debug:
                    print(min_eig,torch.sum(torch.abs(z-(prev_z/torch.sqrt((prev_z*prev_z).sum())))))
                diff = torch.sum(torch.abs(z-(prev_z/torch.sqrt((prev_z*prev_z).sum()))))
                i+=1
            z = self.lowerbounding_hessian_vector(tc,ac,l,u,z,device).transpose(0,1) - shift*z
            min_eig = torch.sqrt((z*z).sum())
            return min_eig, sign_eig*min_eig

    def dom_eig_hessian(self,tc,ac,z_0,device,store=False,debug = False):
        if debug:
            print('Computing min_eig:')
        if not store:
            with torch.no_grad():
                z = torch.rand((1,self.input_size), device = device)
                z = z/(torch.sqrt((z*z).sum()))
                prev_z = torch.ones(self.input_size, device = device)
                diff = 1
                while not diff < 1e-3:
                    '''
                    Each iteration we do two steps because if the dominant eigenvalue is negative,
                    with a single step prev_z = -z and never converges.
                    '''
                    prev_z = z.clone()
                    z = self.hessian_vector(tc,ac,z_0,z).unsqueeze(0)
                    min_eig = torch.sqrt((z*z).sum())
                    #print(z.shape,min_eig.shape)
                    z = z/min_eig
                    z = self.hessian_vector(tc,ac,z_0,z).unsqueeze(0)
                    min_eig = torch.sqrt((z*z).sum())
                    z = z/min_eig
                    if debug:
                        print(min_eig,torch.sum(torch.abs(z-prev_z)))
                    diff = torch.sum(torch.abs(z-prev_z))
                return min_eig
        else:
            hz = torch.autograd.functional.hessian(lambda x : self.verif_objective(x, tc,ac),z_0).squeeze()
            min_eig,_ = UBAB.power_method(hz,device,shift=0,debug=debug)
            return min_eig

    def compute_LH(self,tc, ac, l, u, device):
        with torch.no_grad():
            lxhat = self.l1p(l) + self.l1n(u) + self.l1.bias.data
            uxhat = self.l1p(u) + self.l1n(l) + self.l1.bias.data

            lx = self.activation(lxhat)
            ux = self.activation(uxhat)
            lh_act,uh_act = self.bounds_hess_activation(lxhat,uxhat)
            lg_act,ug_act = self.bounds_grad_activation(lxhat,uxhat)

            w = self.l1.weight.data

            L_ = torch.bmm(w.unsqueeze(-1),w.unsqueeze(1)).transpose(0,2)
            U_ = L_.clone()

            S = torch.cat(((lh_act*L_).unsqueeze(-1),(lh_act*U_).unsqueeze(-1),(uh_act*L_).unsqueeze(-1),(uh_act*U_).unsqueeze(-1)),dim=-1)
            L,_ = torch.min(S,dim=-1)
            U,_ = torch.max(S,dim=-1)

            lg = torch.min(ug_act*self.l1.weight.data.transpose(0,1), lg_act*self.l1.weight.data.transpose(0,1)).transpose(0,1)
            ug = torch.max(ug_act*self.l1.weight.data.transpose(0,1), lg_act*self.l1.weight.data.transpose(0,1)).transpose(0,1)

            for i in range(1,self.n_hidden_layers+1):
                #print(getattr(self,f'lhidden{i}p').weight.data.shape,lg.shape)
                lJW = torch.matmul(getattr(self,f'lhidden{i}p').weight.data,lg) + torch.matmul(getattr(self,f'lhidden{i}n').weight.data,ug)
                uJW = torch.matmul(getattr(self,f'lhidden{i}p').weight.data,ug) + torch.matmul(getattr(self,f'lhidden{i}n').weight.data,lg)
                lxhat = getattr(self,f'lhidden{i}p')(lx) + getattr(self,f'lhidden{i}n')(ux) + getattr(self,f'lhidden{i}').bias.data
                uxhat = getattr(self,f'lhidden{i}p')(ux) + getattr(self,f'lhidden{i}n')(lx) + getattr(self,f'lhidden{i}').bias.data
                lx = self.activation(lxhat)
                ux = self.activation(uxhat)
                lh_act,uh_act = self.bounds_hess_activation(lxhat,uxhat)
                lg_act,ug_act = self.bounds_grad_activation(lxhat,uxhat)

                S = torch.cat(((lJW.transpose(0,1)*lg_act).unsqueeze(-1),(lJW.transpose(0,1)*ug_act).unsqueeze(-1),(uJW.transpose(0,1)*lg_act).unsqueeze(-1),(uJW.transpose(0,1)*ug_act).unsqueeze(-1)),dim = -1)
                lg,_ = torch.min(S, dim = -1)
                ug,_ = torch.max(S, dim = -1)
                lg = lg.transpose(0,1)
                ug = ug.transpose(0,1)

                wp = getattr(self,f'lhidden{i}p').weight.data.transpose(0,1)
                wn = getattr(self,f'lhidden{i}n').weight.data.transpose(0,1)

                L_ = torch.matmul(L,wp) + torch.matmul(U,wn)
                U_ = torch.matmul(U,wp) + torch.matmul(L,wn)

                S = torch.cat(((L_*lg_act).unsqueeze(-1),(L_*ug_act).unsqueeze(-1),(U_*lg_act).unsqueeze(-1),(U_*ug_act).unsqueeze(-1)),dim = -1)

                L,_ = torch.min(S,dim=-1)
                U,_ = torch.max(S,dim=-1)

                #print(lJW.shape, lg.shape, lJW.unsqueeze(1).shape)

                #print((torch.bmm(lJW.unsqueeze(-1),lJW.unsqueeze(1)).transpose(0,2)).shape)

                S = torch.cat(((torch.bmm(lJW.unsqueeze(-1),lJW.unsqueeze(1)).transpose(0,2)).unsqueeze(-1),\
                               (torch.bmm(lJW.unsqueeze(-1),uJW.unsqueeze(1)).transpose(0,2)).unsqueeze(-1),\
                               (torch.bmm(uJW.unsqueeze(-1),lJW.unsqueeze(1)).transpose(0,2)).unsqueeze(-1),\
                               (torch.bmm(uJW.unsqueeze(-1),uJW.unsqueeze(1)).transpose(0,2)).unsqueeze(-1)),dim = -1)
                Ll_,_ = torch.min(S,dim=-1)
                Ul_,_ = torch.max(S,dim=-1)

                #print(Ll_.shape, lh_act.shape)

                S = torch.cat(((lh_act*Ll_).unsqueeze(-1),(lh_act*Ul_).unsqueeze(-1),(uh_act*Ll_).unsqueeze(-1),(uh_act*Ul_).unsqueeze(-1)),dim=-1)
                Ll,_ = torch.min(S,dim=-1)
                Ul,_ = torch.max(S,dim=-1)

                L += Ll
                U += Ul

            C = self.lout.weight.data[tc,:] - self.lout.weight.data[ac,:]
            Cp = C*(C>0)
            Cn = C*(C<0)
            L_final = torch.matmul(L,Cp) + torch.matmul(U,Cn)
            U_final = torch.matmul(U,Cp) + torch.matmul(L,Cn)
            return (L_final+U_final + torch.diag((torch.sum(L_final,-1) - torch.sum(U_final,-1))))/2

    def singla_2020(self,device,tc=-1,ac=-1):
        max_grad = self.max_grad
        max_hess = self.max_hess
        r = [torch.linalg.norm(self.l1.weight,2)]
        for i in range(1,self.n_hidden_layers+1):
            if self.BN:
                r.append(r[-1]*max_grad*torch.linalg.norm(getattr(self,f'lhidden{i}').weight*getattr(self,f'BN{i}').weight.unsqueeze(1)*(1/torch.sqrt(getattr(self,f'BN{i}').running_var.unsqueeze(1) + getattr(self,f'BN{i}').eps)),2))
            else:
                r.append(r[-1]*max_grad*torch.linalg.norm(getattr(self,f'lhidden{i}').weight,2))

        '''
        Start from the last layer and do S = [S(L,L-1),S(L,L-2), ..., S(L,1)]
        S(L,L-i)= S(L,L-i+1)*g*W^{L-i+1}
        '''
        S = [torch.abs(self.lout.weight[tc]-self.lout.weight[ac])]
        for i in range(self.n_hidden_layers,0,-1):
            if self.BN:
                S.append(max_grad*torch.matmul(S[-1],torch.abs(getattr(self,f'lhidden{i}').weight*getattr(self,f'BN{i}').weight.unsqueeze(1)*(1/torch.sqrt(getattr(self,f'BN{i}').running_var.unsqueeze(1) + getattr(self,f'BN{i}').eps)))))
            else:
                S.append(max_grad*torch.matmul(S[-1],torch.abs(getattr(self,f'lhidden{i}').weight)))
        S = S[::-1] #Flip vector

        Bound = 0
        for i in range(len(r)):
            Bound += ((r[i]**2)*(torch.linalg.norm(S[i],float('inf'))))
        return Bound*max_hess


    def forward_bounds_intermediate(self,l,u,level):
        lxhat = self.l1p(l) + self.l1n(u) + self.l1.bias
        uxhat = self.l1p(u) + self.l1n(l) + self.l1.bias
        lx = self.activation(lxhat)
        ux = self.activation(uxhat)
        for i in range(1,level+1):
            lxhat = getattr(self,f'lhidden{i}p')(lx) + getattr(self,f'lhidden{i}n')(ux) + getattr(self,f'lhidden{i}').bias
            uxhat = getattr(self,f'lhidden{i}p')(ux) + getattr(self,f'lhidden{i}n')(lx) + getattr(self,f'lhidden{i}').bias
            lx = self.activation(lxhat)
            ux = self.activation(uxhat)
        return lxhat,uxhat,lx,ux

    def singla_2020_local(self,device,l,u,tc=-1,ac=-1):
        '''
        Only valid if net.BN = False
        '''
        max_grad = self.max_grad
        max_hess = self.max_hess
        lxhat,uxhat,_,_ = self.forward_bounds_intermediate(l,u,0)
        lg,ug = self.bounds_grad_activation(lxhat,uxhat)
        mg = torch.max(torch.abs(lg),torch.abs(ug))
        r = [torch.linalg.norm(self.l1.weight,2)]
        for i in range(1,self.n_hidden_layers+1):
            #print(i)
            #Compute bounds of sigma'(x_hat^{i})
            lxhat,uxhat,_,_ = self.forward_bounds_intermediate(l,u,i-1)
            lg,ug = self.bounds_grad_activation(lxhat,uxhat)
            mg = torch.max(torch.abs(lg),torch.abs(ug))
            r.append(r[-1]*torch.linalg.norm(mg.squeeze(),float('inf'))*torch.linalg.norm(getattr(self,f'lhidden{i}').weight,2))

        S = [torch.abs(self.lout.weight[tc]-self.lout.weight[ac])]
        '''
        Start from the last layer and do S = [S(L,L-1),S(L,L-2), ..., S(L,1)]
        S(L,L-i)= S(L,L-i+1)*g*W^{L-i+1}
        '''
        for i in range(self.n_hidden_layers,0,-1):
            #Compute bounds of sigma'(x_hat^{i})
            lxhat,uxhat,_,_ = self.forward_bounds_intermediate(l,u,i-1)
            lg,ug = self.bounds_grad_activation(lxhat,uxhat)
            mg = torch.max(torch.abs(lg),torch.abs(ug))
            S.append(torch.matmul(S[-1],mg*torch.abs(getattr(self,f'lhidden{i}').weight)))
        S = S[::-1]

        Bound = 0
        for i in range(len(r)):
            lxhat,uxhat,_,_ = self.forward_bounds_intermediate(l,u,i)
            lh,uh = self.bounds_hess_activation(lxhat,uxhat)
            mh = torch.max(torch.abs(lh),torch.abs(uh))
            Bound += torch.linalg.norm(mh.squeeze(),float('inf'))*(r[i]**2)*(torch.linalg.norm(S[i],float('inf')))
        return Bound

    def compute_min_eig(self, tc, ac, l, u, device, momentum = 0, method = 'L', debug = False):
        if debug:
            print('Computing min_eig:')
        if method == 'L':
            a,aa = self.dom_eig_L(tc,ac,l,u,device, momentum = momentum, debug = debug)
            if debug:
                print('Heyyy',a,aa)
            if aa < 0:
                self.min_eig = min(0,0.5*float(aa))
            else:
                b,bb = self.dom_eig_L(tc,ac,l,u,device, momentum = momentum, shift = a, debug = debug)
                if debug:
                    print('HEYYY', a,b)
                self.min_eig = min(0,0.5*float(bb+aa))
        elif method == 'Store':
            LH = self.compute_LH(tc, ac, l, u, device)
            a,aa = UBAB.power_method(LH,device, debug = debug)
            if debug:
                print('Heyyy',a,aa)
            if aa < 0:
                self.min_eig = min(0,0.5*float(aa))
            else:
                b,bb =  UBAB.power_method(LH,device,shift = a, debug = debug)
                if debug:
                    print('HEYYY', a,b)
                self.min_eig = min(0,0.5*float(bb+aa))
        elif method == 'Singla':
            self.min_eig = -float(self.singla_2020(device,tc,ac))/2
        elif method == 'Singla_local':
            self.min_eig = -float(self.singla_2020_local(device,l,u,tc,ac))/2
        else:
            alphas = -self.alphas(tc,ac,l,u,device,v = u-l+0.0001)
            self.min_eig = alphas*(alphas<0)
