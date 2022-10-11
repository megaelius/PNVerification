import os
import sys
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms, datasets
import torchattacks

def load_db(root='./data', name = 'MNIST', batch_size=64, shuffle=True, valid_ratio=0.2, augment = True, resize = 0):
    if name == 'MNIST':
        #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.MNIST(root, train=True, download=True, transform = transform)
        testset = datasets.MNIST(root, train=False, download=True, transform = transform)
        image_size, n_classes, channels_in = 28, 10, 1
    elif name == 'CIFAR10':
        #transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
        #                                transforms.RandomHorizontalFlip(),
        #                                transforms.ToTensor(),
        #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
        #transform_test = transforms.Compose([transforms.ToTensor(),
        #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_test = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.CIFAR10(root, train=True, download=True, transform = transform if augment else transform_test)
        testset = datasets.CIFAR10(root, train=False, download=True, transform = transform_test)
        image_size, n_classes, channels_in = 32, 10, 3
    elif name == 'CIFAR100':
        #transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
        #                                transforms.RandomHorizontalFlip(),
        #                                transforms.ToTensor(),
        #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
        #transform_test = transforms.Compose([transforms.ToTensor(),
        #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_test = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.CIFAR100(root, train=True, download=True, transform = transform if augment else transform_test)
        testset = datasets.CIFAR100(root, train=False, download=True, transform = transform_test)
        image_size, n_classes, channels_in = 32, 100, 3
    elif name == 'STL10':
        #transform = transforms.Compose([transforms.RandomCrop(96, padding=4),
        #                                transforms.RandomHorizontalFlip(),
        #                                transforms.ToTensor(),
        #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image_size, n_classes, channels_in = 96, 10, 3
        if resize:
            transform = transforms.Compose([transforms.RandomCrop(96, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Resize(resize)])
            transform_test = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize(resize)])
            image_size = resize
        else:
            transform = transforms.Compose([transforms.RandomCrop(96, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()])
            transform_test = transforms.Compose([transforms.ToTensor()])

        #transform_test = transforms.Compose([transforms.ToTensor(),
        #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.STL10(root, split = 'train', download=True, transform = transform if augment else transform_test)
        testset = datasets.STL10(root, split = 'test', download=True, transform = transform_test)

    elif name == 'imagenette':
        #transform = transforms.Compose([
        #    transforms.Resize(256),
        #    transforms.CenterCrop(224),
        #    transforms.RandomCrop(224, padding=4),
        #    transforms.RandomHorizontalFlip(),
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #])
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        #transform_test = transforms.Compose([
        #    transforms.Resize(256),
        #    transforms.CenterCrop(224),
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        trainset = datasets.ImageFolder(root+'/imagenette2/train', transform if augment else transform_test)
        testset = datasets.ImageFolder(root+'/imagenette2/val', transform_test)

        image_size, n_classes, channels_in = 224, 10, 3

    if valid_ratio > 0:
        # # divide the training set into validation and training set.
        instance_num = len(trainset)
        indices = list(range(instance_num))
        split_pt = int(instance_num * valid_ratio)
        train_idx, valid_idx = indices[split_pt:], indices[:split_pt]
        train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)
        train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
        valid_loader = None

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader, image_size, n_classes, channels_in


'''
CCP
--------------------------------------------------------------------------------
'''

class CCP(nn.Module):
    def __init__(self, hidden_size, weight_sharing = False, image_size=28, channels_in=1, n_degree=4, bias=False, n_classes=10):
        super(CCP, self).__init__()
        self.image_size = image_size
        self.channels_in = channels_in
        self.total_image_size = self.image_size * self.image_size * channels_in
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_degree = n_degree
        self.bias = bias
        self.weight_sharing = False
        for i in range(1, self.n_degree + 1):
            setattr(self, 'U{}'.format(i), nn.Linear(self.total_image_size, self.hidden_size, bias=bias))
        self.C = nn.Linear(self.hidden_size, self.n_classes, bias=True)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and classname != 'CCP_Conv' and classname != 'NCP_Conv':
            m.weight.data.normal_(0.0, 0.01)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        # print('initializing {}'.format(classname))

    def forward(self, z):
        h = z.view(-1, self.total_image_size)
        out = self.U1(h)
        for i in range(2, self.n_degree + 1):
            if not self.weight_sharing:
                out = getattr(self, 'U{}'.format(i))(h) * out + out
            else:
                out = getattr(self, 'U{}'.format(1))(h) * out + out
            #print('heyyyyy',out.view(-1,self.hidden_size)[0,:10])
        out = self.C(out)
        return out

    def verif_objective(self, z, tc, ac):
        out = self.forward(z)
        return out[:,tc] - out[:,ac]

    def compute_signs(self,device):
        for i in range(1, self.n_degree + 1):
            setattr(self, 'U{}p'.format(i), nn.Linear(self.total_image_size, self.hidden_size, bias=self.bias, device = device))
            setattr(self, 'U{}n'.format(i), nn.Linear(self.total_image_size, self.hidden_size, bias=self.bias, device = device))
            with torch.no_grad():
                getattr(self, 'U{}p'.format(i)).weight.copy_(getattr(self, 'U{}'.format(i)).weight*(getattr(self, 'U{}'.format(i)).weight > 0))
                getattr(self, 'U{}n'.format(i)).weight.copy_(getattr(self, 'U{}'.format(i)).weight*(getattr(self, 'U{}'.format(i)).weight < 0))
        self.Cp = nn.Linear(self.hidden_size, self.n_classes, bias = False,device = device)
        self.Cn = nn.Linear(self.hidden_size, self.n_classes, bias = False,device = device)
        with torch.no_grad():
            self.Cp.weight.copy_(self.C.weight*(self.C.weight > 0))
            self.Cn.weight.copy_(self.C.weight*(self.C.weight < 0))

    def forward_bounds(self,l,u):
        with torch.no_grad():
            lxi = (self.U1p(l) + self.U1n(u)).unsqueeze(-1)
            uxi = (self.U1p(u) + self.U1n(l)).unsqueeze(-1)
            #print(getattr(self, 'U{}'.format(2))(l) * uxi.squeeze(-1) + uxi.squeeze(-1))
            for i in range(2, self.n_degree + 1):

                lxi_ = (getattr(self, 'U{}p'.format(i))(l) + getattr(self, 'U{}n'.format(i))(u) + 1).unsqueeze(-1)
                uxi_ = (getattr(self, 'U{}p'.format(i))(u) + getattr(self, 'U{}n'.format(i))(l) + 1).unsqueeze(-1)
                S = torch.cat((lxi_*lxi.view(lxi_.shape), lxi_*uxi.view(lxi_.shape), uxi_*lxi.view(lxi_.shape), uxi_*uxi.view(lxi_.shape)),dim = -1)
                uxi, _ = torch.max(S,-1, keepdim = True)
                lxi, _ = torch.min(S,-1, keepdim = True)
                #print(uxi, lxi)

            lxi = lxi.squeeze(-1)
            uxi = uxi.squeeze(-1)

            lout = self.Cp(lxi) + self.Cn(uxi) + self.C.bias
            uout = self.Cp(uxi) + self.Cn(lxi) + self.C.bias
            return lout, uout

    def forward_bounds_intermediate(self,l,u, level):
        lxi = (self.U1p(l) + self.U1n(u)).unsqueeze(-1)
        uxi = (self.U1p(u) + self.U1n(l)).unsqueeze(-1)
        #print(getattr(self, 'U{}'.format(2))(l) * uxi.squeeze(-1) + uxi.squeeze(-1))
        for i in range(2, level + 1):

            lxi_ = (getattr(self, 'U{}p'.format(i))(l) + getattr(self, 'U{}n'.format(i))(u) + 1).unsqueeze(-1)
            uxi_ = (getattr(self, 'U{}p'.format(i))(u) + getattr(self, 'U{}n'.format(i))(l) + 1).unsqueeze(-1)
            S = torch.cat((lxi_*lxi.view(lxi_.shape), lxi_*uxi.view(lxi_.shape), uxi_*lxi.view(lxi_.shape), uxi_*uxi.view(lxi_.shape)),dim = -1)
            uxi, _ = torch.max(S,-1)
            lxi, _ = torch.min(S,-1)

        return lxi,uxi

    def bounds_intermediate_grad(self,l,u,level):
        if level == 1:
            return self.U1.weight.data.transpose(0,1), self.U1.weight.data.transpose(0,1)
        else:

            lxl_ = (getattr(self, 'U{}p'.format(level))(l) + getattr(self, 'U{}n'.format(level))(u) + 1)
            uxl_ = (getattr(self, 'U{}p'.format(level))(u) + getattr(self, 'U{}n'.format(level))(l) + 1)

            lg_prev, ug_prev = self.bounds_intermediate_grad(l,u,level-1)

            S = torch.cat(((lxl_*lg_prev).unsqueeze(-1), (lxl_*ug_prev).unsqueeze(-1), (uxl_*lg_prev).unsqueeze(-1), (uxl_*ug_prev).unsqueeze(-1)),dim = -1)
            lright, _ = torch.min(S,-1)
            uright, _ = torch.max(S,-1)
            lright = lright.squeeze(-1)
            uright = uright.squeeze(-1)

            lxl, uxl = self.forward_bounds_intermediate(l,u,level-1)
            lxl = lxl.squeeze()
            uxl = uxl.squeeze()

            lout = lxl*getattr(self, f'U{level}p').weight.data.transpose(0,1) + uxl*getattr(self, f'U{level}n').weight.data.transpose(0,1) + lright
            uout = uxl*getattr(self, f'U{level}p').weight.data.transpose(0,1) + lxl*getattr(self, f'U{level}n').weight.data.transpose(0,1) + uright

            return lout, uout

    def output_intermediate(self,z,level):
        z = z.view(-1, self.total_image_size)
        if level == 1:
            return self.U1(z)
        else:
            return (getattr(self, 'U{}'.format(level))(z) + 1) * self.output_intermediate(z,level-1)

    def gradient_intermediate(self,z,level):
        with torch.no_grad():
            if level == 1:
                #[net.total_image_size,16]
                return self.U1.weight.data.transpose(0,1)
            else:
                #print((getattr(self, 'U{}'.format(level)).weight.data.unsqueeze(0).transpose(1,2)*self.output_intermediate(z,level-1)).shape)
                #print(getattr(self, 'U{}'.format(level)).weight.data.shape, self.gradient_intermediate(z,level-1).shape)
                #print(((getattr(self, 'U{}'.format(level))(z) + torch.ones((1,self.hidden_size))) * self.gradient_intermediate(z,level-1)).shape)
                return (getattr(self, 'U{}'.format(level)).weight.data.unsqueeze(0).transpose(1,2)*self.output_intermediate(z,level-1)).squeeze() + (getattr(self, 'U{}'.format(level))(z) + torch.ones((1,self.hidden_size))) * self.gradient_intermediate(z,level-1)

    def hessian_intermediate(self,z,level):
        with torch.no_grad():
            if level == 1:
                return torch.zeros((self.hidden_size,self.total_image_size,self.total_image_size))
            elif level == 2:
                aux = torch.bmm(self.U1.weight.data.unsqueeze(1).transpose(2,1), self.U2.weight.data.unsqueeze(1))
                return aux + aux.transpose(1,2)
            else:
                aux2 = getattr(self, 'U{}'.format(level))(z) + torch.ones((1,self.hidden_size))
                aux = torch.bmm(getattr(self, 'U{}'.format(level)).weight.data.unsqueeze(1).transpose(2,1), self.gradient_intermediate(z,level-1).unsqueeze(1).transpose(0,2))
                return aux + aux.transpose(1,2) + (self.hessian_intermediate(z,level-1).transpose(0,2)*aux2.squeeze()).transpose(0,2)

    def hessian(self,tc,ac,z):
        return torch.matmul((self.C.weight.data[tc] - self.C.weight.data[ac]), self.hessian_intermediate(z,self.n_degree).transpose(0,1))

    def compute_Qq(self,tc,ac, cpu = False, z = None):
        k = self.hidden_size
        d = self.total_image_size
        self.beta = self.C.bias.cpu().detach().numpy()
        with torch.no_grad():
            if self.weight_sharing:
                self.Q = torch.matmul(self.C.weight.data[tc] - self.C.weight.data[ac], torch.bmm(self.U1.weight.data.unsqueeze(1).transpose(2,1), self.U1.weight.data.unsqueeze(1)).transpose(0,1)).cpu().numpy()
            else:
                self.Q = torch.matmul(self.C.weight.data[tc] - self.C.weight.data[ac], torch.bmm(self.U1.weight.data.unsqueeze(1).transpose(2,1), self.U2.weight.data.unsqueeze(1)).transpose(0,1)).cpu().numpy()
            self.q = torch.matmul(self.C.weight.data[tc] - self.C.weight.data[ac],self.U1.weight.data).cpu().numpy()

    def L_hessian_recursive(self,l,u,lw,uw,z,level,device):
        with torch.no_grad():
            if level == 1:
                return torch.zeros((self.total_image_size,1), device = device),torch.zeros((self.total_image_size,1), device = device),torch.zeros((self.total_image_size,1), device = device),torch.zeros((self.total_image_size,1), device = device)
            else:
                lg_, ug_ = self.bounds_intermediate_grad(l,u,level-1)
                S = torch.cat(((lg_*lw.transpose(0,1)).unsqueeze(-1), (lg_*uw.transpose(0,1)).unsqueeze(-1), (ug_*lw.transpose(0,1)).unsqueeze(-1), (ug_*uw.transpose(0,1)).unsqueeze(-1)),dim = -1)
                lg, _ = torch.min(S,-1)
                ug, _ = torch.max(S,-1)

                LAz = getattr(self,f'U{level}p').weight.data*torch.matmul(lg.transpose(0,1), z.transpose(0,1)) + getattr(self,f'U{level}n').weight.data*torch.matmul(ug.transpose(0,1), z.transpose(0,1)) + lg.transpose(0,1)*torch.matmul(getattr(self,f'U{level}p').weight.data, z.transpose(0,1)) + ug.transpose(0,1)*torch.matmul(getattr(self,f'U{level}n').weight.data, z.transpose(0,1))
                UAz = getattr(self,f'U{level}p').weight.data*torch.matmul(ug.transpose(0,1), z.transpose(0,1)) + getattr(self,f'U{level}n').weight.data*torch.matmul(lg.transpose(0,1), z.transpose(0,1)) + ug.transpose(0,1)*torch.matmul(getattr(self,f'U{level}p').weight.data, z.transpose(0,1)) + lg.transpose(0,1)*torch.matmul(getattr(self,f'U{level}n').weight.data, z.transpose(0,1))
                LA1 = getattr(self,f'U{level}p').weight.data*torch.sum(lg, 0).unsqueeze(-1) + getattr(self,f'U{level}n').weight.data*torch.sum(ug, 0).unsqueeze(-1) + lg.transpose(0,1)*torch.sum(getattr(self,f'U{level}p').weight.data,1, keepdim = True) + ug.transpose(0,1)*torch.sum(getattr(self,f'U{level}n').weight.data, 1, keepdim = True)
                UA1 = getattr(self,f'U{level}p').weight.data*torch.sum(ug, 0).unsqueeze(-1) + getattr(self,f'U{level}n').weight.data*torch.sum(lg, 0).unsqueeze(-1) + ug.transpose(0,1)*torch.sum(getattr(self,f'U{level}p').weight.data,1, keepdim = True) + lg.transpose(0,1)*torch.sum(getattr(self,f'U{level}n').weight.data, 1, keepdim = True)

                UAz = torch.sum(UAz.transpose(0,1),dim = -1, keepdim=True)
                LAz = torch.sum(LAz.transpose(0,1),dim = -1, keepdim=True)
                LA1 = torch.sum(LA1.transpose(0,1),dim = -1, keepdim=True)
                UA1 = torch.sum(UA1.transpose(0,1),dim = -1, keepdim=True)

                lxlev_ = (getattr(self, 'U{}p'.format(level))(l) + getattr(self, 'U{}n'.format(level))(u) + 1).transpose(0,1)
                uxlev_ = (getattr(self, 'U{}p'.format(level))(u) + getattr(self, 'U{}n'.format(level))(l) + 1).transpose(0,1)

                S1 = torch.cat(((lxlev_*lw).unsqueeze(-1), (lxlev_*uw).unsqueeze(-1), (uxlev_*lw).unsqueeze(-1), (uxlev_*uw).unsqueeze(-1)),dim = -1)
                lxlev, _ = torch.min(S1,-1)
                uxlev, _ = torch.max(S1,-1)

                LAz_prev, UAz_prev, LA1_prev, UA1_prev = self.L_hessian_recursive(l,u,lxlev,uxlev,z,level-1,device)

                return LAz + LAz_prev, UAz + UAz_prev, LA1 + LA1_prev, UA1 + UA1_prev

    def evaluate_L_hessian(self,tc,ac,l,u,z,device):
        C = (self.C.weight.data[tc] - self.C.weight.data[ac]).unsqueeze(-1)
        Lz,Uz,ld,ud = self.L_hessian_recursive(l,u,C,C,z,self.n_degree, device = device)
        return ((Lz+Uz)/2 + ((ld-ud)/2)*z.transpose(0,1)).transpose(0,1)

    def dom_eig_L(self,tc,ac,l,u,device,debug = False):
        with torch.no_grad():
            z = torch.rand((1,self.total_image_size), device = device)
            z /= (torch.sqrt((z*z).sum()))
            prev_z = torch.ones(self.total_image_size, device = device)
            while not torch.sum(torch.abs(z-prev_z)) < 1e-5:
                '''
                Each iteration we do two steps because if the dominant eigenvalue is negative,
                with a single step prev_z = -z and never converges.
                '''
                prev_z = z.clone()
                z = self.evaluate_L_hessian(tc,ac,l,u,z,device)
                min_eig = torch.sqrt((z*z).sum())
                z /= min_eig
                z = self.evaluate_L_hessian(tc,ac,l,u,z,device)
                min_eig = torch.sqrt((z*z).sum())
                z /= min_eig
                if debug:
                    print(min_eig,torch.sum(torch.abs(z-prev_z)))
            return min_eig

    def bounds_diag_hess(self,tc,ac,l,u,device):
        lx = self.U1p(l) + self.U1n(u)
        ux = self.U1p(u) + self.U1n(l)

        lg = self.U1.weight.data.transpose(0,1)
        ug = self.U1.weight.data.transpose(0,1)

        Lhp = torch.zeros(lg.shape,device = device)
        Uhp = torch.zeros(lg.shape,device = device)
        Lhn = torch.zeros(lg.shape,device = device)
        Uhn = torch.zeros(lg.shape,device = device)

        for i in range(2,self.n_degree+1):
            lxhat = getattr(self, 'U{}p'.format(i))(l) + getattr(self, 'U{}n'.format(i))(u) + 1
            uxhat = getattr(self, 'U{}p'.format(i))(u) + getattr(self, 'U{}n'.format(i))(l) + 1

            Lhp_ = Lhp*lxhat*(lxhat>0) + Uhn*uxhat*(uxhat<0)
            Lhn_ = Lhn*uxhat*(uxhat>0) + Uhp*lxhat*(lxhat<0)
            Uhp_ = Uhp*uxhat*(uxhat>0) + Lhn*lxhat*(lxhat<0)
            Uhn_ = Uhn*lxhat*(lxhat>0) + Lhp*uxhat*(uxhat<0)

            Lhp = 2*(lg*(lg>0)*getattr(self, 'U{}p'.format(i)).weight.data.transpose(0,1) + ug*(ug<0)*getattr(self, 'U{}n'.format(i)).weight.data.transpose(0,1)) + Lhp_
            Lhn = 2*(lg*(lg<0)*getattr(self, 'U{}p'.format(i)).weight.data.transpose(0,1) + ug*(ug>0)*getattr(self, 'U{}n'.format(i)).weight.data.transpose(0,1)) + Lhn_
            Uhp = 2*(ug*(ug>0)*getattr(self, 'U{}p'.format(i)).weight.data.transpose(0,1) + lg*(lg<0)*getattr(self, 'U{}n'.format(i)).weight.data.transpose(0,1)) + Uhp_
            Uhn = 2*(ug*(ug<0)*getattr(self, 'U{}p'.format(i)).weight.data.transpose(0,1) + lg*(lg>0)*getattr(self, 'U{}n'.format(i)).weight.data.transpose(0,1)) + Uhn_

            S = torch.cat(((lxhat*lg).unsqueeze(-1),(lxhat*ug).unsqueeze(-1),(uxhat*lg).unsqueeze(-1),(uxhat*ug).unsqueeze(-1)), dim = -1)
            lg,_ = torch.min(S,dim = -1)
            ug,_ = torch.max(S,dim = -1)

            lg += getattr(self, 'U{}p'.format(i)).weight.data.transpose(0,1)*lx + getattr(self, 'U{}n'.format(i)).weight.data.transpose(0,1)*ux
            ug += getattr(self, 'U{}p'.format(i)).weight.data.transpose(0,1)*ux + getattr(self, 'U{}n'.format(i)).weight.data.transpose(0,1)*lx

            S = torch.cat(((lxhat*lx).unsqueeze(-1),(lxhat*ux).unsqueeze(-1),(uxhat*lx).unsqueeze(-1),(uxhat*ux).unsqueeze(-1)), dim = -1)
            lx,_ = torch.min(S,dim = -1)
            ux,_ = torch.max(S,dim = -1)

        C = self.C.weight.data[tc,:] - self.C.weight.data[ac,:]
        Cp = C*(C>0)
        Cn = C*(C<0)
        return torch.matmul(Lhp + Lhn,Cp) + torch.matmul(Uhp + Uhn,Cn), torch.matmul(Lhp + Lhn,Cn) + torch.matmul(Uhp + Uhn,Cp)
    '''
    vector of alphas
    '''
    def lowerbound_diag(self,tc,ac,l,u,device):
        lx = self.U1p(l) + self.U1n(u)
        ux = self.U1p(u) + self.U1n(l)

        lg = self.U1.weight.data.transpose(0,1)
        ug = self.U1.weight.data.transpose(0,1)

        Lhp = torch.zeros(lg.shape,device = device)
        Uhp = torch.zeros(lg.shape,device = device)
        Lhn = torch.zeros(lg.shape,device = device)
        Uhn = torch.zeros(lg.shape,device = device)

        for i in range(2,self.n_degree+1):
            lxhat = getattr(self, 'U{}p'.format(i))(l) + getattr(self, 'U{}n'.format(i))(u) + 1
            uxhat = getattr(self, 'U{}p'.format(i))(u) + getattr(self, 'U{}n'.format(i))(l) + 1

            Lhp_ = Lhp*lxhat*(lxhat>0) + Uhn*uxhat*(uxhat<0)
            Lhn_ = Lhn*uxhat*(uxhat>0) + Uhp*lxhat*(lxhat<0)
            Uhp_ = Uhp*uxhat*(uxhat>0) + Lhn*lxhat*(lxhat<0)
            Uhn_ = Uhn*lxhat*(lxhat>0) + Lhp*uxhat*(uxhat<0)

            Lhp = 2*(lg*(lg>0)*getattr(self, 'U{}p'.format(i)).weight.data.transpose(0,1) + ug*(ug<0)*getattr(self, 'U{}n'.format(i)).weight.data.transpose(0,1)) + Lhp_
            Lhn = 2*(lg*(lg<0)*getattr(self, 'U{}p'.format(i)).weight.data.transpose(0,1) + ug*(ug>0)*getattr(self, 'U{}n'.format(i)).weight.data.transpose(0,1)) + Lhn_
            Uhp = 2*(ug*(ug>0)*getattr(self, 'U{}p'.format(i)).weight.data.transpose(0,1) + lg*(lg<0)*getattr(self, 'U{}n'.format(i)).weight.data.transpose(0,1)) + Uhp_
            Uhn = 2*(ug*(ug<0)*getattr(self, 'U{}p'.format(i)).weight.data.transpose(0,1) + lg*(lg>0)*getattr(self, 'U{}n'.format(i)).weight.data.transpose(0,1)) + Uhn_

            S = torch.cat(((lxhat*lg).unsqueeze(-1),(lxhat*ug).unsqueeze(-1),(uxhat*lg).unsqueeze(-1),(uxhat*ug).unsqueeze(-1)), dim = -1)
            lg,_ = torch.min(S,dim = -1)
            ug,_ = torch.max(S,dim = -1)

            lg += getattr(self, 'U{}p'.format(i)).weight.data.transpose(0,1)*lx + getattr(self, 'U{}n'.format(i)).weight.data.transpose(0,1)*ux
            ug += getattr(self, 'U{}p'.format(i)).weight.data.transpose(0,1)*ux + getattr(self, 'U{}n'.format(i)).weight.data.transpose(0,1)*lx

            S = torch.cat(((lxhat*lx).unsqueeze(-1),(lxhat*ux).unsqueeze(-1),(uxhat*lx).unsqueeze(-1),(uxhat*ux).unsqueeze(-1)), dim = -1)
            lx,_ = torch.min(S,dim = -1)
            ux,_ = torch.max(S,dim = -1)

        C = self.C.weight.data[tc,:] - self.C.weight.data[ac,:]
        Cp = C*(C>0)
        Cn = C*(C<0)
        return torch.matmul(Lhp + Lhn,Cp) + torch.matmul(Uhp + Uhn,Cn)

    def sum_abs_out_diag(self,tc,ac,l,u,v,device):
        lx = self.U1p(l) + self.U1n(u)
        ux = self.U1p(u) + self.U1n(l)

        lg = self.U1.weight.data.transpose(0,1)
        ug = self.U1.weight.data.transpose(0,1)

        result = torch.zeros(lg.shape,device = device)

        for i in range(2,self.n_degree+1):
            lxhat = getattr(self, 'U{}p'.format(i))(l) + getattr(self, 'U{}n'.format(i))(u) + 1
            uxhat = getattr(self, 'U{}p'.format(i))(u) + getattr(self, 'U{}n'.format(i))(l) + 1

            maxnorm_g = torch.max(torch.abs(lg), torch.abs(ug))
            maxnorm_w = torch.abs(getattr(self, 'U{}'.format(i)).weight.data.transpose(0,1))
            result_ = []
            for j in range(maxnorm_g.shape[0]):
                v_aux = v.clone()
                v_aux/=v[0,j]
                v_aux[0,j] = 0
                result_.append((maxnorm_g*torch.matmul(v_aux,maxnorm_w))[j,:].unsqueeze(0) + (maxnorm_w*torch.matmul(v_aux,maxnorm_g))[j,:].unsqueeze(0))

            result_ = torch.cat(result_,dim = 0)

            result = torch.max(torch.abs(lxhat),torch.abs(uxhat))*result + result_

            S = torch.cat(((lxhat*lg).unsqueeze(-1),(lxhat*ug).unsqueeze(-1),(uxhat*lg).unsqueeze(-1),(uxhat*ug).unsqueeze(-1)), dim = -1)
            lg,_ = torch.min(S,dim = -1)
            ug,_ = torch.max(S,dim = -1)

            lg += getattr(self, 'U{}p'.format(i)).weight.data.transpose(0,1)*lx + getattr(self, 'U{}n'.format(i)).weight.data.transpose(0,1)*ux
            ug += getattr(self, 'U{}p'.format(i)).weight.data.transpose(0,1)*ux + getattr(self, 'U{}n'.format(i)).weight.data.transpose(0,1)*lx

            S = torch.cat(((lxhat*lx).unsqueeze(-1),(lxhat*ux).unsqueeze(-1),(uxhat*lx).unsqueeze(-1),(uxhat*ux).unsqueeze(-1)), dim = -1)
            lx,_ = torch.min(S,dim = -1)
            ux,_ = torch.max(S,dim = -1)

        C = self.C.weight.data[tc,:] - self.C.weight.data[ac,:]
        return torch.matmul(result,torch.abs(C))

    def alphas(self,tc,ac,l,u,device,v = None):
        with torch.no_grad():
            if v is None:
                v = torch.ones(l.shape,device = device)
            ldiag = self.lowerbound_diag(tc,ac,l,u,device)
            sum_abs = self.sum_abs_out_diag(tc,ac,l,u,v,device)
            return -(ldiag-sum_abs)/2

    def compute_min_eig(self, tc, ac, l, u, device, method = 'L', debug = False):
        if debug:
            print('Computing min_eig:')
        if self.n_degree == 2:
            '''
            Simply do power method because hessian is constant
            '''
            with torch.no_grad():
                z = torch.ones((self.total_image_size), device = device)
                z /= (torch.sqrt((z*z).sum()))
                prev_z = torch.ones(self.total_image_size, device = device)
                while not torch.sum(torch.abs(z-prev_z)) < 1e-5:
                    '''
                    Each iteration we do two steps because if the dominant eigenvalue is negative,
                    with a single step prev_z = -z and never converges.
                    '''
                    prev_z = z.clone()
                    z = torch.matmul(self.C.weight.data[tc] - self.C.weight.data[ac], (torch.mul(self.U1.weight.data.transpose(0,1),self.U2(z)) + torch.mul(self.U2.weight.data.transpose(0,1),self.U1(z))).transpose(0,1))
                    self.min_eig = (torch.sqrt((z*z).sum()))
                    z /= self.min_eig
                    z = torch.matmul(self.C.weight.data[tc] - self.C.weight.data[ac], (torch.mul(self.U1.weight.data.transpose(0,1),self.U2(z)) + torch.mul(self.U2.weight.data.transpose(0,1),self.U1(z))).transpose(0,1))
                    self.min_eig = (torch.sqrt((z*z).sum()))
                    z /= self.min_eig
                    if debug:
                        print(self.min_eig)
                self.min_eig = min(0,-0.5*float(self.min_eig))
        else:
            if method == 'L':
                self.min_eig = min(0,-0.5*float(self.dom_eig_L(tc,ac,l,u,device, debug = debug)))
            elif method == 'Weyl':
                self.min_eig = min(0,-0.5*float(self.dom_eig_weyl(tc,ac,l,u,device)))
            elif method == 'alphas':
                alphas = -self.alphas(tc,ac,l,u,device,v = u-l+0.0001)
                self.min_eig = alphas*(alphas<0)

    def hessian_vector_recursive(self, z0, z, level,device):
        '''
        evaluate at z0 multiply by z
        '''
        with torch.no_grad():
            if level == 1:
                return torch.zeros((self.total_image_size, self.hidden_size),device = device)
            if level == 2:
                return self.U1.weight.data.transpose(0,1)*self.U2(z) + self.U2.weight.data.transpose(0,1)*self.U1(z)
            else:
                grad_prev = self.gradient_intermediate(z0,level-1)
                aux1 = grad_prev*getattr(self, f'U{level}')(z) + getattr(self, f'U{level}').weight.data.transpose(0,1)*torch.matmul(z, grad_prev)
                #print((getattr(self, f'U{level}')(z) + 1).shape)
                #print((self.C.weight.data[tc] - self.C.weight.data[ac]).shape)
                return aux1 + (getattr(self, f'U{level}')(z0) + 1)*self.hessian_vector_recursive(z0, z, level-1, device)

    def eval_hessian_vector(self, tc, ac, z0, z, device):
        with torch.no_grad():
            return torch.matmul(self.hessian_vector_recursive(z0, z, self.n_degree,device), self.C.weight.data[tc] - self.C.weight.data[ac])

    def min_eig_hessian(self, tc, ac, z0, device, debug = False):
        '''
        Returns the dominant eigenvalue of the hessian matrix evaluated at z0
        '''
        min_eig = None
        signed_eig = None
        with torch.no_grad():
            z = torch.rand(z0.shape[1], device = device)
            z = z/(torch.sqrt((z*z).sum()))
            prev_z = torch.ones(z0.shape[1], device = device)
            while not torch.sum(torch.abs(z-prev_z)) < 1e-4:
                prev_z = z.clone()
                z = self.eval_hessian_vector( tc, ac, z0, z,device)
                min_eig = (torch.sqrt((z*z).sum()))
                signed_eig = (z/prev_z)[0]
                z = z/min_eig
                z = self.eval_hessian_vector(tc, ac, z0, z, device)
                min_eig = (torch.sqrt((z*z).sum()))
                z = z/min_eig
                if debug:
                    print(signed_eig, torch.sum(torch.abs(z-prev_z)))
            return signed_eig, z

'''
NCP
--------------------------------------------------------------------------------
'''
class NCP(nn.Module):
    def __init__(self, hidden_size, bn = False, image_size=28, channels_in=1, n_degree=4, bias=False, down_sample = True, n_classes=10):
        super(NCP, self).__init__()
        print('Initializing the NCP model with degree={}.'.format(n_degree))
        self.image_size = image_size
        self.channels_in = channels_in
        self.total_image_size = self.image_size * self.image_size * channels_in
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_degree = n_degree
        self.bias = bias
        self.bn = bn
        self.down_sample = down_sample
        for i in range(1, self.n_degree + 1):
            out_hidden = int(self.hidden_size // 2**(i-1)) if down_sample else self.hidden_size
            in_hidden = int(self.hidden_size // 2**(i-2)) if down_sample else self.hidden_size
            setattr(self, 'A{}'.format(i), nn.Linear(self.total_image_size, out_hidden, bias=bias))
            if i > 1:
                setattr(self, 'S{}'.format(i), nn.Linear(in_hidden, out_hidden, bias=True))
                if bn:
                    setattr(self, 'BN{}'.format(i), nn.BatchNorm1d((out_hidden)))
        if down_sample:
            self.C = nn.Linear(int(self.hidden_size // 2**(i-1)), self.n_classes, bias=True)
        else:
            self.C = nn.Linear(self.hidden_size, self.n_classes, bias=True)

    def forward(self, z):
        h = z.view(-1, self.total_image_size)
        out = self.A1(h)
        for i in range(2, self.n_degree + 1):
            h1 = getattr(self, 'A{}'.format(i))(h)
            out = getattr(self, 'S{}'.format(i))(out) * h1
            if self.bn:
                out = getattr(self, 'BN{}'.format(i))(out)
        out = self.C(out)
        return out

    def verif_objective(self, z, tc, ac):
        out = self.forward(z)
        return out[:,tc] - out[:,ac]

    def compute_signs(self,device):
        for i in range(1, self.n_degree + 1):
            setattr(self, 'A{}p'.format(i), nn.Linear(getattr(self, 'A{}'.format(i)).weight.data.shape[1], getattr(self, 'A{}'.format(i)).weight.data.shape[0], bias=self.bias, device = device))
            setattr(self, 'A{}n'.format(i), nn.Linear(getattr(self, 'A{}'.format(i)).weight.data.shape[1], getattr(self, 'A{}'.format(i)).weight.data.shape[0], bias=self.bias, device = device))
            with torch.no_grad():
                getattr(self, 'A{}p'.format(i)).weight.copy_(getattr(self, 'A{}'.format(i)).weight*(getattr(self, 'A{}'.format(i)).weight > 0))
                getattr(self, 'A{}n'.format(i)).weight.copy_(getattr(self, 'A{}'.format(i)).weight*(getattr(self, 'A{}'.format(i)).weight < 0))
            if i > 1:
                setattr(self, 'S{}p'.format(i), nn.Linear(getattr(self, 'S{}'.format(i)).weight.data.shape[1], getattr(self, 'S{}'.format(i)).weight.data.shape[0], bias=False, device = device))
                setattr(self, 'S{}n'.format(i), nn.Linear(getattr(self, 'S{}'.format(i)).weight.data.shape[1], getattr(self, 'S{}'.format(i)).weight.data.shape[0], bias=False, device = device))
                with torch.no_grad():
                    getattr(self, 'S{}p'.format(i)).weight.copy_(getattr(self, 'S{}'.format(i)).weight*(getattr(self, 'S{}'.format(i)).weight > 0))
                    getattr(self, 'S{}n'.format(i)).weight.copy_(getattr(self, 'S{}'.format(i)).weight*(getattr(self, 'S{}'.format(i)).weight < 0))
                    #getattr(self, 'S{}p'.format(i)).bias.copy_(getattr(self, 'S{}'.format(i)).bias)
                    #getattr(self, 'S{}n'.format(i)).bias.copy_(getattr(self, 'S{}'.format(i)).bias)

        self.Cp = nn.Linear(self.C.weight.data.shape[1], self.C.weight.data.shape[0], bias = False,device = device)
        self.Cn = nn.Linear(self.C.weight.data.shape[1], self.C.weight.data.shape[0], bias = False,device = device)
        with torch.no_grad():
            self.Cp.weight.copy_(self.C.weight*(self.C.weight > 0))
            self.Cn.weight.copy_(self.C.weight*(self.C.weight < 0))

    def forward_bounds(self,l,u):
        with torch.no_grad():
            lxi = (self.A1p(l) + self.A1n(u))
            uxi = (self.A1p(u) + self.A1n(l))
            #print(getattr(self, 'U{}'.format(2))(l) * uxi.squeeze(-1) + uxi.squeeze(-1))
            for i in range(2, self.n_degree + 1):
                lxia = (getattr(self, 'A{}p'.format(i))(l) + getattr(self, 'A{}n'.format(i))(u)).unsqueeze(-1)
                uxia = (getattr(self, 'A{}p'.format(i))(u) + getattr(self, 'A{}n'.format(i))(l)).unsqueeze(-1)
                lxis = (getattr(self, 'S{}p'.format(i))(lxi) + getattr(self, 'S{}n'.format(i))(uxi)).unsqueeze(-1) + getattr(self, 'S{}'.format(i)).bias.data.unsqueeze(-1)
                uxis = (getattr(self, 'S{}p'.format(i))(uxi) + getattr(self, 'S{}n'.format(i))(lxi)).unsqueeze(-1) + getattr(self, 'S{}'.format(i)).bias.data.unsqueeze(-1)

                S = torch.cat((lxia*lxis, lxia*uxis, uxia*lxis, uxia*uxis),dim = -1)
                lxi, _ = torch.min(S,-1)
                uxi, _ = torch.max(S,-1)
                lxi = lxi.squeeze(-1)
                uxi = uxi.squeeze(-1)

            lout = self.Cp(lxi) + self.Cn(uxi) + self.C.bias
            uout = self.Cp(uxi) + self.Cn(lxi) + self.C.bias
            return lout, uout

    def forward_bounds_intermediate(self,l,u, level):
        with torch.no_grad():
            lxi = (self.A1p(l) + self.A1n(u))
            uxi = (self.A1p(u) + self.A1n(l))
            #print(getattr(self, 'U{}'.format(2))(l) * uxi.squeeze(-1) + uxi.squeeze(-1))
            for i in range(2, level + 1):
                lxia = (getattr(self, 'A{}p'.format(i))(l) + getattr(self, 'A{}n'.format(i))(u)).unsqueeze(-1)
                uxia = (getattr(self, 'A{}p'.format(i))(u) + getattr(self, 'A{}n'.format(i))(l)).unsqueeze(-1)
                lxis = (getattr(self, 'S{}p'.format(i))(lxi) + getattr(self, 'S{}n'.format(i))(uxi)).unsqueeze(-1) + getattr(self, 'S{}'.format(i)).bias.data.unsqueeze(-1)
                uxis = (getattr(self, 'S{}p'.format(i))(uxi) + getattr(self, 'S{}n'.format(i))(lxi)).unsqueeze(-1) + getattr(self, 'S{}'.format(i)).bias.data.unsqueeze(-1)
                S = torch.cat((lxia*lxis, lxia*uxis, uxia*lxis, uxia*uxis),dim = -1)
                lxi_, _ = torch.min(S,-1)
                uxi_, _ = torch.max(S,-1)
                lxi = lxi_.squeeze(-1)
                uxi = uxi_.squeeze(-1)
            return lxi,uxi

    def bounds_intermediate_grad(self,l,u,level):
        with torch.no_grad():
            if level == 1:
                return self.A1.weight.data.transpose(0,1), self.A1.weight.data.transpose(0,1)
            else:
                la,ua = self.forward_bounds_intermediate(l,u,level-1)
                lxl_ = (getattr(self, 'S{}p'.format(level))(la) + getattr(self, 'S{}n'.format(level))(ua)) + getattr(self, 'S{}'.format(level)).bias.data
                uxl_ = (getattr(self, 'S{}p'.format(level))(ua) + getattr(self, 'S{}n'.format(level))(la)) + getattr(self, 'S{}'.format(level)).bias.data

                lleft = (getattr(self, 'A{}p'.format(level)).weight.data.transpose(0,1)*lxl_ + getattr(self, 'A{}n'.format(level)).weight.data.transpose(0,1)*uxl_)
                uleft = (getattr(self, 'A{}p'.format(level)).weight.data.transpose(0,1)*uxl_ + getattr(self, 'A{}n'.format(level)).weight.data.transpose(0,1)*lxl_)

                laz = (getattr(self, 'A{}p'.format(level))(l) + getattr(self, 'A{}n'.format(level))(u))
                uaz = (getattr(self, 'A{}p'.format(level))(u) + getattr(self, 'A{}n'.format(level))(l))

                lg_prev, ug_prev = self.bounds_intermediate_grad(l,u,level-1)

                lgg = getattr(self, 'S{}p'.format(level))(lg_prev) + getattr(self, 'S{}n'.format(level))(ug_prev)
                ugg = getattr(self, 'S{}p'.format(level))(ug_prev) + getattr(self, 'S{}n'.format(level))(lg_prev)

                S = torch.cat(((laz*lgg).unsqueeze(-1), (laz*ugg).unsqueeze(-1), (uaz*lgg).unsqueeze(-1), (uaz*ugg).unsqueeze(-1)),dim = -1)
                lright, _ = torch.min(S,-1)
                uright, _ = torch.max(S,-1)
                lright = lright.squeeze(-1)
                uright = uright.squeeze(-1)

                return lleft+lright, uleft+uright

    def output_intermediate(self,z,level):
        z = z.view(-1,self.total_image_size)
        out = self.A1(z)
        for i in range(2, level + 1):
            h1 = getattr(self, 'A{}'.format(i))(z)
            out = getattr(self, 'S{}'.format(i))(out) * h1
        return out

    def gradient_intermediate(self,z,level):
        with torch.no_grad():
            if level == 1:
                #[net.total_image_size,16]
                return self.A1.weight.data.transpose(0,1)
            else:
                left = getattr(self, 'S{}'.format(level))(self.output_intermediate(z,level-1))
                right = (getattr(self, 'A{}'.format(level))(z)) * torch.matmul(self.gradient_intermediate(z,level-1),getattr(self, 'S{}'.format(level)).weight.data.transpose(0,1))
                return getattr(self, 'A{}'.format(level)).weight.data.transpose(0,1)*left.squeeze() + right

    def L_hessian_recursive(self,l,u,lw,uw,z,level,device):
        with torch.no_grad():
            if level == 1:
                return torch.zeros((self.total_image_size,1), device = device),torch.zeros((self.total_image_size,1), device = device),torch.zeros((self.total_image_size,1), device = device),torch.zeros((self.total_image_size,1), device = device)
            else:

                lg__, ug__ = self.bounds_intermediate_grad(l,u,level-1)
                #print(lg__.shape, lw.shape, getattr(self,f'S{level}p').weight.data.shape)

                lg_ = torch.matmul(lg__,getattr(self,f'S{level}p').weight.data.transpose(0,1)) + torch.matmul(ug__,getattr(self,f'S{level}n').weight.data.transpose(0,1))
                ug_ = torch.matmul(ug__,getattr(self,f'S{level}p').weight.data.transpose(0,1)) + torch.matmul(lg__,getattr(self,f'S{level}n').weight.data.transpose(0,1))

                S = torch.cat(((lg_*lw.transpose(0,1)).unsqueeze(-1), (lg_*uw.transpose(0,1)).unsqueeze(-1), (ug_*lw.transpose(0,1)).unsqueeze(-1), (ug_*uw.transpose(0,1)).unsqueeze(-1)),dim = -1)
                lg, _ = torch.min(S,-1)
                ug, _ = torch.max(S,-1)
                lg = lg.transpose(0,1)
                ug = ug.transpose(0,1)

                ones = torch.ones((1,self.total_image_size), device = device)
                #print(getattr(self,f'A{level}p').weight.data.shape, lg.shape, z.shape)
                UAz = (getattr(self,f'A{level}p').weight.data*torch.matmul(ug, z.transpose(0,1)) + getattr(self,f'A{level}n').weight.data*torch.matmul(lg, z.transpose(0,1)) + ug*torch.matmul(getattr(self,f'A{level}p').weight.data, z.transpose(0,1)) + lg*torch.matmul(getattr(self,f'A{level}n').weight.data, z.transpose(0,1))).transpose(0,1)
                LAz = (getattr(self,f'A{level}p').weight.data*torch.matmul(lg, z.transpose(0,1)) + getattr(self,f'A{level}n').weight.data*torch.matmul(ug, z.transpose(0,1)) + lg*torch.matmul(getattr(self,f'A{level}p').weight.data, z.transpose(0,1)) + ug*torch.matmul(getattr(self,f'A{level}n').weight.data, z.transpose(0,1))).transpose(0,1)
                LA1 = (getattr(self,f'A{level}p').weight.data*torch.matmul(lg, ones.transpose(0,1)) + getattr(self,f'A{level}n').weight.data*torch.matmul(ug, ones.transpose(0,1)) + lg*torch.matmul(getattr(self,f'A{level}p').weight.data, ones.transpose(0,1)) + ug*torch.matmul(getattr(self,f'A{level}n').weight.data, ones.transpose(0,1))).transpose(0,1)
                UA1 = (getattr(self,f'A{level}p').weight.data*torch.matmul(ug, ones.transpose(0,1)) + getattr(self,f'A{level}n').weight.data*torch.matmul(lg, ones.transpose(0,1)) + ug*torch.matmul(getattr(self,f'A{level}p').weight.data, ones.transpose(0,1)) + lg*torch.matmul(getattr(self,f'A{level}n').weight.data, ones.transpose(0,1))).transpose(0,1)

                UAz = torch.sum(UAz,dim = -1, keepdim=True)
                LAz = torch.sum(LAz,dim = -1, keepdim=True)
                LA1 = torch.sum(LA1,dim = -1, keepdim=True)
                UA1 = torch.sum(UA1,dim = -1, keepdim=True)
                #print(UAz.shape)
                #print('Holaaa',LAz.shape,getattr(self,f'S{level}p').weight.data.shape)
                #weights for the next recursion
                lxlev_ = torch.matmul(getattr(self, 'A{}p'.format(level)).weight.data,l.transpose(0,1)) + torch.matmul(getattr(self, 'A{}n'.format(level)).weight.data,u.transpose(0,1))
                uxlev_ = torch.matmul(getattr(self, 'A{}p'.format(level)).weight.data,u.transpose(0,1)) + torch.matmul(getattr(self, 'A{}n'.format(level)).weight.data,l.transpose(0,1))
                #if self.down_sample:
                #    lxlev_ = lxlev_.repeat(int(lw.shape[0]/lxlev_.shape[0]),1)
                #    uxlev_ = uxlev_.repeat(int(uw.shape[0]/uxlev_.shape[0]),1)
                #print(lw.shape, lxlev_.shape)
                S1 = torch.cat(((lxlev_*lw).unsqueeze(-1), (lxlev_*uw).unsqueeze(-1), (uxlev_*lw).unsqueeze(-1), (uxlev_*uw).unsqueeze(-1)),dim = -1)
                lxlev, _ = torch.min(S1,-1)
                uxlev, _ = torch.max(S1,-1)
                lxlev = lxlev.transpose(0,1)
                uxlev = uxlev.transpose(0,1)

                LAzs = torch.zeros((self.total_image_size,1), device = device)
                UAzs = torch.zeros((self.total_image_size,1), device = device)
                LA1s = torch.zeros((self.total_image_size,1), device = device)
                UA1s = torch.zeros((self.total_image_size,1), device = device)
                lwm = getattr(self,f'S{level}p').weight.data*lxlev.transpose(0,1) + getattr(self,f'S{level}n').weight.data*uxlev.transpose(0,1)
                uwm = getattr(self,f'S{level}p').weight.data*uxlev.transpose(0,1) + getattr(self,f'S{level}n').weight.data*lxlev.transpose(0,1)
                for i in range(getattr(self,f'S{level}').weight.data.shape[0]):
                    LAz_prev, UAz_prev, LA1_prev, UA1_prev = self.L_hessian_recursive(l,u,lwm[i].unsqueeze(-1),uwm[i].unsqueeze(-1),z,level-1,device)
                    LAzs += LAz_prev
                    UAzs += UAz_prev
                    LA1s += LA1_prev
                    UA1s += UA1_prev
                return LAz + LAzs, UAz + UAzs, LA1 + LA1s, UA1 + UA1s

    def evaluate_L_hessian(self,tc,ac,l,u,z,device):
        '''
        Multiplies by C INSIDE
        '''
        C = (self.C.weight.data[tc] - self.C.weight.data[ac]).unsqueeze(-1)
        Lz,Uz,ld,ud = self.L_hessian_recursive(l,u,C,C,z,self.n_degree, device = device)
        return ((Lz+Uz)/2 + ((ld-ud)/2)*z.transpose(0,1)).transpose(0,1)

    def dom_eig_L(self,tc,ac,l,u,device,debug = False):
        if debug:
            print('Computing min_eig:')
        with torch.no_grad():
            z = torch.rand((1,self.total_image_size), device = device)
            z = z/(torch.sqrt((z*z).sum()))
            prev_z = torch.ones(self.total_image_size, device = device)
            while not torch.sum(torch.abs(z-prev_z)) < 1e-3:
                '''
                Each iteration we do two steps because if the dominant eigenvalue is negative,
                with a single step prev_z = -z and never converges.
                '''
                prev_z = z.clone()
                z = self.evaluate_L_hessian(tc,ac,l,u,z,device)
                min_eig = torch.sqrt((z*z).sum())
                #print(z.shape,min_eig.shape)
                z = z/min_eig
                z = self.evaluate_L_hessian(tc,ac,l,u,z,device)
                min_eig = torch.sqrt((z*z).sum())
                z = z/min_eig
                if debug:
                    print(min_eig,torch.sum(torch.abs(z-prev_z)))
            return min_eig

    '''
    vector of alphas
    '''
    def lowerbound_diag(self,tc,ac,l,u,device):
        lx = self.A1p(l) + self.A1n(u)
        ux = self.A1p(u) + self.A1n(l)

        lg = self.A1.weight.data.transpose(0,1)
        ug = self.A1.weight.data.transpose(0,1)

        Lhp = torch.zeros(lg.shape,device = device)
        Uhp = torch.zeros(lg.shape,device = device)
        Lhn = torch.zeros(lg.shape,device = device)
        Uhn = torch.zeros(lg.shape,device = device)

        for i in range(2,self.n_degree+1):
            lxhat = getattr(self, 'A{}p'.format(i))(l) + getattr(self, 'A{}n'.format(i))(u)
            uxhat = getattr(self, 'A{}p'.format(i))(u) + getattr(self, 'A{}n'.format(i))(l)

            lxx = getattr(self, 'S{}p'.format(i))(lx) + getattr(self, 'S{}n'.format(i))(ux) + getattr(self, 'S{}'.format(i)).bias.data
            uxx = getattr(self, 'S{}p'.format(i))(ux) + getattr(self, 'S{}n'.format(i))(lx) + getattr(self, 'S{}'.format(i)).bias.data

            S = torch.cat(((lxhat*lxx).unsqueeze(-1),(lxhat*uxx).unsqueeze(-1),(uxhat*lxx).unsqueeze(-1),(uxhat*uxx).unsqueeze(-1)), dim = -1)
            lx,_ = torch.min(S,dim = -1)
            ux,_ = torch.max(S,dim = -1)

            lgg = getattr(self, 'S{}p'.format(i))(lg) + getattr(self, 'S{}n'.format(i))(ug)
            ugg = getattr(self, 'S{}p'.format(i))(ug) + getattr(self, 'S{}n'.format(i))(lg)

            Lhp_ = getattr(self, 'S{}p'.format(i))(Lhp)*lxhat*(lxhat>0) + getattr(self, 'S{}n'.format(i))(Uhn)*uxhat*(uxhat<0)
            Lhn_ = getattr(self, 'S{}p'.format(i))(Lhn)*uxhat*(uxhat>0) + getattr(self, 'S{}n'.format(i))(Uhp)*lxhat*(lxhat<0)
            Uhp_ = getattr(self, 'S{}p'.format(i))(Uhp)*uxhat*(uxhat>0) + getattr(self, 'S{}n'.format(i))(Lhn)*lxhat*(lxhat<0)
            Uhn_ = getattr(self, 'S{}p'.format(i))(Uhn)*lxhat*(lxhat>0) + getattr(self, 'S{}n'.format(i))(Lhp)*uxhat*(uxhat<0)

            Lhp = 2*(lgg*(lgg>0)*getattr(self, 'A{}p'.format(i)).weight.data.transpose(0,1) + ugg*(ugg<0)*getattr(self, 'A{}n'.format(i)).weight.data.transpose(0,1)) + Lhp_
            Lhn = 2*(lgg*(lgg<0)*getattr(self, 'A{}p'.format(i)).weight.data.transpose(0,1) + ugg*(ugg>0)*getattr(self, 'A{}n'.format(i)).weight.data.transpose(0,1)) + Lhn_
            Uhp = 2*(ugg*(ugg>0)*getattr(self, 'A{}p'.format(i)).weight.data.transpose(0,1) + lgg*(lgg<0)*getattr(self, 'A{}n'.format(i)).weight.data.transpose(0,1)) + Uhp_
            Uhn = 2*(ugg*(ugg<0)*getattr(self, 'A{}p'.format(i)).weight.data.transpose(0,1) + lgg*(lgg>0)*getattr(self, 'A{}n'.format(i)).weight.data.transpose(0,1)) + Uhn_

            S = torch.cat(((lxhat*lgg).unsqueeze(-1),(lxhat*ugg).unsqueeze(-1),(uxhat*lgg).unsqueeze(-1),(uxhat*ugg).unsqueeze(-1)), dim = -1)
            lg,_ = torch.min(S,dim = -1)
            ug,_ = torch.max(S,dim = -1)

            lg += lxx*getattr(self, 'A{}p'.format(i)).weight.data.transpose(0,1) + uxx*getattr(self, 'A{}n'.format(i)).weight.data.transpose(0,1)
            ug += uxx*getattr(self, 'A{}p'.format(i)).weight.data.transpose(0,1) + lxx*getattr(self, 'A{}n'.format(i)).weight.data.transpose(0,1)

        C = self.C.weight.data[tc,:] - self.C.weight.data[ac,:]
        Cp = C*(C>0)
        Cn = C*(C<0)
        return torch.matmul(Lhp + Lhn,Cp) + torch.matmul(Uhp + Uhn,Cn)

    def sum_abs_out_diag(self,tc,ac,l,u,v,device):
        lx = self.A1p(l) + self.A1n(u)
        ux = self.A1p(u) + self.A1n(l)

        lg = self.A1.weight.data.transpose(0,1)
        ug = self.A1.weight.data.transpose(0,1)

        result = torch.zeros(lg.shape,device = device)

        for i in range(2,self.n_degree+1):
            lxhat = getattr(self, 'A{}p'.format(i))(l) + getattr(self, 'A{}n'.format(i))(u)
            uxhat = getattr(self, 'A{}p'.format(i))(u) + getattr(self, 'A{}n'.format(i))(l)

            lxx = getattr(self, 'S{}p'.format(i))(lx) + getattr(self, 'S{}n'.format(i))(ux) + getattr(self, 'S{}'.format(i)).bias.data
            uxx = getattr(self, 'S{}p'.format(i))(ux) + getattr(self, 'S{}n'.format(i))(lx) + getattr(self, 'S{}'.format(i)).bias.data

            S = torch.cat(((lxhat*lxx).unsqueeze(-1),(lxhat*uxx).unsqueeze(-1),(uxhat*lxx).unsqueeze(-1),(uxhat*uxx).unsqueeze(-1)), dim = -1)
            lx,_ = torch.min(S,dim = -1)
            ux,_ = torch.max(S,dim = -1)

            lgg = getattr(self, 'S{}p'.format(i))(lg) + getattr(self, 'S{}n'.format(i))(ug)
            ugg = getattr(self, 'S{}p'.format(i))(ug) + getattr(self, 'S{}n'.format(i))(lg)

            maxnorm_g = torch.max(torch.abs(lgg), torch.abs(ugg))
            maxnorm_w = torch.abs(getattr(self, 'A{}'.format(i)).weight.data.transpose(0,1))
            result_ = []
            for j in range(maxnorm_g.shape[0]):
                v_aux = v.clone()
                v_aux/=v[0,j]
                v_aux[0,j] = 0
                result_.append((maxnorm_g*torch.matmul(v_aux,maxnorm_w))[j,:].unsqueeze(0) + (maxnorm_w*torch.matmul(v_aux,maxnorm_g))[j,:].unsqueeze(0))

            result_ = torch.cat(result_,dim = 0)

            result = torch.max(torch.abs(lxhat),torch.abs(uxhat))*torch.matmul(result, torch.abs(getattr(self, 'S{}'.format(i)).weight.data.transpose(0,1))) + result_

            S = torch.cat(((lxhat*lgg).unsqueeze(-1),(lxhat*ugg).unsqueeze(-1),(uxhat*lgg).unsqueeze(-1),(uxhat*ugg).unsqueeze(-1)), dim = -1)
            lg,_ = torch.min(S,dim = -1)
            ug,_ = torch.max(S,dim = -1)

            lg += lxx*getattr(self, 'A{}p'.format(i)).weight.data.transpose(0,1) + uxx*getattr(self, 'A{}n'.format(i)).weight.data.transpose(0,1)
            ug += uxx*getattr(self, 'A{}p'.format(i)).weight.data.transpose(0,1) + lxx*getattr(self, 'A{}n'.format(i)).weight.data.transpose(0,1)


        C = self.C.weight.data[tc,:] - self.C.weight.data[ac,:]
        return torch.matmul(result,torch.abs(C))

    def alphas(self,tc,ac,l,u,device,v = None):
        with torch.no_grad():
            if v is None:
                v = torch.ones(l.shape,device = device)
            ldiag = self.lowerbound_diag(tc,ac,l,u,device)
            sum_abs = self.sum_abs_out_diag(tc,ac,l,u,v,device)
            return -(ldiag-sum_abs)/2

    def compute_min_eig(self, tc, ac, l, u, device, method = 'L', debug = False):
        if self.n_degree == 2:
            '''
            Simply do power method because hessian is constant
            '''
            with torch.no_grad():
                z = torch.ones((1,self.total_image_size), device = device)
                z = z/(torch.sqrt((z*z).sum()))
                prev_z = torch.ones(self.total_image_size, device = device)
                while not torch.sum(torch.abs(z-prev_z)) < 1e-5:
                    '''
                    Each iteration we do two steps because if the dominant eigenvalue is negative,
                    with a single step prev_z = -z and never converges.
                    '''
                    prev_z = z.clone()
                    z = torch.matmul(self.C.weight.data[tc] - self.C.weight.data[ac], (torch.matmul(self.A1.weight.data.transpose(0,1),self.S2.weight.data)*self.A2(z) + self.A2.weight.data.transpose(0,1)*torch.matmul(z,torch.matmul(self.A1.weight.data.transpose(0,1),self.S2.weight.data))).transpose(0,1))
                    self.min_eig = (torch.sqrt((z*z).sum()))
                    z = z/self.min_eig
                    z = torch.matmul(self.C.weight.data[tc] - self.C.weight.data[ac], (torch.matmul(self.A1.weight.data.transpose(0,1),self.S2.weight.data)*self.A2(z) + self.A2.weight.data.transpose(0,1)*torch.matmul(z,torch.matmul(self.A1.weight.data.transpose(0,1),self.S2.weight.data))).transpose(0,1))
                    self.min_eig = (torch.sqrt((z*z).sum()))
                    z = z/self.min_eig
                    if debug:
                        print(self.min_eig)
                self.min_eig = min(0,-0.5*float(self.min_eig))
        else:
            if method == 'L':
                self.min_eig = min(0,-0.5*float(self.dom_eig_L(tc,ac,l,u,device,debug = debug)))
            elif method == 'Weyl':
                self.min_eig = min(0,-0.5*float(self.dom_eig_weyl(tc,ac,l,u,device)))
            elif method == 'alphas':
                alphas = -self.alphas(tc,ac,l,u,device,v = u-l+0.0001)
                self.min_eig = alphas*(alphas<0)

    def hessian_vector_recursive(self, z0, z, level, device):
        '''
        evaluate at z0 multiply by z
        '''
        with torch.no_grad():
            if level == 1:
                return torch.zeros((self.total_image_size, self.hidden_size),device = device)
            if level == 2:
                return torch.matmul(self.A1.weight.data.transpose(0,1),self.S2.weight.data.transpose(0,1))*self.A2(z) + self.A2.weight.data.transpose(0,1)*torch.matmul(z,torch.matmul(self.A1.weight.data.transpose(0,1),self.S2.weight.data.transpose(0,1)))
            else:
                grad_prev = self.gradient_intermediate(z0,level-1)
                #print(grad_prev.shape, getattr(self, f'S{level}').weight.data.shape, z.shape, getattr(self, f'A{level}')(z).shape)
                #print(torch.matmul(z, torch.matmul(grad_prev,getattr(self, f'S{level}').weight.data.transpose(0,1))).shape)
                aux1 = torch.matmul(grad_prev,getattr(self, f'S{level}').weight.data.transpose(0,1))*getattr(self, f'A{level}')(z) + getattr(self, f'A{level}').weight.data.transpose(0,1)*torch.matmul(z, torch.matmul(grad_prev,getattr(self, f'S{level}').weight.data.transpose(0,1)))
                #print((getattr(self, f'U{level}')(z) + 1).shape)
                #print((self.C.weight.data[tc] - self.C.weight.data[ac]).shape)
                return aux1 + (getattr(self, f'A{level}')(z0))*torch.matmul(self.hessian_vector_recursive(z0, z, level-1, device),getattr(self, f'S{level}').weight.data.transpose(0,1))

    def eval_hessian_vector(self, tc, ac, z0, z,device):
        with torch.no_grad():
            return torch.matmul(self.hessian_vector_recursive(z0, z, self.n_degree, device), self.C.weight.data[tc] - self.C.weight.data[ac])

    def min_eig_hessian(self, tc, ac, z0, device, debug = False):
        '''
        Returns the dominant eigenvalue of the hessian matrix evaluated at z0
        '''
        min_eig = None
        signed_eig = None
        with torch.no_grad():
            z = torch.rand(z0.shape[1], device = device)
            z = z/(torch.sqrt((z*z).sum()))
            prev_z = torch.ones(z0.shape[1], device = device)
            while not torch.sum(torch.abs(z-prev_z)) < 1e-4:
                prev_z = z.clone()
                z = self.eval_hessian_vector( tc, ac, z0, z, device)
                min_eig = (torch.sqrt((z*z).sum()))
                signed_eig = (z/prev_z)[0]
                z = z/min_eig
                z = self.eval_hessian_vector(tc, ac, z0, z, device)
                min_eig = (torch.sqrt((z*z).sum()))
                z = z/min_eig
                if debug:
                    print(signed_eig, torch.sum(torch.abs(z-prev_z)))
            return signed_eig, z

'''
CCP_conv
--------------------------------------------------------------------------------
'''

class CCP_Conv(nn.Module):
    def __init__(self, n_channels, n_degree=4, kernel_size=7, stride = 1, bias=False, BN = False, downsample_degs=[2, 3], use_preconv=True, use_w = False, n_classes=10, channels_in=1, image_size=28):
        super(CCP_Conv, self).__init__()
        self.n_channels = n_channels
        self.hidden_size = n_channels*image_size
        self.n_classes = n_classes
        self.image_size = image_size
        self.total_image_size = self.image_size * self.image_size * channels_in
        self.before_c_size = int((image_size+2-(kernel_size%2))//stride)
        self.total_size = self.before_c_size * self.before_c_size * n_channels
        self.n_degree = n_degree
        self.downsample_degs = downsample_degs
        self.n_downs = 0
        self.bn = BN
        self.channels_in = channels_in
        self.use_w = use_w
        if self.use_w:
            self.w = nn.Parameter(torch.zeros(1))
        self.use_preconv = use_preconv
        padding = int(kernel_size // 2)
        st_channels = channels_in
        if self.use_preconv:
            self.conv0 = nn.Conv2d(channels_in, self.n_channels, kernel_size=3, stride=1, padding=1, bias=False)
            st_channels = self.n_channels

        for i in range(1, self.n_degree + 1):
            setattr(self, 'conv{}'.format(i), nn.Conv2d(st_channels, self.n_channels, kernel_size=kernel_size,
                                                        stride=stride, padding=padding, bias=bias))
            if BN and i > 1:
                setattr(self, f'BN{i}', nn.BatchNorm2d(self.n_channels,image_size))
        self.down_sample = torch.nn.AvgPool2d(2, 2)
        self.C = torch.nn.Linear(self.total_size, self.n_classes, bias=True)

    def forward(self, z):
        if self.use_preconv:
            z = self.conv0(z)
        out = self.conv1(z)
        # # This is a flag indicating how many times the representations need to be down-sampled.
        self.n_downs = 0
        for i in range(2, self.n_degree + 1):
            temp = getattr(self, 'conv{}'.format(i))(z)
            if self.n_downs > 0:
                # # down-sample appropriately before the Hadamard.
                for k in range(self.n_downs):
                    temp = self.down_sample(temp)
            if self.bn:
                temp = getattr(self, 'BN{}'.format(i))(temp)
            if self.use_w and i > 1:
                out = temp * out * self.w + out
            else:
                out = temp * out + out
            #print('heyyyyy',out.view(-1,self.total_image_size*self.n_channels)[0,:10])
            if i in self.downsample_degs:
                out = self.down_sample(out)
                self.n_downs += 1
        out = out.view(-1, self.total_size)
        out = self.C(out)
        return out

    def output_intermediate(self,z,level):
        if self.use_preconv:
            z = self.conv0(z)
        out = self.conv1(z)
        # # This is a flag indicating how many times the representations need to be down-sampled.
        self.n_downs = 0
        for i in range(2, level + 1):
            temp = getattr(self, 'conv{}'.format(i))(z)
            if self.n_downs > 0:
                # # down-sample appropriately before the Hadamard.
                for k in range(self.n_downs):
                    temp = self.down_sample(temp)
            out = temp * out + out
            if self.bn:
                out = getattr(self, 'BN{}'.format(i))(out)
            #print('heyyyyy',out.view(-1,self.total_image_size*self.n_channels)[0,:10])
            if i in self.downsample_degs:
                out = self.down_sample(out)
                self.n_downs += 1
        return out


'''
NCP_conv
--------------------------------------------------------------------------------
'''
def get_norm(norm_local):
    """ Define the appropriate function for normalization. """
    if norm_local is None or norm_local == 0:
        norm_local = nn.BatchNorm2d
    elif norm_local == 1:
        norm_local = nn.InstanceNorm2d
    elif isinstance(norm_local, int) and norm_local < 0:
        norm_local = lambda a: lambda x: x
    return norm_local

class NCP_Conv(nn.Module):
    def __init__(self, in_planes, planes, stride = 1, use_alpha=True, kernel_sz=3,
                 norm=True, kernel_size_S=1, n_classes = 10, image_size = 28, n_degree = 2, **kwargs):
        """ This class implements a single second degree NCP model. """
        super(NCP_Conv, self).__init__()
        self.norm = norm
        self.n_classes = n_classes
        self.n_degree = n_degree
        self.hidden_size = planes
        self.use_alpha = use_alpha

        pad1 = kernel_sz // 2
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_sz, stride=stride, padding=pad1, bias=False)
        if norm:
            self.bn1 = nn.BatchNorm2d(planes)

        for i in range(1,n_degree):
            pad = kernel_size_S // 2
            setattr(self,f'conv_{i+1}', nn.Conv2d(in_planes, planes, kernel_size=kernel_size_S, stride=stride, padding=pad, bias=False))
            if norm:
                setattr(self,f'bn{i+1}', nn.BatchNorm2d(planes))
            if self.use_alpha:
                setattr(self,f'alpha{i+1}', nn.Parameter(torch.zeros(1)))
        self.C = torch.nn.Linear(planes*image_size*image_size, self.n_classes, bias=True)

    def forward(self, z):
        out = self.conv1(z)
        if self.norm:
            out = self.bn1(out)
        for i in range(1,self.n_degree):
            out1 = getattr(self,f'conv_{i+1}')(z)
            if self.norm:
                out1 = getattr(self,f'bn{i+1}')(out1)
            if self.use_alpha:
                out = out1*(1 + getattr(self,f'alpha{i+1}')*out)
            else:
                out = out1*(1 + out)
        #print(out.shape, self.C.weight.data.shape)
        return self.C(out.reshape(out.shape[0],-1))

    def output_intermediate(self,z,level):
        out = self.conv1(z)
        if self.norm:
            out = self.bn1(out)
        for i in range(1,self.n_degree):
            out1 = getattr(self,f'conv_{i+1}')(z)
            if self.norm:
                out1 = getattr(self,f'bn{i+1}')(out1)
            if self.use_alpha:
                out = out1*(1 + getattr(self,f'alpha{i+1}')*out)
            else:
                out = out1*(1 + out)
        #print(out.shape, self.C.weight.data.shape)
        return out

'''
Prod-Poly
--------------------------------------------------------------------------------
'''
class Prod_Poly(nn.Module):
    def __init__(self, hidden_sizes, degree_list, stride_list, channel_list, use_w = False, kernel_size = 5, BN = False ,architecture = 'NCP', image_size=28, channels_in=1, bias=False, n_classes=10):
        super(Prod_Poly, self).__init__()
        print('Initializing the NCP model with degree={}.'.format(degree_list))
        self.degree_list = degree_list
        self.stride_list = stride_list
        self.channel_list = channel_list
        self.image_size = image_size
        self.channels_in = channels_in
        self.total_image_size = self.image_size * self.image_size * channels_in
        self.hidden_sizes = hidden_sizes
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.architecture = architecture
        self.bias = bias
        self.BN = BN
        self.use_w = use_w

        new_image_size = image_size
        for i,(degree,s,c) in enumerate(zip(degree_list,stride_list,channel_list)):
            if i == 0:
                if architecture == 'CCP':
                    setattr(self, f'Poly{i+1}', CCP(hidden_sizes[0], image_size=image_size, channels_in=channels_in, n_degree=degree, bias=self.bias, n_classes=self.n_classes))
                elif architecture == 'NCP':
                    setattr(self, f'Poly{i+1}', NCP(new_image_size*new_image_size*c, image_size=image_size, channels_in=channels_in, n_degree=degree, bias=self.bias, n_classes=self.n_classes))
                elif architecture == 'CCP_Conv':
                    setattr(self, f'Poly{i+1}', CCP_Conv(n_channels = c, stride = s,use_preconv = True, use_w = self.use_w, downsample_degs = [], n_degree=degree, BN = self.BN, kernel_size=self.kernel_size, n_classes=n_classes, channels_in=channels_in, image_size=image_size))
                    if s > 1:
                        new_image_size = int((new_image_size+2-(kernel_size%2))//s)
                        print(new_image_size)
                elif architecture == 'NCP_Conv':
                    if s > 1:
                        new_image_size = int((new_image_size + 1)//s)
                        print(new_image_size)
                    self.conv0 = nn.Conv2d(channels_in, c, kernel_size=3, stride=1, padding=1, bias=False)
                    if self.BN:
                        self.bn0 = nn.BatchNorm2d(c)
                    setattr(self, f'Poly{i+1}', NCP_Conv(c, c, n_degree = degree, stride=s, use_alpha=True, kernel_sz=self.kernel_size,norm = self.BN, kernel_size_S=1, n_classes = n_classes, image_size = new_image_size))
            else:
                if architecture == 'CCP':
                    setattr(self, f'Poly{i+1}', CCP(hidden_sizes[i], image_size=1, channels_in=getattr(getattr(self, f'Poly{i}'), f'U{self.degree_list[i-1]}').weight.data.shape[0], n_degree=degree, bias=self.bias, n_classes=self.n_classes))
                elif architecture == 'NCP':
                    setattr(self, f'Poly{i+1}', NCP(new_image_size*new_image_size*c, image_size=1, channels_in=getattr(getattr(self, f'Poly{i}'), f'A{self.degree_list[i-1]}').weight.data.shape[0], n_degree=degree, bias=self.bias, n_classes=self.n_classes))
                elif architecture == 'CCP_Conv':
                    #new_image_size = int(image_size // 2 ** (2 if i == 1 else 3))
                    setattr(self, f'Poly{i+1}', CCP_Conv(n_channels = c, stride=s,use_preconv = False, use_w = self.use_w, downsample_degs = [], n_degree=degree, BN = self.BN, kernel_size=self.kernel_size, n_classes=n_classes, channels_in=self.channel_list[i-1], image_size=new_image_size))
                    if s > 1:
                        new_image_size = int((new_image_size+2-(kernel_size%2))//s)
                        print(new_image_size)
                elif architecture == 'NCP_Conv':
                    if s > 1:
                        new_image_size = int((new_image_size + 1)//s)
                        print(new_image_size)
                    setattr(self, f'Poly{i+1}', NCP_Conv(channel_list[i-1], c, n_degree = degree, strides=s, use_alpha=True, kernel_sz=self.kernel_size,norm = self.BN, kernel_size_S=1, n_classes = n_classes, image_size = new_image_size))

                if BN and architecture != 'CCP_Conv' and architecture != 'NCP_Conv':
                    setattr(self, f'BN{i+1}', nn.BatchNorm1d(hidden_size))

    def forward(self, z):
        if self.architecture == 'NCP_Conv':
            h=self.conv0(z)
            if self.BN:
                h = self.bn0(h)
        else: h = z
        for i, deg in enumerate(self.degree_list):
            if i != len(self.degree_list)-1:
                h = getattr(self,f'Poly{i+1}').output_intermediate(h,deg)
                if self.BN and self.architecture != 'CCP_Conv' and self.architecture != 'NCP_Conv':
                    h = getattr(self,f'BN{i+1}')(h)
            else:
                h = getattr(self,f'Poly{i+1}')(h)
        return h

    def verif_objective(self, z, tc, ac):
        out = self.forward(z)
        return out[:,tc] - out[:,ac]

    def compute_signs(self,device):
        for i,deg in enumerate(self.degree_list):
            getattr(self, f'Poly{i+1}').compute_signs(device)

    def forward_bounds(self,l,u):
        for i,deg in enumerate(self.degree_list):
            if i != len(self.degree_list)-1:
                l,u = getattr(self,f'Poly{i+1}').forward_bounds_intermediate(l,u,deg)
            else:
                l,u = getattr(self,f'Poly{i+1}').forward_bounds(l,u)
        return l, u

    def forward_bounds_intermediate(self,l,u,level):
        for i,deg in enumerate(self.degree_list):
            if i < level:
                l,u = getattr(self,f'Poly{i+1}').forward_bounds_intermediate(l,u,deg)
        return l, u

    def output_intermediate(self,z,level):
        for i,deg in enumerate(self.degree_list):
            if i < level:
                z = getattr(self,f'Poly{i+1}').output_intermediate(z,deg)
        return z

    def gradient_intermediate(self,z,level_poly, level_in):
        with torch.no_grad():
            if level_poly == 1:
                return self.Poly1.gradient_intermediate(z,level_in)
            else:
                x = self.output_intermediate(z,level_poly-1)
                gl = getattr(self,f'Poly{level_poly}').gradient_intermediate(x,level_in)
                gl_z = self.gradient_intermediate(z,level_poly-1, self.degree_list[level_poly-2])
                return torch.matmul(gl_z,gl)

    def bounds_intermediate_grad(self,l,u,level_poly, level_in):
        with torch.no_grad():
            if level_poly == 1:
                return self.Poly1.bounds_intermediate_grad(l,u,level_in)
            else:
                l_prev, u_prev = self.forward_bounds_intermediate(l,u,level_poly-1)
                lg_prev, ug_prev = getattr(self, f'Poly{level_poly}').bounds_intermediate_grad(l_prev,u_prev,level_in)
                lg_prev_z, ug_prev_z = self.bounds_intermediate_grad(l,u,level_poly-1, self.degree_list[level_poly-2])

                ls = []
                us = []
                for i in range(lg_prev.shape[1]):
                    S = torch.cat(((lg_prev_z*lg_prev[:,i]).unsqueeze(-1), (lg_prev_z*ug_prev[:,i]).unsqueeze(-1), (ug_prev_z*lg_prev[:,i]).unsqueeze(-1), (ug_prev_z*ug_prev[:,i]).unsqueeze(-1)),dim = -1)
                    l_aux, _ = torch.min(S,-1)
                    u_aux, _ = torch.max(S,-1)
                    ls.append(l_aux.clone().unsqueeze(-1))
                    us.append(u_aux.clone().unsqueeze(-1))
                ls = torch.cat(ls, dim = -1)
                us = torch.cat(us, dim = -1)
                return torch.sum(ls, dim = 1), torch.sum(us, dim = 1)

    def hessian_vector_recursive(self, z0, z, level, device):
        '''
        evaluate at z0 multiply by z
        '''

        '''
        x -> y -> z
        Hz(x) = JT Hz(y) J + sum dz/dy * Hy(x)
        '''
        with torch.no_grad():
            if level == 1:
                return self.Poly1.hessian_vector_recursive(z0,z,self.degree_list[0], device)
            else:
                output_prev_block = self.output_intermediate(z0,level-1)

                grad_prev_block = getattr(self,f'Poly{level}').gradient_intermediate(output_prev_block,self.degree_list[level-1])
                right = torch.matmul(self.hessian_vector_recursive(z0, z, level-1, device), grad_prev_block)

                grad_z = self.gradient_intermediate(z0, level-1, self.degree_list[level-2])
                aux = torch.matmul(z,grad_z)
                hessian_level_level_prev_z =  getattr(self,f'Poly{level}').hessian_vector_recursive(output_prev_block, aux, self.degree_list[level-1], device)

                return torch.matmul(grad_z, hessian_level_level_prev_z) + right

    def eval_hessian_vector(self, tc, ac, z0, z, device):
        last_pol = len(self.degree_list)
        with torch.no_grad():
            return torch.matmul(self.hessian_vector_recursive(z0, z, len(self.degree_list), device), getattr(self,f'Poly{last_pol}').C.weight.data[tc] - getattr(self,f'Poly{last_pol}').C.weight.data[ac]).unsqueeze(1)

    def L_hessian_recursive(self,z,l,u,lw,uw,device,level):
        if level == 1:
            Lz,Uz,ld,ud = self.Poly1.L_hessian_recursive(l,u,lw,uw,z,self.degree_list[0], device = device)
            #print(Lz.shape)

            #return Lz,Uz,ld,ud
            return (torch.sum(Lz+Uz, dim = -1)/2 + (torch.sum(ld-ud, dim = -1)/2)*z.squeeze()).unsqueeze(0)
        else:
            l_prev, u_prev = self.forward_bounds_intermediate(l,u,level-1)
            lg_prev, ug_prev = getattr(self, f'Poly{level}').bounds_intermediate_grad(l_prev,u_prev,self.degree_list[level-1])
            #print(lg_prev.shape)
            S = torch.cat(((lg_prev*lw.transpose(0,1)).unsqueeze(-1),(lg_prev*uw.transpose(0,1)).unsqueeze(-1),(ug_prev*lw.transpose(0,1)).unsqueeze(-1),(ug_prev*uw.transpose(0,1)).unsqueeze(-1)), dim = -1)
            lw_,_ = torch.min(S,dim = -1)
            uw_,_ = torch.max(S,dim = -1)
            lw__ = torch.sum(lw_,dim = -1, keepdim = True)
            uw__ = torch.sum(uw_,dim = -1, keepdim = True)
            #Lz,Uz,ld,ud = self.L_hessian_recursive(z,l,u,lw__,uw__,device,level-1)
            left = self.L_hessian_recursive(z,l,u,lw__,uw__,device,level-1)

            lJ, uJ = self.bounds_intermediate_grad(l,u,level-1,self.degree_list[level-2])
            J_maxnorm = torch.max(torch.abs(lJ), torch.abs(uJ))
            v = torch.matmul(z,J_maxnorm)

            Lv,Uv,ldv,udv = getattr(self, f'Poly{level}').L_hessian_recursive(l_prev,u_prev,lw,uw,v,self.degree_list[level-1], device = device)
            right = torch.sum(Lv+Uv, dim = -1)/2 + (torch.sum(ldv-udv, dim = -1)/2)*v.squeeze()

            return left + torch.matmul(J_maxnorm,right).unsqueeze(0)
            #print(J_maxnorm.shape, Lz.shape, Lz.shape, ld.shape, ldv.shape)
            #return Lz + torch.matmul(J_maxnorm,Lv),Uz + torch.matmul(J_maxnorm,Uv),ld + torch.matmul(J_maxnorm,ldv),ud + torch.matmul(J_maxnorm,udv)

    def evaluate_L_hessian(self,tc,ac,l,u,z,device):
        with torch.no_grad():
            last_poly = len(self.degree_list)
            C = (getattr(self, f'Poly{last_poly}').C.weight.data[tc,:] - getattr(self, f'Poly{last_poly}').C.weight.data[ac,:]).unsqueeze(-1)
            return self.L_hessian_recursive(z,l,u,C,C,device,last_poly)

    def dom_eig_L(self,tc,ac,l,u,device,debug = False):
        if debug:
            print('Computing min_eig:')
        with torch.no_grad():
            z = torch.rand((1,self.total_image_size), device = device)
            z = z/(torch.sqrt((z*z).sum()))
            prev_z = torch.ones(self.total_image_size, device = device)
            while not torch.sum(torch.abs(z-prev_z)) < 1e-5:
                '''
                Each iteration we do two steps because if the dominant eigenvalue is negative,
                with a single step prev_z = -z and never converges.
                '''
                prev_z = z.clone()
                z = self.evaluate_L_hessian(tc,ac,l,u,z,device)
                min_eig = torch.sqrt((z*z).sum())
                z = z/min_eig
                z = self.evaluate_L_hessian(tc,ac,l,u,z,device)
                min_eig = torch.sqrt((z*z).sum())
                z = z/min_eig
                if debug:
                    print(min_eig,torch.sum(torch.abs(z-prev_z)))
            return min_eig

    def compute_min_eig(self, tc, ac, l, u, device, method = 'L', debug = False):
        if method == 'L':
            self.min_eig = min(0,-0.5*float(self.dom_eig_L(tc,ac,l,u,device,debug = debug)))
        elif method == 'alphas':
            alphas = -self.Poly1.alphas(tc,ac,l,u,device,v = u-l+0.0001)
            self.min_eig = alphas*(alphas<0)

    def bounds_diag_hess(self,tc,ac,l,u,device):
        if len(self.degree_list) == 1:
            return self.Poly1.bounds_diag_hess(tc,ac,l,u,device)
'''
Common stuff
--------------------------------------------------------------------------------
'''

def from_CCP_Conv_to_CCP(net, device):
    new_net = CCP(net.total_size, image_size=net.image_size, channels_in = net.channels_in, n_classes=net.n_classes, n_degree = net.n_degree)
    new_net.to(device)
    for i in range(1, new_net.n_degree + 1):
        with torch.no_grad():
            U4 = []
            for l in range(net.channels_in):
                for j in range(net.image_size):
                    for k in range(net.image_size):
                        U2 = torch.zeros((1,net.channels_in, net.image_size, net.image_size), device = device)
                        U2[:,l,j,k] = 1
                        if net.use_preconv:
                            U3 = getattr(net, f'conv{i}')(net.conv0(U2))
                        else:
                            U3 = getattr(net, f'conv{i}')(U2)
                        if net.bn:
                            U3 = getattr(net, f'BN{i}')(U3)
                        U4.append(U3.view(-1,1).clone())
            U4 = torch.cat(U4,dim=-1)
            getattr(new_net, 'U{}'.format(i)).weight.copy_(U4.view(-1,new_net.total_image_size))
    new_net.C.weight.data.copy_(net.C.weight.data)
    new_net.C.bias.data.copy_(net.C.bias.data)
    return new_net

def power_method(u,v,device):
    '''
    Computes dominant eigenvalue of matrix defined by (uvT + vuT)
    '''
    min_eig = None
    signed_eig = None
    with torch.no_grad():
        z = torch.ones(u.shape[0], device = device)
        z = z/(torch.sqrt((z*z).sum()))
        prev_z = torch.ones(u.shape[0], device = device)
        while not torch.sum(torch.abs(z-prev_z)) < 1e-3:
            prev_z = z.clone()
            z = (u*torch.dot(v,z) + v*torch.dot(u,z))
            min_eig = (torch.sqrt((z*z).sum()))
            signed_eig = (z/prev_z)[0]
            z = z/min_eig
            z = (u*torch.dot(v,z) + v*torch.dot(u,z))
            min_eig = (torch.sqrt((z*z).sum()))
            z = z/min_eig
        return signed_eig, z

def power_method_matrix(m,device):
    '''
    Computes dominant eigenvalue of matrix m
    '''
    min_eig = None
    signed_eig = None
    with torch.no_grad():
        z = torch.ones(m.shape[0], device = device)
        z = z/(torch.sqrt((z*z).sum()))
        prev_z = torch.ones(m.shape[0], device = device)
        while not torch.sum(torch.abs(z-prev_z)) < 1e-3:
            prev_z = z.clone()
            z = torch.matmul(m,z)
            min_eig = (torch.sqrt((z*z).sum()))
            signed_eig = (z/prev_z)[0]
            z = z/min_eig
            z = torch.matmul(m,z)
            min_eig = (torch.sqrt((z*z).sum()))
            z = z/min_eig
        return signed_eig, z

def compute_Qq(U1,U2,C,tc,ac):
    k = U1.shape[0]
    d = U1.shape[1]
    Q = np.zeros((d,d))
    q = np.zeros(d)
    for i in range(k):
        #Q = Q + (C[tc,i] - C[ac,i])*np.diag(U1[i,:])@np.ones((d,d))@np.diag(U1[i,:])
        Q = Q + (C[tc,i] - C[ac,i])*np.matmul(U2[i,:][np.newaxis].T,U1[i,:][np.newaxis])
        q = q + (C[tc,i] - C[ac,i])*U1[i,:]
    return Q, q

def evaluate_Q(U1,U2,C,beta,tc,ac,z):
    f = 0
    #evaluate zTQz:
    for i in range(U1.shape[0]):
        f += (C[tc,i] - C[ac,i]) * ((z @ U2[i,:]) + 1) * (U1[i,:] @ z)

        #f += (C[tc,i] - C[ac,i]) * zU1_aux
    return f + (beta[tc] - beta[ac])


'''
Training and testing
--------------------------------------------------------------------------------
'''

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train(train_loader, net, optimizer, criterion, epoch, device, beta=10, cutmix_prob=0.2,attack = 0):
    """ Perform single epoch of the training."""
    net.train()
    train_loss, correct, total = 0, 0, 0
    if attack:
        adversary = torchattacks.attacks.pgd.PGD(net)
    for idx, data_dict in enumerate(train_loader):
        img = data_dict[0]
        label = data_dict[1]
        inputs, label = img.to(device), label.to(device)
        if attack:
            inputs = adversary(inputs, label)
        optimizer.zero_grad()
        r = np.random.rand(1)
        if beta > 0 and r < cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(inputs.size()[0]).to(device)
            target_a = label
            target_b = label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            pred = net(inputs)
            loss = criterion(pred, target_a) * lam + criterion(pred, target_b) * (1. - lam)
        else:
            # compute output
            pred = net(inputs)
            loss = criterion(pred, label)
        assert not torch.isnan(loss), 'NaN loss.'
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(pred.data, 1)
        total += label.size(0)
        correct += predicted.eq(label).cpu().sum()
        if idx % 100 == 0 and idx > 0:

            acc = float(correct) / total
            m2 = ('Epoch: {}, Epoch iters: {} / {}\t'
                  'Loss: {:.04f}, Acc: {:.06f}')
            print(m2.format(epoch, idx, len(train_loader), float(train_loss), acc))
    return train_loss

def test(net, test_loader, device='cuda'):
    """ Perform testing, i.e. run net on test_loader data
        and return the accuracy. """
    net.eval()
    correct, total = 0, 0
    for (idx, data) in enumerate(test_loader):
        sys.stdout.write('\r [%d/%d]' % (idx + 1, len(test_loader)))
        sys.stdout.flush()
        img = data[0].to(device)
        label = data[1].to(device)
        with torch.no_grad():
             pred = net(img)
        _, predicted = pred.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
    return correct / total
