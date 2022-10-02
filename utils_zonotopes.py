import numpy as np
import torch
import utils_PN as PN
import argparse
import utils_BaB as UBAB

def init_zonotope_from_interval(l,u):
    w = torch.diag((u-l).squeeze()/2)
    c = (u+l).squeeze()/2
    return w,c

def abstract_linear(w,c,lin_layer):
    #print('heyyyyy',lin_layer.weight.data.shape, w.shape)
    w2 = torch.matmul(lin_layer.weight.data,w)
    c2 = lin_layer(c)
    return w2,c2

def abstract_mult_fast(w1,c1,w2,c2):
    '''
    Here the parameters can have different hidden dimension/number of epsilons
    '''
    #print(w1.shape, w2.shape)
    norm2 = torch.sum(torch.abs(w2),dim=1)
    #print(norm2.shape, w2.shape)
    result = torch.sum(torch.abs(w1.transpose(0,1)*norm2),dim = 0)
    #print('htrgerwew', result.shape, w1.shape, w2.shape)
    l = -result
    u = result
    l -= torch.sum(torch.abs(w1.transpose(0,1)*c2),0)
    l -= torch.sum(torch.abs(w2.transpose(0,1)*c1),0)
    u += torch.sum(torch.abs(w1.transpose(0,1)*c2),0)
    u += torch.sum(torch.abs(w2.transpose(0,1)*c1),0)
    l += c1*c2
    u += c1*c2
    return l,u

def abstract_mult_fast2(w1,c1,w2,c2):
    '''
    Here the parameters can have different hidden dimension/number of epsilons
    Here instead of losing the variables and making intervals, we introduce new variables
    to represent the addition of a zonotope with the interval.
    '''
    #print(w1.shape, w2.shape)
    norm2 = torch.sum(torch.abs(w2),dim=1)
    #print(norm2.shape, w2.shape)
    result = torch.sum(torch.abs(w1.transpose(0,1)*norm2),dim = 0)
    #print('htrgerwew', result.shape, w1.shape, w2.shape)
    w = torch.cat(((w1.transpose(0,1)*c2).transpose(0,1),(w2.transpose(0,1)*c1).transpose(0,1)),dim = 1)
    c = c1*c2
    #print('tyrthegdsf',result.shape, w.shape)
    ww = torch.cat((w,torch.diag(result)), dim = 1)
    return ww,c

def abstract_mult_precise(w1,c1,w2,c2):
    '''
    Here all the parameters need to have the same shapes
    '''
    h,d = w1.shape
    #equation 6 from fast and precise certification of transformers
    ll = []
    uu = []
    prod = w1*w2
    l = torch.sum((prod < 0)*prod, dim = 1)
    u = torch.sum((prod > 0)*prod, dim = 1)
    for i in range(d):
        for j in range(d):
            if j!=i:
                prod2 = torch.abs(w1[:,i]*w2[:,j])
                l-=prod2
                u+=prod2
    l -= torch.sum(torch.abs(w1.transpose(0,1)*c2),0)
    l -= torch.sum(torch.abs(w2.transpose(0,1)*c1),0)
    u += torch.sum(torch.abs(w1.transpose(0,1)*c2),0)
    u += torch.sum(torch.abs(w2.transpose(0,1)*c1),0)
    l += c1*c2
    u += c1*c2
    return l,u

def abstract_mult(w1,c1,w2,c2, precise = False):
    if precise:
        return abstract_mult_precise(w1,c1,w2,c2)
    else:
        return abstract_mult_fast(w1,c1,w2,c2)

def init_interval_from_zonotope(w,c):
    l = c - torch.sum(torch.abs(w), dim = 1)
    u = c + torch.sum(torch.abs(w), dim = 1)
    return l,u

def abstract_PN(net,l,u, precise = False):
    with torch.no_grad():
        w0,c0 = init_zonotope_from_interval(l,u)
        wn,cn = abstract_linear(w0,c0,net.U1)
        for n in range(1,net.n_degree):
            wn_,cn_ = abstract_linear(w0,c0,getattr(net,f'U{n+1}'))
            cn_ += 1
            #ll, uu = abstract_mult(wn,cn,wn_,cn_,precise = precise)
            #wn, cn = init_zonotope_from_interval(ll,uu)
            wn, cn = abstract_mult_fast2(wn,cn,wn_,cn_)
        w,c = abstract_linear(wn,cn,net.C)
        return w,c


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Complete and sound verification through branch and bound and interval propagation')
    #parser.add_argument('--weights', default='weights_01/Prod_Poly_CCP_Conv_1_0_order_[2]_hidden_[32]_cifar10_best.pt', type=str, help='Path of the PN weights')
    parser.add_argument('--weights', default='weights/Prod_Poly_CCP_0_0_order_[4]_hidden_[100]_mnist_resize_0_best.pt', type=str, help='Path of the PN weights')
    parser.add_argument('--data_root', type=str, default='./data', help='Root data folder')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Root data folder')
    parser.add_argument('--eps', default=0.1, type=float, help='Maximum allowed perturbation in l_inf norm')
    args = parser.parse_args()

    train_loader, valid_loader, test_loader, image_size, n_classes, channels_in = PN.load_db(root = args.data_root, name = args.dataset ,batch_size=64, resize = False)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    #f = open(args.weights, "r+b")
    #f.seek(0)
    net = torch.load(args.weights, map_location = device)
    net.device = device
    #net = pickle.load(f)
    #f.close()
    if args.weights.split('/')[1][:4] == 'Prod':
        if net.architecture == 'CCP_Conv':
            for i in range(1,len(net.degree_list)+1):
                print('Converting convolutions to linear layers:')
                setattr(net,f'Poly{i}' , PN.from_CCP_Conv_to_CCP(getattr(net,f'Poly{i}'), device))
    net.compute_signs(device)

    #define input set
    z, tc = test_loader.dataset[150]
    z = z.to(device)
    with torch.no_grad():
        pred = net(z)
    logit, predicted = pred.max(1)

    z_flat = z.flatten().cpu().numpy()
    l = torch.zeros((1,net.total_image_size),device = device)
    u = torch.zeros((1,net.total_image_size),device = device)
    for i in range(net.total_image_size):
        l[0,i] = max(0, z_flat[i]-args.eps)
        u[0,i] = min(1, z_flat[i]+args.eps)

    pred2 = [(-p,i) for i, p in enumerate(pred[0])]
    (p, ac) = sorted(pred2)[1]
    print(sorted(pred2))
    print(tc,ac,net.total_image_size)

    #w,c = init_zonotope_from_interval(l,u)
    #w,c = abstract_linear(w,c,net.Poly1.U1)
    #w,c = abstract_mult_fast2(w,c,w,c)
    #w,c = abstract_PN(net.Poly1,l,u,precise = False)
    #print(-torch.sum(torch.abs(w[tc] - w[ac])) + c[tc] - c[ac])
    #ll,uu = abstract_PN(net.Poly1,l,u,precise = True)
    #print(ll,uu)
    ll, uu = net.forward_bounds(l,u)
    print(ll,uu)
    print(ll[0,tc] - uu[0,ac])

    net.compute_min_eig(tc,ac,l,u,device, method = 'L', debug = True)
    L, U, _,_,_ = UBAB.get_bounds(net,tc,ac,l,u,'alpha-conv','PGD',device,debug = False, optim = 'PGD')
    print(L,U)
    L, U, _,_,_ = UBAB.get_bounds(net,tc,ac,l,u,'zonotopes','PGD',device,debug = False, optim = 'PGD')
    print(L,U)
