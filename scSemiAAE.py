#aae_pytorch_basic.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 加载必要的库
from time import time
import argparse
from datetime import datetime
from turtle import forward
import torch
import pickle
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.misc
import os
from torch.utils.data import DataLoader, TensorDataset
import  scanpy   as sc
from sklearn.decomposition import PCA
import torchvision
import pandas as pd
# import viz
# from viz import get_X_batch
# from viz import create_reconstruction
# from viz import grid_plot
# from viz import grid_plot2d
import h5py
from sklearn import metrics
from sklearn.metrics.cluster import homogeneity_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from torch.nn.parameter import Parameter
from layers import ZINBLoss, MeanAct, DispAct
from utils import cluster_acc
from sklearn.mixture import GaussianMixture  # 高斯混合模型
from sklearn.decomposition import PCA
import phate
import sklearn.manifold # t-SNE
import scprep
import umap.umap_ as umap
# Training settings
# 训练时的一些设置，若未设置则使用默认值
# 模型描述
parser = argparse.ArgumentParser(description='PyTorch unsupervised MNIST')
# batch_size
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 100)')
# 迭代次数
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)')

args = parser.parse_args()
cuda = torch.cuda.is_available()

# 在神经网络中，参数默认是进行随机初始化的。不同的初始化参数往往会导致不同的结果，当得到比较好的结果时我们通常希望这个结果是可以复现的，
# 在pytorch中，通过设置随机数种子也可以达到这么目的。
seed = 10


kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}   # GPU的一些设置
n_classes =7  # 类别数，在无监督学习中未用到
z_dim = 256
X_dim = 24341
y_dim =7
n_clusters=7
train_batch_size = args.batch_size
valid_batch_size = args.batch_size
N = 512
M=256
epochs = 100   
def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
            if filter_min_counts:
                sc.pp.filter_genes(adata, min_counts=1)
                sc.pp.filter_cells(adata, min_counts=1)
            if size_factors or normalize_input or logtrans_input:
                adata.raw = adata.copy()
            else:
                adata.raw = adata
            if size_factors:
                 sc.pp.normalize_per_cell(adata)
                 adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
            else:
                adata.obs['size_factors'] = 1.0
            if logtrans_input:
               sc.pp.log1p(adata)
            if normalize_input:
               sc.pp.scale(adata)
            return adata
        
def load_data():
    print('loading data!')
    data_mat = h5py.File(r'C:\Users\lenovo\Desktop\romanov.h5')
    X = np.array(data_mat['X'])
    X=X.T
    data = h5py.File(r'C:\Users\lenovo\Desktop\romanov.h5')
    Y = np.array(data['Y'].value.astype('int'))
    data_mat.close() 

    adata = sc.AnnData(X)
    # adata.obs['Group'] = Y
    # adata = sc.read(r'D:\单细胞\code\文献代码\ItClust-master\tutorial\data\pancreas\smartseq2.h5ad')
    # Y=pd.Series(adata.obs["celltype"],dtype="category")
    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    # sc.pp.highly_variable_genes(adata, n_top_genes=4000,min_mean=0.0125, max_mean=3, min_disp=0.5)
    # adata = adata[:, adata.var.highly_variable]
    # data = np.array([])
    # for i in adata.var.highly_variable.index:
    #     if len(data)==0:
    #         data = adata.raw.X[:,int(i)].reshape([-1,1])
    #     else:
    #         data = np.concatenate((data,adata.raw.X[:,int(i)].reshape([-1,1])),axis=1)
    adata_raw_X=X
    print(len(set(Y)))
    print(Y)
    print(X.shape)
    return X,Y,adata_raw_X, adata.obs.size_factors


class AAE_net(nn.Module):
    def __init__(self,X_dim, N,M,z_dim,n_clusters,sigma=1.,alpha=1.,gamma=1.):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.gamma = gamma
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, z_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)        
        self.Q_net = nn.Sequential( 
                                   
            nn.Linear(X_dim, N),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(N, M),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(M, z_dim)  
        )
        
        self.P_net = nn.Sequential( 
                                   
            nn.Linear(z_dim, M),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(M, N)

        )
       
        self.out_net = nn.Sequential(
            nn.Dropout(0.2),

            nn.Linear(N, N),
            nn.Sigmoid(),             
        )
        self._lin_mean = nn.Sequential(nn.Linear(N,X_dim), MeanAct())
        self._lin_disp = nn.Sequential(nn.Linear(N,X_dim), DispAct())
        self._lin_pi = nn.Sequential(nn.Linear(N,X_dim), nn.Sigmoid())
        # degree
        self.v = 1
        self.zinb_loss = ZINBLoss().cuda()
        self.out_y = nn.Linear(z_dim,n_clusters)
        self.D_net_gaus = nn.Sequential(
            nn.Linear(z_dim, N), 
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(N, M),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(M, 1),
            nn.Sigmoid()
        )
        self.n_clusters = n_clusters
        self.loss_ce = nn.CrossEntropyLoss()
    def gen(self,z):
        z = z.to(torch.float32)
        return self.D_net_gaus(z)
    def forward(self,x):
        x  = x.float()
        z_noise  = self.Q_net((x+torch.randn_like(x) * self.sigma).float())
    
        z_noise_2 = self.P_net(z_noise.float())
        h = self.out_net(z_noise_2)
        mean = self._lin_mean(h)
        disp = self._lin_disp(h)
        pi = self._lin_pi(h)      
        y_hat=F.softmax(self.out_y(z_noise),dim=1)#训练使用带噪声的z,预测使用不带噪声的z
        latent = torch.cat((z_noise,y_hat),dim=1)#cat函数 将z和y按行样本合并，共同输入解码器 
         
        z = self.Q_net(x.float())
        return z,mean,disp,pi,z_noise,y_hat,latent

    #对于自编码器进行预训练
    def pretrian(self,x, X_raw, size_factor,epoch =10,batch_size=128,lr=0.001):    
        self.train()
        time = datetime.now().strftime('%Y%m%d')
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epoch):
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch)
                x_raw_tensor = Variable(x_raw_batch)
                sf_tensor = Variable(sf_batch)
                _, mean_tensor, disp_tensor, pi_tensor,_,_ ,_= self.forward(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=sf_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Pretrain epoch [{}/{}], ZINB loss:{:.4f}'.format(batch_idx+1, epoch+1, loss.item()))
        torch.save({
            'ae_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },'./预训练模型/'+time+'.pth.tar')
    def encodeBatch(self, X, batch_size=128):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch)
            z= self.Q_net(inputs.float())
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded
    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)
    def fit(self,x,y,X_raw, sf,lr=0.01,lr_d=0.0001 ,batch_size=128, num_epochs=100,save_dir=""):#234数据集的参数
        #y-=1
        Y = torch.tensor(y).long()
        TINY = 1e-15
        X = torch.tensor(x)
        X_raw = torch.tensor(X_raw)
        sf = torch.tensor(sf)
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)
        #sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=0.001, last_epoch=-1, verbose=False)#学习率衰减
        optimizer_D = optim.Adam(self.D_net_gaus.parameters(),lr=lr_d)
        kmeans = KMeans(self.n_clusters, n_init=20)
        data = self.encodeBatch(X)
        self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
       
        acc = np.round(cluster_acc(y, self.y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
        print('Initializing k-means: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
        
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        np.random.seed(0)
        
        for epoch in range(num_epochs):
            self.eval()
            latent = self.encodeBatch(X)
            y_ = F.softmax(latent,dim=1).data.cpu().numpy()
            newlatent=np.concatenate((latent,y_), axis=1)
            self.final_latent =newlatent
            self.y_pred =GaussianMixture(n_components=nb_classes).fit_predict(newlatent)#只返回预测标签
            acc = np.round(cluster_acc(y, self.y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score (y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print('Clustering   %d: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (epoch+1, acc, nmi, ari))
          

            train_loss = 0.0
            recon_loss_val = 0.0
            self.train()
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                xbatch_use = xbatch[:int(len(xbatch)*0.2)]
                xbatch_unlabel = xbatch[int(len(xbatch)*0.2):]
              
                xrawbatch = X_raw[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                xrawbatch = xrawbatch[int(len(xbatch)*0.2):]
               
                sfbatch = sf[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sfbatch = sfbatch[int(len(xbatch)*0.2):]
               
                y_use = Y[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)] 
                y_use = y_use[:int(len(xbatch)*0.2)]
               
                optimizer.zero_grad()
                inputs = Variable(xbatch_unlabel)
                inputs_label = Variable(xbatch_use)  
                rawinputs = Variable(xrawbatch)
                sfinputs = Variable(sfbatch)
        
                
                z, meanbatch, dispbatch, pibatch,z_noise,_,_= self.forward(inputs)
                recon_loss = self.zinb_loss(rawinputs.float(), meanbatch, dispbatch, pibatch, sfinputs)
                _,_,_,_,_,y_hat_use,_=self.forward(inputs_label)
                loss = recon_loss+self.loss_ce(y_hat_use,y_use)   
                loss.backward()                                                                            
                optimizer.step()
                #sch.step()
                
                if batch_idx % 5 == 0:
                    z = self.Q_net(inputs.float()) 
                    z2 = z.detach().numpy()
                    z_real_gauss = Variable(torch.from_numpy(np.random.normal(float(z2.mean()),float(z2.std()),size = (z.shape[0], z_dim))))
                    D_real_gauss = self.gen(z_real_gauss)
                
                    z, meanbatch, dispbatch, pibatch,z_noise,_,_ = self.forward(inputs)
                    D_fake_gauss = self.gen(z_noise)
                    # D_loss = -(torch.mean(torch.log((D_real_gauss + TINY-1)**2) + torch.log((D_fake_gauss + TINY)**2)))
                    # G_loss = -torch.mean(torch.log((D_fake_gauss + TINY-1)**2))
                    D_loss = -torch.mean(torch.log(D_real_gauss + TINY)) -torch.mean(torch.log((1-D_fake_gauss + TINY)))
                    optimizer_D.zero_grad()
                    D_loss.backward()
                    optimizer_D.step()
                
                z,  meanbatch, dispbatch, pibatch,z_noise,_,_ = self.forward(inputs)
                D_fake_gauss = self.gen(z_noise)
                G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))
                G_loss.backward()
                optimizer.step()
           

                recon_loss_val += recon_loss.data * len(inputs)
                
                train_loss = recon_loss_val+0.5*G_loss+0.5*D_loss
            print("#Epoch %3d: Total: %.4f ZINB Loss: %.4f D_loss: %.4f,G_loss:%.4f" % (
                epoch + 1, train_loss / num, recon_loss_val / num, D_loss,G_loss))
        self.save_checkpoint({'epoch': epoch+1,
                'state_dict': self.state_dict(),
                'y_pred': self.y_pred,
                'y': y
                }, epoch+1, filename=save_dir)
        return self.y_pred, acc, nmi, ari
if __name__ == '__main__':

    x,y,raw_x,sf= load_data()
 
    nb_classes = len(set(y))    
    aae = AAE_net(X_dim,N,M,z_dim,nb_classes)
    print('预训练自编码器')
    
    t0 = time()
    aae.pretrian(x,raw_x,sf)
    y_pred, acc, nmi, ari = aae.fit(x,y,raw_x,sf,save_dir='./数据模型')
    print('Total time: %d seconds.' % int(time() - t0))

    #purity = metric.compute_purity(label_kmeans, Y)
    acc = cluster_acc(y, y_pred)
    nmi = normalized_mutual_info_score(y,y_pred)
    ari = adjusted_rand_score(y,y_pred)
    homogeneity = homogeneity_score(y,y_pred)
    ami = adjusted_mutual_info_score(y,y_pred)
    print('ACC = {},NMI = {}, ARI = {},AMI = {}, Homogeneity = {}'.format(acc,nmi,ari,ami,homogeneity))  
    np.save(r'C:\Users\lenovo\Desktop\hxb\cc_label.npy',y_pred)#用np.load读取，读取为一个数组

    
 
    final_latent = aae.final_latent
    # np.savetxt('wn_label.csv', y_pred, delimiter=",")
    reducer = umap.UMAP(n_components=2,random_state=42)
    X_umap = reducer.fit_transform(final_latent)
   
    pca = PCA(n_components=2,random_state=42)
    X_pca = pca.fit_transform(final_latent)

    phate_operator = phate.PHATE(n_components=2)
    X_phate= phate_operator.fit_transform(final_latent)

    reducer = umap.UMAP(n_components=6,random_state=32)
    X_umap = reducer.fit_transform(final_latent)

    tsne_operator = sklearn.manifold.TSNE(n_components=2)
    X_tsne = tsne_operator.fit_transform(final_latent)

    scprep.plot.scatter2d(X_umap,label_prefix="Umap", title="scAAE",ticks=True, figsize=(8,7),c=y_pred,marker='o',cmap='Spectral')
    scprep.plot.scatter2d(X_pca,label_prefix="PCA", title="scAAE",ticks=True, figsize=(8,7),c=y_pred,marker='o',cmap='Spectral')
    scprep.plot.scatter2d(X_phate,label_prefix="Phate", title="scAAE",ticks=True, figsize=(8,7),c=y_pred,marker='o',cmap='Spectral')
    scprep.plot.scatter2d(X_tsne,label_prefix="Tsne", title="scAAE",ticks=True, figsize=(8,7),c=y_pred,marker='o',cmap='Spectral')
    
   