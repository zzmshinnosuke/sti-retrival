
import argparse
import yaml
from easydict import EasyDict

import sys
import os
import time
from tqdm import tqdm as tqdm
import numpy as np

import torch
import torchvision
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau

import logging
from tensorboardX import SummaryWriter

import ske_text_img_composition_models
import datasets

def load_dataset(opt):
    print('Reading dataset ',opt.dataset)
    size=224
    if opt.img_model=="inceptionv3":
        size=299
    if opt.dataset.name=='STI':
        trainset=datasets.STIDataset(
            path=opt.dataset.dataset_path,
            split='train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(size),
                torchvision.transforms.CenterCrop(size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
        testset=datasets.STIDataset(
            path=opt.dataset.dataset_path,
            split='test',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(size),
                torchvision.transforms.CenterCrop(size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))

    else:
        print('Invalid dataset',opt.dataset)
        sys.exit()

    print('trainset size:',len(trainset))
    print('test size:',len(testset))
    return trainset,testset

def create_model_and_optimizer(opt,texts):
    print('Creates model and optimizer ',opt.model)
    if opt.model=='imgonly':
        model = ske_text_img_composition_models.SkeOnlyModel(texts,embed_dim=opt.embed_dim,img_model=opt.img_model)
    elif opt.model=='textonly':
        model = ske_text_img_composition_models.TextOnlyMode(texts,embed_dim=opt.embed_dim,img_model=opt.img_model)
    elif opt.model=='concat':
        model = ske_text_img_composition_models.Concat(texts,embed_dim=opt.embed_dim,img_model=opt.img_model)
    elif opt.model == 'tirg':
        model = ske_text_img_composition_models.TIRG(texts, embed_dim=opt.embed_dim,img_model=opt.img_model)
    elif opt.model == 'tirg_lastconv':
        model = ske_text_img_composition_models.TIRGLastConv(texts, embed_dim=opt.embed_dim,img_model=opt.img_model)
    else:
        print('Invalid model',opt.model)
        sys.exit()

    model=model.cuda()

    params=[]
    params.append({'params': [p for p in model.parameters()]})
    optimizer=torch.optim.SGD(params, lr=opt.optimizer.learning_rate, momentum=opt.optimizer.momentum, weight_decay=opt.optimizer.weight_decay)
    scheduler=ReduceLROnPlateau(optimizer,
                                mode='min',
                                factor=opt.lr_scheduler.learning_rate_factor,
                                patience=opt.lr_scheduler.learning_rate_decay_frequency)
    return model,optimizer,scheduler

def test(opt,model,testset,test=True):
    model.eval()
    all_imgs = []
    all_captions = []
    all_queries = []
    all_target_captions = []

    ske_names=[]
    img_names=[]
    ske_imgs=[]
    captions=[]
    imgs=[]
    results=[]
    for item in tqdm(testset):
        ske_imgs+=[item['source_ske_img_data']]
        img_names+=[item['target_img_name']]
        ske_names+=[item['source_ske_name']]
        captions +=[item['mod']['str']]
        imgs += [item['target_img_data']]
        if len(imgs)>=opt.batch_size or item is testset[-1]:
            ske_imgs=torch.stack(ske_imgs).float()
            ske_imgs=torch.autograd.Variable(ske_imgs)
            captions=[t for t in captions]
            f=model.compose_img_text(ske_imgs.cuda(),captions).data.cpu().numpy()

            imgs=torch.stack(imgs).float()
            imgs=torch.autograd.Variable(imgs)
            imgs=model.extract_img_feature(imgs.cuda()).data.cpu().numpy()

            all_queries+=[f]
            all_imgs+=[imgs]
            ske_imgs = []
            captions = []
            imgs=[]
    all_imgs=np.concatenate(all_imgs)
    all_queries=np.concatenate(all_queries)

    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(all_imgs.T)
    nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]
#     for i,nns in enumerate(nn_result):
#         if i in nns[:1]:
#             print(ske_names[i])
#             for nn in nns[:10]:
#                 print(img_names[nn])
    out=[]
    top1=0
    for k in [1,5,10,50,100]:
        r=0.0
        for i,nns in enumerate(nn_result):
            if i in nns[:k]:
                r+=1
        r /= len(nn_result)
        out+=[('recall_top' + str(k) + '_correct_composition', r)]
        if k==1:
            top1=r

    return out,top1

def train(opt,logger,trainset,testset,model,optimizer,scheduler,epoch=0):
    print("Begin training!")
    tic=time.time()
    best_top1=0
    
    while epoch<opt.num_epochs:
        print('Epoch ',epoch,' Elapsed time ',round(time.time()-tic,4),opt.logger_comment)
        tic=time.time()

        if epoch%3==1:
            tests = []
            for name, dataset in [('train', trainset), ('test', testset)]:
                t,top1 = test(opt, model, dataset)
                if name=='test' and opt.best_model and top1>best_top1:
                    best_top1=top1
                    torch.save({
                        'epoch':epoch,
                        'opt':opt,
                        'model_state_dict':model.state_dict(),},
                        logger.file_writer.get_logdir()+'/best_checkpoint.pth')

                tests += [(name + ' ' + metric_name, metric_value)
                          for metric_name, metric_value in t]
            for metric_name, metric_value in tests:
                logger.add_scalar(metric_name, metric_value, epoch)
                print('    ', metric_name, round(metric_value, 4))

        torch.save({
            'epoch':epoch,
            'opt':opt,
            'model_state_dict':model.state_dict(),},
            logger.file_writer.get_logdir()+'/latest_checkpoint.pth'
        )

        model.train()

        trainloader = trainset.get_loader(
            batch_size=opt.batch_size,
            shuffle=opt.dataset.shuffle,
            drop_last=opt.dataset.drop_last,
            num_workers=opt.dataset.loader_num_workers)

        def training_1_iter(data):
            assert type(data) is list
            img1 = np.stack([d['source_ske_img_data'] for d in data])
            img1 = torch.from_numpy(img1).float()
            img1 = torch.autograd.Variable(img1).cuda()
            img2 = np.stack([d['target_img_data'] for d in data])
            img2 = torch.from_numpy(img2).float()
            img2 = torch.autograd.Variable(img2).cuda()
            mods = [str(d['mod']['str']) for d in data]
            mods = [t.encode('utf-8').decode('utf-8') for t in mods]

            # compute loss
            losses = []
            if opt.loss == 'soft_triplet':
                loss_value = model.compute_loss(
                    img1, mods, img2, soft_triplet_loss=True)
            elif opt.loss == 'batch_based_classification':
                loss_value = model.compute_loss(
                    img1, mods, img2, soft_triplet_loss=False)
            else:
                print('Invalid loss function', opt.loss)
                sys.exit()
            assert not torch.isnan(loss_value)

            # gradient descend
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            return loss_value.detach().cpu().numpy()

        loss_in_one_epoch=[]
        for data in tqdm(trainloader, desc='Training for epoch ' + str(epoch)):
            loss_in_one_epoch.append(training_1_iter(data))
        
        avg_loss = np.mean(loss_in_one_epoch)
        print('    Loss', opt.loss, round(avg_loss, 4))
        logger.add_scalar(opt.loss, avg_loss, epoch)
        logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step(np.sum(loss_in_one_epoch))
        epoch += 1 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='')
    args=parser.parse_args()
    with open(args.config) as f:
        opt=yaml.load(f,Loader=yaml.FullLoader)
    opt=EasyDict(opt['common'])
    print(opt)
    print("Arguments:")
    for k in list(opt.__dict__.keys()):
        print(k,':',str(opt.__dict__[k]))

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.set_num_threads(3)

    logger=SummaryWriter(comment=opt.logger_comment)
    print('Log files saved to',logger.file_writer.get_logdir())
    for k in list(opt.__dict__.keys()):
        logger.add_text(k, str(opt.__dict__[k]))
    
    trainset,testset=load_dataset(opt)
    model,optimizer,scheduler=create_model_and_optimizer(opt,[t for t in trainset.get_all_texts()])
    start_epoch=0
    if opt.resume:
        if os.path.isfile(opt.resume):
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt.resume, start_epoch))
            
            tests,top1 = test(opt, model, testset)
            for metric_name, metric_value in tests:
                print('    ', metric_name, round(metric_value, 4))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    train(opt,logger,trainset,testset,model,optimizer,scheduler,start_epoch)
    logger.close()

def main_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='')
    args=parser.parse_args()
    with open(args.config) as f:
        opt=yaml.load(f,Loader=yaml.FullLoader)
    opt=EasyDict(opt['common'])
    print(opt)
    print("Arguments:")
    for k in list(opt.__dict__.keys()):
        print(k,':',str(opt.__dict__[k]))

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_num_threads(3)
    
    trainset,testset=load_dataset(opt)
    model,optimizer,scheduler=create_model_and_optimizer(opt,[t for t in trainset.get_all_texts()])
    start_epoch=0
    if opt.resume:
        if os.path.isfile(opt.resume):
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch'] 
            model.load_state_dict(checkpoint['model_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt.resume, start_epoch))
            
            tests,top1 = test(opt, model, testset)
            for metric_name, metric_value in tests:
                print('    ', metric_name, round(metric_value, 4))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

if __name__=='__main__':
      main_test()
