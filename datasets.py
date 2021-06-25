import numpy as np
import torch
import torch.utils.data
import torchvision
import PIL
import skimage.io
from os import listdir
import json

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self):
        super(BaseDataset,self).__init__()
        self.skes_img=[]
        self.imgs=[]

    def get_loader(self,batch_size,shuffle=False,drop_last=False,num_workers=0):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=lambda i: i)

    def get_all_texts(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.generate_random_query_target()

    def generate_random_query_target(self):
        raise NotImplementedError

    def get_img(self,idx,raw_img=False):
        raise NotImplementedError


class STIDataset(BaseDataset):

    def __init__(self,path,split='train',transform=None):
        super(STIDataset,self).__init__()
        self.path=path
        self.transform=transform
        self.split=split

        self.skes_img=[]
        self.texts={}

        if split=='test':
            for f in listdir(path+'/test/'):
                assert (f.endswith('json'))
                name=f.split('.')[0]
                self.skes_img.append({"ske_img_path":path+'/sketch-png/'+name+'.png',
                                      "img_path":path+'/image/'+name+'.jpg',
                                      "img_name":name+'.jpg'})
        
        if split=='train':
            for f in listdir(path+'/train/'):
                assert (f.endswith('json'))
                name=f.split('.')[0]
                self.skes_img.append({"ske_img_path":path+'/sketch-png/'+name+'.png',
                                      "img_path":path+'/image/'+name+'.jpg',
                                      "img_name":name+'.jpg'})

    def __getitem__(self, idx):
        out={}
        out['source_ske_id']=idx
        out['source_ske_name']=self.skes_img[idx]['img_name']
        out['source_ske_img_data']=self.get_img(self.skes_img[idx]['ske_img_path'])
        out['source_caption']=""
        out['target_img_id']=idx
        out['target_img_name']=self.skes_img[idx]['img_name']
        out['target_img_data']=self.get_img(self.skes_img[idx]['img_path'])
        out['mod']={'str':""} 
        return out

    def __len__(self):
        return len(self.skes_img)

    def get_all_texts(self):
        texts=[]
        for key in self.texts.keys():
            values=self.texts[key]
            for v in values:
                texts.append(v['caption'])
        return texts

    def get_img(self,img_path,raw_img=False):
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        if raw_img:
            return img
        if self.transform:
            img = self.transform(img)
        
        return img



