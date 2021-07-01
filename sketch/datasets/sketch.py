#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2021-06-30 15:49:26
# @Author: zzm

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
from category import *
from PIL import Image
import os
from pycocotools.coco import COCO
from shutil import copyfile
import random
from tqdm import tqdm as tqdm

class Stroke:
    def __init__(self,stroke):
        self.thickness=-1
        self.color=-1
        self.id=-1
        self.points=[]
        self.load(stroke)
        
    def load(self,stroke):
        self.thickness=stroke["thickness"]
        self.id=stroke["id"]
        self.color=stroke["color"]
        for point in stroke["points"]:
            self.points.append(point)
            
    def get_points_len(self):
        return len(self.points)
    
    def get_points(self):
        return self.points

class Item:
    def __init__(self, item,color=True):
        self.name=None
        self.id=-1
        self.strokes=[]
        self.category=None
        self.color=[]
        self.show_color=color #显示或为图像的时候是彩色或者黑白
        self.integrity=-1
        self.quality=-1
        self.similarity=-1
        self.direction=""
        self.boundingbox=None #与图像中boundingbox的对应
        self.load(item)
    
    def load(self,item):
        self.name=item["name"]
        self.category=item["category"]
        self.id=item["id"]
        self.color=item["color"]
        self.integrity=item["integrity"]
        self.similarity=item["similarity"]
        self.quality=item["quality"]
        if "boundingbox" in item.keys() :
            self.boundingbox=item["boundingbox"]
        self.direction=item["direction"]
        for stroke in item["strokes"]:
            self.strokes.append(Stroke(stroke))
         
    def get_boundingbox(self):
        all_points=self.get_points()      
        min_xy=np.min(np.array(all_points),0).tolist()
        max_xy=np.max(np.array(all_points),0).tolist()
        min_xy.extend(max_xy)
        return [int(i) for i in min_xy]
    
    def get_resolution(self):
        boundingbox=self.get_boundingbox()
        return [boundingbox[2]-boundingbox[0],boundingbox[3]-boundingbox[1]]
    
    def get_points(self):
        all_points=[]
        for stroke in self.strokes:
            all_points.extend(stroke.get_points())
        return all_points
    
    def get_strokes(self):
        return self.strokes
    
    def get_strokes_len(self):
        return len(self.category)
    
    def get_points_len(self):
        return np.sum([stroke.get_points_len() for stroke in self.strokes])   
    
    def get_item_image(self):
        resolution=self.get_resolution()
        plt.figure(figsize=(resolution[0]/100, resolution[1]/100),dpi=100)
        x = []
        y = []
        count = 0
        if self.show_color :
            color="#"+str(hex(self.color[0]))[-2:]+str(hex(self.color[1]))[-2:]+str(hex(self.color[2]))[-2:]+str(hex(self.color[3]))[-2:]
        else :
            color="black"
        for stroke in self.strokes:
            points = stroke.get_points()
            for point in points:
                x.append(point[0])
                y.append(point[1])
                count = count + 1
            nodes = np.array([x, y])
            x1 = nodes[0]
            y1 = nodes[1]
            plt.plot(x1, y1, color=color,linewidth=float(stroke.thickness), linestyle="-")
            x.clear()
            y.clear()
            count = 0 
        plt.axis('off')
        plt.xticks([])  # ignore xticks
        ax = plt.gca()  # 获取坐标轴信息
        ax.invert_yaxis()  # y轴反向
        ax.spines['top'].set_visible(False)  # 去掉边框
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.yticks([])  # ignore yticks
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        img = np.array(canvas.renderer.buffer_rgba())
        plt.clf()
        plt.cla()
        plt.close("all")
        return img

class Sketch:
    def __init__(self, sketch_json_path=None,image_path=None,useless=True,color=False):
        self.resolution = []
        self.image_name = ""
        self.image_path=image_path
        self.image =None
        self.items = []
        self.captions = []
        self.sketch_json = []
        self.useless = useless  # 是否保留无用的物体,True不保留，False保留
        self.coco_caption = caption_coco
        self.color = color
        self.scene = scene
        if label_end_path:
            self.sketch_json=self.read_json(label_end_path)
            self.load(self.sketch_json)
        if draw_end_path:
            draw_json=self.read_json(draw_end_path)
            self.image_name=draw_json["filename"]
            self.resolution=draw_json["device"][-2:]
            # print(self.image_name,self.resolution)
        if image_path:
            self.image=self.load_reference_image()
        if caption_coco:
            self.captions=self.load_captions()

    @staticmethod
    def show_image(img):
        print(len(cv2.split(img)))
        if img.shape[2]==4:
            b, g, r, a = cv2.split(img)
            img = cv2.merge([r, g, b, a])
        else:
            b, g, r= cv2.split(img)
            img = cv2.merge([r, g, b])
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def get_crop_image(img,boundingbox):
        return img[boundingbox[1]:boundingbox[3],boundingbox[0]:boundingbox[2]]

    @staticmethod
    def save_image(img,target_path):
        im = Image.fromarray(img)
        im.save(target_path)

    def save_json(self,target_path):
        new_json = {"filename" : self.image_name,"resolution":self.resolution,"captions":self.captions,'scene':self.scene,"objects":self.sketch_json}
        with open(target_path,"w") as f:
            json.dump(new_json,f)

    def read_json(self, path):
        with open(path, "r") as f:
            try:
                load_dict = json.load(f)
                return load_dict
            except json.decoder.JSONDecodeError:
                print("read json error!")
        return None

    def load_reference_image(self):
        return np.array(Image.open(os.path.join(self.image_path, self.image_name)))

    def load_captions(self):
        id=int(self.image_name.split(".")[0])
        annIds = self.coco_caption.getAnnIds(imgIds=int(id), iscrowd=None)
        anns = self.coco_caption.loadAnns(annIds)
        return [ann['caption'] for ann in anns]

    def load(self, sketch_json):
        for item in sketch_json:
            if self.useless and item["category"] == "useless":
                continue
            else:
                self.items.append(Item(item,color=self.color))

    def get_resolution(self):
        return self.resolution
    
    def get_items_len(self):
        return len(self.items)

    def get_strokes_len(self):
        return np.sum([item.get_strokes_len() for item in self.items])  

    def get_points_len(self):
        return np.sum([item.get_points_len() for item in self.items])  

    def get_all_category(self):
        all_category=[item.category for item in self.items]
        return list(set(all_category))

    def get_items(self):
        return self.items

    def get_foreground_items(self):
        items = []
        for item in self.items:
            if item.category in Sketch.Foreground_category:
                items.append(item)
        return items

    def get_background_items(self):
        items = []
        for item in self.items:
            if item.category in Sketch.Background_category:
                items.append(item)
        return items
    
    def get_strokes(self):
        strokes=[]
        for item in self.items:
            strokes.extend(item.get_strokes())
        return strokes

    def items_to_image(self,items=None):
        if not items:
            items = self.items
        resolution = self.get_resolution()
        plt.figure(figsize=(resolution[0] / 100, resolution[1] / 100), dpi=100)
        x = []
        y = []
        count = 0
        for item in items:
            if self.color:
                color = "#" + str(hex(item.color[0]))[-2:] + str(hex(item.color[1]))[-2:] + str(hex(item.color[2]))[-2:] + \
                    str(hex(item.color[3]))[-2:]
            else:
                color = "black"
            for stroke in item.strokes:
                points = stroke.get_points()
                for point in points:
                    x.append(point[0])
                    y.append(point[1])
                    count = count + 1
                nodes = np.array([x, y])
                x1 = nodes[0]
                y1 = nodes[1]
                plt.plot(x1, y1, color=color, linewidth=float(stroke.thickness), linestyle="-")
                x.clear()
                y.clear()
                count = 0
        plt.axis('off')
        plt.xticks([])  # ignore xticks
        ax = plt.gca()  # 获取坐标轴信息
        ax.invert_yaxis()  # y轴反向
        ax.spines['top'].set_visible(False)  # 去掉边框
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.yticks([])  # ignore yticks
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        img = np.array(canvas.renderer.buffer_rgba())
        plt.clf()
        plt.cla()
        plt.close("all")
        return img

    def get_image(self):
        return self.items_to_image()

    def get_foreground_image(self):
        return self.items_to_image(items=self.get_foreground_items())

    def get_background_image(self):
        return self.items_to_image(items=self.get_background_items())

    def get_five_point(self):
        pass

    def get_svg(self):
        pass

def main():
    print("load data:")
    # sketch = Sketch(label_end_path="/home/zzm/data-new/label-end/field/results1/100.json",image_path="/home/zzm/datasets/coco2017/train2017"
    #                 ,draw_end_path="/home/zzm/data-new/draw-end/field/results1/100.json",caption_path='/home/zzm/datasets/coco2017/annotations/captions_train2017.json',useless=True)
    # print(sketch.get_points_len(),sketch.get_all_category())

    # Sketch.show_image(img=sketch.items[0].get_item_image())
    # Sketch.show_image(img=sketch.get_background_image())
    # Sketch.show_image(img=sketch.get_foreground_image()
    # Sketch.show_image(img=sketch.get_image())
    # Sketch.show_image(img=sketch.get_crop_image(sketch.load_reference_image(),sketch.items[0].get_boundingbox()))
    # Sketch.show_image(img=sketch.items[0].get_item_image())
    # sketch.save_json("./1.json")
    # print(sketch.captions)
    save_model_json_image()
    get_train_test()

if __name__ == '__main__':
    main()
