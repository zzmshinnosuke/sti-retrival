
# RUN:  
source activate pt17-cu11-py37  
python main.py --dataset=STI --dataset_path=./STI --num_iters=160000 --model=concat --loss=soft_triplet --learning_rate_decay_frequency=50000 --num_iters=160000 --weight_decay=5e-5 --comment=sti_concat

python main.py --dataset=STI --dataset_path=./STI --num_iters=160000 --model=textonly --loss=soft_triplet --learning_rate_decay_frequency=50000 --num_iters=160000 --weight_decay=5e-5 --comment=sti_textonly

python main.py --dataset=STI --dataset_path=./STI --num_iters=160000 --model=imgonly --loss=soft_triplet --learning_rate_decay_frequency=50000 --num_iters=160000 --weight_decay=5e-5 --comment=sti_imgonly

python main.py --dataset=STI --dataset_path=./STI --num_iters=160000 --model=tirg --loss=soft_triplet --learning_rate_decay_frequency=50000 --num_iters=160000 --weight_decay=5e-5 --comment=sti_tirg

python main.py --dataset=STI --dataset_path=./STI --num_iters=160000 --model=tirg_lastconv --loss=soft_triplet --learning_rate_decay_frequency=50000 --num_iters=160000 --weight_decay=5e-5 --comment=sti_tirg_lastconv

python main.py --dataset=STI --dataset_path=./STI --num_iters=200000 --model=imgonly --img_model=inceptionv3 --loss=soft_triplet --learning_rate_decay_frequency=300 --learning_rate_factor=5e-1  --weight_decay=5e-5 --comment=sti_imgonly_inceptionv3_newlr 

python main.py --dataset=STI --dataset_path=./STI --num_iters=200000 --model=imgonly --img_model=resnet18 --loss=soft_triplet --learning_rate_decay_frequency=300 --learning_rate_factor=5e-1  --weight_decay=5e-5 --embed_dim=1024 --comment=sti_imgonly_embed_dim_1024

python main.py --dataset=STI --dataset_path=./STI --num_iters=200000 --model=textonly --img_model=resnet18 --loss=soft_triplet --learning_rate_decay_frequency=300 --learning_rate_factor=5e-1  --weight_decay=5e-5 --embed_dim=1024 --comment=sti_textonly_retrivalSketch

# backup
stimodel-v1 
在tirg基础上抽出了其中文本和图片检索图片的方法。修改了相应数据加载的地方以及测试的地方。
将图片替换成了草图。

stimodel-v1.1
把10.0.0.67上改的模型都复制到了10.0.0.62,这个是在佳欣师兄改的学习率，并增加freeze基础上改的。

stimodel-v1.2
模型添加了自动调整学习率的的功能。

stimodel-v1.3
模型添加yaml配置参数的方法，感觉要比后面直接加参数更方便很多。记录loss计算total_loss 和 triplet_loss 重复了，没有存在的必要，先去掉了，只保存一个就够了。看看要不要添加logging，目前看来没有任何必要。训练了一下准确率有所下降，不知道是改模型出现的，还是正常出现的。

stimodel-v1.4
添加保存最优模型参数的功能，已经加载模型继续训练的功能,20200204修改完了，训练一下试试有没有问题

stimodel-v1.5
把全部数据都整理了一下，数据读取的地方重新改改

