from fastai import *
from fastai.vision import *
import os

os.environ['CUDA_VISIBLE_DEVICE'] = '0,1'
import torch

bs = 128

path = 'test'

tfms = get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4, p_affine=1., p_lighting=1.)

src = ImageItemList.from_folder(path).random_split_by_pct(0.2, seed=2)

def get_data(size, bs, padding_mode='reflection'):
    return (src.label_from_re('([^/]+)_\d+.png$').transform(tfms, size=size, padding_mode=padding_mode).databunch(bs=bs).normalize(imagenet_stats))

data = get_data(224, bs, 'zeros')

def _plot(i,j,ax):
    x,y = data.train_ds[9]
    x.show(ax, y=y)

# plot_multi(_plot, 3, 3, figsize=(8,8))
# plt.show()



'''
    TRAIN A MODEL
'''

gc.collect()

learn = create_cnn(data, models.resnet101, metrics=accuracy, bn_final=True).to_fp16()
# learn = learn.to_fp16()
learn.model = torch.nn.DataParallel(learn.model)


# learn.lr_find()
# learn.recorder.plot()
# plt.show()

learn.fit_one_cycle(3, slice(3e-2), pct_start=0.8)

learn.unfreeze()
# learn.lr_find()
# learn.recorder.plot()
# plt.show()
# plt.close()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4), pct_start=0.8)

print("--- change to bigger images size ---")
sz_big = 299
data = get_data(sz_big, bs)
learn.data = data

# learn.lr_find()
# learn.recorder.plot()
# plt.show()
# plt.close()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
