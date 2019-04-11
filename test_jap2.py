from fastai import *
from fastai.vision import *
import os

os.environ['CUDA_VISIBLE_DEVICE'] = '0,1,2'
import torch

bs = 128


import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", type=str, required=True)
args = vars(ap.parse_args())

path = args['name']


tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
#tfms = get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4, p_affine=1., p_lighting=1.)

src = ImageList.from_folder(path).split_by_rand_pct(0.2, seed=2)

def get_data(suff, size, bs, padding_mode='reflection'):
    return (src.label_from_re('([^/]+)_\d+.{}$'.format(suff)).transform(tfms, size=size, padding_mode=padding_mode).databunch(bs=bs).normalize(imagenet_stats))

formatting = ['jpg', 'png', 'jpeg']

for img_ft in formatting:

    try:
        data = get_data(img_ft, 256, bs, 'zeros')
        break
    except AssertionError:
        continue

#data = get_data(224, bs, 'zeros')

# def _plot(i,j,ax):
    # x,y = data.train_ds[9]
    # x.show(ax, y=y)

# plot_multi(_plot, 3, 3, figsize=(8,8))
# plt.show()

'''
    TRAIN A MODEL
'''

gc.collect()

learn = cnn_learner(data, models.resnet50, metrics=accuracy, bn_final=True).to_fp16()
# learn = learn.to_fp16()
#learn.model = torch.nn.DataParallel(learn.model)


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
learn.save('small')

print("--- change to bigger images size ---")

sz_big = 352

for img_ft in formatting:

    try:
        data = get_data(img_ft, sz_big, bs)
        break
    except AssertionError:
        continue

# data = get_data(sz_big, bs)
learn = cnn_learner(data, models.resnet50, metrics=accuracy, bn_final=True).load('small').to_fp16()

# learn.lr_find()
# learn.recorder.plot()
# plt.show()
# plt.close()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
#learn.save('{}'.format(str(path)))
learn.save('big')
learn.load('big')
######
print("--- INFERENCING ---")
#learn_load = cnn_learner(data, models.resnet50).load('{}'.format(str(path)), strict=False).fp_16()
#learn.export(fname='../models/{}.pkl'.format(str(path)))
learn.export()
print("--- LOAD MODEL : {} ---".format(str(path)))
#learn = load_learner(path, fname='../models/{}.pkl'.format(str(path))).fp_16()
learn = load_learner(path).to_fp16()
img = data.train_ds[0][0]
print(learn.predict(img))










