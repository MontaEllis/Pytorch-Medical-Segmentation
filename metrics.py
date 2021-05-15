import torchio as tio
from pathlib import Path
import torch
import numpy as np
import copy
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)



predict_dir = '/data0/my_project/med/seg_3d/results_6-10'
labels_dir = '/data2/zkndataset/med/unet/test'

# predict_dir = '/data0/my_project/med/seg_3d/results'
# labels_dir = '/data2/zkndataset/med/unet/label'


def do_subject(image_paths, label_paths):
    for (image_path, label_path) in zip(image_paths, label_paths):
        subject = tio.Subject(
            pred=tio.ScalarImage(image_path),
            gt=tio.LabelMap(label_path),
        )
        subjects.append(subject)

images_dir = Path(predict_dir)
labels_dir = Path(labels_dir)

image_paths = sorted(images_dir.glob('*.mhd'))
label_paths = sorted(labels_dir.glob('*/*.mhd'))


subjects = []
do_subject(image_paths, label_paths)

training_set = tio.SubjectsDataset(subjects)


toc = ToCanonical()

for i,subj in enumerate(training_set.subjects):
    gt = subj['gt'][tio.DATA]

    # subj = toc(subj)
    pred = subj['pred'][tio.DATA]#.permute(0,1,3,2)

    # preds.append(pred)
    # gts.append(gt)




    preds = pred.numpy()
    gts = gt.numpy()



    pred = preds.astype(int)  # float data does not support bit_and and bit_or
    gdth = gts.astype(int)  # float data does not support bit_and and bit_or
    fp_array = copy.deepcopy(pred)  # keep pred unchanged
    fn_array = copy.deepcopy(gdth)
    gdth_sum = np.sum(gdth)
    pred_sum = np.sum(pred)
    intersection = gdth & pred
    union = gdth | pred
    intersection_sum = np.count_nonzero(intersection)
    union_sum = np.count_nonzero(union)

    tp_array = intersection

    tmp = pred - gdth
    fp_array[tmp < 1] = 0

    tmp2 = gdth - pred
    fn_array[tmp2 < 1] = 0

    tn_array = np.ones(gdth.shape) - union

    tp, fp, fn, tn = np.sum(tp_array), np.sum(fp_array), np.sum(fn_array), np.sum(tn_array)

    smooth = 0.001
    precision = tp / (pred_sum + smooth)
    recall = tp / (gdth_sum + smooth)

    false_positive_rate = fp / (fp + tn + smooth)
    false_negtive_rate = fn / (fn + tp + smooth)

    jaccard = intersection_sum / (union_sum + smooth)
    dice = 2 * intersection_sum / (gdth_sum + pred_sum + smooth)

    print(false_positive_rate)
    print(false_negtive_rate)
    print(dice)