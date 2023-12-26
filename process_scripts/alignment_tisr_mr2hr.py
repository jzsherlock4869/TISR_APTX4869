import numpy as np
import cv2
import os
import os.path as osp
from glob import glob
from tqdm import tqdm

from utils_align import eccAlign

# ============= 2023 settings ============== #
# lq_dir = "/home/datasets/TISR23/tk1/challengedataset/train/320_axis_mr"
# hr_dir = "/home/datasets/TISR23/tk1/challengedataset/train/640_flir_hr"

lq_dir = "/home/datasets/TISR23/tk1/challengedataset/validation/320_axis_mr"
hr_dir = "/home/datasets/TISR23/tk1/challengedataset/validation/640_flir_hr"


postfix = 'ecc_aligned'
n_iter = 1000
ter_eps = 1e-8
# ============= settings ============== #

aligned_dir = lq_dir.rstrip('/') + f'_{postfix}'
os.makedirs(aligned_dir, exist_ok=True)
print(aligned_dir)

lq_fnames = sorted(glob(osp.join(lq_dir, '*.jpg')))
hr_fnames = sorted(glob(osp.join(hr_dir, '*.jpg')))

assert len(lq_fnames) == len(hr_fnames), "lq num: {}, hr num: {}".format(len(lq_fnames), len(hr_fnames))

pbar = tqdm(range(len(lq_fnames)), total=len(lq_fnames))
# for idx in range(len(lq_fnames)):
for idx in pbar:
    lq_fname = lq_fnames[idx]
    hr_fname = hr_fnames[idx]
    imid = osp.basename(hr_fname).split('.')[0]
    assert osp.basename(lq_fname).startswith(imid), f'Unpaired images {osp.basename(hr_fname)} and {osp.basename(lq_fname)}'
    lq = cv2.imread(lq_fname)[:, :, :] # keep BGR
    hr = cv2.imread(hr_fname)[:, :, :]
    hr_down = cv2.resize(hr, dsize=(lq.shape[1], lq.shape[0]), interpolation=cv2.INTER_AREA)
    lq_t, _ = eccAlign(hr_down, lq, number_of_iterations=n_iter, termination_eps=ter_eps)
    save_path = osp.join(aligned_dir, osp.basename(lq_fname))
    cv2.imwrite(save_path, lq_t)
    pbar.set_postfix(Image_id=idx, FileName=osp.basename(lq_fname))

print("[jzsherlock] all done, everything ok")
