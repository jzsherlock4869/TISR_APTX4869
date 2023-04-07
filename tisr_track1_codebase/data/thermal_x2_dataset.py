import cv2
import os
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.degradations import add_jpg_compression
from basicsr.data.transforms import augment, mod_crop, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY

from glob import glob

"""
Track 2: paired MR and HR as lq and gt
        [240 x 320] -> [480 x 640]    x2
        note: in training, considering boarder caused by alignment
"""

@DATASET_REGISTRY.register()
class ThermalX2Dataset(data.Dataset):
    """Example dataset.

    1. Read GT image
    2. Read corresonding LQ (Low Quality) image

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(ThermalX2Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        # it now only supports folder mode, for other modes such as lmdb and meta_info file, please see:
        # https://github.com/xinntao/BasicSR/blob/master/basicsr/data/
        # self.paths = [os.path.join(self.gt_folder, v) for v in list(scandir(self.gt_folder))]
        self.gt_paths = sorted(glob(os.path.join(self.gt_folder, '*.jpg')))
        self.lq_paths = sorted(glob(os.path.join(self.lq_folder, '*.jpg')))
        assert len(self.gt_paths) == len(self.lq_paths)

        self.boarder_max_h = 20  # one-side boarder max, shrinked by 2*boarder_max_h in height
        self.boarder_max_w = 40  # same as above

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.gt_paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        
        lq_path = self.lq_paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        if self.opt['phase'] == 'train':
            img_gt = img_gt[self.boarder_max_h * scale: -self.boarder_max_h * scale,
                        self.boarder_max_w * scale: -self.boarder_max_w * scale, :]
            img_lq = img_lq[self.boarder_max_h: -self.boarder_max_h,
                            self.boarder_max_w: -self.boarder_max_w, :]

        # generate lq image
        # downsample
        H, W = img_gt.shape[0:2]
        h, w = img_lq.shape[0:2]
        assert H % scale == 0 and W % scale == 0, f'size ({H}, {W}) not divided by {scale}'
        assert H // h == scale and W // w == scale, f'H, W, h, w, scale: {H}, {W}, {h}, {w}, {scale}'

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': gt_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.gt_paths)
