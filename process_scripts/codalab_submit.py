import enum
from logging import root
import os
from datetime import datetime
from glob import glob
import sys
import cv2

# ======================================== #
track1_exp = 'x4_test_000_swinir_baseline'
track2_exp = 'x2_test_000_swinir_baseline'
no_submit = 1
# ======================================== #

track1_dir = os.path.join('../results', track1_exp)
track2_dir = os.path.join('../results', track2_exp)

now = datetime.now().strftime('%y_%m_%d') + f'_v{no_submit}'
os.makedirs(f'../__submit/{now}', exist_ok=True)

root_dir = f'../__submit/{now}'
eval1_dir = os.path.join(root_dir, 'evaluation1/x4')
eval2_dir = os.path.join(root_dir, 'evaluation2/x2')
os.makedirs(eval1_dir)
os.makedirs(eval2_dir)

track1_imgs = sorted(glob(os.path.join(track1_dir, 'visualization', 'ThermalX4Test', '*.png')))
track2_imgs = sorted(glob(os.path.join(track2_dir, 'visualization', 'ThermalX2Test', '*.png')))

for imid, img_path in enumerate(track1_imgs):
    target_imname = f'ev1_{imid+1:03}.jpg'
    assert int(os.path.basename(img_path).split('_')[1]) == imid + 1
    img = cv2.imread(img_path)
    cv2.imwrite(os.path.join(eval1_dir, target_imname), img)
    print(imid, img_path, target_imname, eval1_dir)

for imid, img_path in enumerate(track2_imgs):
    target_imname = f'ev2_{imid+1:03}.jpg'
    assert int(os.path.basename(img_path).split('_')[0]) == imid + 1
    img = cv2.imread(img_path)
    cv2.imwrite(os.path.join(eval2_dir, target_imname), img)

print("[jzsherlock] all done, everything ok")