import os
from PIL import Image
import re

""" rp = 'E:\Dataset\KittiDepthCompletion_mannul'
for dirs in os.listdir(rp):
    print(dirs)

logs = '*'*10
print(logs)

for root, dirs, files in os.walk(rp):
    print(root) """

""" # 读取图片
depth_raw = Image.open('E:\\Dataset\\KittiDepthCompletion_mannul\\train\\2011_09_26_drive_0001_sync\\proj_depth\\velodyne_raw\\image_02\\0000000005.png')
depth_gt = Image.open('E:\\Dataset\\KittiDepthCompletion_mannul\\train\\2011_09_26_drive_0001_sync\\proj_depth\\groundtruth\\image_02\\0000000005.png')
  
image = depth_gt

# 展示图片
image.show()
 """
root = 'E:\\Dataset\\KittiDepthCompletion_mannul\\Raw\\RGB_synced_rectified\\city\\train\\2011_09_26_drive_0001_sync\\2011_0926\\2011_09_26_drive_0001_sync\\image_02\\data'
prepath = 'E:\\Dataset\\KittiDepthCompletion_mannul'
date_pattern = r'(\d{4}_\d{2}_\d{2})'
index_pattern = r'drive_(\d{4})_sync'

phase = 'train' if 'train' in root else 'val'
date_match = re.search(date_pattern, root)
if date_match:
    date = date_match.group(1)  # 获取匹配的日期信息
    print('date:', date)
else:
    assert 0


index_match = re.search(index_pattern, root)
if index_match:
    index = index_match.group(1)  # 获取匹配的序号信息
    print('index:', index)
else:
    assert 0
cam = 'image_02' if 'image_02' in root else 'image_03'
# E:\Dataset\KittiDepthCompletion_mannul\train\2011_09_26_drive_0009_sync\proj_depth\velodyne_raw\image_02
match_root = os.path.join(prepath,phase,date+'_'+'drive_' + index + '_sync', 'proj_depth', 'velodyne_raw', cam)
print('match_root:',match_root )