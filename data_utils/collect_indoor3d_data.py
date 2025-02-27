import os
import sys
from indoor3d_util import DATA_PATH, collect_point_label
'''---------------------------给每个点获取标签并保存文件------------------'''
#os.path.dirname()获取父级目录
#os.path.abspath(__file__)将 __file__ 转换为它的绝对路径，即当前文件的完整路径  __file__ 是当前脚本的路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#提取给定路径的目录部分
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
# DATA_PATH = os.path.join(ROOT_DIR, 'data','xuGong_data', 'Stanford3dDataset_v1.2_Aligned_Version')
#rstrip() 方法用于去除字符串末尾的 空白字符（包括空格、换行符 \n、制表符等）
#Area_1/copyRoom_1/Annotations 数据文件名
anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
#每一个点云的文件路径
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]
print(anno_paths)
output_folder = os.path.join(ROOT_DIR, 'data/stanford_indoor3d')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in anno_paths:
    print(anno_path)
    try:
        elements = anno_path.split('/')
        out_filename = elements[-3]+'_'+elements[-2]+'.npy' # Area_1_hallway_1.npy
        #输入（单个点云路径，要设置标签的文件路径）保存的是一整个点云，如hallway_1
        collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
    except:
        print(anno_path, 'ERROR!!')
