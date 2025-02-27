import os
import glob
import numpy as np
import glob
import os
import sys
# 设置文件夹路径和匹配模式
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
folder_path = './data/xuGong_data/Stanford3dDataset_v1.2_Aligned_Version'  # 修改为你的文件夹路径
pattern = '*/'  # 匹配所有子文件夹，'*/' 表示所有文件夹

# 获取所有符合条件的文件夹路径
folders = glob.glob(os.path.join(folder_path,pattern))
folder_path_1 = os.path.join(folder_path,'Area_1')
folders_1 = glob.glob(os.path.join(folder_path_1,pattern))
folder_path_2 = os.path.join(folder_path,'Area_2')
folders_2 = glob.glob(os.path.join(folder_path_2,pattern))
combined_folders = folders_1 + folders_2
print(combined_folders)
output = []
for path in combined_folders:
    # 使用os.path.split获取路径最后两个部分
    path = os.path.normpath(path)
    head, tail = os.path.split(path)  # tail是文件名，head是文件夹路径
    head2, tail2 = os.path.split(head)  # 再次分割head，获取最后两个部分
    output.append(f"{tail2}/{tail}")

output_file = "./data_utils/meta/anno_paths.txt"

# 将结果写入文件
with open(output_file, 'w') as f:
    for line in output:
        f.write(line +"/Annotations"+'\n')