import os
import shutil
import glob
# 设置文件夹路径
folder_path = "/home/xds/PycharmProjects/pythonProject1/xuGong/data/xuGong_data/Stanford3dDataset_v1.2_Aligned_Version/Area_1"
print(os.path.exists(folder_path))
pattern = '*/'
folders = glob.glob(os.path.join(folder_path,pattern))
print(f"folders{folders}")
for folder in folders:

    annotations_folder = os.path.join(folder, "Annotations")
# 如果Annotations文件夹不存在，创建它
    if not os.path.exists(annotations_folder):
        os.makedirs(annotations_folder)

# 获取文件夹中的所有pcd文件
    pcd_files = [f for f in os.listdir(folder) if f.endswith('.pcd')]

# 移动前三个pcd文件到Annotations文件夹
    for pcd_file in pcd_files:  # 只移动前三个文件
        if pcd_file[0].isdigit():
            continue
        print(pcd_file)
        src = os.path.join(folder, pcd_file)
        dest = os.path.join(annotations_folder, pcd_file)
        shutil.move(src, dest)