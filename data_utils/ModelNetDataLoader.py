'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    #对点云数据进行标准化的一个实现，它的目标是去除点云数据的偏移并缩放点云，使得点云数据的中心位于原点并且最大距离为 1
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        #-1，沿着最后一个维度求和
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        if self.num_category == 10:
            #文件是每个类别的名字
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        #ine.rstrip()：rstrip() 是字符串方法，用来去除每行末尾的空白字符（包括换行符）。这样确保每行末尾没有不必要的空格或换行符
        #cet是列表，保存每个类的名字
        self.cat = [line.rstrip() for line in open(self.catfile)]
        #zip() 函数将两个序列（self.cat 和 range(len(self.cat))）打包成一个元组的序列，其中每个元组包含一个类别名称和它对应的数字索引
        #dict转换为字典
        '''
        'airplane': 0,
        'car': 1,
        'chair': 2,
        '''
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        '''
        shape_ids = {
        'train': ['airplane_01', 'car_01', 'chair_01'],
        'test': ['airplane_02', 'car_02', 'chair_02']
        }
        '''
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            #读取的是每个点云文件的名称，一个数据集分为训练集和测试集
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        #assert 语句来确保传入的 split 参数的值只能是 'train' 或 'test'。如果 split 的值不是这两者之一，程序会抛出 AssertionError，停止执行
        assert (split == 'train' or split == 'test')
        #'_'.join() 只有在列表中有多个元素时，才会真正插入下划线作为分隔符
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        #datapath是一个元组，(airplane,path)
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.pkl' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.pkl' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    #从文件中获取数据，用逗号隔开，形成一个二维数组
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        #这个过程的核心目标是保证代表性和覆盖性
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls
                with open(self.save_path,'wb') as f:
                    #pickle.dump，以二进制格式保存到一个文件路径 self.save_path 中
                    pickle.dump([self.list_of_points,self.list_of_labels],f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path,'rb') as f:
                    self.list_of_points,self.list_of_labels = pickle.load(f)


    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('../data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
