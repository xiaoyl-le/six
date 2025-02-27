import argparse
import os

from matplotlib import pyplot as plt

from data_utils.S3DISDataLoader import S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
#excavator ground soil

classes = ['excavator', 'ground', 'soil']
#enumerate(classes) 是一个内建函数，它返回一个枚举对象。这个对象会生成一对一对的元素  索引i 值 cls
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat  #{ 0: "chair"}

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_sem_seg', help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=2, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--seg_path', type=str, default='/home/xds/PycharmProjects/pythonProject1/xuGong/data/stanford_indoor3d',
                                        help='Segment Data Path')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    #datetime.datetime.now() 获取当前的日期和时间。
    #strftime('%Y-%m-%d_%H-%M') 将当前日期和时间格式化为一个字符串，格式为 年-月-日_小时-分钟（例如：2025-02-18_14-30）
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
     #如果目录已经存在，则不抛出异常（exist_ok=True）
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    
    # Create TensorBoard writer
    tb_dir = experiment_dir.joinpath('tensorboard/')
    tb_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str("xugon_logs"))

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    ''' %(asctime)s: 日志记录的时间戳
        %(name)s: 日志记录器的名字
        %(levelname)s: 日志级别（如 INFO、ERROR 等）
        %(message)s: 实际的日志消息
    '''
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    #root = '/home/xds/PycharmProjects/pythonProject1/Pointnet_Pointnet2_pytorch-master/data/stanford_indoor3d'
    NUM_CLASSES = 3
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    '''
        batch_size：控制每个批次包含多少个样本。这里的 BATCH_SIZE 变量定义了这个值。
        shuffle：当设置为 True 时，数据在每个 epoch 开始时都会被打乱，这样可以避免模型每次看到数据的顺序相同。
        num_workers：指定用于加载数据的子进程数量。在这个例子中，设置为 10，可以提高数据加载的性能，特别是在处理大型数据集时。
        pin_memory：当设置为 True 时，会将数据加载到固定内存中，这样可以加速 GPU 上的数据传输，尤其是在使用 CUDA 时。
        drop_last：当设置为 True 时，如果最后一个批次的数据量少于 batch_size，则丢弃这个批次。
        worker_init_fn：这是一个初始化每个工作线程的函数。这里使用了 lambda 函数，基于当前时间和线程索引来设置每个工作线程的随机种子，以确保每个线程的随机数生成器是独立的，从而避免不同工作线程之间的随机数冲突。
    '''
    print("start loading training data ...")
    TRAIN_DATASET = S3DISDataset(split='train', data_root=args.seg_path, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    print("start loading test data ...")
    TEST_DATASET = S3DISDataset(split='test', data_root=args.seg_path, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True)
    #表示每个类别标签的权重值。通常这些权重用于加权损失函数中的各类别的贡献，例如在类别不平衡时使用较大的权重来增强少数类的影响
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    #动态导入指定的模型模块，类似于import
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).cuda()#模型
    criterion = MODEL.get_loss().cuda()#损失函数
    #直接运行relu，改变张量，而不是生成一个新的变量
    #apply 方法会自动将模型中的每一层作为参数传递给 inplace_relu
    classifier.apply(inplace_relu)#应用relu

    #初始化权重
    def weights_init(m):
        #classname 是通过 m.__class__.__name__ 获取的模块名，表示当前被遍历到的每一层的类名
        classname = m.__class__.__name__
        #返回子字符串 'Conv2d' 在类名中的位置索引
        if classname.find('Conv2d') != -1:
            #xavier_normal_ 初始化会使权重值服从正态分布，均值为 0，方差为 2/(输入节点数 + 输出节点数)，
            # 这样可以让输入和输出的方差保持一致，避免梯度消失或爆炸问题
            torch.nn.init.xavier_normal_(m.weight.data)
            #torch.nn.init.constant_ 将偏置项初始化为常数，这里设为 0.0
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        #预训练模型的权重参数
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)
    #更新momentum
    def bn_momentum_adjust(m, momentum):
        #isinstance(m, ...)：检查 m 是否是指定类型的实例
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum
    #Learning Rate Clip，用于限制学习率最小值的常量
    LEARNING_RATE_CLIP = 1e-5
    '''MOMENTUM_ORIGINAL 定义了动量的初始值，用于优化算法中的动量参数。
    动量是一种加速梯度下降的方法，通过累积之前梯度的方向来减少振荡，使得梯度更新更为稳定。特别是在深度的损失面中，动量能够帮助模型快速跨越平坦的区域。
    初始动量设置为 0.1，意味着训练刚开始时，动量的影响较小，主要依赖于当前的梯度更新。
    '''
    MOMENTUM_ORIGINAL = 0.1
    #表示每经过一个特定的步长后，动量会乘以这个因子来减少
    MOMENTUM_DECCAY = 0.5
    #决定了每隔多少个 epoch 才会对动量进行一次调整
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0
    
    # Lists to store metrics for final plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_ious = []
    val_accuracies = []

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        # ** 代表削弱了几次
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        #遍历优化器的所有参数组 (param_groups)，并将它们的学习率设置为当前计算出的值 (lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        #x 代表模型中的每个子模块（例如卷积层、批量归一化层等）
        #apply 方法会自动将模型中的每一层作为参数传递给 inplace_relu，由于存在momentum所以需要lambda
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        #启用 Dropout 层和 BatchNorm 层的训练行为
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            #对点云进行随机旋转，以便增加数据的多样性（数据增强），使模型对旋转更加鲁棒
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            '''
            对于二维卷积，输入的格式通常是 (batch_size, channels, height, width)；而对于一维卷积，输入格式为 (batch_size, channels, length)。
            在 PointNet 中，点云处理类似于对每个点应用一维卷积操作，而点的特征（如 x, y, z）作为通道。因此，需要将通道维度置于第二维，以便使用卷积层处理这些点的特征
            '''
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            #.contiguous() 用于将张量的内存布局变为连续的，
            #将 seg_pred 调整为符合交叉熵损失函数要求的形状，使每个点的预测结果能够逐点与真实标签比较，计算损失
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            #将二维张量 (65536, 1) 转换为一维张量 (65536,)
            #[:, 0]：这个操作会将二维张量中的第一个维度（列）提取出来，实际上就是把标签转换为一维数组的格式
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            #用于计算模型参数的梯度
            loss.backward()
            optimizer.step()
            #max(1) 返回两个值：最大值（即最大概率）。最大值所在的索引（即预测的类别标签）。
            #max（1），表示第一维度，就是每一个点，0表示第0维度，就是所有点
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
            
            # Log batch loss to TensorBoard
            writer.add_scalar('Batch/Train_Loss', loss.item(), epoch * num_batches + i)
            
        # Calculate epoch metrics
        train_loss = loss_sum / num_batches
        train_accuracy = total_correct / float(total_seen)
        
        # Add to tracking lists
        train_losses.append(train_loss.item())
        train_accuracies.append(train_accuracy)
        
        # Log epoch metrics to TensorBoard
        writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Train_Accuracy', train_accuracy, epoch)
        
        log_string('Training mean loss: %f' % train_loss)
        log_string('Training accuracy: %f' % train_accuracy)

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                #classifier.state_dict() 保存的是模型 classifier 的所有参数，包括每一层的权重和偏置等
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                #使用了 np.histogram() 函数，用于统计 batch_label 中不同类别标签的数量分布。它返回了一个直方图，其中包含了各类别在 batch_label 中出现的次数
                #NUM_CLASSES + 1 ：边界值的数量
                #0 到 1 对应类别 0，1 到 2 对应类别 1，2 到 3 对应类别 2，等等
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
                    
                # Log batch validation loss to TensorBoard
                writer.add_scalar('Batch/Val_Loss', loss.item(), epoch * num_batches + i)
                
            #是对 labelweights 中的各类别数量进行归一化，使得它们的总和为 1，从而得到每个类别的相对权重
            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            #IoU，即Intersection over Union，用于衡量预测区域和实际区域之间的重合程度，是计算语义分割模型性能的一个重要指标
            #mIoU 是对所有类别的 IoU 取平均值
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
            
            # Calculate validation metrics
            val_loss = loss_sum / float(num_batches)
            val_accuracy = total_correct / float(total_seen)
            
            # Add to tracking lists
            val_losses.append(val_loss.item())
            val_ious.append(mIoU)
            val_accuracies.append(val_accuracy)
            
            # Log epoch validation metrics to TensorBoard
            writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            writer.add_scalar('Epoch/Val_Accuracy', val_accuracy, epoch)
            writer.add_scalar('Epoch/Val_mIoU', mIoU, epoch)
            
            # Log class-wise IoU to TensorBoard
            for l in range(NUM_CLASSES):
                class_iou = total_correct_class[l] / float(total_iou_deno_class[l])
                writer.add_scalar(f'Class_IoU/{seg_label_to_cat[l]}', class_iou, epoch)
            
            log_string('eval mean loss: %f' % val_loss)
            log_string('eval point avg class IoU: %f' % (mIoU))
            #总的正确率，对于每一个点
            log_string('eval point accuracy: %f' % val_accuracy)
            #类的正确率，对于每一个类的点
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % val_loss)
            log_string('Eval accuracy: %f' % val_accuracy)

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath) 
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1
        
    # Create and save final metrics plots
    plot_dir = experiment_dir.joinpath('plots/')
    plot_dir.mkdir(exist_ok=True)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(str(plot_dir) + '/loss_curves.png')
    plt.close()
    
    # Plot validation mIoU
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(val_ious)), val_ious)
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('Validation mIoU')
    plt.savefig(str(plot_dir) + '/miou_curve.png')
    plt.close()
    
    # Plot training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_accuracies)), train_accuracies, label='Training Accuracy')
    plt.plot(range(len(val_accuracies)), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(str(plot_dir) + '/accuracy_curves.png')
    plt.close()
    
    # Close TensorBoard writer
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)