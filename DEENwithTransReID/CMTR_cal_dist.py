from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, LLCMData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net
from model_ViT import TransReID, vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from utils import *
import pdb
import scipy.io
import torchvision
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: llcm, regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline: resnet50')
parser.add_argument('--resume', '-r', default='llcm_cmtr_lr_p4_n8_lr_0.001_seed_0_AdamW_best.t', type=str, help='resume from checkpoint')
# parser.add_argument('--resume', '-r', default='sysu_cmtr_lr_p4_n8_lr_0.001_seed_0_AdamW_best.t', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model_bak/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--log_dist_path', default='log_dist/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log_dist/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
# parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
# parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--img_w', default=128, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=256, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=16, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=32, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='awg', type=str, metavar='m', help='method type: base or awg')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu') # SYSU-MM01
parser.add_argument('--tvsearch', default=True, help='whether thermal to visible search on RegDB') # RegDB

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset
mode = args.mode
if dataset == 'sysu':
    data_path = '../Datasets/SYSU-MM01/'
    # data_path = '/home/guohangyu/data/datasets/SYSU-MM01'
    n_class = 395
    test_mode = [1, 2]
    pool_dim = 768
elif dataset =='regdb':
    data_path = '../Datasets/RegDB/'
    # data_path = '/home/guohangyu/data/datasets/RegDB'
    n_class = 206
    test_mode = [1, 2]
    pool_dim = 768
elif dataset =='llcm':
    data_path = '../Datasets/LLCM/'
    # data_path = '/home/guohangyu/data/datasets/LLCM'
    n_class = 713
    # test_mode = [1, 2]  # [1, 2]: IR to VIS; [2, 1]: VIS to IR;
    test_mode = [1, 2]  # [1, 2]: IR to VIS; [2, 1]: VIS to IR;
    pool_dim = 768
 
# json file path
suffix = args.log_dist_path + 'CMTR_lr/' 
if dataset == 'llcm':
    suffix = suffix + 'llcm/'
    if not os.path.isdir(suffix):
        os.makedirs(suffix)
    if test_mode == [1, 2]:
        suffix = suffix + 'I2V'
    elif test_mode == [2, 1]:
        suffix = suffix + 'V2I'
elif dataset == 'sysu':
    suffix = suffix + 'sysu/'
    if not os.path.isdir(suffix):
        os.makedirs(suffix)
    if mode == 'all':
        suffix = suffix + 'all'
    elif mode == 'indoor':
        suffix = suffix + 'indoor'
elif dataset == 'regdb':
    suffix = suffix + 'regdb/'
    if not os.path.isdir(suffix):
        os.makedirs(suffix)
    if test_mode == [1, 2]:
        suffix = suffix + 'I2V'
    elif test_mode == [2, 1]:
        suffix = suffix + 'V2I'

path = suffix + '.json'

batch = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 
print('==> Building model..')
net = vit_base_patch16_224_TransReID(n_class, dataset)
net.to(device)    

cudnn.benchmark = True

checkpoint_path = args.model_path

if args.method =='id':
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h,args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip



def extract_gall_feat(gall_loader):
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, pool_dim))
    gall_feat_att = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            if batch_idx == batch:
                break
            batch_num = input.size(0)
            input = Variable(input.cuda())
            # feat, feat_att = net(input, input, test_mode[0])
            feat, feat_att, out = net(input)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat, gall_feat_att
    
def extract_query_feat(query_loader):
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, pool_dim))
    query_feat_att = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            if batch_idx == batch:
                break
            batch_num = input.size(0)
            input = Variable(input.cuda())
            # feat, feat_att = net(input, input, test_mode[1])
            feat, feat_att, out = net(input)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num       
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat, query_feat_att


if dataset == 'llcm':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    # queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat, query_feat_att = extract_query_feat(query_loader)

    dist_dict = {}

    for trial in range(1):
        gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=trial)

        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat, gall_feat_att = extract_gall_feat(trial_gall_loader)

        # 截取前batch*args.test_batch个特征
        query_feat = query_feat[:batch*args.test_batch]
        query_feat_att = query_feat_att[:batch*args.test_batch]
        gall_feat = gall_feat[:batch*args.test_batch]
        gall_feat_att = gall_feat_att[:batch*args.test_batch]

        feat = np.concatenate((query_feat, gall_feat), axis=0)
        feat_att = np.concatenate((query_feat_att, gall_feat_att), axis=0)

        distmat = np.matmul(feat, np.transpose(feat))
        distmat_att = np.matmul(feat_att, np.transpose(feat_att))

        a = 0.1
        dist1= distmat + distmat_att
        dist2 = a * distmat + (1 - a) * distmat_att
        
        dist_dict = {}
                    
        # 遍历每个query图像
        # for i in range(nquery):
        for i in range(batch*args.test_batch):
            id = int(query_label[i])
            cam = int(query_cam[i])
            dist_dict[i] = {'query_id': id, 'query_cam': cam, 'dist': []}

            # 遍历每个query与query和gallery的距离
            # for j in range(nquery+ngall)
            num = 0
            for j in range(batch*args.test_batch*2):
                # if j < nquery:
                if j < batch*args.test_batch:
                    id = int(query_label[j])
                    cam = int(query_cam[j])
                    dist_tmp = dist1[i][j]
                    dist_dict[i]['dist'].append({'num': num, 'query_id': id, 'query_cam': cam, 'dist': dist_tmp})
                else:
                    # id = gall_label[j-nquery]
                    # cam = gall_cam[j-nquery]
                    id = int(gall_label[j-batch*args.test_batch])
                    cam = int(gall_cam[j-batch*args.test_batch])
                    dist_tmp = dist1[i][j]
                    dist_dict[i]['dist'].append({'num': num, 'gall_id': id, 'gall_cam': cam, 'dist': dist_tmp})
                num = num+1

        # 将 dist_dict 转换为 JSON 格式的字符串，注意dist_dict中有int64类型的数据，需要转换为int类型
        # dist_dict_int = {}
        # for key, value in dist_dict.items():
        #     dist_dict_int[key] = {}
        #     dist_dict_int[key]['id'] = int(value['id'])
        #     dist_dict_int[key]['cam'] = int(value['cam'])
        #     dist_dict_int[key]['dist'] = []
        #     for item in value['dist']:
        #         if 'query_id' in item:
        #             item['query_id'] = int(item['query_id'])
        #             item['query_cam'] = int(item['query_cam'])
        #         elif 'gall_id' in item:
        #             item['gall_id'] = int(item['gall_id'])
        #             item['gall_cam'] = int(item['gall_cam'])
        #         item['dist'] = float(item['dist'])
        #         dist_dict_int[key]['dist'].append(item)

        # json_str = json.dumps(dist_dict_int, indent=4)


        json_str = json.dumps(dist_dict, indent=4)
        # 将 JSON 格式的字符串写入文件
        with open(path, 'w') as f:
            f.write(json_str)
        
        plt.imshow(dist1, cmap='Blues', interpolation='nearest')
        # 设置 x 轴和 y 轴的间隔
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(2))
        plt.colorbar()
        img_path = suffix + '_dist.png'
        plt.savefig(img_path)

elif dataset == 'sysu':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat, query_feat_att = extract_query_feat(query_loader)

    dist_dict = {}
    
    for trial in range(1):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial)

        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat, gall_feat_att = extract_gall_feat(trial_gall_loader)


         # 截取前batch*args.test_batch个特征
        query_feat = query_feat[:batch*args.test_batch]
        query_feat_att = query_feat_att[:batch*args.test_batch]
        gall_feat = gall_feat[:batch*args.test_batch]
        gall_feat_att = gall_feat_att[:batch*args.test_batch]

        feat = np.concatenate((query_feat, gall_feat), axis=0)
        feat_att = np.concatenate((query_feat_att, gall_feat_att), axis=0)

        distmat = np.matmul(feat, np.transpose(feat))
        distmat_att = np.matmul(feat_att, np.transpose(feat_att))

        a = 0.1
        dist1= distmat + distmat_att
        dist2 = a * distmat + (1 - a) * distmat_att
        
        dist_dict = {}
                    
        # 遍历每个query图像
        # for i in range(nquery):
        for i in range(batch*args.test_batch):
            id = int(query_label[i])
            cam = int(query_cam[i])
            dist_dict[i] = {'query_id': id, 'query_cam': cam, 'dist': []}

            # 遍历每个query与query和gallery的距离
            # for j in range(nquery+ngall):
            num = 0
            for j in range(batch*args.test_batch*2):
                # if j < nquery:
                if j < batch*args.test_batch:
                    id = int(query_label[j])
                    cam = int(query_cam[j])
                    dist_tmp = dist1[i][j]
                    dist_dict[i]['dist'].append({'num': num, 'query_id': id, 'query_cam': cam, 'dist': dist_tmp})
                else:
                    # id = gall_label[j-nquery]
                    # cam = gall_cam[j-nquery]
                    id = int(gall_label[j-batch*args.test_batch])
                    cam = int(gall_cam[j-batch*args.test_batch])
                    dist_tmp = dist1[i][j]
                    dist_dict[i]['dist'].append({'num': num, 'gall_id': id, 'gall_cam': cam, 'dist': dist_tmp})
            num = num+1
        
        json_str = json.dumps(dist_dict, indent=4)


        # 将 JSON 格式的字符串写入文件
        with open(path, 'w') as f:
            f.write(json_str)
        
        plt.imshow(dist1, cmap='Blues', interpolation='nearest')
        # 设置 x 轴和 y 轴的间隔
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(5))
        plt.colorbar()
        img_path = suffix + '_dist.png'
        plt.savefig(img_path)
        # plt.show()



elif dataset == 'regdb':

    for trial in range(10):
        test_trial = trial +1
        model_path = checkpoint_path + 'regdb_agw_p4_n6_lr_0.1_seed_0_trial_{}_best.t'.format(test_trial)
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])

        # training set
        trainset = RegDBData(data_path, test_trial, transform=transform_train)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

        # testing set
        query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal='thermal')
        gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal='visible')

        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        nquery = len(query_label)
        ngall = len(gall_label)

        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        print('Data Loading Time:\t {:.3f}'.format(time.time() - end))


        query_feat, query_feat_att = extract_query_feat(query_loader)
        gall_feat, gall_feat_att = extract_gall_feat(gall_loader)

        if args.tvsearch:
            distmat = np.matmul(query_feat, np.transpose(gall_feat))
            distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))
            a = 0.1
            distmat7 = distmat + distmat_att
            distmat8 = a * distmat + (1 - a) * distmat_att

            cmc7, mAP7, mINP7 = eval_regdb(-distmat7, gall_label, query_label)
            cmc8, mAP8, mINP8 = eval_regdb(-distmat8, gall_label, query_label)

        else:
            distmat = np.matmul(query_feat, np.transpose(gall_feat))
            distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))
            a = 0.1
            distmat7 = distmat + distmat_att
            distmat8 = a * distmat + (1 - a) * distmat_att

            cmc7, mAP7, mINP7 = eval_regdb(-distmat7, gall_label, query_label)
            cmc8, mAP8, mINP8 = eval_regdb(-distmat8, gall_label, query_label)

        if trial == 0:
            all_cmc7 = cmc7
            all_mAP7 = mAP7
            all_mINP7 = mINP7

            all_cmc8 = cmc8
            all_mAP8 = mAP8
            all_mINP8 = mINP8

        else:
            all_cmc7 = all_cmc7 + cmc7
            all_mAP7 = all_mAP7 + mAP7
            all_mINP7 = all_mINP7 + mINP7

            all_cmc8 = all_cmc8 + cmc8
            all_mAP8 = all_mAP8 + mAP8
            all_mINP8 = all_mINP7 + mINP8

        print('Test Trial: {}'.format(trial))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc7[0], cmc7[4], cmc7[9], cmc7[19], mAP7, mINP7))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc8[0], cmc8[4], cmc8[9], cmc8[19], mAP8, mINP8))

