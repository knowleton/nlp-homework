# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network,output_tester
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Computer-based Patient Record Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert,TextCNN,TextRNN')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'CPRTXT'  # 数据集
    '''textcnn和bilstm的embedding'''
    if args.model != 'Bert':
        embedding = 'embedding_SougouNews.npz' #搜狐新闻的预训练embedding
        if args.embedding == 'random':
            embedding = 'random'
    model_name = args.model  #模型参数
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    #记录好时间，导入数据，dataloader
    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)

    #输出作业时候用，记得要把train注释了
    output_tester(config, model, train_iter, dev_iter, test_iter)
