# -*- coding: utf-8 -*-
import copy
from time import time 
from datetime import datetime
import numpy as np 
import pandas as pd
import torch
import pickle
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from sklearn.metrics import roc_curve, auc

torch.manual_seed(2)  
np.random.seed(3)
from argparse import ArgumentParser
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder

from collections import deque
import logging

logging.basicConfig(filename='output.txt', level=logging.DEBUG, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def print_to_log(message):
    #print(message)  # if you need print all to shell. Please delete the '#'
    logger.info(message)
def array_to_flat_string(array):
    array = np.round(array, 3)
    return ', '.join(str(x) for x in array.flat)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

parser = ArgumentParser(description='MolTrans Training.')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, metavar='LR', help='initial learning rate', dest='lr')

def get_task():
    return './dataset/sBioSNAP'

def test(data_generator, model, flag_test):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    for i in range(flag_test):
        d, p, d_mask, p_mask, label = data_generator.popleft()
        score = model(d.cuda(), p.cuda(), d_mask.long().cuda(), p_mask.long().cuda())

        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()

        label = label.float().cuda()
        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()
        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

        data_generator.append((d, p, d_mask, p_mask, label))

    if count > 0:
        loss = loss_accumulate / count
    else:
        loss = 0 

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr)
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    thred_optim = thresholds[5:][np.argmax(f1[5:])]
    
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_label, y_pred)
    
    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
    auc_k = auc(fpr, tpr)

    cm1 = confusion_matrix(y_label, y_pred_s)
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])

    cm1 = confusion_matrix(y_label, y_pred_s)
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])

    message = ('Initial Testing sensitivity1: {:.4f} , specificity1: {:.4f} '.format(sensitivity1, specificity1))
    print_to_log(message)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), y_pred, loss.item(), tpr, fpr, accuracy1, precision_score(y_label, y_pred_s), recall_score(y_label, y_pred_s), str(thred_optim)

def test2(data_generator, model, flag_test):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    for i in range(flag_test):
        d, p, d_mask, p_mask, label = data_generator.popleft()
        score = model(d.cuda(), p.cuda(), d_mask.long().cuda(), p_mask.long().cuda())

        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()

        label = label.float().cuda()
        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()
        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

        data_generator.append((d, p, d_mask, p_mask, label))

    if count > 0:
        loss = loss_accumulate / count
    else:
        loss = 0 

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr)
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    thred_optim = thresholds[5:][np.argmax(f1[5:])]
    
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_label, y_pred)
    
    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
    auc_k = auc(fpr, tpr)

    cm1 = confusion_matrix(y_label, y_pred_s)
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    
    print_to_log('--------------------▼--------------------')
    print_to_log('--------------------▼--------------------')
    print_to_log(fpr)
    print_to_log(tpr)
    array_str1 = array_to_flat_string(recall_curve)
    array_str2 = array_to_flat_string(precision_curve) 
    print_to_log(array_str1)
    print_to_log(array_str2)
    print_to_log('there are my evaluations......')
    print_to_log(str(thred_optim))
    print_to_log(thred_optim)
    print_to_log(precision_score(y_label, y_pred_s))
    
    print_to_log(recall_score(y_label, y_pred_s))
    print_to_log(f1_score(y_label, y_pred_s))
    print_to_log(accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print_to_log(sensitivity1)
    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print_to_log(specificity1)
    print_to_log('There were my evaluation')
    print_to_log(y_pred)
    print_to_log('--------------------▲--------------------')
    print_to_log('--------------------▲--------------------')

    cm1 = confusion_matrix(y_label, y_pred_s)
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])

    message = ('Initial Testing sensitivity1: {:.4f} , specificity1: {:.4f} '.format(sensitivity1, specificity1))
    print_to_log(message)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), y_pred, loss.item(), tpr, fpr, accuracy1, precision_score(y_label, y_pred_s), recall_score(y_label, y_pred_s), str(thred_optim)

def save_data_to_file(deque_obj, variable_value, filename):
    data_to_save = {'deque': deque_obj, 'variable': variable_value}
    with open(filename, 'wb') as file:
        pickle.dump(data_to_save, file)

def load_data_from_file(filename):
    try:
        with open(filename, 'rb') as file:
            loaded_data = pickle.load(file)
            loaded_queue = loaded_data.get('deque', deque())
            loaded_variable = loaded_data.get('variable', 0)
            return loaded_queue, loaded_variable
    except FileNotFoundError:
        return deque(), 0  

def main():
    message='test'
    print_to_log(message)
    config = BIN_config_DBPE()
    args = parser.parse_args()
    config['batch_size'] = args.batch_size

    loss_history = []

    model = BIN_Interaction_Flat(**config)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('--- Data Preparation ---')
    print_to_log('--- Data Preparation ---')
    params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.workers, 'drop_last': True}

    dataFolder = get_task()

    df_train = pd.read_csv(dataFolder + '/train.csv')
    df_val = pd.read_csv(dataFolder + '/val.csv')
    df_test = pd.read_csv(dataFolder + '/test.csv')

    training_set = BIN_Data_Encoder(df_train.index.values, df_train.Label.values, df_train)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = BIN_Data_Encoder(df_val.index.values, df_val.Label.values, df_val)
    validation_generator = data.DataLoader(validation_set, **params)

    testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)

    max_auc = 0
    model_max = copy.deepcopy(model)

    data_queue_test = deque()
    bench_flag_test = 0
    data_queue = deque()
    bench_flag = 0
    data_queue_val = deque()
    bench_flag_val = 0
    data_queue_test, bench_flag_test = load_data_from_file('./data_queue_val-12-9.pkl')
    data_queue, bench_flag = load_data_from_file('./data_queue_val-12-9.pkl')
    data_queue_val, bench_flag_val = load_data_from_file('./data_queue_val-12-9.pkl')

    if bench_flag_test == 0:
        for i, (d, p, d_mask, p_mask, label) in enumerate(testing_generator):
            print(d.shape)
            print(p.shape)
            data_queue_test.append((d, p, d_mask, p_mask, label))
            bench_flag_test += 1
        save_data_to_file(data_queue_test, bench_flag_test, './data_queue_test.pkl')

    if bench_flag == 0:
        for i, (d, p, d_mask, p_mask, label) in enumerate(training_generator):
            print(d.shape)
            print(p.shape)
            data_queue.append((d, p, d_mask, p_mask, label))
            bench_flag += 1
        save_data_to_file(data_queue, bench_flag, './data_queue.pkl')

    if bench_flag_val == 0:
        for i, (d, p, d_mask, p_mask, label) in enumerate(validation_generator):
            print(d.shape)
            print(p.shape)
            data_queue_val.append((d, p, d_mask, p_mask, label))
            bench_flag_val += 1
        save_data_to_file(data_queue_val, bench_flag_val, './data_queue_val.pkl')

    max_auc = 0
    model_max = copy.deepcopy(model)

    with torch.set_grad_enabled(False):
        auc, auprc, f1, logits, loss, tpr, fpr, acc, pre, recall, thresholds = test(data_queue_test, model_max, bench_flag_test)
        message = ('Initial Testing AUROC: {:.2f} , AUPRC: {:.2f} , F1: {:.2f} , Test loss: {:.2f}'.format(auc, auprc, f1, loss))
        print_to_log(message)

    print('--- Go for Training ---')
    print_to_log('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True

    epoch_list=[]
    loss_0_list=[]
    loss_100_list=[]
    acc_list=[]
    pre_list=[]
    recall_list=[]
    tpr_list=[]
    fpr_list=[]
    auc_list=[]
    auprc_list=[]
    thr_list=[]
    f1_list=[]

    epo_time_list=[]

    for epo in range(args.epochs):      
        start = time()
        model.train()
        if len(data_queue) > 0:
            for i in range(bench_flag):
                d, p, d_mask, p_mask, label = data_queue.popleft()
                score = model(d.to(device), p.to(device), d_mask.long().to(device), p_mask.long().to(device))
                label = label.float().cuda()
                loss_fct = torch.nn.BCELoss()
                m = torch.nn.Sigmoid()
                n = torch.squeeze(m(score))

                loss = loss_fct(n, label)
                loss_history.append(loss)

                opt.zero_grad()
                loss.backward()
                opt.step()
                data_queue.append((d, p, d_mask, p_mask, label))  

                if (i % 100 == 0):
                    lo=loss.cpu().detach().numpy()
                    if i==0:
                        loss_0_list.append(lo)
                    elif i==100:
                        loss_100_list.append(lo)
                        
                    message = 'Training at Epoch {} iteration {} with loss {:.4f}'.format(epo + 1, i, lo)
                    print_to_log(message)

                    epoch_list.append(epo+1)

        else:
            for i, (d, p, d_mask, p_mask, label) in enumerate(training_generator):
                data_queue.append((d, p, d_mask, p_mask, label))
                bench_flag = bench_flag + 1
            for i in range(bench_flag):
                d, p, d_mask, p_mask, label = data_queue.popleft()
                score = model(d.to(device), p.to(device), d_mask.long().to(device), p_mask.long().to(device))
                label = label.float().cuda()
                loss_fct = torch.nn.BCELoss()
                m = torch.nn.Sigmoid()
                n = torch.squeeze(m(score))

                loss = loss_fct(n, label)
                loss_history.append(loss)

                opt.zero_grad()
                loss.backward()
                opt.step()

                data_queue.append((d, p, d_mask, p_mask, label))

                if (i % 100 == 0):
                    lo=loss.cpu().detach().numpy()
                    if i==0:
                        loss_0_list.append(lo)
                    elif i==100:
                        loss_100_list.append(lo)

                    message = 'Training at Epoch {} iteration {} with loss {:.4f}'.format(epo + 1, i, lo)
                    print_to_log(message)

                    epoch_list.append(epo+1)
        
        # every epoch test
        with torch.set_grad_enabled(False):
            auc, auprc, f1, logits, loss, tpr, fpr, acc, pre, recall, thresholds  = test(data_queue_val, model_max, bench_flag_val)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            acc_list.append(acc)
            pre_list.append(pre)
            recall_list.append(recall)
            auc_list.append(auc)
            auprc_list.append(auprc)
            thr_list.append(thresholds)
            f1_list.append(f1)
            if auc > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = auc
            message = ('Validation at Epoch {} , AUROC: {:.2f} , AUPRC: {:.2f} , F1: {:.2f}'.format(epo + 1, auc, auprc, f1))
            print_to_log(message)
        end = time()
        epo_time_list.append((end-start))
    del data_queue

    torch.save(model_max.state_dict(), 'dict_model.pth') 

    print('--- Go for Testing ---')
    print_to_log('--- Go for Testing ---')
    try:
        with torch.set_grad_enabled(False):
            auc, auprc, f1, logits, loss, tpr, fpr, acc, pre, recall, thresholds = test2(data_queue_test, model_max, bench_flag_test)
            print('this is thresholds :',thresholds)
            print_to_log(thresholds)
            message = ('Testing AUROC: {:.4f} , AUPRC: {:.4f} , F1: {:.2f} , Test loss: {:.4f}'.format(auc, auprc, f1, loss))
            print_to_log(message)
            print(message)
    except:
        print_to_log('testing failed')

    print('Finished !')
    max_length = max(len(tpr) for tpr in tpr_list)
    tpr_list_adjusted = [np.interp(np.linspace(0, 1, max_length), np.linspace(0, 1, len(tpr)), tpr) for tpr in tpr_list]
    fpr_list_adjusted = [np.interp(np.linspace(0, 1, max_length), np.linspace(0, 1, len(fpr)), fpr) for fpr in fpr_list]
    auc_value = np.mean(auc_list)
    tpr_all = np.mean(tpr_list_adjusted, axis=0)
    fpr_all = np.mean(fpr_list_adjusted, axis=0)
    print("Every epoch time",epo_time_list)
    return model_max, loss_history

s = time()
model_max, loss_history = main()
e = time()
print(e - s)