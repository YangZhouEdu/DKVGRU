import torch
import argparse
from  data_loader import DATA
from model import MODEL
from run import train, test
import numpy as np
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
import  pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='DKVGRU-SUNHAO')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu will be used, e.g "0,1,2,3"')
    parser.add_argument('--max_iter', type=int, default=60, help='number of iterations')
    parser.add_argument('--decay_epoch', type=int, default=20, help='number of iterations')
    parser.add_argument('--test', type=bool, default=False, help='enable testing')
    parser.add_argument('--train_test', type=bool, default=True, help='enable testing')
    parser.add_argument('--show', type=bool, default=True, help='print progress')
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--dataset', type=str, default='ASSISTments2009')
    parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.75, help='learning rate decay')
    parser.add_argument('--final_lr', type=float, default=1E-5,
                        help='learning rate will not decrease after hitting this threshold')

    parser.add_argument('--max_grad_norm', type=float, default=50.0, help='maximum gradient norm')
    parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
    parser.add_argument('--qa_embed_dim', type=int, default=200, help='answer and question embedding dimensions')
    # 动量法的提出是为了解决梯度下降
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
    dataset = "statics2011"  # synthetic / junyiacademy / ASSISTment2015 / ASSISTments2009 / statics2011

    if dataset == 'ASSISTments2009':
        parser.add_argument('--batch_size', type=int, default=64, help='the batch size')
        parser.add_argument('--n_question', type=int, default=123, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='./data/ASSISTments2009', help='data directory')
        parser.add_argument('--data_name', type=str, default='ASSISTments2009', help='data set name')
        parser.add_argument('--load', type=str, default='ASSISTments2009', help='model file to load')
        parser.add_argument('--save', type=str, default='ASSISTments2009', help='path to save model')
        parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')
        parser.add_argument('--memory_size', type=int, default=20, help='memory size')

    if dataset == 'ASSISTment2015':
        parser.add_argument('--batch_size', type=int, default=64, help='the batch size')
        parser.add_argument('--n_question', type=int, default=100,
                            help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='./data/ASSISTment2015', help='data directory')
        parser.add_argument('--data_name', type=str, default='ASSISTment2015', help='data set name')
        parser.add_argument('--load', type=str, default='ASSISTment2015', help='model file to load')
        parser.add_argument('--save', type=str, default='ASSISTment2015', help='path to save model')
        parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')
        parser.add_argument('--memory_size', type=int, default=20, help='memory size')

    elif dataset == "junyiacademy":
        parser.add_argument('--batch_size', type=int, default=64, help='the batch size')
        parser.add_argument('--n_question', type=int, default=1326,
                            help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='./data/junyiacademy', help='data directory')
        parser.add_argument('--data_name', type=str, default='junyiacademy', help='data set name')
        parser.add_argument('--load', type=str, default='junyiacademy', help='model file to load')
        parser.add_argument('--save', type=str, default='junyiacademy', help='path to save model')
        parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')
        parser.add_argument('--memory_size', type=int, default=20, help='memory size')

    elif dataset == "synthetic":
        parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
        parser.add_argument('--n_question', type=int, default=50, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=50, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='./data/synthetic', help='data directory')
        parser.add_argument('--data_name', type=str, default='synthetic', help='data set name')
        parser.add_argument('--load', type=str, default='synthetic', help='model file to load')
        parser.add_argument('--save', type=str, default='synthetic', help='path to save model')
        parser.add_argument('--memory_size', type=int, default=5, help='memory size')
        parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')

    elif dataset == 'statics2011':
        parser.add_argument('--batch_size', type=int, default=10, help='the batch size')
        parser.add_argument('--n_question', type=int, default=1224,
                            help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='./data/statics2011', help='data directory')
        parser.add_argument('--data_name', type=str, default='statics2011', help='data set name')
        parser.add_argument('--load', type=str, default='statics2011', help='model file to load')
        parser.add_argument('--save', type=str, default='statics2011', help='path to save model')
        parser.add_argument('--memory_size', type=int, default=20, help='memory size')
        parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')

    params = parser.parse_args()
    params.lr = params.init_lr  # 初始的学习率
    params.memory_key_state_dim = params.q_embed_dim  # 键矩阵维度
    params.memory_value_state_dim = params.qa_embed_dim  # 值矩阵维度
    params.dataset = dataset   # 数据集
    print(params)  # 打印参数

    # 读取数据

    dat = DATA(n_question=params.n_question,seqlen=params.seqlen,separate_char=',')
    if params.dataset != 'synthetic':
        train_data_path = params.data_dir + "/train_set.csv"
        valid_data_path = params.data_dir + "/test_set.csv"

    else:
        train_data_path = params.data_dir + "/naive_c5_q50_s4000_v0train_set.csv"
        valid_data_path = params.data_dir + "/naive_c5_q50_s4000_v0test_set.csv"

    train_q_data, train_qa_data,train_repeated_time_gap_dataArray, train_past_trail_counts_dataArray, train_seq_time_gap_dataArray = dat.load_data(train_data_path)
    valid_q_data, valid_qa_data,valid_repeated_time_gap_dataArray, valid_past_trail_counts_dataArray, valid_seq_time_gap_dataArray = dat.load_data(valid_data_path)


    model = MODEL(n_question=params.n_question,
                  batch_size=params.batch_size,
                  q_embed_dim=params.q_embed_dim,
                  qa_embed_dim=params.qa_embed_dim,
                  memory_size=params.memory_size,
                  memory_key_state_dim=params.memory_key_state_dim,
                  memory_value_state_dim=params.memory_value_state_dim,
                  final_fc_dim=params.final_fc_dim,
                  seqlen=params.seqlen

                  )

    model.init_embeddings()
    model.init_params()
    optimizer = optim.Adam(params=model.parameters(), lr=params.lr, betas=(0.9, 0.9))
    if params.gpu >= 0:
        print('device: ' + str(params.gpu))
        torch.cuda.set_device(params.gpu)
        model.cuda()

    best_valid_auc = 0
    correspond_train_auc = 0
    correspond_test_auc = 0
    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    best_epoch = 0
    best_valid_acc = 0
    best_valid_loss = 0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []

    for idx in range(params.max_iter):
        train_loss, train_accuracy, train_auc = train(model, params, optimizer, train_q_data, train_qa_data,train_repeated_time_gap_dataArray, train_past_trail_counts_dataArray, train_seq_time_gap_dataArray)
        print('Epoch %d/%d, loss : %3.5f, auc : %3.5f, accuracy : %3.5f' % (
        idx + 1, params.max_iter, train_loss, train_auc, train_accuracy))
        valid_loss, valid_accuracy, valid_auc = test(model, params, optimizer, valid_q_data, valid_qa_data,valid_repeated_time_gap_dataArray, valid_past_trail_counts_dataArray, valid_seq_time_gap_dataArray)
        print('Epoch %d/%d, valid auc : %3.5f, valid accuracy : %3.5f' % (
        idx + 1, params.max_iter, valid_auc, valid_accuracy))
        train_loss_all.append(train_loss)
        val_loss_all.append(valid_loss)
        train_acc_all.append(train_accuracy)
        val_acc_all.append(valid_accuracy)
        all_train_auc[idx + 1] = train_auc
        all_train_accuracy[idx + 1] = train_accuracy
        all_train_loss[idx + 1] = train_loss

        all_valid_loss[idx + 1] = valid_loss
        all_valid_accuracy[idx + 1] = valid_accuracy
        all_valid_auc[idx + 1] = valid_auc

        # output the epoch with the best validation auc
        if valid_auc > best_valid_auc:
            print('%3.4f to %3.4f' % (best_valid_auc, valid_auc))
            best_valid_auc = valid_auc
            best_epoch = idx + 1
            best_valid_acc = valid_accuracy
            best_valid_loss = valid_loss
    train_process = pd.DataFrame(
        data={
            "epoch": range(params.max_iter),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all,
            "train_acc_all": train_acc_all,
            "val_acc_all": val_acc_all
        })
    # 可视化模型训练过程
    plt.figure(figsize=(18,6))
    plt.subplot(1,2,1)
    plt.plot(train_process.epoch,train_process.train_loss_all,"r.-",label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss_all, "bs-", label="val loss")
    plt.legend()
    plt.xlabel("Epoch number",size=13)
    plt.ylabel("Loss value",size=13)
    plt.subplot(1,2,2)
    plt.plot(train_process.epoch, train_process.train_acc_all, "r.-", label="Train acc")
    plt.plot(train_process.epoch, train_process.val_acc_all, "bs-", label="val acc")
    plt.legend()
    plt.xlabel("Epoch number", size=13)
    plt.ylabel("ACC", size=13)

    plt.show()


    if not os.path.isdir('result'):
        os.makedirs('result')
    if not os.path.isdir(os.path.join('result', params.save)):
        os.makedirs(os.path.join('result', params.save))
    f_save_log = open(os.path.join('result', params.save, params.save), 'w')
    f_save_log.write("valid_auc:\n" + str(all_valid_auc) + "\n\n")
    f_save_log.write("train_auc:\n" + str(all_train_auc) + "\n\n")
    f_save_log.write("valid_loss:\n" + str(all_valid_loss) + "\n\n")
    f_save_log.write("train_loss:\n" + str(all_train_loss) + "\n\n")
    f_save_log.write("valid_accuracy:\n" + str(all_valid_accuracy) + "\n\n")
    f_save_log.write("train_accuracy:\n" + str(all_train_accuracy) + "\n\n")
    f_save_log.close()
    print("best outcome: best epoch: %.4f" % (best_epoch))
    print(
        "valid_auc: %.4f\tvalid_accuracy: %.4f\tvalid_loss: %.4f\t" % (
        best_valid_auc, best_valid_acc, best_valid_loss))

    if not os.path.isdir('result'):
        os.makedirs('result')
    if not os.path.isdir(os.path.join('result', params.save)):
        os.makedirs(os.path.join('result', params.save))
    f_save_log = open(os.path.join('result', params.save, params.save), 'a')
    f_save_log.write("best outcome: best epoch: %.4f" % (best_epoch) + "\n\n")
    f_save_log.write(
        "valid_auc: %.4f\tvalid_accuracy: %.4f\tvalid_loss: %.4f\t" % (
        best_valid_auc, best_valid_acc, best_valid_loss) + "\n\n")
    f_save_log.close()


if __name__ == "__main__":
    main()
