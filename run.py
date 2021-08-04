import numpy as np
import math
import torch
import random
from torch import nn
import utils as utils
from sklearn import metrics
from tqdm import tqdm


def train(model, params, optimizer, q_data, qa_data,repeated_time_gap, past_trail_counts, seq_time_gap):
    N = int(math.floor(len(q_data) / params.batch_size))  # batch的数量

    # shuffle data
    shuffle_index = np.random.permutation(q_data.shape[0])
    q_data = q_data[shuffle_index]
    qa_data = qa_data[shuffle_index]
    repeated_time_gap = repeated_time_gap[shuffle_index]
    past_trail_counts = past_trail_counts[shuffle_index]
    seq_time_gap = seq_time_gap[shuffle_index]
    # correct_answer_rate = correct_answer_rate[shuffle_index]


    pred_list = []
    target_list = []
    epoch_loss = 0
    model.train()

    for idx in range(N):
        # 根据batch-size分割数据集
        q_one_seq = q_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        qa_batch_seq = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        target = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]

        target = (target - 1) / params.n_question
        target = np.floor(target)  # 向下取整  真实的学生反应
        input_q = utils.variable(torch.LongTensor(q_one_seq), params.gpu)
        input_qa = utils.variable(torch.LongTensor(qa_batch_seq), params.gpu)
        target = utils.variable(torch.FloatTensor(target), params.gpu)
        target_to_1d = torch.chunk(target, params.batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(params.batch_size)], 1)
        target_1d = target_1d.permute(1, 0)  # 维度换位
        repeated_time_gap_seq = repeated_time_gap[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        past_trail_counts_seq = past_trail_counts[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        seq_time_gap_seq = seq_time_gap[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        input_repeated_time_gap = utils.variable(torch.FloatTensor(repeated_time_gap_seq), params.gpu)
        input_past_trail_counts = utils.variable(torch.FloatTensor(past_trail_counts_seq), params.gpu)
        input_seq_time_gap = utils.variable(torch.FloatTensor(seq_time_gap_seq), params.gpu)
        # correct_answer_rate_seq = correct_answer_rate[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        # input_correct_answer_rate = utils.variable(torch.FloatTensor(correct_answer_rate_seq), params.gpu)

        model.zero_grad()
        loss, filtered_pred, filtered_target = model(input_q, input_qa, target_1d,input_repeated_time_gap, input_past_trail_counts, input_seq_time_gap)
        loss.backward()  # 每一个batch做一次反向传播
        nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)
        optimizer.step()
        epoch_loss += utils.to_scalar(loss)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        pred_list.append(right_pred)
        target_list.append(right_target)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    # if (idx + 1) % params.decay_epoch == 0:
    #     utils.adjust_learning_rate(optimizer, params.init_lr * params.lr_decay)
    # print('lr: ', params.init_lr / (1 + 0.75))
    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)
    # f1 = metrics.f1_score(all_target, all_pred)

    return epoch_loss/N, accuracy, auc

def test(model, params, optimizer, q_data, qa_data,repeated_time_gap, past_trail_counts, seq_time_gap):
    N = int(math.floor(len(q_data) / params.batch_size))

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.eval()

    for idx in range(N):

        q_one_seq = q_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        qa_batch_seq = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        target = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]

        target = (target - 1) / params.n_question
        target = np.floor(target)

        input_q = utils.variable(torch.LongTensor(q_one_seq), params.gpu)
        input_qa = utils.variable(torch.LongTensor(qa_batch_seq), params.gpu)
        target = utils.variable(torch.FloatTensor(target), params.gpu)

        target_to_1d = torch.chunk(target, params.batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(params.batch_size)], 1)
        target_1d = target_1d.permute(1, 0)

        repeated_time_gap_seq = repeated_time_gap[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        past_trail_counts_seq = past_trail_counts[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        seq_time_gap_seq = seq_time_gap[idx * params.batch_size:(idx + 1) * params.batch_size, :]

        input_repeated_time_gap = utils.variable(torch.FloatTensor(repeated_time_gap_seq), params.gpu)
        input_past_trail_counts = utils.variable(torch.FloatTensor(past_trail_counts_seq), params.gpu)
        input_seq_time_gap = utils.variable(torch.FloatTensor(seq_time_gap_seq), params.gpu)
        # correct_answer_rate_seq = correct_answer_rate[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        # input_correct_answer_rate = utils.variable(torch.FloatTensor(correct_answer_rate_seq), params.gpu)

        loss, filtered_pred, filtered_target = model.forward(input_q, input_qa, target_1d,input_repeated_time_gap, input_past_trail_counts, input_seq_time_gap)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        pred_list.append(right_pred)
        target_list.append(right_target)
        epoch_loss += utils.to_scalar(loss)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)

    return epoch_loss/N, accuracy, auc









