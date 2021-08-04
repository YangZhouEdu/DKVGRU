import torch
import torch.nn as nn

from memory import DKVMN
import  numpy as np

class MODEL(nn.Module):

    def __init__(self, n_question, batch_size, q_embed_dim, qa_embed_dim,
                 memory_size, memory_key_state_dim, memory_value_state_dim, final_fc_dim,seqlen, student_num=None,forget_dim = 0):
        super(MODEL, self).__init__()
        self.n_question = n_question  # 问题维度
        self.batch_size = batch_size
        self.q_embed_dim = q_embed_dim   # 问题矩阵维度
        self.qa_embed_dim = qa_embed_dim  # 值矩阵维度
        self.memory_size = memory_size  # 记忆矩阵槽个数
        self.memory_key_state_dim = memory_key_state_dim  # 键矩阵维度
        self.memory_value_state_dim = memory_value_state_dim  # 值矩阵维度
        self.final_fc_dim = final_fc_dim  # 学习率
        self.student_num = student_num  # 学生数量
        self.forget_dim = forget_dim
        self.input_embed_linear = nn.Linear(self.q_embed_dim, self.q_embed_dim, bias=True)  # 线性层
        self.read_embed_linear = nn.Linear(self.memory_value_state_dim + self.q_embed_dim + self.forget_dim, self.final_fc_dim, bias=True)
        self.predict_linear = nn.Linear(self.final_fc_dim, 1, bias=True)  # 预测层
        self.init_memory_key = nn.Parameter(torch.randn(self.memory_size, self.memory_key_state_dim)) # 初始化键值矩阵
        nn.init.kaiming_normal_(self.init_memory_key)
        self.init_memory_value = nn.Parameter(torch.randn(self.memory_size, self.memory_value_state_dim))
        nn.init.kaiming_normal_(self.init_memory_value)
        self.mem = DKVMN(memory_size=self.memory_size,
                   memory_key_state_dim=self.memory_key_state_dim,
                   memory_value_state_dim=self.memory_value_state_dim, init_memory_key=self.init_memory_key)

        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        # 题目序号从1开始
        # nn.embedding输入是一个下标的列标，输出是对应的嵌入
        self.seqlen = seqlen
        self.q_embed = nn.Embedding(self.n_question + 1, self.q_embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(self.seqlen, self.q_embed_dim)
        self.qa_embed = nn.Embedding(2 * self.n_question + 1, self.qa_embed_dim, padding_idx=0)

        self.C = nn.Parameter(torch.randn(self.batch_size, self.seqlen, self.forget_dim), requires_grad=True)

        # self.C1 = nn.Parameter(torch.randn(self.batch_size, self.seqlen, self.forget_dim-1), requires_grad=True)
        # self.C2 = nn.Parameter(torch.randn(self.batch_size, self.seqlen, self.forget_dim - 9), requires_grad=True)


    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.kaiming_normal_(self.read_embed_linear.weight)
        nn.init.constant_(self.read_embed_linear.bias, 0)
        nn.init.constant_(self.predict_linear.bias, 0)
        # nn.init.constant(self.input_embed_linear.bias, 0)
        # nn.init.normal(self.input_embed_linear.weight, std=0.02)

    def init_embeddings(self):
        nn.init.kaiming_normal_(self.q_embed.weight)  # A
        nn.init.kaiming_normal_(self.qa_embed.weight)  # B

    def forward(self, q_data, qa_data, target,repeated_time_gap_seq, past_trail_counts_seq, seq_time_gap_seq):

        batch_size = q_data.shape[0]   # 32
        seqlen = q_data.shape[1]   # 200
        # correct_answer_rate_seq = correct_answer_rate_seq[:,:-1,:]
        # pos_id = torch.arange(q_data.size(1)).unsqueeze(0).to(q_data.device)
        # pos_x = self.pos_embedding(pos_id)
        # pos_x1 = torch.Tensor(np.zeros((batch_size, self.seqlen, self.q_embed_dim))).to(q_data.device) + pos_x

        # c = torch.cat([], 2)
        # c_t = c[:, :, :]
        # c_t = torch.mul(self.C, c_t)


        #  qt && (q,a) embedding
        q_embed_data = self.q_embed(q_data)
        qa_embed_data = self.qa_embed(qa_data)

        ## copy mk batch times for dkvmn
        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        ## slice data for seqlen times by axis 1
        # torch.chunk(tensor, chunk_num, dim)
        slice_q_data = torch.chunk(q_data, seqlen, 1)
        slice_q_embed_data = torch.chunk(q_embed_data, seqlen, 1)
        slice_qa_embed_data = torch.chunk(qa_embed_data, seqlen, 1)


        value_read_content_l = []
        input_embed_l = []

        for i in range(seqlen):
            ## Attention
            q = slice_q_embed_data[i].squeeze(1)
            correlation_weight = self.mem.attention(q)

            ## Read Process
            read_content = self.mem.read(correlation_weight)

            ## save intermedium data
            value_read_content_l.append(read_content)
            input_embed_l.append(q)

            ## Write Process 写过程
            qa = slice_qa_embed_data[i].squeeze(1)
            new_memory_value = self.mem.write(correlation_weight, qa)


        # Projection
        all_read_value_content = torch.cat([value_read_content_l[i].unsqueeze(1) for i in range(seqlen)], 1)
        input_embed_content = torch.cat([input_embed_l[i].unsqueeze(1) for i in range(seqlen)], 1)

        ## Project rt
        input_embed_content = input_embed_content.view(batch_size * seqlen, -1)
        input_embed_content = torch.tanh(self.input_embed_linear(input_embed_content))
        input_embed_content = input_embed_content.view(batch_size, seqlen, -1)



        ## Concat Read_Content and input_embedding_value
        predict_input_before = torch.cat([all_read_value_content, input_embed_content], 2)
        # predict_input = torch.cat([predict_input_before,c_t], 2).view(batch_size*seqlen, -1)
        predict_input = predict_input_before.view(batch_size*seqlen, -1)
        read_content_embed = torch.tanh(self.read_embed_linear(predict_input))


        pred = self.predict_linear(read_content_embed)
        # predicts = torch.cat([predict_logs[i] for i in range(seqlen)], 1)
        target_1d = target                   # [batch_size * seq_len, 1]
        # mask = target_1d.ge(0)               # [batch_size * seq_len, 1]
        mask = q_data.gt(0).view(-1, 1)
        # pred_1d = predicts.view(-1, 1)           # [batch_size * seq_len, 1]
        pred_1d = pred.view(-1, 1)           # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target)

        return loss, torch.sigmoid(filtered_pred), filtered_target