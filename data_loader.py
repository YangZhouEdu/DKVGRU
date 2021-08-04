import numpy as np
import math

class DATA(object):
    def __init__(self,n_question,seqlen,separate_char,name = "data"):
        self.separate_char = separate_char  # 数据集的分割符 csv为','
        self.n_question = n_question  # 问题的维度
        self.seqlen = seqlen  # 序列长度

    ### data format
    ### 15 # 序列长度
    ### 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54 # skill_id
    ### 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0  # 学生反应

    def load_data(self,path):
        f_data = open(path,'r')
        q_data = []
        qa_data = []
        repeated_time_gap = []
        past_trail_counts = []
        # correct_answer_rate = []
        for lineID, line in enumerate(f_data):
            if lineID % 3 ==1:  # 第2行
                Q = line.split(self.separate_char)
                if len( Q[len(Q)-1] ) == 0: # 如果Q-1的长度等于 0
                    Q = Q[:-1]
            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len(A[len(A) - 1]) == 0:
                    A = A[:-1]

                # 对于长度大于设置的序列长度时 对序列进行分割存入q_data
                n_split = 1
                if len(Q) > self.seqlen:
                    # math.floor 表示小于或等于指定数字的最大整数的数字
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen: # eg. 201 % 100 =1
                        n_split = n_split + 1
                for k in range(n_split):  # 0,1,2,3 ...
                    question_sequence = []
                    answer_sequence = []
                    target_sequence = [] # 答题反应序列
                    if k == n_split - 1:
                        endIndex = len(A)
                    else:
                        endIndex = (k+1) * self.seqlen
                        # 序列的起始 和 终点
                    for i in range(k*self.seqlen,endIndex):
                        if len(Q[i]) > 0 :
                            # Xindex 为特征值
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            answer_sequence.append(Xindex)
                            target_sequence.append(int(A[i]))
                        else:
                            print(Q[i])
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
###################################################  修改部分  #####################################
                    # rtg 重复时间间隔 Repeated time gap
                    # ptc 练习次数 Past exercise counts
                    # car 答题正确率 correct answer rate


                    sub_rtg = []
                    sub_ptc = [] # 所有都是log2
                    # sub_car = [0]
                    # car = [0] * (self.n_question + 1)
                    rtg = [0] * (self.n_question + 1)
                    ptc = [0] * (self.n_question + 1)
                    for i in range(len(question_sequence)):
                        sub_ptc.append(math.floor(np.log2(ptc[question_sequence[i]])) if ptc[question_sequence[i]] != 0 else 0)
                        ptc[question_sequence[i]] += 1
                        j = i - 1
                        sum = 0
                        while j >= 0:
                            sum += answer_sequence[j]
                            if question_sequence[j] == question_sequence[i]:
                                rtg[question_sequence[i]] = i - j
                                sub_rtg.append(math.floor(np.log2(rtg[question_sequence[i]])))
                                break
                            if j == 0:
                                sub_rtg.append(4)
                            j -= 1
                        # if sum / (i + 1) >= 0.6:
                        #     sub_car.append(2)
                        # elif sum / (i + 1) > 0.4 and sum / (i + 1) < 0.6:
                        #     sub_car.append(1)
                        # else:
                        #     sub_car.append(0)

                    repeated_time_gap.append(sub_rtg)
                    past_trail_counts.append(sub_ptc)
                    # correct_answer_rate.append(sub_car)
########################################################################################################
        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        ### convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat
        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat
        # dataArray: [ array([[],[],..])] Shape: (3633, 200)
        #####################################################################################
        repeated_time_gap_dataArray = np.zeros((len(repeated_time_gap), self.seqlen, 8))
        for i in range(len(repeated_time_gap)):
            for j in range(len(repeated_time_gap[i])):
                repeated_time_gap_dataArray[i, j, repeated_time_gap[i][j]] = 1

        past_trail_counts_dataArray = np.zeros((len(past_trail_counts), self.seqlen, 8))  # （学生数，序列数+1 方便取seqlen长度，后面的8是独热编码）
        for i in range(len(past_trail_counts)):
            for j in range(len(past_trail_counts[i])):
                past_trail_counts_dataArray[i, j, past_trail_counts[i][j]] = 1

        seq_time_gap_dataArray = np.ones((len(repeated_time_gap), self.seqlen, 1))
        # correct_answer_rate_dataArray = np.zeros((len(correct_answer_rate), self.seqlen + 1, 3))
        # for i in range(len(correct_answer_rate)):
        #     for j in range(len(correct_answer_rate[i])):
        #         correct_answer_rate_dataArray[i, j, correct_answer_rate[i][j]] = 1

##############################################################################################
        return q_dataArray, qa_dataArray ,repeated_time_gap_dataArray, past_trail_counts_dataArray, seq_time_gap_dataArray

    def generate_all_index_data(self, batch_size):
        n_question = self.n_question
        batch = math.floor(n_question / self.seqlen)
        if self.n_question % self.seqlen:
            batch += 1

        seq = np.arange(1, self.seqlen * batch + 1)
        zeroindex = np.arange(n_question, self.seqlen * batch)
        zeroindex = zeroindex.tolist()
        seq[zeroindex] = 0
        q = seq.reshape((batch, self.seqlen))
        q_dataArray = np.zeros((batch_size, self.seqlen))
        q_dataArray[0:batch, :] = q
        return q_dataArray