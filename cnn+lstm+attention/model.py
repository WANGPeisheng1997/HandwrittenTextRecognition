import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

GO = 0
EOS_TOKEN = 1              # 结束标志的标签

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output
    

class encoder(nn.Module):
    # CNN+BiLstm做特征提取
    def __init__(self, imgH, nc, nh):
        super(encoder, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.cnn = nn.Sequential(
                      nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 64x16x50
                      nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 128x8x25
                      nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), # 256x8x25
                      nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)), # 256x4x25
                      nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), # 512x4x25
                      nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)), # 512x2x25
                      nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)) # 512x1x25
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features calculate
        encoder_outputs = self.rnn(conv)          # seq * batch * n_classes// 25 × batchsize × 256（隐藏节点个数）
        
        return encoder_outputs


class AttentionDecoder(nn.Module):
    # 采用seq to seq模型，修改注意力权重的计算方式
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # test
        self.vat = nn.Linear(hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)  # 前一次的输出进行词嵌入
        embedded = self.dropout(embedded)

        # test
        batch_size = encoder_outputs.shape[1]
        alpha = hidden + encoder_outputs  # 特征融合采用+/concat其实都可以
        alpha = alpha.view(-1, alpha.shape[-1])
        attn_weights = self.vat(torch.tanh(alpha))  # 将encoder_output:batch*seq*features,将features的维度降为1
        attn_weights = attn_weights.view(-1, 1, batch_size).permute((2, 1, 0))
        attn_weights = F.softmax(attn_weights, dim=2)

        attn_applied = torch.matmul(attn_weights,
                                    encoder_outputs.permute((1, 0, 2)))  # 矩阵乘法，bmm（8×1×56，8×56×256）=8×1×256
        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)  # 上一次的输出和attention feature，做一个线性+GRU
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)  # 最后输出一个概率
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))

        return result


class decoder(nn.Module):

    def __init__(self, nh=256, nclass=13, dropout_p=0.1):
        super(decoder, self).__init__()
        self.hidden_size = nh
        self.decoder = AttentionDecoder(nh, nclass, dropout_p)

    def forward(self, input, hidden, encoder_outputs):
        return self.decoder(input, hidden, encoder_outputs)

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result
