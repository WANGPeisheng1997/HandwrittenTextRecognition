import torch
import torch.utils.data
import src.utils as utils
from src.utils import alphabet
from src.utils import strLabelConverterForAttention as converter

def predict(encoder, decoder, criterion, batchsize, dataset, max_iter=100, workers=2, max_width=53):
    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = False
        d.requires_grad = False
    encoder.eval()
    decoder.eval()

    data_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batchsize, num_workers=workers)
    iterator = iter(data_loader)
    max_iter = min(max_iter, len(data_loader))

    n_correct = 0  # correct characters (including EOS)
    n_total = 0  # total characters (including EOS)
    loss_avg = utils.averager()

    EOS_TOKEN = 1  # end of sequence

    for _ in range(max_iter):
        data = iterator.next()
        cpu_images, cpu_texts = data
        b = cpu_images.size(0)
        image = torch.FloatTensor(batchsize, 3, 1, 1)
        image = image.cuda()
        utils.loadData(image, cpu_images)
        target_variable = converter(alphabet).encode(cpu_texts)
        target_variable = target_variable.cuda()

        encoder_outputs = encoder(image)  # cnn+biLstm做特征提取
        decoder_input = target_variable[0].cuda()  # 初始化decoder的开始,从0开始输出
        decoder_hidden = decoder.initHidden(b).cuda()
        loss = 0.0

        decoded_words = []
        decoded_labels = []
        flag = [True] * batchsize

        for _ in range(batchsize):
            new_list = []
            decoded_words.append(new_list)
            new_list = []
            decoded_labels.append(new_list)

        for di in range(1, target_variable.shape[0]):  # 最大字符串的长度
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])  # 每次预测一个字符
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze()
            decoder_input = ni

            for count in range(batchsize):
                if flag[count]:
                    if ni[count] == EOS_TOKEN:
                        decoded_words[count].append('<EOS>')
                        decoded_labels[count].append(EOS_TOKEN)
                        flag[count] = False
                    else:
                        decoded_words[count].append(converter(alphabet).decode(ni[count]))
                        decoded_labels[count].append(ni[count])

        loss_avg.add(loss)

        # 计算正确个数
        for count in range(batchsize):
            n_total += len(cpu_texts[count]) + 1  # EOS included
            for pred, target in zip(decoded_labels[count], target_variable[1:,count]):
                if pred == target:
                    n_correct += 1

            texts = cpu_texts[count]
            print('Pred:%-20s, GT: %-20s' % (decoded_words[count], texts))

    accuracy = n_correct / float(n_total)
    print('Loss: %f, accuracy: %f' % (loss_avg.val(), accuracy))