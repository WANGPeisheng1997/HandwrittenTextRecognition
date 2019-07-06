import argparse
import torch
import torch.utils.data
import src.utils as utils
from src.utils import alphabet
from src.utils import strLabelConverterForAttention as converter
import src.dataset as dataset
import model

parser = argparse.ArgumentParser()
parser.add_argument('--testList',  default='label/test_label.txt')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')

parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
parser.add_argument('--gpuid', type=int, default=0, help='which GPU to use')

parser.add_argument('--height', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--width', type=int, default=208, help='the width of the input image to network')

parser.add_argument('--encoder', type=str, default='', help="path to encoder (to continue training)")
parser.add_argument('--decoder', type=str, default='', help='path to decoder (to continue training)')
parser.add_argument('--loadModelEpoch', type=int, default=0, help='load model from epoch n to continue training, override the previous two')
opt = parser.parse_args()

if opt.cuda:
    torch.cuda.set_device(opt.gpuid)


def predict(encoder, decoder, criterion, batchsize, dataset, workers=2):
    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = False
        d.requires_grad = False
    encoder.eval()
    decoder.eval()

    data_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batchsize, num_workers=workers)
    iterator = iter(data_loader)

    n_correct = 0  # correct characters (including EOS)
    n_total = 0  # total characters (including EOS)
    n_current = 0 # current position
    loss_avg = utils.averager()

    EOS_TOKEN = 1  # end of sequence

    for _ in range(len(data_loader)):
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

        for count in range(batchsize):
            n_total += len(cpu_texts[count]) + 1  # EOS included
            for pred, target in zip(decoded_labels[count], target_variable[1:,count]):
                if pred == target:
                    n_correct += 1

            texts = cpu_texts[count]
            print('%d Pred:%-20s, GT: %-20s' % (n_current, decoded_words[count], texts))
            n_current += 1

    accuracy = n_correct / float(n_total)
    print('Loss: %f, Accuracy: %f' % (loss_avg.val(), accuracy))


if __name__ == '__main__':
    test_dataset = dataset.listDataset(list_file=opt.testList, transform=dataset.resizeNormalize((opt.width, opt.height)))
    nclass = len(alphabet) + 3
    nc = 1
    criterion = torch.nn.NLLLoss()
    encoder = model.encoder(opt.height, nc=nc, nh=256)
    decoder = model.decoder(nh=256, nclass=nclass, dropout_p=0.1)

    if opt.encoder:
        print('loading pretrained encoder model from %s' % opt.encoder)
        encoder.load_state_dict(torch.load(opt.encoder))
    if opt.decoder:
        print('loading pretrained decoder model from %s' % opt.decoder)
        decoder.load_state_dict(torch.load(opt.decoder))
    if opt.loadModelEpoch > 0:
        encoder_path = 'model/encoder_%d.pth' % opt.loadModelEpoch
        print('loading pretrained encoder model from %s' % encoder_path)
        encoder.load_state_dict(torch.load(encoder_path))
        decoder_path = 'model/decoder_%d.pth' % opt.loadModelEpoch
        print('loading pretrained decoder model from %s' % decoder_path)
        decoder.load_state_dict(torch.load(decoder_path))

    if opt.cuda:
        encoder.cuda()
        decoder.cuda()
        criterion = criterion.cuda()

    print("Testing:")
    predict(encoder, decoder, criterion, opt.batchSize, dataset=test_dataset)
