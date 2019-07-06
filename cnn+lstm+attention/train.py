from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np

import src.utils as utils
import src.dataset as dataset
from src.utils import alphabet
from src.utils import weights_init

import model

parser = argparse.ArgumentParser()
parser.add_argument('--trainList',  default='label/train_label.txt')
parser.add_argument('--valList',  default='label/val_label.txt')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--epochs', type=int, default=501, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for Critic, default=0.00005')

parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
parser.add_argument('--gpuid', type=int, default=0, help='which GPU to use')

parser.add_argument('--height', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--width', type=int, default=208, help='the width of the input image to network')

parser.add_argument('--encoder', type=str, default='', help="path to encoder (to continue training)")
parser.add_argument('--decoder', type=str, default='', help='path to decoder (to continue training)')
parser.add_argument('--loadModelEpoch', type=int, default=0, help='load model from epoch n to continue training, override the previous two')
parser.add_argument('--savePath', default='model', help='Where to store samples and models')

parser.add_argument('--displayInterval', type=int, default=10, help='batch(s) between display')
parser.add_argument('--valInterval', type=int, default=1, help='epoch(s) between validations')
parser.add_argument('--valDisplayInterval', type=int, default=1, help='validation display interval batch(s)')
parser.add_argument('--saveInterval', type=int, default=10, help='epoch(s) between model savings')

parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', default=True, action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--max_width', type=int, default=53, help='the width of the featuremap out from cnn')
opt = parser.parse_args()

SOS_TOKEN = 0
EOS_TOKEN = 1   # end of sequence
BLANK = 2       # blank character

opt.manualSeed = random.randint(1, 10000)  # fix seed
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if opt.cuda:
    torch.cuda.set_device(opt.gpuid)

transform = None
train_dataset = dataset.listDataset(list_file =opt.trainList, transform=transform)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=False, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(height=opt.height, width=opt.width, keep_ratio=opt.keep_ratio))

val_dataset = dataset.listDataset(list_file=opt.valList, transform=dataset.resizeNormalize((opt.width, opt.height)))

nclass = len(alphabet) + 3          # decoder的时候，需要的类别数,3 for SOS,EOS和blank 
nc = 1

converter = utils.strLabelConverterForAttention(alphabet)
criterion = torch.nn.NLLLoss()              # 最后的输出要为log_softmax


encoder = model.encoder(opt.height, nc=nc, nh=256)
decoder = model.decoder(nh=256, nclass=nclass, dropout_p=0.1)

# continue training or use the pretrained model to initial the parameters of the encoder and decoder
encoder.apply(weights_init)
decoder.apply(weights_init)
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

image = torch.FloatTensor(opt.batchSize, 3, opt.height, opt.height)
text = torch.LongTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    encoder.cuda()
    decoder.cuda()
    image = image.cuda()
    text = text.cuda()
    criterion = criterion.cuda()

# loss averager
loss_avg = utils.averager()

# setup optimizer
encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.lr,
                       betas=(0.5, 0.999))
decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr,
                    betas=(0.5, 0.999))


def val(encoder, decoder, criterion, batchsize, dataset, max_iter=100):
    print('Validate:')

    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = False
        d.requires_grad = False
    encoder.eval()
    decoder.eval()

    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=batchsize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)
    max_iter = min(max_iter, len(data_loader))

    n_correct = 0  # correct characters (including EOS)
    n_total = 0  # total characters (including EOS)
    loss_avg = utils.averager()

    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        b = cpu_images.size(0)
        target_variable = converter.encode(cpu_texts)
        utils.loadData(image, cpu_images)
        n_total += len(cpu_texts[0]) + 1  # EOS included

        decoded_words = []
        decoded_label = []
        decoder_attentions = torch.zeros(len(cpu_texts[0]) + 1, opt.max_width)

        encoder_outputs = encoder(image)            # cnn+biLstm做特征提取
        target_variable = target_variable.cuda()
        decoder_input = target_variable[0].cuda()   # 初始化decoder的开始,从0开始输出
        decoder_hidden = decoder.initHidden(b).cuda()
        loss = 0.0

        for di in range(1, target_variable.shape[0]):  # 最大字符串的长度
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])  # 每次预测一个字符
            loss_avg.add(loss)
            decoder_attentions[di-1] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze(1)
            decoder_input = ni
            if ni == EOS_TOKEN:
                decoded_words.append('<EOS>')
                decoded_label.append(EOS_TOKEN)
                break
            else:
                decoded_words.append(converter.decode(ni))
                decoded_label.append(ni)

        # 计算正确个数
        for pred, target in zip(decoded_label, target_variable[1:,:]):
            if pred == target:
                n_correct += 1

        if i % opt.valDisplayInterval == 0:
            texts = cpu_texts[0]
            print('pred:%-20s, gt: %-20s' % (decoded_words, texts))

    accuracy = n_correct / float(n_total)
    print('Val loss: %f, accuracy: %f' % (loss_avg.val(), accuracy))


def trainBatch(encoder, decoder, criterion, encoder_optimizer, decoder_optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    b = cpu_images.size(0)
    target_variable = converter.encode(cpu_texts)
    utils.loadData(image, cpu_images)

    encoder_outputs = encoder(image)               # cnn+biLstm做特征提取
    target_variable = target_variable.cuda()
    decoder_input = target_variable[0].cuda()      # 初始化decoder的开始,从0开始输出
    decoder_hidden = decoder.initHidden(b).cuda()
    loss = 0.0

    for di in range(1, target_variable.shape[0]):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_variable[di])  # 每次预测一个字符
        topv, topi = decoder_output.data.topk(1)
        ni = topi.squeeze()
        decoder_input = ni

    encoder.zero_grad()
    decoder.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss


if __name__ == '__main__':
    for epoch in range(opt.epochs):
        # train
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader)-1:
            for e, d in zip(encoder.parameters(), decoder.parameters()):
                e.requires_grad = True
                d.requires_grad = True
            encoder.train()
            decoder.train()
            cost = trainBatch(encoder, decoder, criterion, encoder_optimizer, 
                              decoder_optimizer)
            loss_avg.add(cost)
            i += 1

            if i % opt.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                    (epoch, opt.epochs, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

        # validate
        if epoch % opt.valInterval == 0:
            val(encoder, decoder, criterion, opt.batchSize, dataset=val_dataset)

        # save model
        if epoch % opt.saveInterval == 0:
            torch.save(encoder.state_dict(), '{0}/encoder_{1}.pth'.format(opt.savePath, epoch))
            torch.save(decoder.state_dict(), '{0}/decoder_{1}.pth'.format(opt.savePath, epoch))