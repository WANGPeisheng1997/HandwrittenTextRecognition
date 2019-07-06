import torch
from torch.autograd import Variable
from PIL import Image

from src.utils import alphabet
import src.dataset
import model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path',  default='dataset/train/0.png')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')

parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
parser.add_argument('--gpuid', type=int, default=0, help='which GPU to use')

parser.add_argument('--height', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--width', type=int, default=208, help='the width of the input image to network')

parser.add_argument('--encoder', type=str, default='', help="path to encoder (to continue training)")
parser.add_argument('--decoder', type=str, default='', help='path to decoder (to continue training)')
parser.add_argument('--loadModelEpoch', type=int, default=0, help='load model from epoch n to continue training, override the previous two')
parser.add_argument('--max_width', type=int, default=53, help='the width of the featuremap out from cnn')
opt = parser.parse_args()

if opt.cuda:
    torch.cuda.set_device(opt.gpuid)

img_path = opt.img_path
max_length = 10
EOS_TOKEN = 1

nclass = len(alphabet) + 3
encoder = model.encoder(opt.height, 1, 256)
decoder = model.decoder(256, nclass)

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

converter = src.utils.strLabelConverterForAttention(alphabet)
transformer = src.dataset.resizeNormalize((opt.width, opt.height))

image = Image.open(img_path).convert('L')
image = transformer(image)

if opt.cuda:
    encoder.cuda()
    decoder.cuda()
    image = image.cuda()

image = image.view(1, *image.size())
image = Variable(image)

encoder.eval()
decoder.eval()

encoder_out = encoder(image)
decoded_words = []
prob = 1.0
decoder_attentions = torch.zeros(max_length, opt.max_width)
decoder_input = torch.zeros(1).long()
decoder_hidden = decoder.initHidden(1)

if opt.cuda:
    decoder_input = decoder_input.cuda()
    decoder_hidden = decoder_hidden.cuda()

for di in range(max_length):
    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_out)
    probs = torch.exp(decoder_output)
    decoder_attentions[di] = decoder_attention.data
    topv, topi = decoder_output.data.topk(1)
    ni = topi.squeeze(1)
    decoder_input = ni
    prob *= probs[:, ni]
    if ni == EOS_TOKEN:
        break
    else:
        decoded_words.append(converter.decode(ni))

words = ''.join(decoded_words)
prob = prob.item()
print('predict_str:%-20s => prob:%-20s' % (words, prob))
