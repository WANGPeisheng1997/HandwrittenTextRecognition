from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image, ImageOps
import math


def grayscale(image):
    data = list(image.getdata())
    a = sorted(data)[int(len(data) * 0.3)]
    b = max(data)
    image = image.point(lambda i: (i - a) / (b - a) * 255.0 if i > a else 0)
    # image.show()
    return image


def preprocessing(image):
    image = image.convert('L')
    image = image.resize((28,28))
    image = ImageOps.invert(image)
    image = grayscale(image)
    return image


def predict(model, image):
    model.eval()
    data_transforms = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = data_transforms(image)
    image = image.view(1, 1, 28, 28)
    image = Variable(image)
    output = model(image)
    pred = int(output.data.max(1, keepdim=True)[1][0][0])
    confidence = math.exp(output[0][pred])
    # target = torch.from_numpy(np.arrray(target))
    # loss = F.nll_loss(output.view(1,10),Variable(torch.from_numpy(np.array(target))).view(1))
    # print("predict: " + str(int(pred[0][0])) + " truth: " + str(target))
    # print("loss ", float(loss))
    return pred, confidence