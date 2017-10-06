import torch
from torch.autograd import Variable

cuda_is_available = torch.cuda.is_available()


def cuda(x):
    return x.cuda() if cuda_is_available else x


def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x.cuda(async=True), volatile=volatile))


def long_tensor(x):
    return x.type(torch.LongTensor).cuda()