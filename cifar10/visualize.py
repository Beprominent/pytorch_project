from torchviz import make_dot
from models import *
import torch as t
from torch.autograd import Variable

model = MobileNetV2()
x = Variable(t.randn(1, 3, 32, 32))
y = model(x)
dot = make_dot(y.mean(), params=dict(list(model.named_parameters()) + [('x', x)]))
dot.format = 'svg'
dot.render()