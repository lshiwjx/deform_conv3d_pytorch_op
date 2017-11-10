# -a*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import time

s = time.time()
a = torch.zeros((100, 100)).cuda()
s1 = time.time()
b = a.new((100, 100)).zero_()
s2 = time.time()
c = torch.zeros((100, 100)).cuda()
s3 = time.time()
d = torch.zeros(a)
s4 = time.time()
e = a.new(a.size()).zero_()
s5 = time.time()
print(s1 - s, s2 - s1, s3 - s2, s4 - s3, s5 - s4)
print(a.type(), b.type(), c.type(), d.type(), e.type())
