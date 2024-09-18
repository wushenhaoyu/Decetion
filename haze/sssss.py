import torch
print(torch.backends.cudnn.version())
from torch.backends import cudnn
cudnn.is_available()
a=torch.tensor(1.)
cudnn.is_acceptable(a.cuda())