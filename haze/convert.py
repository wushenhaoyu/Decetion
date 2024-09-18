import paddle
import torch
import numpy as np

import re

# PyTorch 权重键
torch_keys = [
    'module.a.conv_in.1.conv.weight', 'module.a.conv_in.2.conv.0.0.weight', 'module.a.conv_in.2.conv.0.0.bias',
    'module.a.conv_in.2.conv.1.0.weight', 'module.a.conv_in.2.conv.1.0.bias', 'module.a.conv_in.2.conv.2.0.weight',
    'module.a.conv_in.2.conv.2.0.bias', 'module.a.conv_in.3.conv.weight', 'module.a.conv_in.4.conv.weight',
    'module.a.conv_in.5.weight', 'module.a.conv_in.5.bias', 'module.t.conv1_1.0.conv.weight', 'module.t.conv1_1.1.conv.weight',
    'module.t.conv2_1.1.conv.weight', 'module.t.conv2_1.2.conv.weight', 'module.t.conv2_1.3.conv.weight',
    'module.t.conv2_1.4.conv.weight', 'module.t.conv3_1.1.conv.weight', 'module.t.conv3_1.2.conv.weight',
    'module.t.conv3_1.3.conv.weight', 'module.t.conv3_1.4.conv.weight', 'module.t.conv4_0.conv.weight',
    'module.t.conv4_1.0.conv.weight', 'module.t.conv4_1.1.conv.weight', 'module.t.conv4_1.2.conv.weight',
    'module.t.conv5_0.conv.weight', 'module.t.conv5_1.0.conv.weight', 'module.t.conv5_1.1.conv.weight',
    'module.t.conv5_1.2.conv.weight', 'module.t.conv5_2.weight', 'module.t.conv5_2.bias', 'module.f.conv1.conv.weight',
    'module.f.conv2.conv.weight', 'module.f.conv3.conv.weight', 'module.f.conv4.conv.weight', 'module.f.conv5.conv.weight',
    'module.r.conv.0.conv.weight', 'module.r.conv.1.conv.0.conv.weight', 'module.r.conv.1.conv.1.conv.weight',
    'module.r.conv.2.conv.0.conv.weight', 'module.r.conv.2.conv.1.conv.weight', 'module.r.conv.3.conv.0.conv.weight',
    'module.r.conv.3.conv.1.conv.weight', 'module.r.conv.4.conv.0.conv.weight', 'module.r.conv.4.conv.1.conv.weight',
    'module.r.conv.5.conv.0.conv.weight', 'module.r.conv.5.conv.1.conv.weight', 'module.r.conv.6.weight'
]
paddle_keys = [
    'a.conv_in.1.conv.weight', 'a.conv_in.2.conv.0.0.weight', 'a.conv_in.2.conv.0.0.bias',
      'a.conv_in.2.conv.1.0.weight', 'a.conv_in.2.conv.1.0.bias', 'a.conv_in.2.conv.2.0.weight',
      'a.conv_in.2.conv.2.0.bias', 'a.conv_in.3.conv.weight', 'a.conv_in.4.conv.weight', 
      'a.conv_in.5.weight', 'a.conv_in.5.bias', 't.conv1_1.0.conv.weight', 't.conv1_1.1.conv.weight',
        't.conv2_1.1.conv.weight', 't.conv2_1.2.conv.weight', 't.conv2_1.3.conv.weight', 
        't.conv2_1.4.conv.weight', 't.conv3_1.1.conv.weight', 't.conv3_1.2.conv.weight', 
        't.conv3_1.3.conv.weight', 't.conv3_1.4.conv.weight', 't.conv4_0.conv.weight', 
        't.conv4_1.0.conv.weight', 't.conv4_1.1.conv.weight', 't.conv4_1.2.conv.weight',
          't.conv5_0.conv.weight', 't.conv5_1.0.conv.weight', 't.conv5_1.1.conv.weight', 
          't.conv5_1.2.conv.weight', 't.conv5_2.weight', 't.conv5_2.bias', 'f.conv1.conv.weight',
            'f.conv2.conv.weight', 'f.conv3.conv.weight', 'f.conv4.conv.weight', 'f.conv5.conv.weight', 
            'r.conv.0.conv.weight', 'r.conv.1.conv.0.conv.weight', 'r.conv.1.conv.1.conv.weight', 
            'r.conv.2.conv.0.conv.weight', 'r.conv.2.conv.1.conv.weight', 'r.conv.3.conv.0.conv.weight',
            'r.conv.3.conv.1.conv.weight', 'r.conv.4.conv.0.conv.weight', 'r.conv.4.conv.1.conv.weight', 
            'r.conv.5.conv.0.conv.weight', 'r.conv.5.conv.1.conv.weight', 'r.conv.6.weight'
]

# PaddlePaddle 权重键

result_dict = dict(zip(torch_keys, paddle_keys))




print(result_dict )

torch_model_path = "GNet.tar"
torch_state_dict_all = torch.load(torch_model_path,map_location=torch.device('cpu'))
torch_state_dict = torch_state_dict_all['state_dict']
paddle_model_path = "paddle.pdparams"
paddle_state_dict = {}

keys_dict = result_dict
for torch_key in torch_state_dict:
    paddle_key = torch_key
    for k in keys_dict:
        if k in paddle_key:
            paddle_key = paddle_key.replace(k, keys_dict[k])

    if ('linear' in paddle_key) or ('proj' in  paddle_key) or ('vocab' in  paddle_key and 'weight' in  paddle_key) or ("dense.weight" in paddle_key) or ('transform.weight' in paddle_key) or ('seq_relationship.weight' in paddle_key):
        paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy().transpose())
    else:
        paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy())

    print("torch: ", torch_key,"\t", torch_state_dict[torch_key].shape)
    print("paddle: ", paddle_key, "\t", paddle_state_dict[paddle_key].shape, "\n")

paddle.save(paddle_state_dict, paddle_model_path)