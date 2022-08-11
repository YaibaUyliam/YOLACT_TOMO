from yolact_edge.yolact import Yolact
from torch2trt import torch2trt
import torch 
import torch.backends.cudnn as cudnn

Json_config = '/home/crist_tienngoc/TOMO/Multi_object/yolact_edge/yolact_edge/config.json'
pretrained = '/home/crist_tienngoc/TOMO/Multi_object/yolact_edge/multi_object_3/weight/yolact_edge_25_12844.pth'

from yolact_edge.utils.loadJSON_config import LoadJSON_config

cfg = LoadJSON_config(Json_config)._load_config()
with torch.no_grad():
    model = Yolact(training=False,cfg=cfg)
    model.load_weights(pretrained)
    model.eval()
    model = model.cuda()

    x = torch.zeros((1, 3, 550, 550)).cuda().float()
    cudnn.benchmark = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    extras = {"backbone": "full", "interrupt": False, "keep_statistics": False, "moving_statistics": None}
    model_trt = torch2trt(model, [x])