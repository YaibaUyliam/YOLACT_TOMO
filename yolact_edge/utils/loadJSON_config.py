import json
from yolact_edge.data import *

class LoadJSON_config:
    ''' 
    json_file: path to json config file 
    '''

    def __init__(self, json_file):
        self.json_file = json_file
        self.config = self.__read_json()
        #==============Dataset config =====================
        dataset_config = dataset_base.copy(self.config['dataset_config'])
        yolact_config = yolact_base_config.copy({'dataset': dataset_config,
                                                 'num_classes': len(dataset_config.class_names) + 1,
                                                    })  
        yolact_config = yolact_config.copy(self.config['Yolact_config'])
        self.yolact_edge_config = yolact_config.copy(self.config['Yolact_edge_config'])

        # self.yolact_edge_resnet50_config = self.yolact_edge_config.copy({
        #     'name': 'yolact_edge_resnet50',
        #     'backbone': yolact_resnet50_config.backbone
        # })

        #self.yolact_edge_resnet50_config.backbone.pred_scales = self.config["Backbone_config"]["pred_scales"]
        self.yolact_edge_config.backbone.pred_scales = self.config["Backbone_config"]["pred_scales"]
        #==============

    def __read_json(self):
        with open(self.json_file) as f:
            config = json.load(f)
        return config

    def _load_config(self):
        return self.yolact_edge_config
        #return self.yolact_edge_resnet50_config
