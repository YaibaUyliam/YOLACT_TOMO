
export CUDA_VISIBLE_DEVICES=1
python eval.py\
    --score_threshold=0.2\
    --top_k=20\
    --images="/home/finn/yolact_edge/square_data/test:/home/crist_tienngoc/TOMO/Multi_object/Dataset/results"\
    --fast_nms=True\
    --trained_model="/home/finn/yolact_edge/Square_5-8/weight/yolact_edge_22_115.pth"\
    --display_masks=True\
    --JSON_config="/home/finn/yolact_edge/yolact_edge/square_tomo.json"


