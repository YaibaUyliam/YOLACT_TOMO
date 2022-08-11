export CUDA_VISIBLE_DEVICES=3

python3 train.py\
 --JSON_config="yolact_edge/config.json"\
 --batch_size=8\
 --num_gpus=1\
 --learning_rate=0.00025\
 --momentum=0.9\
 --decay=0.0001\
 --validation_epoch=1\
 --logs="multi_object_3/"\
 --resume="/home/crist_tienngoc/TOMO/Multi_object/yolact_edge/multi_object_2/weight/yolact_edge_52_27136.pth"\

# python train.py\
#  --batch_size=8\
#  --num_gpus=1\
#  --learning_rate=0.001\
#  --momentum=0.9\
#  --decay=0.0001\
#  --validation_epoch=1\
#  --log_folder="logs_model4.7/logs"\
#  --save_interval=2000\
#  --save_folder="logs_model4.7/weight"\
#  --resume='/home/jay2/TOMO/yolact_edge_blister_1_to_5/logs_model4.6/weight/yolact_edge_20_60000.pth'\

# python train.py\
#  --batch_size=8\
#  --num_gpus=1\
#  --learning_rate=0.001\
#  --momentum=0.9\
#  --decay=0.0001\
#  --validation_epoch=1\
#  --log_folder="logs_model4.5/logs"\
#  --save_interval=2000\
#  --save_folder="logs_model4.5/weight"\
#  --resume='/home/jay2/TOMO/yolact_edge_blister_1_to_5/logs_model4.4/weight/yolact_edge_20_60000.pth'\


# Model 4.2


# python train.py\
#  --batch_size=8\
#  --num_gpus=1\
#  --learning_rate=0.001\
#  --momentum=0.9\
#  --decay=0.0001\
#  --validation_epoch=1\
#  --log_folder="logs_model4.2/logs"\
#  --save_interval=2000\
#  --save_folder="logs_model4.2/weight"\
#  --resume='/home//TOMO/yolact_edge_blister_1_to_5/logs_model3.2/weight/yolact_edge_25_66000.pth'\

# # Train blister 1-5