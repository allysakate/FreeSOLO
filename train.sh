# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

#export GLOO_SOCKET_IFNAME=eth0
python train_net.py \
	--num-gpus 1 \
	--config configs/freesolo/freesolo_30k.yaml \
	OUTPUT_DIR training_dir/FreeSOLO\
