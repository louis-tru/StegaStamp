#!/usr/bin/env bash

base=`dirname $0`

cd $base/..

image=$1

# if [ "$image" == "" ]; then
# 	image="out/WechatIMG105_target.png"
# fi

python3.7 detector_image.py \
	--detector_model detector_models/stegastamp_detector \
	--decoder_model saved_models/stegastamp_pretrained \
	--image $image