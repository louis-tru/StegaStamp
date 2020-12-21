#!/usr/bin/env bash

base=`dirname $0`

cd $base/..

video=$1

if [ "$video" == "" ]; then
	video="out/WechatIMG77.mp4"
fi

python3.7 detector.py \
	--detector_model detector_models/stegastamp_detector \
	--decoder_model saved_models/stegastamp_pretrained \
	--video $video