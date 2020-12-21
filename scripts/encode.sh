#!/usr/bin/env bash

base=`dirname $0`

cd $base/..

image=$1
text=$2

if [ "$image" == "" ]; then
	image="out/WechatIMG77.png"
fi

if [ "$text" == "" ]; then
	text="Hello"
fi

python3.7 encode_image.py \
	saved_models/stegastamp_pretrained \
	--image $image  --save_dir out \
	--secret $text