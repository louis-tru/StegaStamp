#!/usr/bin/env bash

base=`dirname $0`

cd $base/..

image=$1

if [ "$image" == "" ]; then
	image="out/WechatIMG77_hidden.png"
fi

python3.7 decode_image.py \
	saved_models/stegastamp_pretrained \
	--image $image