import bchlib
import glob
import os
from PIL import Image,ImageOps,ImageChops
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('model', type=str)
	parser.add_argument('--image', type=str, default=None)
	parser.add_argument('--images_dir', type=str, default=None)
	parser.add_argument('--save_dir', type=str, default=None)
	parser.add_argument('--secret', type=str, default='Stega!!')
	args = parser.parse_args()

	if args.image is not None:
		files_list = [args.image]
	elif args.images_dir is not None:
		files_list = glob.glob(args.images_dir + '/*')
	else:
		print('Missing input image')
		return

	sess = tf.InteractiveSession(graph=tf.Graph())

	model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

	input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
	input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)

	input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
	input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

	output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
	output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)

	output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
	output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

	# print('----------', model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY])

	width = 400
	height = 400

	bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

	# print('--------', args.secret, len(args.secret))

	data = bytearray(args.secret, 'utf-8')

	if len(data) > 7:
		print('Error: Can only encode 56bits (7 characters) with ECC')
		return

	# bytearray
	data = data + bytes(' '*(7-len(data)),'utf-8')
	# print('--------', data)
	ecc = bch.encode(data)
	packet = data + ecc

	# print('packet', len(packet))

	packet_binary = ''.join(format(x, '08b') for x in packet)
	secret = [int(x) for x in packet_binary]
	secret.extend([0,0,0,0])

	# print('secret', len(secret))

	if args.save_dir is not None:
		if not os.path.exists(args.save_dir):
			os.makedirs(args.save_dir)
		size = (width, height)
		for filename in files_list:
			rawImage = Image.open(filename)
			image = rawImage.convert("RGB")
			# print('----------------', image)
			# print('image size', rawSize)
			# image = np.array(ImageOps.fit(image, size, Image.BICUBIC, 0.0, (1, 1)),dtype=np.float32)
			image = np.array(image.resize(size,Image.BICUBIC), dtype=np.float32)
			image /= 255.

			feed_dict = { input_secret: [secret], input_image: [image] }

			save_name = filename.split('/')[-1].split('.')[0]
			# raw_img = (image * 255).astype(np.uint8)

			hidden_img, residual = sess.run([output_stegastamp, output_residual], feed_dict=feed_dict)

			rescaled = hidden_img[0]
			rescaled = (rescaled * 255).astype(np.uint8)
			im = Image.fromarray(np.array(rescaled))
			im.save(args.save_dir + '/' + save_name + '_hidden.png')

			residual = residual[0]+.5
			residual = (residual * 255).astype(np.uint8)
			# print('----', len(residual), len(residual[0]), len(residual[0][0]), residual)
			im = Image.fromarray(np.squeeze(np.array(residual)))
			im.save(args.save_dir + '/' + save_name + '_residual.png')

			### 保存最终 image
			im = im.convert("RGBA")
			im.putalpha(int(255 * 0.3)) # 设置每个像素点颜色的透明度 = 30%, 这个值越大识别率越高
			im = im.resize(rawImage.size)
			im = ImageChops.hard_light(rawImage, im) # 使用强光模式叠加
			im = Image.alpha_composite(rawImage, im)

			im.save(args.save_dir + '/' + save_name + '_target.png')

if __name__ == "__main__":
	main()
