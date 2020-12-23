import bchlib, cv2, glob
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def get_intersect(p1, p2, p3, p4):
	s = np.vstack([p1,p2,p3,p4])
	h = np.hstack((s, np.ones((4, 1))))
	l1 = np.cross(h[0], h[1])
	l2 = np.cross(h[2], h[3])
	x, y, z = np.cross(l1, l2)
	if z == 0:
		print('invalid')
		return (0,0)
	return (x/z, y/z)

def poly_area(poly):
	return 0.5*np.abs(np.dot(poly[:,0],np.roll(poly[:,1],1))-np.dot(poly[:,1],np.roll(poly[:,0],1)))

def order_points(pts):
	rect = np.zeros((4, 2), dtype=np.float32)
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def writeImage(buf, name, i = -1, j = -1):
	im = Image.fromarray(buf)
	if i > -1:
		name += '_' + str(i)
	if j > -1:
		name += '_' + str(j)
	im.save('out/' + name + '.png')

def main():
	import argparse
	parser = argparse.ArgumentParser()
	# parser.add_argument('model', type=str)
	parser.add_argument('--detector_model', type=str, required=True)
	parser.add_argument('--decoder_model', type=str, required=True)
	parser.add_argument('--image', type=str, default=None)
	parser.add_argument('--images_dir', type=str, default=None)
	# parser.add_argument('--secret_size', type=int, default=100)
	args = parser.parse_args()

	if args.image is not None:
		files_list = [args.image]
	elif args.images_dir is not None:
		files_list = glob.glob(args.images_dir + '/*')
	else:
		print('Missing input image')
		return

	# Initializing network
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	detector_graph = tf.Graph()
	decoder_graph = tf.Graph()

	with detector_graph.as_default():
		detector_sess = tf.Session()
		detector_model = tf.saved_model.loader.load(detector_sess, [tag_constants.SERVING], args.detector_model)

		detector_input_name = detector_model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
		detector_input = detector_graph.get_tensor_by_name(detector_input_name)

		detector_output_name = detector_model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['detections'].name
		detector_output = detector_graph.get_tensor_by_name(detector_output_name)

	with decoder_graph.as_default():
		decoder_sess = tf.Session()
		decoder_model = tf.saved_model.loader.load(decoder_sess, [tag_constants.SERVING], args.decoder_model)

		decoder_input_name = decoder_model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
		decoder_input = decoder_graph.get_tensor_by_name(decoder_input_name)

		decoder_output_name = decoder_model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
		decoder_output = decoder_graph.get_tensor_by_name(decoder_output_name)

	bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

	ii = 0

	for filename in files_list:
		image = Image.open(filename).convert("RGB")
		# np.array(image.resize((1024, 1024),Image.BICUBIC))
		# image_rgb =  cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
		image_rgb = np.array(image.resize((1024, 1024), Image.BICUBIC))
		# image_rgb =  cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
		# image_rgb = cv2.resize(image_rgb, (1024,1024))
		detector_image_input = np.expand_dims(image_rgb,axis=0) / 255.

		# print('detector_image_input', detector_image_input)

		writeImage(image_rgb, 'aa_image_rgb')

		output_image = detector_sess.run(detector_output,feed_dict={detector_input:detector_image_input})

		output_image = np.array(output_image[0,:,:,:])
		output_image = x = np.argmax(output_image, axis = -1)

		color_codes = np.array([[255,255,255],[0,0,0]])
		out_vis_image = color_codes[output_image.astype(int)]

		writeImage(out_vis_image.astype(np.uint8), 'aa')

		mask_im = cv2.resize(np.float32(out_vis_image), image.size)

		print('image.size', image.size)

		ii += 1
		jj = 0

		contours, _ = cv2.findContours(cv2.cvtColor(mask_im, cv2.COLOR_BGR2GRAY).astype(np.uint8),1,2)
		extrema = np.zeros((8,2))
		corners = np.zeros((4,2))

		print('contours.len', len(contours))

		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area < 1000:
				# print('area < 1000', area)
				continue

			# print('area >= 1000', area)

			hull = cv2.convexHull(cnt)
			if len(hull) < 4:
				# print('len(hull) < 4', len(hull))
				continue

			extrema[0,:] = hull[np.argmax(hull[:,0,0]),0,:]
			extrema[1,:] = hull[np.argmax(hull[:,0,0]+hull[:,0,1]),0,:]
			extrema[2,:] = hull[np.argmax(hull[:,0,1]),0,:]
			extrema[3,:] = hull[np.argmax(-hull[:,0,0]+hull[:,0,1]),0,:]
			extrema[4,:] = hull[np.argmax(-hull[:,0,0]),0,:]
			extrema[5,:] = hull[np.argmax(-hull[:,0,0]-hull[:,0,1]),0,:]
			extrema[6,:] = hull[np.argmax(-hull[:,0,1]),0,:]
			extrema[7,:] = hull[np.argmax(hull[:,0,0]-hull[:,0,1]),0,:]

			extrema_lines = extrema - np.roll(extrema, shift=1, axis=0)
			extrema_len = extrema_lines[:,0]**2 + extrema_lines[:,1]**2
			line_idx = np.sort(extrema_len.argsort()[-4:])
			for c in range(4):
				p1 = extrema[line_idx[(c-1)%4],:]
				p2 = extrema[(line_idx[(c-1)%4]-1)%8,:]
				p3 = extrema[line_idx[c],:]
				p4 = extrema[(line_idx[c]-1)%8,:]
				corners[c,:] = get_intersect(p1, p2, p3, p4)

			new_area = poly_area(corners)
			if new_area / area > 1.5:
				continue

			corners = order_points(corners)
			corners_full_res = corners

			pts_dst = np.array([[0,0],[399,0],[399,399],[0,399]])
			h, status = cv2.findHomography(corners_full_res, pts_dst)
			try:
				warped_im = cv2.warpPerspective(image_rgb, h, (400,400))
				# print('warped_im', warped_im)
				w_im = warped_im.astype(np.float32)
				# w_im = np.array(warped_im, dtype=np.float32)
				w_im /= 255.
			except:
				continue

			print('decode index', ii, jj)
			# print('decode w_im', ii, jj, w_im)

			jj += 1

			writeImage(warped_im, 'aa', ii, jj)

			for im_rotation in range(1):
				w_rotated = np.rot90(w_im, im_rotation)
				recovered_secret = decoder_sess.run([decoder_output],feed_dict={decoder_input:[w_rotated]})[0][0]
				recovered_secret = list(recovered_secret)
				recovered_secret = [int(i) for i in recovered_secret]

				# print('im_rotation', im_rotation, ii, jj)

				packet_binary = "".join([str(bit) for bit in recovered_secret[:96]])
				footer = recovered_secret[96:]
				if np.sum(footer) > 0:
					continue
				packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
				packet = bytearray(packet)

				data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

				bitflips = bch.decode_inplace(data, ecc)

				if bitflips != -1:
					print('Num bits corrected: ', bitflips)
					try:
						print('----', data.decode("utf-8"))
						code = data.decode("utf-8")
					except:
						continue

if __name__ == "__main__":
	main()
