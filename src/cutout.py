import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
from skimage.transform import resize
from skimage import img_as_float
from skimage.io import imshow, imread

# big thanks to this answer for the sketch
# https://stackoverflow.com/a/63647647/1176872

def show(im):
	im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	plt.figure()
	plt.axis("off")
	plt.imshow(im)
	wm = plt.get_current_fig_manager()
	wm.window.state('zoomed')
	plt.show()

def cluster(im, n_clusters):
	im = im.reshape((im.shape[0] * im.shape[1], 3))
	km = KMeans(n_clusters=n_clusters, random_state=0)
	km.fit(im)
	
	counts = {}
	reps = km.cluster_centers_

	# count colors per label
	for i in range(len(im)):
		if km.labels_[i] not in counts:
			counts[km.labels_[i]] = {}
		rgb = tuple(im[i])
		if rgb not in counts[km.labels_[i]]:
			counts[km.labels_[i]][rgb] = 0	
		counts[km.labels_[i]][rgb] += 1

	# remap representative to most prominent color for ea label
	for label, hist in counts.items():
		flat = sorted(hist.items(), key=lambda x: x[1], reverse=True)
		reps[label] = flat[0][0]

	return km.cluster_centers_, km.labels_

def remap_colors(im, reps, labels):
	orig_shape = im.shape
	im = im.reshape((im.shape[0] * im.shape[1], 3))
	for i in range(len(im)):
		im[i] = reps[labels[i]]
	return im.reshape(orig_shape)

def find_contours(im, reps, min_area): 
	contours = []
	for rep in reps:
		mask = cv2.inRange(im, rep-1, rep+1)
		# show(mask)
		conts, _ = cv2.findContours(
			mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		for cont in conts:
			area = cv2.contourArea(cont)
			if area >= min_area:
				contours.append((area, cont, rep))
	contours.sort(key=lambda x: x[0], reverse=True)
	return contours

def cutout_filter():
	argp = argparse.ArgumentParser(description='Cutout filter.')
	argp.add_argument('-in-file', type=str, required=True)
	argp.add_argument(
		'-out-file', type=str, 
		help='If empty output is displayed with pyplot.')
	argp.add_argument(
		'-n-clusters', type=int, default=3,
		help='Number of colors.')
	argp.add_argument(
		'-blur-kernel', type=int, default=1,
		help='The size of the blur kernel.')
	argp.add_argument(
		'-min-area', type=int, default=3,
		help='Contours with areas smaller than this are dropped.')
	argp.add_argument(
		'-poly-epsilon', type=float, default=3,
		help='Maximum distance between original contour and its drawing.')
	argp.add_argument(
		'-quiet', action='store_true', default=False,
		help='Do not print progress.')
	argp.add_argument(
		'-final-blur', action='store_true', default=False,
		help='3 pixel blur on the output to clean up the jaggies.')
	argp.add_argument(
		'-slice', action='store_true', default=False,
		help='Output N layers masked to their representative color.')
	args = argp.parse_args()

	if args.blur_kernel % 2 != 1:
		print('-blur-kernel must be an odd number')
		return 1

	if args.min_area < 1:
		print('-min-area must be at least 1')
		return 1

	if not args.quiet:
		print(f'Reading file {args.in_file}...')

	orig = cv2.imread(args.in_file)
	im = orig.copy()
	# show(im)

	if not args.quiet:
		print(f'Blurring with size {args.blur_kernel}...')

	im = cv2.GaussianBlur(im, (args.blur_kernel, args.blur_kernel), 0)
	# show(im)

	if not args.quiet:
		print(f'Clustering around {args.n_clusters} colors...')

	reps, labels = cluster(im, args.n_clusters)

	if not args.quiet:
		print('Remapping image to representative colors...')

	im = remap_colors(im, reps, labels)
	# show(im)

	if not args.quiet:
		print(f'Finding contours with area gte {args.min_area}...')

	contours = find_contours(im, reps, args.min_area)

	if not args.quiet:
		print(f'Drawing...')

	canvas = np.zeros(orig.shape, np.uint8)
	for area, cont, rep in contours:
		approx = cv2.approxPolyDP(cont, args.poly_epsilon, True)
		cv2.drawContours(canvas, [approx], -1, rep, -1)

	if args.final_blur:
		canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

	return canvas, orig

def color_burn(base, blend):
    result = 1 - (1 - base) / np.maximum(blend, 1e-10)  # Avoid division by zero
    return np.clip(result, 0, 1)