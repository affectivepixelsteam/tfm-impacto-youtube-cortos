# USAGE
# python static_saliency.py --image images/neymar.jpg

# import the necessary packages
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to input image")
# args = vars(ap.parse_args())

# load the input image
# image = cv2.imread(args["image"])

def salistatic(years, path):

	for y in years:
		path_to_years = os.path.join(path, y)
		dirs = os.listdir(path_to_years)
		for dire in dirs:
			path_to_dir = os.path.join(path_to_years, dire)
			path_to_frames = os.path.join(path_to_dir, 'frames')
			folder_to_save_saliency_sf = os.path.join(path_to_dir, 'output_saliency_fine')
			folder_to_save_saliency_sr = os.path.join(path_to_dir, 'output_saliency_residual')
			folder_to_save_thres = os.path.join(path_to_dir, 'output_thres')

			if (not os.path.exists(folder_to_save_saliency_sf) and not os.path.exists(folder_to_save_saliency_sr) and
					not os.path.exists(folder_to_save_thres)):
				os.makedirs(folder_to_save_saliency_sf)
				os.makedirs(folder_to_save_saliency_sr)
				os.makedirs(folder_to_save_thres)
				frames = os.listdir(path_to_frames)
				print('frames check ' + dire)
				for f in frames:
					file = os.path.join(path_to_frames, f)
					image = cv2.imread(file)
					# Initialize OpenCV's static saliency spectral residual detector and compute the saliency map
					saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
					(success, saliencyMap_sr) = saliency.computeSaliency(image)
					saliencyMap_sr = (saliencyMap_sr * 255).astype("uint8")
					#cv2.imshow("Image", image)
					#cv2.imshow("Output", saliencyMap)
					#cv2.waitKey(0)

					# Initialize OpenCV's static fine grained saliency detector and compute the saliency map
					saliency = cv2.saliency.StaticSaliencyFineGrained_create()
					(success, saliencyMap_sf) = saliency.computeSaliency(image)
					saliencyMap_sf = (saliencyMap_sf * 255).astype("uint8")

					# If we would like a *binary* map that we could process for contours, compute convex hull's, extract
					# bounding boxes, etc., we can additionally threshold the saliency map
					threshMap = cv2.threshold(saliencyMap_sf.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

					# Save the images
					name_sf = f[0:16] + '_saliency_sf.jpg'
					name_sr = f[0:16] + '_saliency_sr.jpg'
					name_t = f[0:16] + '_thres.jpg'
					filename_saliency_sf = os.path.join(folder_to_save_saliency_sf, name_sf)
					filename_saliency_sr = os.path.join(folder_to_save_saliency_sr, name_sr)
					filename_thres = os.path.join(folder_to_save_thres, name_t)

					cv2.imwrite(filename_saliency_sf, saliencyMap_sf)
					cv2.imwrite(filename_saliency_sr, saliencyMap_sr)
					cv2.imwrite(filename_thres, threshMap)

					# Show the images
					# cv2.imshow("Image", image)
					# cv2.imshow("Output", saliencyMap)
					# cv2.imshow("Thresh", threshMap)
					# cv2.waitKey(0)


years_no_finalistas = ['2017', '2018', '2019']
years_finalistas = ['2014', '2015', '2016', '2017', '2018', '2019']

fin = '/mnt/pgth04b/DATABASES_CRIS/FINALISTAS_ORIGINAL/DATABASES'
no_fin = '/mnt/pgth04b/DATABASES_CRIS/NO_FINALISTAS/DATABASES'

salistatic(years_finalistas, fin)
salistatic(years_no_finalistas, no_fin)