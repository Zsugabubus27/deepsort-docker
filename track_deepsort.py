import argparse
import os
import time
import pandas as pd
import cv2

from deep_sort import DeepSort
from util import draw_bboxes, draw_dead_bboxes, draw_frameNum
from deep_sort import coord_mapper
import pickle

import natsort
import glob

class Detector(object):
	def __init__(self, detections_file : str, resolution : tuple, fps : int, input_images_dir : str, 
				output_video_path : str, output_result_path : str, use_cuda : bool, lambdaParam : float,
				max_dist : float, min_confidence : float, nms_max_overlap : float,max_iou_distance : float, 
				max_age : int, n_init : int, nn_budget : int, model_path = 'deep_sort/deep/checkpoint/ckpt.t7'):
		
		self.detections_file = detections_file # A pickle fájl amiben az összes detekció benne van
		self.input_images_dir = input_images_dir # A mappa ahol a 2.5K-s képek vannak {frameNum}.jpg formátumban
		self.output_video_path = output_video_path # Ahova a vizualizálandó videót mentem
		self.output_result_path = output_result_path # Ahová a kimenetet mentem CSV formátumba
		
		assert self.output_result_path is not None and self.detections_file is not None
		
		self._use_cuda = use_cuda
		self.fps = fps
		self.resolution = resolution
		# Initialize coordinate mapper
		self.myCoordMapper = coord_mapper.CoordMapperCSG(match_code='HUN-BEL 1. Half')

		self.deepsort = DeepSort(model_path=model_path, lambdaParam=lambdaParam, coordMapper=self.myCoordMapper, 
						max_dist=max_dist, min_confidence=min_confidence, nms_max_overlap=nms_max_overlap, 
						max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init, nn_budget=nn_budget, 
						use_cuda=self._use_cuda, resolution = (self.resolution[0] * 2, self.resolution[1]))


	def initVideoOutput(self):
		if self.input_images_dir is None or self.output_video_path is None:
			return
		
		# Itt minden kép 2.5K-s
		imgList = natsort.natsorted(glob.glob(self.input_images_dir))
		self.dict_frame2path = {int(path.split('/')[-1].split('.')[0]) : path for path in imgList}

		self.out_vid_height, self.out_vid_width = self.resolution[1], self.resolution[0] * 2

		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		self.output = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (self.out_vid_width, self.out_vid_height))

	def writeVideoOutput(self, frameNum, list_detections, tracks):
		if self.input_images_dir is None or self.output_video_path is None:
			return
		# Beolvasom a megfelelő képkockát
		img = cv2.imread(self.dict_frame2path[frameNum])
		# Resizeolom
		img = cv2.resize(img, (self.out_vid_width, self.out_vid_height), interpolation = cv2.INTER_AREA)
		
		# Detection Boxot rajzolok rá...
		bb_xyxy = [det['box'] for det in list_detections]
		all1 = [None]*len(bb_xyxy)
		img = draw_bboxes(img, bb_xyxy, all1)

		# Trackeket rajzolok rá
		if len(tracks) > 0:
			bbox_xyxy = tracks[:, :4]
			identities = tracks[:, -1]
			img = draw_bboxes(img, bbox_xyxy, identities)

		self.out_vid_height, self.out_vid_width
		# Frame Numbert is felrajzolom
		img = draw_frameNum(img, (self.out_vid_width // 2, self.out_vid_height // 10), frameNum)
		
		# Write to file
		self.output.write(img)

	def closeVideoOutput(self):
		if self.input_images_dir is None or self.output_video_path is None:
			return
		self.output.release()

	def doTrackingOnDetectionFile(self):
		'''
		A detectionons pickle fájl így néz ki:
		dict( frameNum : List[dict_detection])

		dict_detection = {'worldXY' : tuple(X, Y), 'box' : [xTL, yTL, xBR, yBR], 
							'bigBox' : [xTL, yTL, xBR, yBR], 'score' : float, 'image' : np.array(NxM)}
		'''
		# Calc frame skipping
		assert 30 % self.fps == 0 
		stepFrame = 60 // self.fps
		
		print('Reading detections pickle')
		# Read in detection pickle
		with open(self.detections_file, 'rb') as handle:
			dict_detections = pickle.load(handle)
		print('Done')
		
		self.initVideoOutput()

		for frameNum in sorted(dict_detections.keys()):
			if (frameNum % stepFrame) != 0:
				continue
			list_dets = dict_detections[frameNum]
			print('Frame', frameNum)
			self.doTrackingForOneFrame(frameNum, list_dets)
			if frameNum >= 1800:
				break

		# Végül bezárom a videót ha van
		self.closeVideoOutput()

	def doTrackingForOneFrame(self, frameNum, list_of_detections):
		'''
		list_of_detections : List[
									{'worldXY' : tuple(X, Y), 'box' : [xTL, yTL, xBR, yBR], 
									'bigBox' : [xTL, yTL, xBR, yBR], 'score' : float, 'image' : np.array(NxM)}
								]
		'''

		# Létrehozom a BBoxokat, átalakítva, úgy hogy cX, cY, W, H legyen
		# FONTOS: Mivel ki fogom plotolni ezért a kisképen lévő bboxok kellenek
		bbox_xcycwh = [det['box'] for det in list_of_detections]
		bbox_xcycwh = [[(xBR + xTL) / 2, (yBR + yTL) / 2, (xBR - xTL), (yBR - yTL) ] for xTL, yTL, xBR, yBR in bbox_xcycwh]
		cls_conf = [det['score'] for det in list_of_detections]
		bbox_imgs = [det['image'] for det in list_of_detections]
		worldCoordXY = [det['worldXY'] for det in list_of_detections]
		
		#print(bbox_xcycwh, cls_conf, worldCoordXY)


		outputs, deadtracks = self.deepsort.update(bbox_xcycwh, cls_conf, bbox_imgs, worldCoordXY)

		self.writeVideoOutput(frameNum, list_of_detections, outputs)
		
		# TODO: Save results to file
		# Úgy ahogy a workernél a pandasos mókát csináltam


# def parse_args():
# 	# TODO:
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument("--video_path", type=str, default=None)
# 	parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
# 	parser.add_argument("--save_path", type=str, default="demo.avi")
# 	parser.add_argument("--use_cuda", type=str, default="True")
# 	return parser.parse_args()


if __name__ == "__main__":
	# args = parse_args()
	# if args.video_path is None:
	# 	print('Debugging...')
	# 	args.use_cuda = "False"
	# 	args.save_path = "/mnt/data/mlsa20_cr/out/HUN_BEL_second_half.avi"
	# 	args.result_path = "/mnt/data/mlsa20_cr/out/HUN_BEL_second_half.txt"
	# 	#args.video_path = "/home/dobreff/work/Dipterv/MLSA20/data/video_46000_47000.avi"
	# 	args.imgs_path = "/mnt/data/mlsa20_cr/src/masodik_felido/*.png"
	# 	with Detector(args) as det:
	# 		det.detect()
	# else:
	# 	with Detector(args) as det:
	# 		det.detect()

	# Mérési paraméterek
	fps = 10
	resolution = (2560, 1440) # TODO: Okossabban 
	detections_file = f'/mnt/match_videos/dobreff/detections/{resolution[0]}_30fps/detections.pickle' # TODO: Okosabban
	output_video_path = f'/mnt/match_videos/dobreff/outputs/{resolution[0]}_{fps}.avi' 
	output_result_path = f'/mnt/match_videos/dobreff/outputs/{resolution[0]}_{fps}.csv'
	
	# DeepSort paraméterek
	use_cuda = False
	lambdaParam  = 0.6
	max_dist = 1.0
	min_confidence = 0.1
	nms_max_overlap = 0.7
	max_iou_distance = 0.7
	max_age = fps*3
	n_init = 3
	nn_budget = 50
	
	# Konstans paraméterek
	model_path = 'deep_sort/deep/checkpoint/ckpt.t7'
	input_images_dir = '/mnt/match_videos/dobreff/images/images/*.jpg'

	myTracker = Detector(detections_file=detections_file, resolution=resolution, fps=fps, input_images_dir=input_images_dir, 
						output_video_path=output_video_path, output_result_path=output_result_path, use_cuda=use_cuda,
						lambdaParam=lambdaParam, max_dist=max_dist, min_confidence=min_confidence, 
						nms_max_overlap=nms_max_overlap, max_iou_distance=max_iou_distance, max_age=max_age, 
						n_init=n_init, nn_budget=nn_budget, model_path=model_path)
	
	myTracker.doTrackingOnDetectionFile()