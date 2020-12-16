import argparse
import os
import time
import pandas as pd
import cv2
import numpy as np

from deep_sort import DeepSort
from util import draw_bboxes, draw_dead_bboxes, draw_frameNum
from deep_sort import coord_mapper
import pickle

import natsort
import glob
from itertools import product

class Detector(object):
	def __init__(self, detections_file : str, resolution : tuple, fps : int, input_images_dir : str, 
				output_video_path : str, output_result_path : str, use_cuda : bool, lambdaParam : float,
				max_dist : float, min_confidence : float, nms_max_overlap : float,max_iou_distance : float, 
				max_age : int, n_init : int, nn_budget : int, model_path = 'deep_sort/deep/checkpoint/ckpt.t7',
				early_stopping = None):
		
		self.detections_file = detections_file # A pickle fájl amiben az összes detekció benne van
		self.input_images_dir = input_images_dir # A mappa ahol a 2.5K-s képek vannak {frameNum}.jpg formátumban
		self.output_video_path = output_video_path # Ahova a vizualizálandó videót mentem
		self.output_result_path = output_result_path # Ahová a kimenetet mentem CSV formátumba
		self.early_stopping = early_stopping
		
		assert self.output_result_path is not None and self.detections_file is not None
		
		self._use_cuda = use_cuda
		self.fps = fps
		self.resolution = resolution
		# Initialize coordinate mapper
		self.myCoordMapper = coord_mapper.CoordMapperCSG(match_code='HUN-BEL 1. Half')

		self.deepsort = DeepSort(model_path=model_path, lambdaParam=lambdaParam, coordMapper=self.myCoordMapper, 
						max_dist=max_dist, min_confidence=min_confidence, nms_max_overlap=nms_max_overlap, 
						max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init, nn_budget=nn_budget, 
						use_cuda=self._use_cuda, resolution = (self.resolution[0] * 2, self.resolution[1]), fps=self.fps)


	def initVideoOutput(self):
		if self.input_images_dir is None or self.output_video_path is None:
			return
		
		# Itt minden kép 2.5K-s
		imgList = natsort.natsorted(glob.glob(self.input_images_dir))
		self.dict_frame2path = {int(path.split('/')[-1].split('.')[0]) : path for path in imgList}

		self.out_vid_height, self.out_vid_width = self.resolution[1], self.resolution[0] * 2

		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		self.output = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (self.out_vid_width, self.out_vid_height))

	def writeVideoOutput(self, frameNum, list_detections, tracks, deadtracks, 
							draw_detections=True, draw_tracks=True, draw_deadtracks=True):
		if self.input_images_dir is None or self.output_video_path is None:
			return
		
		# Beolvasom a megfelelő képkockát
		img = cv2.imread(self.dict_frame2path[frameNum])
		# Resizeolom
		img = cv2.resize(img, (self.out_vid_width, self.out_vid_height), interpolation = cv2.INTER_AREA)
		
		# Detection Boxot rajzolok rá...
		if draw_detections:
			bb_xyxy = [det['box'] for det in list_detections]
			all1 = [None]*len(bb_xyxy)
			img = draw_bboxes(img, bb_xyxy, all1)

		resizeFactor = self.resolution[0] / 2560

		# Trackeket rajzolok rá
		if len(tracks) > 0 and draw_tracks:
			bbox_xyxy = tracks[:, :4] * resizeFactor
			identities = tracks[:, 4]
			img = draw_bboxes(img, bbox_xyxy, identities)

		# Draw boxes for dead tracks for debugging
		if len(deadtracks) > 0 and draw_deadtracks:
			bbox_xyxy = [x[:4] for x in deadtracks]
			bbox_xyxy = [x * resizeFactor for x in bbox_xyxy]
			labels = [x[4] for x in deadtracks]
			img = draw_dead_bboxes(img, bbox_xyxy, labels)

		# Frame Numbert is felrajzolom
		img = draw_frameNum(img, (self.out_vid_width // 2, self.out_vid_height // 10), frameNum)
		
		# Write to file
		self.output.write(img)

	def closeVideoOutput(self):
		if self.input_images_dir is None or self.output_video_path is None:
			return
		self.output.release()
	
	def writeResults(self, frameNum, tracks, ts_start, ts_end):
		'''
		tracks : np.array = List[ [x1, y1, x2, y2, tID, xWorld, yWorld] ]
		'''
		if len(tracks) == 0:
			return
		
		list_tracks = [ {'frame' : frameNum, 'ts_start' : ts_start, 'ts_end' : ts_end, 
							'xTL' : xTL, 'yTL' : yTL, 'xBR' : xBR, 
							'yBR' : yBR, 'tID' : tID, 'xWorld' : xWorld, 'yWorld' : yWorld}
						for xTL, yTL, xBR, yBR, tID, xWorld, yWorld
						in tracks
						]
	
		pd.DataFrame(list_tracks).to_csv(self.output_result_path, mode='a', index=None, 
											header=(not os.path.exists(self.output_result_path)))

	def doTrackingOnDetectionFile(self):
		'''
		A detectionons pickle fájl így néz ki:
		dict( frameNum : List[dict_detection])

		dict_detection = {'worldXY' : tuple(X, Y), 'box' : [xTL, yTL, xBR, yBR], 
							'bigBox' : [xTL, yTL, xBR, yBR], 'score' : float, 'image' : np.array(NxM),
							'team' = ['red', 'yellow', 'other', 'more player from different team']}
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
			#list_dets = dict_detections[frameNum]
			# Leszűröm csak a hazai detekciókat
			list_dets = [x for x in dict_detections[frameNum] if x['team'] in ['red']]
			print('Frame', frameNum)
			self.doTrackingForOneFrame(frameNum, list_dets)
			
			if self.early_stopping is not None and frameNum >= self.early_stopping:
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

		ts_start = time.time()

		# Létrehozom a BBoxokat, átalakítva, úgy hogy cX, cY, W, H legyen
		# FONTOS: Mivel ki fogom plotolni ezért a kisképen lévő bboxok kellenek
		bbox_xcycwh = [det['bigBox'] for det in list_of_detections]
		bbox_xcycwh = [[(xBR + xTL) / 2, (yBR + yTL) / 2, (xBR - xTL), (yBR - yTL) ] for xTL, yTL, xBR, yBR in bbox_xcycwh]
		cls_conf = [det['score'] for det in list_of_detections]
		bbox_imgs = [det['image'] for det in list_of_detections]
		worldCoordXY = [det['worldXY'] for det in list_of_detections]

		outputs, deadtracks = self.deepsort.update(bbox_xcycwh, cls_conf, bbox_imgs, worldCoordXY)

		ts_end = time.time()

		self.writeVideoOutput(frameNum, list_of_detections, outputs, deadtracks)
		
		self.writeResults(frameNum, outputs, ts_start, ts_end)

		# TODO: Save results to file
		# Úgy ahogy a workernél a pandasos mókát csináltam


def runOneCombination(fps = 6, resolution = (2560, 1440), lambdaParam  = 1.0, max_age=12, nn_budget = 50, early_stopping = 1800, video_output=False):
		# Konstans paraméterek
	model_path = '/mnt/ckpt.t7'
	input_images_dir = '/mnt/images/*.jpg'
	
	# Mérési paraméterek
	# early_stopping = 30*60 # sec * 60FPS

	# DeepSort paraméterek
	use_cuda = True
	
	max_dist = 1.0
	min_confidence = 0.0 # Fölösleges, mert a detekció során már megcsináltam
	nms_max_overlap = 1.1 # Fölösleges, mert már a detekció során mindent kiszűrtem
	max_iou_distance = 0.7
	n_init = 3

	detections_file = f'/mnt/outputs/{resolution[0]}_30fps/detections_v4.pickle'
	output_result_path = f'/mnt/outputs/{resolution[0]}_30fps/results/{resolution[0]}@fps={fps}@lambda={lambdaParam}@maxage={max_age}@nnbudget={nn_budget}.csv'
	
	if video_output:
		output_video_path = f'/mnt/outputs/{resolution[0]}_30fps/results/{resolution[0]}@fps={fps}@lambda={lambdaParam}@maxage={max_age}@nnbudget={nn_budget}.avi' 
	else:
		output_video_path = None
	

	print(f'{resolution[0]}@fps={fps}@lambda={lambdaParam}@maxage={max_age}@nnbudget={nn_budget}')
	myTracker = Detector(detections_file=detections_file, resolution=resolution, fps=fps, input_images_dir=input_images_dir, 
						output_video_path=output_video_path, output_result_path=output_result_path, use_cuda=use_cuda,
						lambdaParam=lambdaParam, max_dist=max_dist, min_confidence=min_confidence, 
						nms_max_overlap=nms_max_overlap, max_iou_distance=max_iou_distance, max_age=max_age, 
						n_init=n_init, nn_budget=nn_budget, model_path=model_path, early_stopping=early_stopping)
	
	myTracker.doTrackingOnDetectionFile()


if __name__ == "__main__":
	list_resolution = [(2560, 1440), (2048, 1152), (1920, 1080), (1536, 864), (1280, 720)]
	list_fps = [30, 15, 10, 6]
	list_lambda = list(np.arange(0.0, 1.1, 0.25))
	list_maxAgeSec = [1, 2, 3]
	list_nnbudget = [1, 3, 5]

	list_combinations = list(product(list_resolution, list_fps, list_lambda, list_maxAgeSec, list_nnbudget))
	print(len(list_combinations))
	# for actResolution, actFps, actLambda, actMaxAge, actNNBudget) in list_combinations:
	# 	runOneCombination(resolution=actResolution, fps=actFps, lambdaParam=actLambda, 
	# 						max_age=actFps*actMaxAge, nn_budget=actNNBudget*actFps, early_stopping=None)
	# FPS = 6
	# for lambdaP in np.arange(0.0, 1.1, 0.25):
	# 	for maxAgeSec in [1, 2, 3]:
	# 		for nnBudgetSec in [1, 3, 5]:
	# 			runOneCombination(fps=FPS, lambdaParam=lambdaP, max_age=FPS*maxAgeSec, nn_budget=nnBudgetSec*FPS)
	# FPS = 15
	# runOneCombination(fps=FPS, lambdaParam=1.0, max_age=FPS*2, nn_budget=FPS*3)
	# FPS = 6
	# runOneCombination(fps=FPS, lambdaParam=1.0, max_age=FPS*2, nn_budget=FPS*3)
	FPS = 6
	runOneCombination(resolution=(1280, 720), fps=FPS, lambdaParam=1.0, max_age=FPS*2, nn_budget=FPS*3, video_output=True)
	# runOneCombination(fps=FPS, lambdaParam=0.75, max_age=12, nn_budget=18)
	# runOneCombination(fps=FPS, lambdaParam=0.5, max_age=12, nn_budget=18)