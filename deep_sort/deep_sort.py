import numpy as np

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ['DeepSort']


class DeepSort(object):
	def __init__(self, model_path, lambdaParam, coordMapper, resolution, fps, max_dist=0.2, min_confidence=0.3, 
				nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
		'''
			Parameters
			----------
			min_confidence : float
				Detections with confidence below this threshold are not tracked.
			nms_max_overlap : float
				Used for non-max suppression. ROIs that overlap more than this values are suppressed.
			nn_budget : int
				Size of the array that stores the feature vectors of tracks. Lk in paper.
			max_iou_distance : int
				Miután appearance és KF alapján matcheltük a trackeket és a detectionöket, 
				maradnak olyan detectionök és trackek amiket nem matcheltünk.
				Ekkor az olyan nem matchelt trackeket amik 1 idősek VAGY unconfirmedek (újak) 
				és a maradék detectiont IOU távolság alapján matchelünk.
				Maximum ilyen messze lehetnek egymástól.
			n_init : int
				Number of consecutive detections before the track is confirmed. The
				track state is set to `Deleted` if a miss occurs within the first
				`n_init` frames.
			lambdaParam : int
				Ha 1 -> KF alapú
				Ha 0 -> kép alapú
			max_dist : int
				Appearance feature vectorok távolságának maximuma.
				Minden trackhez kiszámoljuk, hogy a detekciók közül melyik hasonlít rá a legjobban.
				Ezt a track összes! Mind az {nn_budget} darab képéhez nézzük. 
				És a sok kép (history) közül a leghasonlóbb távolságát adja a track[i] - detections[j] távolságaként.
				Azokat a párosokat ahol ezen {max_dist}-nél nagyobb a távolság "végtelennel" helyettesíti.
				Ha ez 1.0 akkor semmit se helyettesít végtelennel
			resolution : tuple(width, height)
				Resolution of the video! Not 2560x1440 but 5120x1440
		'''
		
		self.width, self.height = resolution

		self.min_confidence = min_confidence 
		self.nms_max_overlap = nms_max_overlap
		self.lambdaParam = lambdaParam

		# To map coordinates on image to coordinates in world
		self.coordMapper = coordMapper

		# Appearance feature extractor CNN
		self.extractor = Extractor(model_path, use_cuda=use_cuda)

		max_cosine_distance = max_dist
		Lk = nn_budget
		metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, lambdaParam,Lk)
		self.tracker = Tracker(metric, lambdaParam=lambdaParam, max_iou_distance=max_iou_distance,
								 max_age=max_age, n_init=n_init, coordMapper=coordMapper, fps=fps)

	def update(self, bbox_xywh, confidences, bbox_imgs, worldCoordXY):
		'''
		bbox_xywh = xc, yc, w, h
		confidences = list of confidence scores
		bbox_imgs = List[bbox image]
		worldCoordXY = List[ tuple(X, Y) ]
		'''
		# generate detections
		# 1. Generates features from the cropped BB images
		# 2. Transforms the BB coordinates from XcYcWH to TopLeftWH
		# 3. Creates Detection object from detections AND filters only detections which detection confidence is higher than min_confidence
		features = self._get_features(bbox_imgs)
		bbox_tlwh = self._xywh_to_tlwh(np.array(bbox_xywh))
		detections = [Detection(tlwh, conf, feat, worldXY) for tlwh, conf, feat, worldXY in 
						zip(bbox_tlwh, confidences, features, worldCoordXY) if conf > self.min_confidence]
		assert len(detections) == len(worldCoordXY), f'Min-confidence {self.min_confidence} fucks it up'

		# run on non-maximum supression
		# 1. Creates a list of the TopLeftWH coordinates of BBs
		# 2. Creates a list of the detection confidence scores
		# 3. Runs Non-Max suppression which eliminates the overlapping BBs
		# 4. overwrites the detections list with a list that only contains the selected ones
		boxes = np.array([d.tlwh for d in detections])
		scores = np.array([d.confidence for d in detections])
		indices = non_max_suppression( boxes, self.nms_max_overlap, scores)
		assert len(indices) == len(detections), f'NMS with {self.nms_max_overlap} fucks it up'
		detections = [detections[i] for i in indices]

		# update tracker
		self.tracker.predict()
		self.tracker.update(detections)

		# output bbox identities
		# TODO: Debug trackeket kivenni.
		outputs = []
		deadtracks = []
		for track in self.tracker.tracks:
			if not track.is_confirmed() or track.time_since_update >= 1:
				box = track.to_tlwh()
				x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
				xWorld, yWorld = track.to_worldXY()
				track_id = str(track.track_id) + " " + str(track.time_since_update)
				deadtracks.append([x1,y1,x2,y2,track_id, xWorld, yWorld])
				continue
			box = track.to_tlwh()
			xWorld, yWorld = track.to_worldXY()
			x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
			track_id = track.track_id
			outputs.append(np.array([x1,y1,x2,y2,track_id, xWorld, yWorld], dtype=np.float))
		if len(outputs) > 0:
			outputs = np.stack(outputs,axis=0)
		return outputs, deadtracks
	"""
	TODO:
		Convert bbox from xc_yc_w_h to xtl_ytl_w_h
	Thanks JieChen91@github.com for reporting this bug!
	"""
	@staticmethod
	def _xywh_to_tlwh(bbox_xywh):
		_bbox_xywh = bbox_xywh.copy()
		_bbox_xywh[:,0] = _bbox_xywh[:,0] - _bbox_xywh[:,2]/2.
		_bbox_xywh[:,1] = _bbox_xywh[:,1] - _bbox_xywh[:,3]/2.
		return _bbox_xywh


	def _xywh_to_xyxy(self, bbox_xywh):
		'''
		Convert BB coords from XcYcWH to TopLeftXY;BottomRightXY
		'''
		x,y,w,h = bbox_xywh
		x1 = max(int(x-w/2),0)
		x2 = min(int(x+w/2),self.width-1)
		y1 = max(int(y-h/2),0)
		y2 = min(int(y+h/2),self.height-1)
		return x1,y1,x2,y2
	
	def _tlwh_to_footXY(self, bbox_tlwh):
		'''
		Convert BB coords from TopLeftWH to player's foot XY
		'''
		# TODO: Kell ide hogy int legyen?
		x,y,w,h = bbox_tlwh
		x_foot = x + w/2
		y_foot = y + h
		return x_foot, y_foot

	def _tlwh_to_xyxy(self, bbox_tlwh):
		"""
		TODO:
			Convert bbox from xtl_ytl_w_h to xc_yc_w_h
		Thanks JieChen91@github.com for reporting this bug!
		"""
		x,y,w,h = bbox_tlwh
		x1 = max(int(x),0)
		x2 = min(int(x+w),self.width-1)
		y1 = max(int(y),0)
		y2 = min(int(y+h),self.height-1)
		return x1,y1,x2,y2

	def _xyxy_to_tlwh(self, bbox_xyxy):
		x1,y1,x2,y2 = bbox_xyxy

		t = x1
		l = y1
		w = int(x2-x1)
		h = int(y2-y1)
		return t,l,w,h

	def _get_features(self, im_crops):
		'''
		im_crops = List[np.array]
		'''
		if im_crops:
			features = self.extractor(im_crops)
		else:
			features = np.array([])
		return features


	# def _get_features(self, bbox_xywh, ori_img):
	# 	'''
	# 	bbox_xywh = Xc, Yc, W, H
	# 	'''
	# 	im_crops = []
	# 	for box in bbox_xywh:
	# 		# For every BB 
	# 		# 1. Convert from XcYcWH to TLBR coords
	# 		# 2. crop the BB images from the original images
	# 		x1,y1,x2,y2 = self._xywh_to_xyxy(box)
	# 		im = ori_img[y1:y2,x1:x2]
	# 		im_crops.append(im)
	# 	if im_crops:
	# 		features = self.extractor(im_crops)
	# 	else:
	# 		features = np.array([])
	# 	return features


