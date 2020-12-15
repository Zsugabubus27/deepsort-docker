# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter_world
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
	"""
	This is the multi-target tracker.

	Parameters
	----------
	metric : nn_matching.NearestNeighborDistanceMetric
		A distance metric for measurement-to-track association.
	max_age : int
		Maximum number of missed misses before a track is deleted.
	n_init : int
		Number of consecutive detections before the track is confirmed. The
		track state is set to `Deleted` if a miss occurs within the first
		`n_init` frames.

	Attributes
	----------
	metric : nn_matching.NearestNeighborDistanceMetric
		The distance metric used for measurement to track association.
	max_age : int
		Maximum number of missed misses before a track is deleted.
	n_init : int
		Number of frames that a track remains in initialization phase.
	kf : kalman_filter.KalmanFilter
		A Kalman filter to filter target trajectories in image space.
	tracks : List[Track]
		The list of active tracks at the current time step.

	"""

	def __init__(self, metric, lambdaParam, max_iou_distance, max_age, n_init, coordMapper, fps):
		self.metric = metric
		self.max_iou_distance = max_iou_distance
		self.max_age = max_age
		self.n_init = n_init
		self.lambdaParam = lambdaParam
		self.coordMapper = coordMapper

		#self.kf = kalman_filter.KalmanFilter()
		self.kf = kalman_filter_world.KalmanFilterWorldCoordinate(fps=fps)
		self.tracks = []
		self._next_id = 1

	def predict(self):
		"""Propagate track state distributions one time step forward.

		This function should be called once every time step, before `update`.
		"""
		for track in self.tracks:
			track.predict(self.kf)

	def update(self, detections):
		"""Perform measurement update and track management.

		Parameters
		----------
		detections : List[deep_sort.detection.Detection]
			A list of detections at the current time step.

		"""
		# Print the state of every track and every detection
		# ------------------------------
		if False:
			#DEBUG:
			for track in self.tracks:
				trackVelo = np.sum(np.array(track.mean[4:6]) ** 2) ** 0.5 * 25 * 0.1 * 3.6 
				print('tID: {}, v={}, mean:{}, covariance: {}'.format(track.track_id, trackVelo, track.mean, track.covariance))

			for i, det in enumerate(detections):
				print('dID: {}, worldxyah: {}'.format(i, det.to_worldxyah()))

		# ------------------------------

		# Run matching cascade.
		matches, unmatched_tracks, unmatched_detections = \
			self._match(detections)
		
		# Update track set.
		# 1. A párosított trackeket updateli
		# 2. A nem párosított trackeket Missingnek jelöli (Deletedhez kell)
		# 3. A nem párosított detekciókból új trackeket csinál
		# 4. A törölt trackeket törli
		for track_idx, detection_idx in matches:
			self.tracks[track_idx].update(
				self.kf, detections[detection_idx])
		for track_idx in unmatched_tracks:
			self.tracks[track_idx].mark_missed()
		for detection_idx in unmatched_detections:
			self._initiate_track(detections[detection_idx])
		

		self.tracks = [t for t in self.tracks if not t.is_deleted()]

		# Update distance metric.
		# 1. Készít két listát mely sorfolytonosan tartalmazza, hogy melyik feature vektor melyik trackID-hoz tartozik
		#    Ezzel a két listával updateli majd a metric objektumot
		# 2. A pertial_fit fv segítségével frissíti a metric objektumban tárolt memóriát a trackekről és feature vectorjukról
		#    A legutolsó nn_budget darabot tárolja csak, és csak azokat az indexet tárolja amik még élnek.
		active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
		features, targets = [], []
		for track in self.tracks:
			if not track.is_confirmed():
				continue
			features += track.features
			targets += [track.track_id for _ in track.features]
			track.features = []
		self.metric.partial_fit(
			np.asarray(features), np.asarray(targets), active_targets)

	def _match(self, detections):

		def gated_metric(tracks, dets, track_indices, detection_indices):
			features = np.array([dets[i].feature for i in detection_indices])
			targets = np.array([tracks[i].track_id for i in track_indices])
			cost_matrix = self.metric.distance(features, targets)
			cost_matrix = linear_assignment.gate_cost_matrix(
				self.kf, cost_matrix, tracks, dets, track_indices,
				detection_indices, self.lambdaParam, only_position=False)
			if False:
				# DEBUG:
				print('CostMx', cost_matrix)
			return cost_matrix

		# Split track set into confirmed and unconfirmed tracks.
		confirmed_tracks = [ i for i, t in enumerate(self.tracks) if t.is_confirmed() ]
		unconfirmed_tracks = [ i for i, t in enumerate(self.tracks) if not t.is_confirmed() ]

		# Associate confirmed tracks using appearance features.
		# FIXME: ne legyen beégetve konstans
		# Max distance itt az infinity number
		matches_a, unmatched_tracks_a, unmatched_detections = \
			linear_assignment.matching_cascade(
				gated_metric, 1e+5, self.max_age,
				self.tracks, detections, confirmed_tracks)
		if False:
			# DEBUG:
			print('Tracker._match::matches_a:', [(self.tracks[k].track_id, d) for k, d in matches_a], 
					'unmatched_tracks_a:', [self.tracks[k].track_id for k in unmatched_tracks_a],
					'unmatched_detections_a', unmatched_detections)
				
		# Associate remaining tracks together with unconfirmed tracks using IOU.
		# 1. az unconfirmed trackek és azon unmatched trackek kiválasztása akiknek a kora 1
		# 2. minden többi unmatched tracket elment
		# 3. IOU matchning végrehajtása ezeken a tracekeken és az unmatched detectionokon
		iou_track_candidates = unconfirmed_tracks + [
			k for k in unmatched_tracks_a if
			self.tracks[k].time_since_update == 1]
		unmatched_tracks_a = [
			k for k in unmatched_tracks_a if
			self.tracks[k].time_since_update != 1]
		matches_b, unmatched_tracks_b, unmatched_detections = \
			linear_assignment.min_cost_matching(
				iou_matching.iou_cost, self.max_iou_distance, self.tracks,
				detections, iou_track_candidates, unmatched_detections)
		if False:
			# DEBUG:
			print('Tracker._match::matches_b:', [(self.tracks[k].track_id, d) for k, d in matches_b], 
			'unmatched_tracks_b:', [self.tracks[k].track_id for k in unmatched_tracks_b],
			'unmatched_detections_b', unmatched_detections)
		
		matches = matches_a + matches_b
		unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
		return matches, unmatched_tracks, unmatched_detections

	def _initiate_track(self, detection):
		#mean, covariance = self.kf.initiate(detection.to_xyah())
		mean, covariance = self.kf.initiate(detection.to_worldxyah())
		self.tracks.append(Track(
			mean, covariance, self._next_id, self.n_init, self.max_age,
			detection.feature, self.coordMapper))
		self._next_id += 1
