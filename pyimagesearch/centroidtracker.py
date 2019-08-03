# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import math

class CentroidTracker():
	def __init__(self, maxDisappeared=50, minAppeared=20):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.minUsedObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.appearedOnce = OrderedDict()
		self.deletedObjects = OrderedDict()


		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared
		self.minAppeared = minAppeared
		self.distanceDelta = 1000
		self.TEN_METERS = 10000

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.appearedOnce[self.nextObjectID] = 1
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		self.deletedObjects[objectID] = self.objects[objectID]
		del self.objects[objectID]
		del self.disappeared[objectID]
		if objectID in self.appearedOnce:
			del self.appearedOnce[objectID]

	def update(self, rects, depthList):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			temp = self.disappeared.copy()
			for objectID in temp.keys():
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			temp = self.appearedOnce.copy()
			for objectID in self.appearedOnce:
				if self.appearedOnce[objectID]:
					self.deregister(objectID)


			# return early as there are no centroids or tracking info
			# to update
			return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 7), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			cD  = int(depthList[i])
			if cD == -1.0:
				cD = 50.0
			inputCentroids[i] = (cX, cY, cD, startX, startY, endX, endY)

			# depth of the center of X-coordinates

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())


			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			# print (rects)
			# print (type(rects))
			# print ("===========\n")
			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				# val
				if row in usedRows or col in usedCols:
					continue


				# if the distance betweeen the new object and the previous object
				# is bigger than some delta, we get skip it.

				# depthScaling =  math.log((self.TEN_METERS)/((objectCentroids[row][2] + inputCentroids[col][2])/2))
				# if D[row][col] > self.distanceDelta * depthScaling:
				# 	print ("DISTANCE too big: ", D[row][col])
				# 	print ("should be less than: ",  self.distanceDelta * depthScaling)
				# 	continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0
				temp = self.appearedOnce.copy()
				if objectID in temp:
					self.appearedOnce[objectID]+=1
					if self.appearedOnce[objectID] > self.minAppeared:
						del self.appearedOnce[objectID]

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)



			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID]+= 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

					temp = self.appearedOnce.copy()
					if objectID in temp:
						if self.appearedOnce[objectID]:
							self.deregister(objectID)


			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# return the set of trackable objects
		returnDict = OrderedDict()
		for val in self.objects:
			if val not in self.appearedOnce:
				returnDict[val] = self.objects[val]
			else:
				returnDict[val] = [None, None, None, None, None, None, None]
		return returnDict
		# return self.objects
