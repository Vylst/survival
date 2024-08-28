import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import cv2
import numpy as np
import imutils
import pickle
import keras
import time

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, Reshape, Flatten, LeakyReLU, Conv2DTranspose
from keras.optimizers import Adam

keras.backend.clear_session()

'''-----------------------------------------------------------------------
		CLASS
-----------------------------------------------------------------------'''
class robot:


	def __init__(self, battery_reduc, light_state, face):
		
		'''---- Discriminator ----'''
		d_model = Sequential()
		d_model.add(Conv2D(9, (3,3), padding='same', input_shape=(4,4,3)))
		d_model.add(LeakyReLU(alpha=0.2))
		d_model.add(Conv2D(3, (2,2), strides=(2,2), padding='same'))
		d_model.add(LeakyReLU(alpha=0.2))
		d_model.add(Flatten())
		#d_model.add(Dropout(0.4))
		d_model.add(Dense(1, activation='sigmoid'))
		d_model.summary()

		'''---- Generator ----'''
		g_model = Sequential()
		g_model.add(Dense(40, input_dim=3))
		g_model.add(LeakyReLU(alpha=0.2))
		g_model.add(Reshape((2, 2, 10)))
		g_model.add(Conv2DTranspose(3, (3,3), strides=(2,2), padding='same'))
		g_model.add(LeakyReLU(alpha=0.2))
		g_model.summary()
		
		'''---- GAN ---'''
		d_model.trainable = False
		gan_model = Sequential()
		gan_model.add(g_model)
		gan_model.add(d_model)
		opt = Adam(lr=0.0002, beta_1=0.5)
		gan_model.compile(loss='binary_crossentropy', optimizer=opt)
		gan_model.summary()
		
		'''
		emo_states = Input(shape = (3,3,3))
		mid_3 = Conv2D(3, kernel_size=(2, 2), activation="relu")(emo_states)
		mid_4 = Flatten()(mid_3)
		light = Dense(1, activation='sigmoid')(mid_4)
		self.model_B = Model(inputs = emo_states, outputs = light)
		#model_B.summary()
		(self.model_B).compile(optimizer='sgd', loss='mse')

		inputA = Input(shape = (3,))
		outputA = self.model_A(inputA)
		outputB = self.model_B(outputA)
		self.model_C = Model(inputA, outputB)
		'''
		
		
		self.battery_reduc = battery_reduc
		self.light_state = light_state
		self.face = face
		
		self.real_data_X = np.empty((100,3))
		self.real_data_y = np.empty((100,1))
		
		self.fake_data = np.array([light_state, battery_reduc, face])
		
	def get_batt_reduc(self):
		return self.battery_reduc
		
	def get_light_state(self):
		return self.light_state[0][0]
	
	def get_face(self):
		return self.face
	
	def update_batt_reduc(self, batt_reduc):
		self.battery_reduc = batt_reduc
	
	def update_light_state(self, light_state):
		self.light_state = light_state
		
	def update_face(self, face):
		self.face = face
		
	def empty_experiences(self, arr):
		return np.delete(arr, np.s_[0:100], 0)

	#def process_A(self):
	
	#def process_B(self):
	#	emotion = (self.model_A).predict(self.A_input)
	#	(self.model_B).fit(emotion, self.A_input[:,0], batch_size=8, epochs=100)

	def cause_pain(self):
		random_reduc = 75 + 25*(2*np.random.rand(1)[0]-1)
		self.update_batt_reduc(random_reduc)
		print("Ouch! Battery reduction rate at:", random_reduc, "%")
		
		
	def birth(self):
		
		face = 0							#Detect no faces at birth
		for i in range(50):
			battery_reduc = 75 + 25*(2*np.random.rand(1)[0]-1)	#Generate random reduction over 50% (very high at birth)
			light_state = 1					#React to show unpleasantness
			self.real_data_X[i] = np.array([light_state, battery_reduc, face])
			self.real_data_y[i] = light_state

			battery_reduc = 25 + 25*(2*np.random.rand(1)[0]-1)	#Generate random reduction under 50% (lowers post birth)
			light_state = 0					#No need to react if all is good
			self.real_data_X[i] = np.array([light_state, battery_reduc, face])
			self.real_data_y[i] = light_state		
		


	def assimilate(self):
		self.A_input = np.vstack((self.A_input,np.array([self.light_state, self.battery_reduc, self.face])))
		
		
		
		#print(np.shape(np.array([self.light_state, self.battery_reduc, self.face]).reshape((1, 3))))
		reaction = (self.model_C).predict(  (np.array([self.light_state, self.battery_reduc, self.face]).reshape( (1, 3) )).astype('float32') )
		self.update_light_state(reaction)


		if(len(self.A_input) == 100):
			print("Assimilating experiences...")
	
			time.sleep(10)
	
	
	
	
	
			self.fake_data = self.empty_experiences(self.fake_data)



'''-----------------------------------------------------------------------
		INITIALIZATIONS
-----------------------------------------------------------------------'''
protoPath = 'models/deploy.prototxt'
modelPath = 'models/res10_300x300_ssd_iter_140000.caffemodel'
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch('models/openface_nn4.small2.v1.t7')
recognizer = pickle.loads(open('train_on_face/output/recognizer.pkl', "rb").read())
faces = np.flip((pickle.loads(open('train_on_face/output/le.pkl', "rb").read())).classes_)	#0 must be forced as the unknown class here
lights = [(255,255,255),(0,255,255)]


cp = cv2.VideoCapture(0)
canvas = np.zeros((500,1300,3), np.uint8)

r = robot(0, 0, 0)
r.birth()




'''-----------------------------------------------------------------------
		LOOP
-----------------------------------------------------------------------'''
'''
while(1):
	
	k = cv2.waitKey(1) 
	if(k == 27 or k == ord('q')):
		break
	
	if(k == ord('p')):
		r.cause_pain()

	
	face_id = 0		#0 indicates no face or unknown face detected
	[s, f] = cp.read()
	f = cv2.flip(f,1)
	(h, w) = f.shape[:2]
	imageBlob = cv2.dnn.blobFromImage(cv2.resize(f, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
	detector.setInput(imageBlob)
	detections = detector.forward()
	confidence = detections[0, 0, 0, 2]
	
	if confidence > 0.4:

		# Compute the (x, y)-coordinates of the bounding box for the detected face
		box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		
		# extract the face ROI
		face = f[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]
		# ensure the face width and height are sufficiently large
		if fW < 20 or fH < 20:
			continue

		#Recognize face
		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()
		preds = recognizer.predict_proba(vec)[0]
		face_id = np.argmax(preds)
		proba = preds[face_id]
		name = faces[face_id]
		text = "Detecting: {} - {:.2f}%".format(name, proba * 100)
		#print(name, "    ", face_id)

		# draw the bounding box of the face
		cv2.rectangle(f,(startX,startY),(endX,endY),(0,255,0),2)
		
	
	r.update_face(face_id)
	r.update_batt_reduc(25 + 3*(2*np.random.rand(1)[0]-1))
	r.assimilate()
	
	
	
	
	
		
	#Draw scene
	canvas[10:490,10:650] = f
	cv2.putText(canvas, text, (400, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
	cv2.rectangle(canvas, (930,250), (1020,400), (128,128,128), -1)
	cv2.rectangle(canvas, (950,400), (1000,420), (128,128,128), -1)
	print("Light state:", r.get_light_state())
	cv2.circle(canvas, (975,250), 80, lights[int(round(r.get_light_state()))], cv2.FILLED)
	cv2.imshow('demo', canvas)
	

#Clean up
cp.release()
cv2.destroyAllWindows()
	
	'''
	
	
	
	
	
	
	
