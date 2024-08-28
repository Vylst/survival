
import sched
import time
import cv2

class robot:
	def __init__(self, battery_reduc, light_state, face):
		self.battery_reduc = battery_reduc
		self.light_state = light_state
		self.face = face
		
	def get_batt_reduc(self):
		return self.battery_reduc
		
	def get_light_state(self):
		return self.light_state
	
	def get_face(self):
		return self.face
	
	#def update_batt_reduc(self):
		
	
	#def update_light_state(self):
	
	
	#def update_face(self):
		
	
	
lightbulb_on = cv2.imread('on.jpg')
lightbulb_off = cv2.imread('off.jpg')
cv2.imshow('on',lightbulb_on)
cv2.imshow('off',lightbulb_off)
	
	
sonny = robot(90, 1, 0)



scheduler = sched.scheduler(time.time, time.sleep)

def usage(): 
	print("I'm alive!")
	
	#For the next n seconds
	t_end = time.time() + 5
	while time.time() < t_end:
		battery_reduc = read_batt_reduc()
		light_state = read_light_state()
		face = read_face()
		
	a = 0
	scheduler.enter(0, 1, assimilation, (a,))
	
def assimilation(a):
	print("Assimilating knowledge...")
	
	scheduler.enter(1, 1, usage)


scheduler.enter(1, 1, usage)
scheduler.run()


