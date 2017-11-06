import pygame
import pygame.camera
from pygame.locals import *
import time as time

pygame.init()
pygame.camera.init()
class Capture(object):
	def __init__(self):
		self.size = (640,480)
		# create a display surface. standard pygame stuff
		self.display = pygame.display.set_mode(self.size, 0)

		# this is the same as what we saw before
		self.clist = pygame.camera.list_cameras()
		if not self.clist:
		    raise ValueError("Sorry, no cameras detected.")
		self.cam = pygame.camera.Camera(self.clist[0], self.size)
		self.cam.start()

		# create a surface to capture to.  for performance purposes
		# bit depth is the same as that of the display surface.
		self.snapshot = pygame.surface.Surface(self.size, 0, self.display)
		self.get_and_flip()

	def get_and_flip(self):
		# if you don't want to tie the framerate to the camera, you can check 
		# if the camera has an image ready.  note that while this works
		# on most cameras, some will never return true.
		if self.cam.query_image():
		    self.snapshot = self.cam.get_image(self.snapshot)

		# blit it to the display surface.  simple!
		self.display.blit(self.snapshot, (0,0))
		pygame.display.flip()

	def main(self):
		going = True
		while going:
			events = pygame.event.get()
			for e in events:
				if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
					# close the camera safely
					self.cam.stop()
					going = False

			self.get_and_flip()

# def loopVideo():
# 	while(1):
# size = (640,480)
# # create a display surface. standard pygame stuff
# display = pygame.display.set_mode(size, 0)

# # this is the same as what we saw before
# clist = pygame.camera.list_cameras()
# if not clist:
# 	raise ValueError("Sorry, no cameras detected.")
# cam = pygame.camera.Camera(clist[0], size)
# cam.start()



newStream = Capture()
newStream.get_and_flip()
time.sleep(10)