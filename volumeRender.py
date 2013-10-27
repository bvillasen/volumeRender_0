#!/usr/bin/env python
# -*- coding: utf-8 -*-
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *
import numpy as np, Image
import sys, time, os
import pycuda.driver as cuda
import pycuda.gl as cuda_gl
from pycuda.compiler import SourceModule
from pycuda import cumath
import pycuda.gpuarray as gpuarray
import pyglew as glew

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
myToolsDirectory = parentDirectory + "/myTools"
volRenderDirectory = parentDirectory + "/volumeRender"
sys.path.extend( [myToolsDirectory, volRenderDirectory] )
from cudaTools import np3DtoCudaArray, np2DtoCudaArray

nWidth = 128
nHeight = 128
nDepth = 128
nData = nWidth*nHeight*nDepth

plotData_h = np.random.rand(nData)

def stepFunc():
  print "Default step function"

#globals()["stepFunc"] = stepFunction

width_GL = 512
height_GL = 512

dataMax = plotData_h.max()
plotData_h = (256.*plotData_h/dataMax).astype(np.uint8).reshape(nDepth, nHeight, nWidth)
plotData_dArray = None
transferFuncArray_d = None

viewRotation =  np.zeros(3).astype(np.float32)
viewTranslation = np.array([0., 0., -4.])
invViewMatrix_h = np.arange(12).astype(np.float32)


density = 0.05
brightness = 1.0
transferOffset = 0.0
transferScale = 1.0
#linearFiltering = True
density = np.float32(density)
brightness = np.float32(brightness)
transferOffset = np.float32(transferOffset)
transferScale = np.float32(transferScale)

block2D_GL = (16, 16, 1)
grid2D_GL = (width_GL/block2D_GL[0], height_GL /block2D_GL[1] ) 

gl_tex = None
gl_PBO = None
cuda_PBO = None


frameCount = 0
fpsCount = 0
fpsLimit = 8
timer = 0.0

#CUDA device variables
c_invViewMatrix = None

#CUDA Kernels
renderKernel = None


#CUDA Textures
tex = None
transferTex = None

def computeFPS():
    global frameCount, fpsCount, fpsLimit, timer
    frameCount += 1
    fpsCount += 1
    if fpsCount == fpsLimit:
        ifps = 1.0 /timer
        glutSetWindowTitle("Volume Render: %f fps" % ifps)
        fpsCount = 0

def render():
  global invViewMatrix_h, c_invViewMatrix
  global gl_PBO, cuda_PBO
  global width_GL, height_GL, density, brightness, transferOffset, transferScale
  global block2D_GL, grid2D_GL
  global tex, transferTex
  global testData_d
  cuda.memcpy_htod( c_invViewMatrix,  invViewMatrix_h)
  # map PBO to get CUDA device pointer
  cuda_PBO_map = cuda_PBO.map()
  cuda_PBO_ptr, cuda_PBO_size = cuda_PBO_map.device_ptr_and_size()
  cuda.memset_d32( cuda_PBO_ptr, 0, width_GL*height_GL )
  renderKernel( np.intp(cuda_PBO_ptr), np.int32(width_GL), np.int32(height_GL), density, brightness, transferOffset, transferScale, grid=grid2D_GL, block = block2D_GL, texrefs=[tex, transferTex] )
  cuda_PBO_map.unmap()
  
def display():
  global viewRotation, viewTranslation, invViewMatrix_h
  global gl_tex, gl_PBO
  global timer
  
  stepFunc()
  
  
  timer = time.time()
  modelView = np.ones(16)
  glMatrixMode(GL_MODELVIEW)
  glPushMatrix()
  glLoadIdentity()
  glRotatef(-viewRotation[0], 1.0, 0.0, 0.0)
  glRotatef(-viewRotation[1], 0.0, 1.0, 0.0)
  glTranslatef(-viewTranslation[0], -viewTranslation[1], -viewTranslation[2])
  modelView = glGetFloatv(GL_MODELVIEW_MATRIX )
  modelView = modelView.reshape(16).astype(np.float32)
  glPopMatrix()
  invViewMatrix_h[0] = modelView[0]
  invViewMatrix_h[1] = modelView[4]
  invViewMatrix_h[2] = modelView[8]
  invViewMatrix_h[3] = modelView[12]
  invViewMatrix_h[4] = modelView[1]
  invViewMatrix_h[5] = modelView[5]
  invViewMatrix_h[6] = modelView[9]
  invViewMatrix_h[7] = modelView[13]
  invViewMatrix_h[8] = modelView[2]
  invViewMatrix_h[9] = modelView[6]
  invViewMatrix_h[10] = modelView[10]
  invViewMatrix_h[11] = modelView[14]
  render()
   # display results
  glClear(GL_COLOR_BUFFER_BIT)
   # draw image from PBO
  glDisable(GL_DEPTH_TEST)
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
   # draw using texture
   # copy from pbo to texture
  glBindBufferARB( glew.GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO)
  glBindTexture(GL_TEXTURE_2D, gl_tex)
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_GL, height_GL, GL_RGBA, GL_UNSIGNED_BYTE, None)
  glBindBufferARB(glew.GL_PIXEL_UNPACK_BUFFER_ARB, 0)
   # draw textured quad
  glEnable(GL_TEXTURE_2D)
  glBegin(GL_QUADS)
  glTexCoord2f(0, 0)
  glVertex2f(0, 0)
  glTexCoord2f(1, 0)
  glVertex2f(1, 0)
  glTexCoord2f(1, 1)
  glVertex2f(1, 1)
  glTexCoord2f(0, 1)
  glVertex2f(0, 1)
  glEnd()
  glDisable(GL_TEXTURE_2D)
  glBindTexture(GL_TEXTURE_2D, 0)
  glutSwapBuffers();
  timer = time.time() - timer
  computeFPS()


def iDivUp( a, b ):
  if a%b != 0:
    return a/b + 1
  else:
    return a/b
    


def initGL():	
  glutInit()
  glutInitDisplayMode(GLUT_RGB |GLUT_DOUBLE )
  glutInitWindowSize(width_GL, height_GL)
  #glutInitWindowPosition(50, 50)
  glutCreateWindow("Volume Render")
  glew.glewInit()
  print "OpenGL initialized"
  
def initPixelBuffer():
  global gl_PBO, cuda_PBO, gl_tex   
  gl_PBO = glGenBuffers(1)
  glBindBufferARB(glew.GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO)
  glBufferDataARB(glew.GL_PIXEL_UNPACK_BUFFER_ARB, width_GL*height_GL*4, None, GL_STREAM_DRAW_ARB)
  glBindBufferARB(glew.GL_PIXEL_UNPACK_BUFFER_ARB, 0)
  cuda_PBO = cuda_gl.RegisteredBuffer(long(gl_PBO))
  #print "Buffer Created" 
  #Create texture which we use to display the result and bind to gl_tex
  #glEnable(GL_TEXTURE_2D)
  gl_tex = glGenTextures(1)
  glBindTexture(GL_TEXTURE_2D, gl_tex)
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_GL, height_GL, 0, 
		GL_RGBA, GL_UNSIGNED_BYTE, None);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
  glBindTexture(GL_TEXTURE_2D, 0)
  #print "Texture Created"

def initCUDA():
  global plotData_h
  global plotData_dArray
  global tex, transferTex
  global transferFuncArray_d
  global testData_d
  global c_invViewMatrix
  global renderKernel
  #print "Compiling CUDA code for volumeRender"
  cudaCodeFile = open(volRenderDirectory + "/CUDAvolumeRender.cu","r")
  cudaCodeString = cudaCodeFile.read() 
  cudaCodeStringComplete = cudaCodeString
  cudaCode = SourceModule(cudaCodeStringComplete, no_extern_c=True)
  tex = cudaCode.get_texref("tex")
  transferTex = cudaCode.get_texref("transferTex")
  c_invViewMatrix = cudaCode.get_global('c_invViewMatrix')[0]
  renderKernel = cudaCode.get_function("d_render")

  if not plotData_dArray: plotData_dArray = np3DtoCudaArray( plotData_h )
  tex.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
  tex.set_filter_mode(cuda.filter_mode.LINEAR)
  tex.set_address_mode(0, cuda.address_mode.CLAMP)
  tex.set_address_mode(1, cuda.address_mode.CLAMP)
  tex.set_array(plotData_dArray)
  
  transferFunc = np.array([
    [  0.0, 0.0, 0.0, 0.0, ],
    [  1.0, 0.0, 0.0, 1.0, ],
    [  1.0, 0.5, 0.0, 1.0, ],
    [  1.0, 1.0, 0.0, 1.0, ],
    [  0.0, 1.0, 0.0, 1.0, ],
    [  0.0, 1.0, 1.0, 1.0, ],
    [  0.0, 0.0, 1.0, 1.0, ],
    [  1.0, 0.0, 1.0, 1.0, ],
    [  0.0, 0.0, 0.0, 0.0, ]]).astype(np.float32)
  transferFuncArray_d, desc = np2DtoCudaArray( transferFunc )
  transferTex.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
  transferTex.set_filter_mode(cuda.filter_mode.LINEAR)
  transferTex.set_address_mode(0, cuda.address_mode.CLAMP)
  transferTex.set_address_mode(1, cuda.address_mode.CLAMP)
  transferTex.set_array(transferFuncArray_d)  
  print "CUDA volumeRender initialized"

  
def keyboard(*args):
  global transferScale, brightness, density
  ESCAPE = '\033'
  # If escape is pressed, kill everything.
  if args[0] == ESCAPE:
    print "Ending Simulation"
    #cuda.gl.Context.pop()
    sys.exit()
  if args[0] == '6':
    transferScale += np.float32(0.01)
    print "Image Transfer Scale: ",transferScale
  if args[0] == '3':
    transferScale -= np.float32(0.01)
    print "Image Transfer Scale: ",transferScale
  if args[0] == '5':
    brightness += np.float32(0.1)
    print "Image Brightness : ",brightness
  if args[0] == '2':
    brightness -= np.float32(0.1)
    print "Image Brightness : ",brightness
  if args[0] == '4':
    density += np.float32(0.01)
    print "Image Density : ",density    
  if args[0] == '1':
    density -= np.float32(0.01)
    print "Image Density : ",density    


ox = 0
oy = 0
buttonState = 0
def mouse(button, state, x , y):
  global ox, oy, buttonState
  if state == GLUT_DOWN:
    buttonState |= 1<<button
  elif state == GLUT_UP:
    buttonState = 0
  ox = x
  oy = y
  glutPostRedisplay()

def motion(x, y):
  global viewRotation, viewTranslation
  global ox, oy, buttonState
  dx = x - ox
  dy = y - oy 
  if buttonState == 4:
    viewTranslation[2] += dy/100.
  elif buttonState == 2:
    viewTranslation[0] += dx/100.
    viewTranslation[1] -= dy/100.
  elif buttonState == 1:
    viewRotation[0] += dy/5.
    viewRotation[1] += dx/5.
  ox = x
  oy = y
  glutPostRedisplay()

def reshape(w, h):
  global width_GL, height_GL
  global grid2D_GL, block2D_GL
  initPixelBuffer()
  grid2D_GL = ( iDivUp(width_GL, block2D_GL[0]), iDivUp(height_GL, block2D_GL[1]) )
  glViewport(0, 0, w, h)
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
  
  
  
def startGL():
  glutDisplayFunc(display)
  glutKeyboardFunc(keyboard)
  glutMouseFunc(mouse)
  glutMotionFunc(motion)
  glutReshapeFunc(reshape)
  glutIdleFunc(glutPostRedisplay)
  glutMainLoop()

#OpenGL main
def animate():
  print "Starting Volume Render"
  #initGL()
  #import pycuda.gl.autoinit
  initCUDA()
  initPixelBuffer()
  startGL()


