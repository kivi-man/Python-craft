import pyglet
from pyglet.gl import *
import ctypes
import numpy as np

def test():
    window = pyglet.window.Window(visible=False)
    
    firsts = np.array([0, 3], dtype=np.int32)
    counts = np.array([3, 3], dtype=np.int32)
    
    try:
        glMultiDrawArrays(GL_TRIANGLES, 
                         firsts.ctypes.data_as(ctypes.POINTER(GLint)), 
                         counts.ctypes.data_as(ctypes.POINTER(GLsizei)), 
                         2)
        print("glMultiDrawArrays works!")
    except Exception as e:
        print("Error:", e)

test()
