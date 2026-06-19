import ctypes
import time
import numpy as np
import pyglet
from pyglet.gl import *

def main():
    window = pyglet.window.Window(visible=False)
    
    # Create 1000 empty VAOs
    vaos = []
    for _ in range(1000):
        vao = GLuint(0)
        glGenVertexArrays(1, ctypes.byref(vao))
        vaos.append(vao)
        
    t = time.perf_counter()
    for vao in vaos:
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, 0)
    elapsed = time.perf_counter() - t
    
    print(f"1000 glDrawArrays loop took: {elapsed*1000:.2f} ms")
    
if __name__ == "__main__":
    main()
