import ctypes
import pyglet
from pyglet.gl import *

def check_vram():
    window = pyglet.window.Window(visible=False)
    
    total_kb = GLint(0)
    
    # Try NVIDIA
    try:
        glGetIntegerv(0x9048, ctypes.byref(total_kb))
        if total_kb.value > 0:
            print(f"NVIDIA Total VRAM: {total_kb.value / 1024:.2f} MB")
            return
    except Exception as e:
        pass
        
    # Try AMD
    try:
        vbo_free = (GLint * 4)()
        glGetIntegerv(0x87FB, vbo_free)
        if vbo_free[0] > 0:
            print(f"AMD Available VRAM: {vbo_free[0] / 1024:.2f} MB")
            return
    except Exception as e:
        pass
        
    print("Could not detect VRAM via OpenGL extensions.")

check_vram()
