import pyglet
pyglet.options['search_local_libs'] = True

try:
    config = pyglet.gl.Config(major_version=4, minor_version=3, forward_compatible=True)
    window = pyglet.window.Window(width=1, height=1, config=config, visible=False)
except Exception as e:
    print(f"Failed to create GL 4.3 context: {e}")
    import sys
    sys.exit(1)

from pyglet.gl import *
print("Context created successfully.")

print("GL_VERSION:", ctypes.cast(glGetString(GL_VERSION), ctypes.c_char_p).value.decode())

try:
    print("glDispatchCompute exists:", bool(glDispatchCompute))
    print("glMultiDrawArraysIndirect exists:", bool(glMultiDrawArraysIndirect))
except Exception as e:
    print(e)

window.close()
