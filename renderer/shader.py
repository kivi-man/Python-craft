import ctypes
from pyglet.gl import *

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    source_bytes = source.encode('utf-8')
    src_buffer = ctypes.create_string_buffer(source_bytes)
    buf_pointer = ctypes.cast(ctypes.pointer(ctypes.pointer(src_buffer)),
                              ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
    length = ctypes.c_int(len(source_bytes))
    glShaderSource(shader, 1, buf_pointer, ctypes.byref(length))
    glCompileShader(shader)
    
    # Error checking
    status = ctypes.c_int(0)
    glGetShaderiv(shader, GL_COMPILE_STATUS, ctypes.byref(status))
    if not status.value:
        log_length = ctypes.c_int(0)
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, ctypes.byref(log_length))
        log = ctypes.create_string_buffer(log_length.value)
        glGetShaderInfoLog(shader, log_length, None, log)
        raise RuntimeError(f"Shader compile error (Your GPU may not support the required OpenGL version): {log.value.decode()}")
    return shader

def create_shader_program(vert_path, frag_path):
    with open(vert_path, 'r') as f:
        vert_src = f.read()
    with open(frag_path, 'r') as f:
        frag_src = f.read()
        
    vs = compile_shader(vert_src, GL_VERTEX_SHADER)
    fs = compile_shader(frag_src, GL_FRAGMENT_SHADER)
    
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    
    status = ctypes.c_int(0)
    glGetProgramiv(program, GL_LINK_STATUS, ctypes.byref(status))
    if not status.value:
        log_length = ctypes.c_int(0)
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, ctypes.byref(log_length))
        log = ctypes.create_string_buffer(log_length.value)
        glGetProgramInfoLog(program, log_length, None, log)
        raise RuntimeError(f"Shader link error (Try updating your GPU drivers): {log.value.decode()}")
    
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program
