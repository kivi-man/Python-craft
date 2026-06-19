import re
import sys
import os

with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# We will extract methods using regex. 
# We need to be careful with indentation.

def extract_method(method_name, text):
    # Matches `    def method_name(self...):` and all indented lines following it
    pattern = r"(\n    def " + method_name + r"\(self.*?\):\n(?:        .*?\n|    \n|\n)+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1), text.replace(match.group(1), "\n")
    return "", text

gui_methods = [
    '_create_3d_block_sprite',
    '_init_gui',
    '_update_gui_positions',
    '_init_hand_blocks'
]

chunk_methods = [
    '_init_world_system',
    'log',
    '_unload_chunk',
    '_update_chunk_loading',
    '_process_chunk_queues',
    '_apply_chunk_mesh',
    'get_block',
    'set_block'
]

input_methods = [
    '_handle_mouse_action',
    'on_mouse_press',
    'on_mouse_release',
    'on_mouse_motion',
    'on_mouse_drag',
    'on_key_press',
    'on_mouse_scroll',
    'on_resize'
]

gui_code = ""
for m in gui_methods:
    code, content = extract_method(m, content)
    gui_code += code

chunk_code = ""
for m in chunk_methods:
    code, content = extract_method(m, content)
    chunk_code += code

input_code = ""
for m in input_methods:
    code, content = extract_method(m, content)
    input_code += code

# Create gui_mixin.py
gui_mixin_content = """import os
import ctypes
import pyglet
from pyglet.gl import *
import numpy as np
from PIL import Image

# Import CACTUS which is used in _init_gui
from world.terrain import CACTUS

class GUIMixin:
""" + gui_code
with open('renderer/gui_mixin.py', 'w', encoding='utf-8') as f:
    f.write(gui_mixin_content)

# Create chunk_mixin.py
# We also need to extract `async_log`
async_log_pattern = r"def async_log\(message\):\n(?:    .*?\n)+"
async_log_match = re.search(async_log_pattern, content)
async_log_code = async_log_match.group(0) if async_log_match else ""
content = content.replace(async_log_code, "")

chunk_mixin_content = """import concurrent.futures
import numpy as np
import math
import ctypes
from pyglet.gl import *
from world.mc_terrain import load_or_generate_chunk, CHUNK_SIZE, CHUNK_HEIGHT
from core.world_db import save_chunk
from renderer.mesh_builder import build_chunk_mesh
from world.terrain import CACTUS, SAND

""" + async_log_code + """

class ChunkMixin:
""" + chunk_code
with open('world/chunk_mixin.py', 'w', encoding='utf-8') as f:
    f.write(chunk_mixin_content)

# Create input_mixin.py
input_mixin_content = """import pyglet
import math
from pyglet.window import key, mouse
from pyglet.gl import glViewport
from core.raycast import raycast

class InputMixin:
""" + input_code
with open('core/input_mixin.py', 'w', encoding='utf-8') as f:
    f.write(input_mixin_content)

# Add Mixins to main.py
imports = """
from renderer.gui_mixin import GUIMixin
from world.chunk_mixin import ChunkMixin
from core.input_mixin import InputMixin
"""

content = content.replace("class PythonCraftEngine(pyglet.window.Window):", imports + "\nclass PythonCraftEngine(pyglet.window.Window, InputMixin, ChunkMixin, GUIMixin):")

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Refactoring complete.")
