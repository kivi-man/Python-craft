import os
import math
from PIL import Image
import numpy as np

# Map from mesh_builder.py:
# 0: top, 1: bottom, 2: right, 3: left, 4: front, 5: back
FACE_TOP = 0
FACE_BOTTOM = 1
FACE_RIGHT = 2
FACE_LEFT = 3
FACE_FRONT = 4
FACE_BACK = 5

class TextureManager:
    def __init__(self, texture_dir):
        self.texture_dir = texture_dir
        self.tex_size = 16
        
        # Layer 0 is the fallback white texture
        self.textures = [] # list of raw RGBA bytes
        fallback = Image.new('RGBA', (self.tex_size, self.tex_size), (255, 255, 255, 255))
        self.textures.append(fallback.tobytes())
        
        self.tex_names_to_layer = {"fallback": 0}
        
        # 1: STONE, 2: DIRT, 3: GRASS, 4: WATER, 5: SAND
        self.block_defs = {
            1: {"all": "stone.png"},
            2: {"all": "dirt.png"},
            3: {"top": "grass_top.png", "bottom": "dirt.png", "side": "grass_side.png"},
            4: {"all": "water.png"},
            5: {"all": "sand.png"},
            6: {"all": "snow.png"},
            7: {"all": "ice.png"},
            8: {"all": "gravel.png"},
            9: {"all": "sandstone.png"},
            10: {"top": "mycelium_top.png", "bottom": "dirt.png", "side": "mycelium_side.png"},
            11: {"top": "log_oak_top.png", "bottom": "log_oak_top.png", "side": "log_oak.png"},
            12: {"all": "leaves_oak.png"},
            13: {"top": "cactus_top.png", "bottom": "cactus_bottom.png", "side": "cactus_side.png"},
            14: {"top": "log_birch_top.png", "bottom": "log_birch_top.png", "side": "log_birch.png"},
            15: {"top": "log_spruce_top.png", "bottom": "log_spruce_top.png", "side": "log_spruce.png"},
            16: {"all": "leaves_birch.png"},
            17: {"all": "leaves_spruce.png"},
            20: {"all": "glass.png"},
            31: {"all": "tallgrass.png"},
            37: {"all": "flower_dandelion.png"},
            38: {"all": "flower_rose.png"},
            175: {"all": "double_plant_grass_bottom.png"},
            176: {"all": "double_plant_grass_top.png"},
            177: {"all": "double_plant_rose_bottom.png"},
            178: {"all": "double_plant_rose_top.png"}
        }
        
        # Overlays
        self.block_overlays = {
            3: {"top": "grass_top_overlay.png", "side": "grass_side_overlay.png"}
        }
        
    def load_textures(self):
        if not os.path.exists(self.texture_dir):
            os.makedirs(self.texture_dir)
            
        for f in os.listdir(self.texture_dir):
            if f.endswith(".png"):
                self._load_texture(f)
                
        print(f"[TEXTURE] Loaded {len(self.textures)} texture layers.")
        
    def _load_texture(self, filename):
        path = os.path.join(self.texture_dir, filename)
        try:
            img = Image.open(path).convert("RGBA")
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if img.size != (self.tex_size, self.tex_size):
                img = img.resize((self.tex_size, self.tex_size), Image.NEAREST)
            self.tex_names_to_layer[filename] = len(self.textures)
            self.textures.append(img.tobytes())
        except Exception as e:
            print(f"[TEXTURE] Failed to load {filename}: {e}")
            
    def get_texture_array_data(self):
        # Returns raw bytes for glTexImage3D
        return b''.join(self.textures), len(self.textures)
            
    def get_uvs_for_blocks(self):
        # returns [b, f] -> layer_idx
        # shape (256, 6)
        layers = np.zeros((256, 6), dtype=np.float32)
        
        for block_id, faces in self.block_defs.items():
            for face_idx in range(6):
                tex_name = None
                if "all" in faces: tex_name = faces["all"]
                else:
                    if face_idx == FACE_TOP and "top" in faces: tex_name = faces["top"]
                    elif face_idx == FACE_BOTTOM and "bottom" in faces: tex_name = faces["bottom"]
                    elif face_idx in (FACE_RIGHT, FACE_LEFT, FACE_FRONT, FACE_BACK) and "side" in faces: tex_name = faces["side"]
                    
                if tex_name and tex_name in self.tex_names_to_layer:
                    layers[block_id, face_idx] = self.tex_names_to_layer[tex_name]
                else:
                    layers[block_id, face_idx] = 0 # fallback
                    
        return layers

    def get_overlays_for_blocks(self):
        layers = np.zeros((256, 6), dtype=np.float32)
        for block_id, faces in self.block_overlays.items():
            for face_idx in range(6):
                tex_name = None
                if "all" in faces: tex_name = faces["all"]
                else:
                    if face_idx == FACE_TOP and "top" in faces: tex_name = faces["top"]
                    elif face_idx == FACE_BOTTOM and "bottom" in faces: tex_name = faces["bottom"]
                    elif face_idx in (FACE_RIGHT, FACE_LEFT, FACE_FRONT, FACE_BACK) and "side" in faces: tex_name = faces["side"]
                    
                if tex_name and tex_name in self.tex_names_to_layer:
                    layers[block_id, face_idx] = self.tex_names_to_layer[tex_name]
                else:
                    layers[block_id, face_idx] = 0
        return layers
