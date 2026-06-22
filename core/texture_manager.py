import os
import math
from PIL import Image
import numpy as np
from world.terrain import BLOCK_REGISTRY

# Map from mesh_builder.py:
# 0: top, 1: bottom, 2: right, 3: left, 4: front, 5: back
FACE_TOP = 0
FACE_BOTTOM = 1
FACE_RIGHT = 2
FACE_LEFT = 3
FACE_FRONT = 4
FACE_BACK = 5


class TextureManager:
    def __init__(self, texture_dir, fast_leaves=False):
        self.texture_dir = texture_dir
        self.fast_leaves = fast_leaves
        self.tex_size = 16
        
        # Layer 0 is the fallback white texture
        self.textures = [] # list of raw RGBA bytes
        self.alpha_masks = []
        fallback = Image.new('RGBA', (self.tex_size, self.tex_size), (255, 255, 255, 255))
        self.textures.append(fallback.tobytes())
        self.alpha_masks.append(np.ones((self.tex_size, self.tex_size), dtype=bool))
        self.tex_names_to_layer = {"fallback": 0}
        
        
        # Dinamik olarak BLOCK_REGISTRY'den block_defs oluştur
        self.block_defs = {}
        for name, data in BLOCK_REGISTRY.items():
            tex = data["texture"]
            if tex is not None:
                if isinstance(tex, str):
                    self.block_defs[data["id"]] = {"all": tex}
                elif isinstance(tex, dict):
                    self.block_defs[data["id"]] = tex
        
        # Overlays
        self.block_overlays = {
            3: {"top": "grass_top_overlay.png", "side": "grass_side_overlay.png"}
        }
        
    def load_textures(self):
        if not os.path.exists(self.texture_dir):
            os.makedirs(self.texture_dir)
            
        for root, dirs, files in os.walk(self.texture_dir):
            for f in files:
                if f.endswith(".png"):
                    self._load_texture(os.path.join(root, f), f)
                
        print(f"[TEXTURE] Loaded {len(self.textures)} texture layers.")
        
    def _load_texture(self, filepath, filename):
        try:
            img = Image.open(filepath).convert("RGBA")
            
            if getattr(self, "fast_leaves", False) and filename.startswith("leaves_"):
                data = np.array(img)
                # alpha threshold < 128 becomes opaque black
                mask = data[:, :, 3] < 128
                data[mask] = [0, 0, 0, 255]
                img = Image.fromarray(data)
                
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if img.size != (self.tex_size, self.tex_size):
                img = img.resize((self.tex_size, self.tex_size), Image.NEAREST)
            self.tex_names_to_layer[filename] = len(self.textures)
            self.alpha_masks.append(np.array(img)[:, :, 3] > 0)
            self.textures.append(img.tobytes())
        except Exception as e:
            print(f"[TEXTURE] Failed to load {filename}: {e}")
            
    def get_texture_array_data(self):
        # Returns raw bytes for glTexImage3D
        return b''.join(self.textures), len(self.textures)
            
    def get_uvs_for_blocks(self):
        # returns [b, f] -> layer_idx
        # shape (2048, 6)
        layers = np.zeros((2048, 6), dtype=np.float32)
        
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
        layers = np.zeros((2048, 6), dtype=np.float32)
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
