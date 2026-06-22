import pyglet
from pyglet.gl import *
import ctypes
import os
from renderer.entity_renderer import EntityRenderer
from renderer.models.humanoid_model import HumanoidModel

class PlayerRenderer(EntityRenderer):
    def __init__(self):
        super().__init__(HumanoidModel(y_offset=0.0))
        try:
            tex_path = 'assets/textures/entity/char.png'
            if not os.path.exists(tex_path):
                # Fallback to something if char doesn't exist
                tex_path = 'assets/textures/entity/pig/pig.png'
                
            image = pyglet.image.load(tex_path)
            
            # Inject actual texture size into model for correct UV generation
            self.model.tex_w = float(image.width)
            self.model.tex_h = float(image.height)
            
            tex_data = image.get_image_data().get_data('RGBA', image.width * 4)
            
            self.texture_id = GLuint(0)
            glGenTextures(1, ctypes.byref(self.texture_id))
            glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_id)
            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            
            from pyglet.gl import glTexImage3D
            tex_buffer = (ctypes.c_ubyte * len(tex_data)).from_buffer_copy(tex_data)
            glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, image.width, image.height, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_buffer)
            glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
        except Exception as e:
            print("Failed to load player texture:", e)
            self.texture_id = None
        
    def bind_texture(self, entity):
        if hasattr(self, 'texture_id') and self.texture_id is not None:
            glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_id)
