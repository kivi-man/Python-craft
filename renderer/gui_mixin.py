import os
import ctypes
import math
import random
import time
import pyglet
from pyglet.window import key
from pyglet.gl import *
import numpy as np
from core.math_utils import get_hand_cube_vertices, get_item_sprite_vertices, normalize_vec, cross_vec
from world.terrain import CACTUS, BLOCK_HARDNESS_ARRAY, PORKCHOP_RAW, get_break_time
from PIL import Image

# Import CACTUS which is used in _init_gui
from world.terrain import CACTUS

class GUIMixin:

    def _create_3d_block_sprite(self, top_name, left_name, right_name):
        from PIL import Image
        import pyglet.image
        import pyglet.sprite
        import numpy as np
        
        blocks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'textures', 'blocks')
        
        def load_numpy_img(name):
            p = os.path.join(blocks_dir, name)
            if os.path.exists(p):
                return np.array(Image.open(p).convert("RGBA"), dtype=np.float32)
            return None
            
        top_img = load_numpy_img(top_name)
        left_img = load_numpy_img(left_name)
        right_img = load_numpy_img(right_name)
        
        if top_img is None and left_img is None and right_img is None:
            # Fallback
            fallback = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
            raw_data = fallback.tobytes()
            p_img = pyglet.image.ImageData(64, 64, 'RGBA', raw_data)
            tex = p_img.get_texture()
            sprite = pyglet.sprite.Sprite(img=tex)
            return sprite
            
        if top_img is None: top_img = left_img if left_img is not None else right_img
        if left_img is None: left_img = top_img if top_img is not None else right_img
        if right_img is None: right_img = top_img if top_img is not None else left_img
        
        dest = np.zeros((64, 64, 4), dtype=np.uint8)
        
        Y, X = np.mgrid[0:64, 0:64]
        
        # Masks
        top_mask = (X + 2*Y >= 32) & (X - 2*Y <= 32) & (X - 2*Y >= -32) & (X + 2*Y <= 96) & (Y < 32)
        left_mask = (X >= 0) & (X < 32) & (Y >= 16 + X/2.0) & (Y < 48 + X/2.0)
        right_mask = (X >= 32) & (X < 64) & (Y >= 48 - X/2.0) & (Y < 80 - X/2.0)
        
        # Resolving overlaps like original `elif`
        left_mask = left_mask & ~top_mask
        right_mask = right_mask & ~top_mask & ~left_mask
        
        # Top Mapping
        x_prime = (X[top_mask] - 32) / 32.0
        y_prime = (Y[top_mask] - 16) / 16.0
        u_top = np.clip(np.floor(8 * (x_prime + y_prime + 1)).astype(int), 0, 15)
        v_top = np.clip(np.floor(8 * (-x_prime + y_prime + 1)).astype(int), 0, 15)
        # Original PIL coordinates are transposed compared to numpy arrays
        # In original: dest_pix[x, y] = top_pix[u, v]. PIL image data is usually accessed (x,y)
        # In NumPy, image shape is (H, W, 4), so array[y, x] corresponds to PIL [x, y]
        dest[Y[top_mask], X[top_mask]] = top_img[v_top, u_top]
        
        # Left Mapping
        u_left = np.clip(np.floor(X[left_mask] / 2.0).astype(int), 0, 15)
        v_left = np.clip(np.floor((Y[left_mask] - (16 + X[left_mask]/2.0)) / 2.0).astype(int), 0, 15)
        left_vals = left_img[v_left, u_left].copy()
        left_vals[:, 0:3] *= 0.6
        dest[Y[left_mask], X[left_mask]] = left_vals.astype(np.uint8)
        
        # Right Mapping
        u_right = np.clip(np.floor((X[right_mask] - 32) / 2.0).astype(int), 0, 15)
        v_right = np.clip(np.floor((Y[right_mask] - (48 - X[right_mask]/2.0)) / 2.0).astype(int), 0, 15)
        right_vals = right_img[v_right, u_right].copy()
        right_vals[:, 0:3] *= 0.8
        dest[Y[right_mask], X[right_mask]] = right_vals.astype(np.uint8)
        
        # Flip vertically for Pyglet coordinates compatibility (flipud)
        dest = np.flipud(dest)
        
        # Make contiguous in memory to avoid pyglet issues
        dest = np.ascontiguousarray(dest)
        raw_data = dest.tobytes()
        pyglet_img = pyglet.image.ImageData(64, 64, 'RGBA', raw_data)
        tex = pyglet_img.get_texture()
        
        # Apply linear filtering to make cube edges smooth like a high-res 3D model
        glBindTexture(tex.target, tex.id)
        glTexParameteri(tex.target, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(tex.target, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(tex.target, 0)
        
        tex.min_filter = GL_LINEAR
        tex.mag_filter = GL_LINEAR
        
        sprite = pyglet.sprite.Sprite(img=tex)
        return sprite

    def _create_2d_item_sprite(self, texture_name):
        import os
        from PIL import Image
        import pyglet.image
        import pyglet.sprite
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        items_dir = os.path.join(base_dir, 'assets', 'textures', 'items')
        blocks_dir = os.path.join(base_dir, 'assets', 'textures', 'blocks')
        
        p = os.path.join(items_dir, texture_name)
        if not os.path.exists(p):
            p = os.path.join(blocks_dir, texture_name)
            if not os.path.exists(p):
                return None
            
        img = Image.open(p).convert("RGBA")
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        raw_data = img.tobytes()
        pyglet_img = pyglet.image.ImageData(img.width, img.height, 'RGBA', raw_data)
        tex = pyglet_img.get_texture()
        
        glBindTexture(tex.target, tex.id)
        glTexParameteri(tex.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(tex.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(tex.target, 0)
        
        sprite = pyglet.sprite.Sprite(img=tex)
        sprite.original_width = float(img.width)
        return sprite

    def _init_gui(self):
        import pyglet.image
        import pyglet.sprite
        
        try:
            icons_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'textures', 'gui', 'icons.png')
            crosshair_img = pyglet.image.load(icons_path)
            # The top-left 16x16 region in icons.png is the crosshair
            # Pyglet coordinates start from bottom-left, so we set y = height - 16
            region = crosshair_img.get_region(0, crosshair_img.height - 16, 16, 16)
            texture = region.get_texture()
            
            # Apply nearest-neighbor filtering to prevent texture blurring
            glBindTexture(texture.target, texture.id)
            glTexParameteri(texture.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(texture.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glBindTexture(texture.target, 0)
            
            texture.min_filter = GL_NEAREST
            texture.mag_filter = GL_NEAREST
            
            self.crosshair_sprite = pyglet.sprite.Sprite(img=texture)
            self.crosshair_sprite.scale = 2
            
            # Health and Hunger Icons (9x9)
            heart_y = crosshair_img.height - 9
            hunger_y = crosshair_img.height - 36
            
            def get_icon_tex(x, y):
                reg = crosshair_img.get_region(x, y, 9, 9).get_texture()
                glBindTexture(reg.target, reg.id)
                glTexParameteri(reg.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(reg.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                glBindTexture(reg.target, 0)
                return reg
                
            self.tex_heart_empty = get_icon_tex(16, heart_y)
            self.tex_heart_half = get_icon_tex(61, heart_y)
            self.tex_heart_full = get_icon_tex(52, heart_y)
            
            self.tex_hunger_empty = get_icon_tex(16, hunger_y)
            self.tex_hunger_half = get_icon_tex(61, hunger_y)
            self.tex_hunger_full = get_icon_tex(52, hunger_y)
            
            # Bubble Icons (9x9)
            bubble_y = crosshair_img.height - 27
            self.tex_bubble_full = get_icon_tex(16, bubble_y)
            self.tex_bubble_popped = get_icon_tex(25, bubble_y)
            
            self.ui_batch = pyglet.graphics.Batch()
            
            # Create sprite pools for batched drawing
            self.heart_bg_sprites = [pyglet.sprite.Sprite(img=self.tex_heart_empty, batch=self.ui_batch) for _ in range(10)]
            self.heart_fg_sprites = [pyglet.sprite.Sprite(img=self.tex_heart_full, batch=self.ui_batch) for _ in range(10)]
            self.hunger_bg_sprites = [pyglet.sprite.Sprite(img=self.tex_hunger_empty, batch=self.ui_batch) for _ in range(10)]
            self.hunger_fg_sprites = [pyglet.sprite.Sprite(img=self.tex_hunger_full, batch=self.ui_batch) for _ in range(10)]
            self.bubble_sprites = [pyglet.sprite.Sprite(img=self.tex_bubble_full, batch=self.ui_batch) for _ in range(10)]
            
            for spr in self.heart_bg_sprites + self.heart_fg_sprites + self.hunger_bg_sprites + self.hunger_fg_sprites + self.bubble_sprites:
                spr.scale = 2
                spr.visible = False
            
            # XP Bar (182x5)
            xp_empty_y = crosshair_img.height - 69
            xp_full_y = crosshair_img.height - 74
            
            reg_xp_empty = crosshair_img.get_region(0, xp_empty_y, 182, 5).get_texture()
            glBindTexture(reg_xp_empty.target, reg_xp_empty.id)
            glTexParameteri(reg_xp_empty.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(reg_xp_empty.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glBindTexture(reg_xp_empty.target, 0)
            self.tex_xp_empty = reg_xp_empty
            
            reg_xp_full = crosshair_img.get_region(0, xp_full_y, 182, 5).get_texture()
            glBindTexture(reg_xp_full.target, reg_xp_full.id)
            glTexParameteri(reg_xp_full.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(reg_xp_full.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glBindTexture(reg_xp_full.target, 0)
            self.tex_xp_full = reg_xp_full
            
            self.spr_xp_empty = pyglet.sprite.Sprite(img=self.tex_xp_empty)
            self.spr_xp_full = pyglet.sprite.Sprite(img=self.tex_xp_full)
            
            # Load Hotbar Textures
            gui_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'textures', 'gui', 'gui.png')
            gui_img = pyglet.image.load(gui_path)
            
            # Hotbar Background (182x22)
            bg_region = gui_img.get_region(0, gui_img.height - 22, 182, 22)
            bg_tex = bg_region.get_texture()
            glBindTexture(bg_tex.target, bg_tex.id)
            glTexParameteri(bg_tex.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(bg_tex.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            self.hotbar_bg_sprite = pyglet.sprite.Sprite(img=bg_tex)
            self.hotbar_bg_sprite.scale = 2
            
            # Active Slot Selection Frame (24x24)
            sel_region = gui_img.get_region(0, gui_img.height - 46, 24, 24)
            sel_tex = sel_region.get_texture()
            glBindTexture(sel_tex.target, sel_tex.id)
            glTexParameteri(sel_tex.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(sel_tex.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            self.hotbar_sel_sprite = pyglet.sprite.Sprite(img=sel_tex)
            self.hotbar_sel_sprite.scale = 2

            # Load Full Inventory Background
            inv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'textures', 'gui', 'container', 'inventory.png')
            if os.path.exists(inv_path):
                inv_img = pyglet.image.load(inv_path)
                # The active GUI region in inventory.png is 176x166 (top-left aligned, but Pyglet is bottom-left, so y=height-166)
                inv_region = inv_img.get_region(0, inv_img.height - 166, 176, 166)
                inv_tex = inv_region.get_texture()
                glBindTexture(inv_tex.target, inv_tex.id)
                glTexParameteri(inv_tex.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(inv_tex.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                self.inventory_bg_sprite = pyglet.sprite.Sprite(img=inv_tex)
            else:
                self.inventory_bg_sprite = None

            # Load Crafting Table Background
            craft_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'textures', 'gui', 'container', 'crafting_table.png')
            if os.path.exists(craft_path):
                craft_img = pyglet.image.load(craft_path)
                craft_region = craft_img.get_region(0, craft_img.height - 166, 176, 166)
                craft_tex = craft_region.get_texture()
                glBindTexture(craft_tex.target, craft_tex.id)
                glTexParameteri(craft_tex.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(craft_tex.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                self.crafting_bg_sprite = pyglet.sprite.Sprite(img=craft_tex)
            else:
                self.crafting_bg_sprite = None

            
            # Load Block Icons as 3D Isometric Cubes
            self.block_icon_sprites = {}
            from world.terrain import BLOCK_REGISTRY
            sprite_blocks = {31, 37, 38, 175, 176, 177, 178}
            
            for name, data in BLOCK_REGISTRY.items():
                b_id = data["id"]
                if b_id == 0: continue
                
                tex = data.get("texture")
                sprite = None
                
                if b_id == 64:
                    sprite = self._create_2d_item_sprite("door_wood.png")
                elif b_id == 71:
                    sprite = self._create_2d_item_sprite("door_iron.png")
                elif b_id >= 256 or b_id in sprite_blocks:
                    if isinstance(tex, str):
                        sprite = self._create_2d_item_sprite(tex)
                
                if sprite is None:
                    if not tex:
                        sprite = self._create_3d_block_sprite('stone.png', 'stone.png', 'stone.png')
                    elif isinstance(tex, str):
                        sprite = self._create_3d_block_sprite(tex, tex, tex)
                    elif isinstance(tex, dict):
                        top = tex.get('top', 'stone.png')
                        side = tex.get('side', top)
                        sprite = self._create_3d_block_sprite(top, side, side)
                    else:
                        sprite = self._create_3d_block_sprite('stone.png', 'stone.png', 'stone.png')
                        
                if sprite:
                    self.block_icon_sprites[b_id] = sprite
            
            
            # Position user interface elements
            self._update_gui_positions(self.width, self.height)
        except Exception as e:
            print(f"[GUI] Failed to load user interface: {e}")
            self.crosshair_sprite = None
            self.hotbar_bg_sprite = None
            self.hotbar_sel_sprite = None


    def _update_gui_positions(self, width, height):
        if hasattr(self, 'crosshair_sprite') and self.crosshair_sprite is not None:
            # Round coordinates to integers to prevent sub-pixel rendering blurriness
            self.crosshair_sprite.x = int((width - self.crosshair_sprite.width) // 2)
            self.crosshair_sprite.y = int((height - self.crosshair_sprite.height) // 2)
            
        if hasattr(self, 'hotbar_bg_sprite') and self.hotbar_bg_sprite is not None:
            # Center hotbar horizontally, keep 10 pixels margin from the bottom
            self.hotbar_bg_sprite.x = int((width - self.hotbar_bg_sprite.width) // 2)
            self.hotbar_bg_sprite.y = 10

        if hasattr(self, 'inventory_bg_sprite') and self.inventory_bg_sprite is not None:
            self.inventory_bg_sprite.x = int((width - self.inventory_bg_sprite.width) // 2)
            self.inventory_bg_sprite.y = int((height - self.inventory_bg_sprite.height) // 2)

        if hasattr(self, 'crafting_bg_sprite') and self.crafting_bg_sprite is not None:
            self.crafting_bg_sprite.x = int((width - self.crafting_bg_sprite.width) // 2)
            self.crafting_bg_sprite.y = int((height - self.crafting_bg_sprite.height) // 2)


    def _init_hand_blocks(self):
        self.hand_block_vaos = {}
        
        def create_hand_vao(mesh):
            vao = GLuint(0)
            glGenVertexArrays(1, ctypes.byref(vao))
            glBindVertexArray(vao)
            vbo = GLuint(0)
            glGenBuffers(1, ctypes.byref(vbo))
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, mesh.nbytes, mesh.ctypes.data, GL_STATIC_DRAW)
            
            stride = 15 * 4
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(36))
            glEnableVertexAttribArray(3)
            glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(48))
            glEnableVertexAttribArray(4)
            glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(52))
            glEnableVertexAttribArray(5)
            glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(56))
            glEnableVertexAttribArray(6)
            
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)
            num_verts = len(mesh) // 15
            return vao, vbo, num_verts

        from world.terrain import BLOCK_REGISTRY
        from core.special_blocks import is_slab, is_stairs
        
        # Define block IDs that should be rendered as flat 2D item sprites in hand
        # 31: Tallgrass, 37: Dandelion, 38: Rose, 175-178: Double plants, 1000: Raw Porkchop
        # 256-1033: Tools and items
        SPRITE_BLOCKS = {31, 37, 38, 162, 163, 175, 176, 177, 178} | set(range(256, 1034))
        
        for name, data in BLOCK_REGISTRY.items():
            b_id = data["id"]
            if b_id > 0:
                if b_id in SPRITE_BLOCKS:
                    layer_idx = 0
                    if b_id < len(self.block_layers):
                        layer_idx = self.block_layers[b_id, 0]
                    if b_id == 64 and "door_wood.png" in self.texture_manager.tex_names_to_layer:
                        layer_idx = self.texture_manager.tex_names_to_layer["door_wood.png"]
                    elif b_id == 71 and "door_iron.png" in self.texture_manager.tex_names_to_layer:
                        layer_idx = self.texture_manager.tex_names_to_layer["door_iron.png"]
                    elif b_id >= len(self.block_layers) and hasattr(self.texture_manager, 'item_layers') and b_id in self.texture_manager.item_layers:
                         pass
                    
                    mask = self.texture_manager.alpha_masks[int(layer_idx)] if hasattr(self.texture_manager, 'alpha_masks') and int(layer_idx) < len(self.texture_manager.alpha_masks) else None
                    if mask is not None:
                        mesh = get_item_sprite_vertices(layer_idx, mask)
                        if mesh is not None and len(mesh) > 0:
                            vao, vbo, num_verts = create_hand_vao(mesh)
                            self.hand_block_vaos[b_id] = (vao, vbo, num_verts)
                else:
                    aabbs = None
                    if is_slab(b_id):
                        aabbs = [[-0.5, -0.5, -0.5, 0.5, 0.0, 0.5]]
                    elif is_stairs(b_id):
                        # Base + step
                        aabbs = [
                            [-0.5, -0.5, -0.5, 0.5, 0.0, 0.5],
                            [0.0, 0.0, -0.5, 0.5, 0.5, 0.5]
                        ]
                    elif b_id == 81: # Cactus
                        aabbs = [[-0.4375, -0.5, -0.4375, 0.4375, 0.5, 0.4375]]
                        
                    mesh = get_hand_cube_vertices(b_id, self.block_layers, aabbs=aabbs)
                    if mesh is not None and len(mesh) > 0:
                        vao, vbo, num_verts = create_hand_vao(mesh)
                        self.hand_block_vaos[b_id] = (vao, vbo, num_verts)

    def _get_slot_rect(self, slot_idx):
        if getattr(self, 'crafting_open', False) and hasattr(self, 'crafting_bg_sprite'):
            bg_x = self.crafting_bg_sprite.x
            bg_y = self.crafting_bg_sprite.y
            scale = self.crafting_bg_sprite.scale
        else:
            if not hasattr(self, 'inventory_bg_sprite') or self.inventory_bg_sprite is None:
                return 0, 0, 0, 0
                
            bg_x = self.inventory_bg_sprite.x
            bg_y = self.inventory_bg_sprite.y
            scale = self.inventory_bg_sprite.scale
        
        if 0 <= slot_idx <= 8:
            px = 8 + slot_idx * 18
            py = 8
        elif 9 <= slot_idx <= 35:
            rel = slot_idx - 9
            col = rel % 9
            row = rel // 9
            px = 8 + col * 18
            py = 30 + (2 - row) * 18
        elif 36 <= slot_idx <= 39:
            px = 8
            py = 88 + (slot_idx - 36) * 18
        elif 40 <= slot_idx <= 43:
            rel = slot_idx - 40
            col = rel % 2
            row = rel // 2
            px = 88 + col * 18
            py = 124 - row * 18
        elif slot_idx == 44:
            px = 144
            py = 114
        elif 45 <= slot_idx <= 53:
            rel = slot_idx - 45
            col = rel % 3
            row = rel // 3
            px = 30 + col * 18
            py = 133 - row * 18
        elif slot_idx == 54:
            px = 124
            py = 115
        else:
            return 0, 0, 0, 0
            
        return bg_x + px * scale, bg_y + py * scale, 16 * scale, 16 * scale

    def _draw_inventory_gui(self):
        if hasattr(self, 'inventory_bg_sprite') and self.inventory_bg_sprite is not None:
            self.inventory_bg_sprite.draw()
            bg_scale = self.inventory_bg_sprite.scale
            
            # Init labels if not present
            if not hasattr(self, 'count_labels'):
                font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'fonts', 'Minecraftia-Regular.ttf')
                try:
                    pyglet.font.add_file(font_path)
                    target_font = 'Minecraftia'
                except Exception:
                    target_font = 'Arial'
                self.count_labels = [pyglet.text.Label("", font_name=target_font, font_size=8, anchor_x="right", anchor_y="bottom") for _ in range(55)]
                self.crafting_label = pyglet.text.Label("Crafting", font_name=target_font, font_size=8, color=(64, 64, 64, 255), anchor_x="left", anchor_y="bottom")
                
            elif len(self.count_labels) < 55:
                target_font = self.count_labels[0].font_name
                self.count_labels.extend([pyglet.text.Label("", font_name=target_font, font_size=8, anchor_x="right", anchor_y="bottom") for _ in range(55 - len(self.count_labels))])

            # Update and draw crafting label
            if hasattr(self, 'crafting_label'):
                self.crafting_label.font_size = max(8, int(8 * bg_scale))
                # Pyglet coordinates for Crafting grid start at px=88. The text should be at px=98, py=142 roughly
                self.crafting_label.x = int(self.inventory_bg_sprite.x + 98 * bg_scale)
                self.crafting_label.y = int(self.inventory_bg_sprite.y + 144 * bg_scale)
                self.crafting_label.draw()                
            # Hovered slot
            mouse_x, mouse_y = getattr(self, 'mouse_pos', (0, 0))
            hovered_slot = -1
            
            # Draw items and labels
            for slot_idx, b_id in enumerate(self.inventory_blocks[:45]):
                x, y, w, h = self._get_slot_rect(slot_idx)
                
                # Check hover
                if x <= mouse_x <= x + w and y <= mouse_y <= y + h:
                    hovered_slot = slot_idx
                    
                if b_id > 0 and b_id in getattr(self, 'block_icon_sprites', {}):
                    sprite = self.block_icon_sprites[b_id]
                    sprite_size = getattr(sprite, 'original_width', 64.0)
                    sprite.scale = (16.0 * bg_scale) / sprite_size
                    sprite.x = int(x)
                    sprite.y = int(y)
                    sprite.draw()
                    
                    count = self.inventory_counts[slot_idx]
                    if count > 0:
                        lbl = self.count_labels[slot_idx]
                        target_size = max(8, int(8 * bg_scale))
                        if lbl.font_size != target_size:
                            lbl.font_size = target_size
                        if lbl.text != str(count):
                            lbl.text = str(count)
                        lbl.x = int(x + 16 * bg_scale)
                        lbl.y = int(y)
                        lbl.draw()
                        
            # Draw Hover highlight
            if hovered_slot != -1:
                hx, hy, hw, hh = self._get_slot_rect(hovered_slot)

                hover_rect = pyglet.shapes.Rectangle(hx, hy, hw, hh, color=(255, 255, 255, 100))
                hover_rect.draw()
                
            # Draw cursor item
            cursor_id = getattr(self, 'cursor_item_id', 0)
            if cursor_id > 0 and cursor_id in getattr(self, 'block_icon_sprites', {}):
                sprite = self.block_icon_sprites[cursor_id]
                sprite_size = getattr(sprite, 'original_width', 64.0)
                sprite.scale = (16.0 * bg_scale) / sprite_size
                sprite.x = int(mouse_x - 8 * bg_scale)
                sprite.y = int(mouse_y - 8 * bg_scale)
                sprite.draw()
                
                count = getattr(self, 'cursor_item_count', 0)
                if count > 0:
                    lbl = getattr(self, 'cursor_label', None)
                    if not lbl:
                        self.cursor_label = pyglet.text.Label("", font_name='Arial', font_size=8, anchor_x="right", anchor_y="bottom")
                        lbl = self.cursor_label
                    target_size = max(8, int(8 * bg_scale))
                    if lbl.font_size != target_size:
                        lbl.font_size = target_size
                    if lbl.text != str(count):
                        lbl.text = str(count)
                    lbl.x = int(sprite.x + 16 * bg_scale)
                    lbl.y = int(sprite.y)
                    lbl.draw()

    def _draw_crafting_gui(self):
        if hasattr(self, 'crafting_bg_sprite') and self.crafting_bg_sprite is not None:
            self.crafting_bg_sprite.draw()
            bg_scale = self.crafting_bg_sprite.scale
            
            if not hasattr(self, 'count_labels'):
                font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'fonts', 'Minecraftia-Regular.ttf')
                try:
                    pyglet.font.add_file(font_path)
                    target_font = 'Minecraftia'
                except Exception:
                    target_font = 'Arial'
                self.count_labels = [pyglet.text.Label("", font_name=target_font, font_size=8, anchor_x="right", anchor_y="bottom") for _ in range(55)]
                self.crafting_label = pyglet.text.Label("Crafting", font_name=target_font, font_size=8, color=(64, 64, 64, 255), anchor_x="left", anchor_y="bottom")
            elif len(self.count_labels) < 55:
                target_font = self.count_labels[0].font_name
                self.count_labels.extend([pyglet.text.Label("", font_name=target_font, font_size=8, anchor_x="right", anchor_y="bottom") for _ in range(55 - len(self.count_labels))])

            if hasattr(self, 'crafting_label'):
                self.crafting_label.font_size = max(8, int(8 * bg_scale))
                self.crafting_label.x = int(self.crafting_bg_sprite.x + 30 * bg_scale)
                self.crafting_label.y = int(self.crafting_bg_sprite.y + 144 * bg_scale)
                self.crafting_label.draw()
                
            mouse_x, mouse_y = getattr(self, 'mouse_pos', (0, 0))
            hovered_slot = -1
            
            # Draw player inventory (0-35) + Crafting grid & output (45-54)
            slots_to_draw = list(range(36)) + list(range(45, 55))
            
            for slot_idx in slots_to_draw:
                b_id = self.inventory_blocks[slot_idx]
                x, y, w, h = self._get_slot_rect(slot_idx)
                
                if x <= mouse_x <= x + w and y <= mouse_y <= y + h:
                    hovered_slot = slot_idx
                    
                if b_id > 0 and b_id in getattr(self, 'block_icon_sprites', {}):
                    sprite = self.block_icon_sprites[b_id]
                    sprite_size = getattr(sprite, 'original_width', 64.0)
                    sprite.scale = (16.0 * bg_scale) / sprite_size
                    sprite.x = int(x)
                    sprite.y = int(y)
                    sprite.draw()
                    
                    count = self.inventory_counts[slot_idx]
                    if count > 0:
                        lbl = self.count_labels[slot_idx]
                        target_size = max(8, int(8 * bg_scale))
                        if lbl.font_size != target_size:
                            lbl.font_size = target_size
                        if lbl.text != str(count):
                            lbl.text = str(count)
                        lbl.x = int(x + 16 * bg_scale)
                        lbl.y = int(y)
                        lbl.draw()
                        
            if hovered_slot != -1:
                hx, hy, hw, hh = self._get_slot_rect(hovered_slot)
                hover_rect = pyglet.shapes.Rectangle(hx, hy, hw, hh, color=(255, 255, 255, 100))
                hover_rect.draw()
                
            cursor_id = getattr(self, 'cursor_item_id', 0)
            if cursor_id > 0 and cursor_id in getattr(self, 'block_icon_sprites', {}):
                sprite = self.block_icon_sprites[cursor_id]
                sprite_size = getattr(sprite, 'original_width', 64.0)
                sprite.scale = (16.0 * bg_scale) / sprite_size
                sprite.x = int(mouse_x - 8 * bg_scale)
                sprite.y = int(mouse_y - 8 * bg_scale)
                sprite.draw()
                
                count = getattr(self, 'cursor_item_count', 0)
                if count > 0:
                    lbl = getattr(self, 'cursor_label', None)
                    if not lbl:
                        self.cursor_label = pyglet.text.Label("", font_name='Arial', font_size=8, anchor_x="right", anchor_y="bottom")
                        lbl = self.cursor_label
                    target_size = max(8, int(8 * bg_scale))
                    if lbl.font_size != target_size:
                        lbl.font_size = target_size
                    if lbl.text != str(count):
                        lbl.text = str(count)
                    lbl.x = int(sprite.x + 16 * bg_scale)
                    lbl.y = int(sprite.y)
                    lbl.draw()

    def _draw_block_highlight(self, proj, view):
    # RENDER BLOCK HIGHLIGHT
        if getattr(self, 'targeted_block', None) is not None:
            bx, by, bz = self.targeted_block
            glUseProgram(self.line_program)
            glUniformMatrix4fv(self.u_line_proj, 1, GL_FALSE, proj)
            glUniformMatrix4fv(self.u_line_view, 1, GL_FALSE, view)
            
            s_x, s_y, s_z = 1.0, 1.0, 1.0
            t_x, t_y, t_z = 0.0, 0.0, 0.0
            block_id, block_data = self.get_block_info(bx, by, bz)
            from core.special_blocks import is_door, is_slab
            if is_door(block_id):
                is_upper = (block_data & 8) != 0
                lower_data = block_data
                upper_data = block_data
                if is_upper:
                    adj_id, adj_data = self.get_block_info(bx, by - 1, bz)
                    if adj_id == block_id: lower_data = adj_data
                else:
                    adj_id, adj_data = self.get_block_info(bx, by + 1, bz)
                    if adj_id == block_id: upper_data = adj_data
                has_right_hinge = (upper_data & 1) != 0
                dir_val = lower_data & 3
                is_open = (lower_data & 4) != 0
                r = 3.0 / 16.0
                minX, minZ, maxX, maxZ = 0.0, 0.0, 1.0, 1.0
                if dir_val == 0:
                    if is_open:
                        if not has_right_hinge: maxZ = r
                        else: minZ = 1.0 - r
                    else: maxX = r
                elif dir_val == 1:
                    if is_open:
                        if not has_right_hinge: minX = 1.0 - r
                        else: maxX = r
                    else: maxZ = r
                elif dir_val == 2:
                    if is_open:
                        if not has_right_hinge: minZ = 1.0 - r
                        else: maxZ = r
                    else: minX = 1.0 - r
                elif dir_val == 3:
                    if is_open:
                        if not has_right_hinge: maxX = r
                        else: minX = 1.0 - r
                    else: minZ = 1.0 - r
                s_x, s_z = (maxX - minX), (maxZ - minZ)
                t_x, t_z = minX, minZ
            elif is_slab(block_id):
                s_y = 0.5
                if (block_data & 4) != 0: t_y = 0.5
            
            s = 1.005
            model_mat = np.array([
                [s_x * s, 0, 0, 0],
                [0, s_y * s, 0, 0],
                [0, 0, s_z * s, 0],
                [bx + t_x - (s_x * s - s_x) / 2.0, by + t_y - (s_y * s - s_y) / 2.0, bz + t_z - (s_z * s - s_z) / 2.0, 1]
            ], dtype=np.float32)
            
            glUniformMatrix4fv(self.u_line_model, 1, GL_FALSE, (GLfloat * 16)(*model_mat.flatten()))
            
            glBindVertexArray(self.line_vao)
            glDrawArrays(GL_LINES, 0, 24)
            glBindVertexArray(0)
            glUseProgram(0)
            
            # Break animation overlay
            if getattr(self, 'breaking_pos', None) == (bx, by, bz) and getattr(self, 'breaking_progress', 0.0) > 0.0:
                block_id = self.get_block(bx, by, bz)
                if block_id > 0:
                    hardness = BLOCK_HARDNESS_ARRAY[block_id]
                    if hardness > 0:
                        req_time = get_break_time(block_id, self.selected_block_id)
                        stage = int((self.breaking_progress / req_time) * 10)
                        if stage > 9: stage = 9
                        layer_idx = self.destroy_stages[stage]
                        if layer_idx > 0:
                            glUseProgram(self.break_program)
                            glUniformMatrix4fv(self.u_break_proj, 1, GL_FALSE, proj)
                            glUniformMatrix4fv(self.u_break_view, 1, GL_FALSE, view)
                            glUniformMatrix4fv(self.u_break_model, 1, GL_FALSE, (GLfloat * 16)(*model_mat.flatten()))
                            glUniform1f(self.u_break_layer, float(layer_idx))
                            
                            glEnable(GL_BLEND)
                            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                            glActiveTexture(GL_TEXTURE0)
                            glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_id)
                            glUniform1i(self.u_break_texture, 0)
                            
                            glBindVertexArray(self.break_vao)
                            glDrawArrays(GL_TRIANGLES, 0, 36)
                            glBindVertexArray(0)
                            glDisable(GL_BLEND)
                            glUseProgram(0)
        

    def _draw_water_overlay(self, proj_view):
    # WATER OVERLAY PASS
        cx, cy, cz = self.camera.x, self.camera.y, self.camera.z
        cy_int = int(math.floor(cy))
        
        # Find closest water surface
        u_water_surface_y = -1000.0
        if self.get_block(cx, cy_int + 1, cz) in (8, 9):
            u_water_surface_y = cy_int + 2.0
        elif self.get_block(cx, cy_int, cz) in (8, 9):
            u_water_surface_y = cy_int + 1.0
        elif self.get_block(cx, cy_int - 1, cz) in (8, 9):
            u_water_surface_y = float(cy_int)
            
        if u_water_surface_y > -999.0:
            glUseProgram(self.water_overlay_program)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_DEPTH_TEST) # Overlay covers screen
            
            # Inverse proj*view for world reconstruction
            if self._camera_matrix_dirty:
                self._cached_inv_pv = np.linalg.inv(proj_view.reshape(4,4)).flatten().astype(np.float32)
                self._camera_matrix_dirty = False
            inv_pv = self._cached_inv_pv
            glUniformMatrix4fv(self.u_inv_proj_view_overlay, 1, GL_FALSE, (ctypes.c_float * 16)(*inv_pv))
            glUniform1f(self.u_water_surface_y_overlay, u_water_surface_y)
            
            glBindVertexArray(self.dummy_vao)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            glBindVertexArray(0)
            
            glEnable(GL_DEPTH_TEST)
            glDisable(GL_BLEND)
            glUseProgram(0)


    def _draw_held_block(self, proj, view):
    # 3D Held Block Viewmodel Rendering
        if hasattr(self, 'hand_block_vaos') and self.selected_block_id in self.hand_block_vaos and getattr(self.camera, 'third_person_mode', 0) == 0:
            glClear(GL_DEPTH_BUFFER_BIT)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_CULL_FACE)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            glUseProgram(self.program)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_id)
            
            # Bobbing and swing animations
            bob_x = math.sin(self.bob_time) * 0.02
            bob_y = math.cos(self.bob_time * 2.0) * 0.015 - 0.01
            
            # Movement checks
            dx, dz = 0.0, 0.0
            front = self.camera.get_front()
            flat_front = normalize_vec([front[0], 0, front[2]])
            right = normalize_vec(cross_vec(flat_front, [0, 1, 0]))
            if self.keys[key.W]: dx += flat_front[0]; dz += flat_front[2]
            if self.keys[key.S]: dx -= flat_front[0]; dz -= flat_front[2]
            if self.keys[key.A]: dx -= right[0]; dz -= right[2]
            if self.keys[key.D]: dx += right[0]; dz += right[2]
            is_moving = math.sqrt(dx*dx + dz*dz) > 0.001
            
            if is_moving and not self.player.is_flying:
                bob_x = math.sin(self.bob_time) * 0.04
                bob_y = math.cos(self.bob_time * 2.0) * 0.03 - 0.02
                
            swing_tx = 0.0
            swing_ty = 0.0
            swing_tz = 0.0
            swing_rx = 0.0
            swing_ry = 0.0
            swing_rz = 0.0
            
            swing1 = 0.0
            swing2 = 0.0
            swing3 = 0.0
            
            if self.swing_time > 0.0:
                swing = self.swing_time
                swing1 = math.sin(swing * math.pi)
                swing2 = math.sin(math.sqrt(swing) * math.pi)
                swing3 = math.sin(swing * swing * math.pi)
                
                swing_tx = -swing2 * 0.4
                swing_ty = math.sin(math.sqrt(swing) * math.pi * 2.0) * 0.2
                swing_tz = -swing1 * 0.2
                
                swing_rx = -swing2 * 80.0
                swing_ry = -swing3 * 20.0
                swing_rz = -swing2 * 20.0
                
            tx = 0.48 + bob_x + swing_tx
            ty = -0.45 + bob_y + swing_ty
            tz = -0.75 + swing_tz
            
            scale = 0.36
            
            if self.selected_block_id == PORKCHOP_RAW:
                # Use Player gui ItemInHandRenderer transform
                # glTranslatef(0.7f * d, -0.65f * d - (1 - h) * 0.6f, -0.9f * d);
                # glRotatef(45, 0, 1, 0);
                # glRotatef(-swing3 * 20, 0, 1, 0);
                # glRotatef(-swing2 * 20, 0, 0, 1);
                # glRotatef(-swing2 * 80, 1, 0, 0);
                # glScalef(0.4, 0.4, 0.4)
                tx = 0.7 * 0.8 - swing2 * 0.4 * 0.8
                ty = -0.65 * 0.8 + math.sin(math.sqrt(self.swing_time)*math.pi*2)*0.2*0.8 + bob_y
                tz = -0.9 * 0.8 - swing1 * 0.2 * 0.8
                
                scale = 0.4 * 1.5
                
                rad_y = math.radians(45 - swing3 * 20)
                cy, sy = math.cos(rad_y), math.sin(rad_y)
                ry = np.array([
                    [cy, 0, sy, 0],
                    [0, 1, 0, 0],
                    [-sy, 0, cy, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                rad_x = math.radians(-swing2 * 80)
                cx_val, sx_val = math.cos(rad_x), math.sin(rad_x)
                rx = np.array([
                    [1, 0, 0, 0],
                    [0, cx_val, -sx_val, 0],
                    [0, sx_val, cx_val, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                rad_z = math.radians(-swing2 * 20)
                cz, sz = math.cos(rad_z), math.sin(rad_z)
                rz = np.array([
                    [cz, -sz, 0, 0],
                    [sz, cz, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                t_mat = np.array([
                    [1, 0, 0, tx],
                    [0, 1, 0, ty],
                    [0, 0, 1, tz],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                s_mat = np.array([
                    [scale, 0, 0, 0],
                    [0, scale, 0, 0],
                    [0, 0, scale, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                # Extra item rotations to tilt it forward and right
                rad_y_item = math.radians(50)
                cy_item, sy_item = math.cos(rad_y_item), math.sin(rad_y_item)
                ry_item = np.array([
                    [cy_item, 0, sy_item, 0],
                    [0, 1, 0, 0],
                    [-sy_item, 0, cy_item, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                rad_z_item = math.radians(335)
                cz_item, sz_item = math.cos(rad_z_item), math.sin(rad_z_item)
                rz_item = np.array([
                    [cz_item, -sz_item, 0, 0],
                    [sz_item, cz_item, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                hand_matrix = t_mat @ ry @ rx @ rz @ s_mat @ ry_item @ rz_item
            else:
                rad_y = math.radians(15 + swing_ry)
                cy, sy = math.cos(rad_y), math.sin(rad_y)
                ry = np.array([
                    [cy, 0, sy, 0],
                    [0, 1, 0, 0],
                    [-sy, 0, cy, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                rad_x = math.radians(-15 + swing_rx)
                cx_val, sx_val = math.cos(rad_x), math.sin(rad_x)
                rx = np.array([
                    [1, 0, 0, 0],
                    [0, cx_val, -sx_val, 0],
                    [0, sx_val, cx_val, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                rad_z = math.radians(8 + swing_rz)
                cz, sz = math.cos(rad_z), math.sin(rad_z)
                rz = np.array([
                    [cz, -sz, 0, 0],
                    [sz, cz, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                t_mat = np.array([
                    [1, 0, 0, tx],
                    [0, 1, 0, ty],
                    [0, 0, 1, tz],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                s_mat = np.array([
                    [scale, 0, 0, 0],
                    [0, scale, 0, 0],
                    [0, 0, scale, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                hand_matrix = t_mat @ ry @ rx @ rz @ s_mat
                
            flat_hand_matrix = hand_matrix.T.flatten()
            
            glUniformMatrix4fv(self.u_projection, 1, GL_FALSE, proj)
            hand_view = (GLfloat * 16)(*flat_hand_matrix)
            glUniformMatrix4fv(self.u_view, 1, GL_FALSE, hand_view)
            
            h_vao, _, num_verts = self.hand_block_vaos[self.selected_block_id]
            glBindVertexArray(h_vao)
            glDrawArrays(GL_TRIANGLES, 0, num_verts)
            
            glBindVertexArray(0)
            glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
            glDisable(GL_BLEND)
            glUseProgram(0)
            

    def _draw_hud(self):
    # 2D GUI Elements
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        # UI/Inventory tick
        if getattr(self, 'crafting_open', False):
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            overlay_rect = pyglet.shapes.Rectangle(0, 0, self.width, self.height, color=(0, 0, 0, 153))
            overlay_rect.draw()
            self._draw_crafting_gui()
        elif getattr(self, 'inventory_open', False):
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            overlay_rect = pyglet.shapes.Rectangle(0, 0, self.width, self.height, color=(0, 0, 0, 153))
            overlay_rect.draw()
            self._draw_inventory_gui()
        else:
            # 1. Crosshair (using color inversion blending)
            glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ONE_MINUS_SRC_COLOR)
            if hasattr(self, 'crosshair_sprite') and self.crosshair_sprite is not None:
                self.crosshair_sprite.draw()
            
            # 2. Hotbar ve Blok Simgeleri (Standart Alpha Blending ile)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
            if hasattr(self, 'hotbar_bg_sprite') and self.hotbar_bg_sprite is not None:
                self.hotbar_bg_sprite.draw()
            
                # Render block icons (scaled from 64x64 to 13*scale and centered in slots)
                bg_scale = self.hotbar_bg_sprite.scale
                if not hasattr(self, 'count_labels_main'):
                    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'fonts', 'Minecraftia-Regular.ttf')
                    try:
                        pyglet.font.add_file(font_path)
                        target_font = 'Minecraftia'
                    except Exception:
                        target_font = 'Arial'
                    self.count_labels_shadow = [pyglet.text.Label("", font_name=target_font, font_size=8, anchor_x="right", anchor_y="bottom", batch=self.ui_batch, color=(63, 63, 63, 255)) for _ in range(9)]
                    self.count_labels_main = [pyglet.text.Label("", font_name=target_font, font_size=8, anchor_x="right", anchor_y="bottom", batch=self.ui_batch, color=(255, 255, 255, 255)) for _ in range(9)]
                
                # Render active slot selection frame FIRST (behind items and text)
                if hasattr(self, 'hotbar_sel_sprite') and self.hotbar_sel_sprite is not None:
                    scale = self.hotbar_sel_sprite.scale
                    self.hotbar_sel_sprite.x = int(self.hotbar_bg_sprite.x - 1 * scale + self.selected_slot * 20 * scale)
                    self.hotbar_sel_sprite.y = int(self.hotbar_bg_sprite.y - 1 * scale)
                    self.hotbar_sel_sprite.draw()

                # Reset labels visibility
                for i in range(9):
                    self.count_labels_shadow[i].text = ""
                    self.count_labels_main[i].text = ""

                for slot_idx, b_id in enumerate(self.inventory_blocks[:9]):
                    if b_id > 0 and b_id in self.block_icon_sprites:
                        sprite = self.block_icon_sprites[b_id]
                        sprite_size = getattr(sprite, 'original_width', 64.0)
                        sprite.scale = (13.0 * bg_scale) / sprite_size
                        sprite.x = int(self.hotbar_bg_sprite.x + (3 + slot_idx * 20) * bg_scale + 1.5 * bg_scale)
                        sprite.y = int(self.hotbar_bg_sprite.y + 3 * bg_scale + 1.5 * bg_scale)
                        sprite.draw()
                    
                        count = self.inventory_counts[slot_idx]
                        if count > 0:
                            target_size = max(8, int(8 * bg_scale))
                            base_x = self.hotbar_bg_sprite.x + (20 + slot_idx * 20) * bg_scale
                            base_y = self.hotbar_bg_sprite.y + 2 * bg_scale
                            offset = max(1, int(1 * bg_scale))
                            
                            lbl_shadow = self.count_labels_shadow[slot_idx]
                            lbl_main = self.count_labels_main[slot_idx]
                            
                            if lbl_shadow.font_size != target_size:
                                lbl_shadow.font_size = target_size
                                lbl_main.font_size = target_size
                            
                            s_count = str(count)
                            if lbl_shadow.text != s_count:
                                lbl_shadow.text = s_count
                                lbl_main.text = s_count
                        
                            lbl_shadow.x = base_x + offset
                            lbl_shadow.y = base_y - offset
                            lbl_main.x = base_x
                            lbl_main.y = base_y

                # Render XP Bar (just for visual for now, 30% full)
                if hasattr(self, 'spr_xp_empty') and hasattr(self, 'spr_xp_full'):
                    xp_x = self.hotbar_bg_sprite.x
                    xp_y = int(self.hotbar_bg_sprite.y + 24 * self.hotbar_bg_sprite.scale)
                
                    self.spr_xp_empty.scale = self.hotbar_bg_sprite.scale
                    self.spr_xp_empty.x = xp_x
                    self.spr_xp_empty.y = xp_y
                    self.spr_xp_empty.draw()
                
                    # XP bar is completely empty for now
                    # self.spr_xp_full.draw()

                # Reset batched sprites visibility
                for spr in self.heart_bg_sprites + self.heart_fg_sprites + self.hunger_bg_sprites + self.hunger_fg_sprites + self.bubble_sprites:
                    spr.visible = False

                # Render Health and Hunger Bars
                if hasattr(self, 'heart_bg_sprites'):
                    # Draw Health (10 hearts)
                    heart_start_x = self.hotbar_bg_sprite.x
                    bar_y = int(self.hotbar_bg_sprite.y + 32 * self.hotbar_bg_sprite.scale)
                
                    player_health = getattr(self.player, 'health', 20.0)
                    player_hunger = getattr(self.player, 'hunger', 20.0)
                    tick_count = int(time.perf_counter() * 20)
                
                    heart_offset_index = -1
                    if player_health < 20.0 and player_hunger >= 18.0:
                        heart_offset_index = tick_count % 25
                    
                    for i in range(10):
                        yo = bar_y
                        if player_health <= 4.0:
                            yo += random.randint(0, 1) * int(self.hotbar_bg_sprite.scale)
                        if i == heart_offset_index:
                            yo += 2 * int(self.hotbar_bg_sprite.scale)
                        
                        hx = heart_start_x + i * 8 * self.heart_bg_sprites[i].scale
                        
                        bg_spr = self.heart_bg_sprites[i]
                        bg_spr.x = hx
                        bg_spr.y = yo
                        bg_spr.visible = True
                    
                        # Inner heart
                        h_val = player_health - (i * 2)
                        if h_val >= 1:
                            fg_spr = self.heart_fg_sprites[i]
                            fg_spr.x = hx
                            fg_spr.y = yo
                            fg_spr.image = self.tex_heart_full if h_val >= 2 else self.tex_heart_half
                            fg_spr.visible = True
                    
                    # Draw Hunger (10 icons, right-aligned)
                    hunger_start_x = self.hotbar_bg_sprite.x + self.hotbar_bg_sprite.width - 9 * self.hunger_bg_sprites[0].scale
                    for i in range(10):
                        yo = bar_y
                        if player_hunger <= 0.0 and tick_count % 20 < 10:
                             yo += random.randint(0, 1) * int(self.hotbar_bg_sprite.scale)
                         
                        hx = hunger_start_x - i * 8 * self.hunger_bg_sprites[i].scale
                        
                        bg_spr = self.hunger_bg_sprites[i]
                        bg_spr.x = hx
                        bg_spr.y = yo
                        bg_spr.visible = True
                    
                        # Inner food
                        f_val = player_hunger - (i * 2)
                        if f_val >= 1:
                            fg_spr = self.hunger_fg_sprites[i]
                            fg_spr.x = hx
                            fg_spr.y = yo
                            fg_spr.image = self.tex_hunger_full if f_val >= 2 else self.tex_hunger_half
                            fg_spr.visible = True

                    # Render Bubbles if underwater
                    if getattr(self.player, 'is_head_in_water', False):
                        bubble_y = bar_y + 10 * scale
                        air_supply = getattr(self.player, 'air_supply', 300.0)
                        air_scale = 10.0 / 300.0
                        air_scaled = air_supply * air_scale
                        count = int(math.ceil((air_supply - 2) * air_scale))
                        extra = int(math.ceil(air_scaled)) - count
                    
                        for i in range(min(10, count + extra)):
                            bx = hunger_start_x - (i * 8 * scale)
                            bspr = self.bubble_sprites[i]
                            bspr.x = bx
                            bspr.y = bubble_y
                            bspr.image = self.tex_bubble_full if i < count else self.tex_bubble_popped
                            bspr.visible = True

                # Draw UI Batch! (One draw call for all labels, hearts, hunger, and bubbles)
                if hasattr(self, 'ui_batch'):
                    self.ui_batch.draw()


