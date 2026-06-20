import pyglet
import math
from pyglet.window import key, mouse
from pyglet.gl import glViewport
from core.raycast import raycast
from world.terrain import CACTUS, SAND

class InputMixin:

    def _handle_mouse_action(self, button):
        eye_pos = self.player.get_eye_position()
        direction = self.camera.get_front()
        hx, hy, hz, px, py, pz = raycast(eye_pos, direction, self.get_block)
        
        if hx is None: return
        target_x, target_y, target_z = None, None, None
        new_block_id = 0
        
        if button == mouse.LEFT:
            # Entity Raycast
            hit_entity = None
            min_t = 5.0
            
            for entity in getattr(self, 'entities', []):
                if getattr(entity, 'dead', False): continue
                hw = entity.width / 2.0
                min_b = (entity.x - hw - 0.2, entity.y - 0.1, entity.z - hw - 0.2)
                max_b = (entity.x + hw + 0.2, entity.y + entity.height + 0.1, entity.z + hw + 0.2)
                
                tx1 = (min_b[0] - eye_pos[0]) / (direction[0] + 1e-8)
                tx2 = (max_b[0] - eye_pos[0]) / (direction[0] + 1e-8)
                tmin = min(tx1, tx2)
                tmax = max(tx1, tx2)
                
                ty1 = (min_b[1] - eye_pos[1]) / (direction[1] + 1e-8)
                ty2 = (max_b[1] - eye_pos[1]) / (direction[1] + 1e-8)
                tmin = max(tmin, min(ty1, ty2))
                tmax = min(tmax, max(ty1, ty2))
                
                tz1 = (min_b[2] - eye_pos[2]) / (direction[2] + 1e-8)
                tz2 = (max_b[2] - eye_pos[2]) / (direction[2] + 1e-8)
                tmin = max(tmin, min(tz1, tz2))
                tmax = min(tmax, max(tz1, tz2))
                
                if tmax >= tmin and tmin < min_t and tmax > 0:
                    min_t = tmin
                    hit_entity = entity
            
            if hit_entity is not None:
                hit_entity.hurt(4) # Player hits for 4 damage
                # Apply Knockback
                hit_entity.dx += direction[0] * 0.4
                hit_entity.dy = 0.3
                hit_entity.dz += direction[2] * 0.4
                return
                
            target_x, target_y, target_z = hx, hy, hz
            new_block_id = 0
        elif button == mouse.RIGHT and px is not None:
            # Collision prevention: Prevent player from placing a solid block inside their own body
            if self.selected_block_id > 0 and self.selected_block_id != 4:
                player_aabb = self.player._get_player_aabb(self.player.x, self.player.y, self.player.z)
                block_aabb = (px, py, pz, px + 1.0, py + 1.0, pz + 1.0)
                
                # AABB intersection check
                intersects = (
                    player_aabb[0] < block_aabb[3] and player_aabb[3] > block_aabb[0] and
                    player_aabb[1] < block_aabb[4] and player_aabb[4] > block_aabb[1] and
                    player_aabb[2] < block_aabb[5] and player_aabb[5] > block_aabb[2]
                )
                if intersects:
                    return
            
            target_x, target_y, target_z = px, py, pz
            new_block_id = self.selected_block_id
            
            # Cactus placement rules
            if new_block_id == CACTUS:
                below = self.get_block(px, py - 1, pz)
                if below not in (SAND, CACTUS):
                    return
                for dx, dz in [(1,0), (-1,0), (0,1), (0,-1)]:
                    adj = self.get_block(px + dx, py, pz + dz)
                    if adj != 0 and adj != 4: # Not AIR and not WATER
                        return
                 
        if target_x is not None:
            self.set_block(target_x, target_y, target_z, new_block_id)
            

    def on_mouse_press(self, x, y, button, modifiers):
        if button in (mouse.LEFT, mouse.RIGHT):
            self.mouse_held[button] = True
            self.swing_time = 0.01 # Swing animasyonunu tetikle
            self._handle_mouse_action(button)
            self.mouse_action_cooldown = 0.20 # 4 tick (0.2s) cooldown
            

    def on_mouse_release(self, x, y, button, modifiers):
        if button in (mouse.LEFT, mouse.RIGHT):
            self.mouse_held[button] = False
            

    def on_mouse_motion(self, x, y, dx, dy):
        self.camera.yaw += dx * self.camera.sensitivity
        self.camera.pitch += dy * self.camera.sensitivity
        self.camera.pitch = max(-89.0, min(89.0, self.camera.pitch))
        

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.camera.yaw += dx * self.camera.sensitivity
        self.camera.pitch += dy * self.camera.sensitivity
        self.camera.pitch = max(-89.0, min(89.0, self.camera.pitch))
    

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.set_exclusive_mouse(False)
            pyglet.app.exit()
        elif symbol == key.TAB:
            self.player.is_flying = not self.player.is_flying
            mode = "ENABLED" if self.player.is_flying else "DISABLED"
            if self.debug_mode:
                print(f"[PLAYER] Flight Mode: {mode}")
        elif symbol == key._1:
            self.selected_slot = 0
            self.selected_block_id = self.hotbar_blocks[0]
            if self.debug_mode:
                print("[PLAYER] Selected Block: STONE")
        elif symbol == key._2:
            self.selected_slot = 1
            self.selected_block_id = self.hotbar_blocks[1]
            if self.debug_mode:
                print("[PLAYER] Selected Block: GRASS")
        elif symbol == key._3:
            self.selected_slot = 2
            self.selected_block_id = self.hotbar_blocks[2]
            if self.debug_mode:
                print("[PLAYER] Selected Block: GLASS")
        elif symbol == key._4:
            self.selected_slot = 3
            self.selected_block_id = self.hotbar_blocks[3]
            if self.debug_mode:
                print("[PLAYER] Selected Block: LEAVES")
        elif symbol == key._5:
            self.selected_slot = 4
            self.selected_block_id = self.hotbar_blocks[4]
            if self.debug_mode:
                print("[PLAYER] Selected Block: WATER")
        elif symbol == key._6:
            self.selected_slot = 5
            self.selected_block_id = self.hotbar_blocks[5]
            if self.debug_mode:
                print("[PLAYER] Selected Block: CACTUS")
            

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        # scroll_y: +1 for scrolling up, -1 for scrolling down
        # We have 6 selectable slots (0-5)
        direction = -1 if scroll_y > 0 else 1
        self.selected_slot = (self.selected_slot + direction) % 6
        self.selected_block_id = self.hotbar_blocks[self.selected_slot]
        
        block_names = {
            1: "STONE",
            3: "GRASS",
            20: "GLASS",
            12: "LEAVES",
            4: "WATER",
            CACTUS: "CACTUS"
        }
        name = block_names.get(self.selected_block_id, "UNKNOWN")
        if self.debug_mode:
            print(f"[PLAYER] Selected Block: {name}")
    

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)
        self._update_gui_positions(width, height)
        return pyglet.event.EVENT_HANDLED
    
