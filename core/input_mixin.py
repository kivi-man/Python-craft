import pyglet
import math
from pyglet.window import key, mouse
from pyglet.gl import glViewport
from core.raycast import raycast
from world.terrain import CACTUS, SAND, BLOCK_MAX_STACK_ARRAY

class InputMixin:

    def add_to_inventory(self, block_id):
        if block_id <= 0: return False
        
        max_stack = BLOCK_MAX_STACK_ARRAY[block_id]
        
        for i in range(36):
            if self.inventory_blocks[i] == block_id and self.inventory_counts[i] < max_stack:
                self.inventory_counts[i] += 1
                if self.selected_slot == i:
                    self.selected_block_id = block_id
                return True
                
        for i in range(36):
            if self.inventory_blocks[i] == 0:
                self.inventory_blocks[i] = block_id
                self.inventory_counts[i] = 1
                if self.selected_slot == i:
                    self.selected_block_id = block_id
                return True
                
        return False

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
                from core.entities.item_entity import ItemEntity
                if isinstance(entity, ItemEntity): continue
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
                import math
                block_dist = 5.0
                if px is not None and py is not None and pz is not None:
                    block_dist = math.sqrt((px - eye_pos[0])**2 + (py - eye_pos[1])**2 + (pz - eye_pos[2])**2)
                    
                if min_t < block_dist:
                    hit_entity.hurt(4) # Player hits for 4 damage
                    # Apply Knockback
                hit_entity.dx += direction[0] * 0.4
                hit_entity.dy = 0.3
                hit_entity.dz += direction[2] * 0.4
                
                # Play hit sound
                if hasattr(self, 'sound_system'):
                    from core.entities.pig import Pig
                    if isinstance(hit_entity, Pig):
                        if hit_entity.dead:
                            self.sound_system.play("eSoundType_MOB_PIG_DEATH", x=hit_entity.x, y=hit_entity.y, z=hit_entity.z, volume=0.7)
                        else:
                            self.sound_system.play("eSoundType_MOB_PIG_AMBIENT", x=hit_entity.x, y=hit_entity.y, z=hit_entity.z, volume=0.5, pitch=1.2) # Pitch up for hurt
                    else:
                        self.sound_system.play("eSoundType_DAMAGE_HURT", x=hit_entity.x, y=hit_entity.y, z=hit_entity.z, volume=0.5)
                    return
                
            # Block breaking is handled progressively in main loop update()
            return
            
        elif button == mouse.RIGHT and px is not None:
            hit_id, hit_data = self.get_block_info(hx, hy, hz)
            from core.special_blocks import is_door
            if is_door(hit_id) and hit_id != 163: # 163 = IRON_DOOR
                is_upper = (hit_data & 8) != 0
                if is_upper:
                    lower_id, lower_data = self.get_block_info(hx, hy - 1, hz)
                    if lower_id == hit_id:
                        new_data = lower_data ^ 4
                        self.set_block(hx, hy - 1, hz, hit_id, new_data)
                        self.set_block(hx, hy, hz, hit_id, hit_data) # Trigger update
                    else:
                        new_data = lower_data # Fallback
                else:
                    new_data = hit_data ^ 4
                    self.set_block(hx, hy, hz, hit_id, new_data)
                    upper_id, upper_data = self.get_block_info(hx, hy + 1, hz)
                    if upper_id == hit_id:
                        self.set_block(hx, hy + 1, hz, hit_id, upper_data) # Trigger update
                        
                if hasattr(self, 'sound_system'):
                    is_open = (new_data & 4) != 0
                    sound_name = "eSoundType_RANDOM_DOOR_OPEN" if is_open else "eSoundType_RANDOM_DOOR_CLOSE"
                    self.sound_system.play(sound_name, x=hx, y=hy, z=hz, volume=0.7)
                return
                
            from world.terrain import CRAFTING_TABLE
            if hit_id == CRAFTING_TABLE:
                self.crafting_open = True
                self.set_exclusive_mouse(False)
                if hasattr(self, '_evaluate_crafting'): self._evaluate_crafting()
                return
                
            if self.selected_block_id == 0 or self.inventory_counts[self.selected_slot] <= 0:
                return
                
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
            if button == mouse.RIGHT and 0 < new_block_id < 256:
                self.inventory_counts[self.selected_slot] -= 1
                if self.inventory_counts[self.selected_slot] <= 0:
                    self.inventory_blocks[self.selected_slot] = 0
                    self.selected_block_id = 0
                    
                data = 0
                from core.special_blocks import is_stairs, is_slab, is_door, SLAB_TO_FULL
                
                if is_door(new_block_id):
                    if target_y >= 255: return
                    below_id = self.get_block(target_x, target_y - 1, target_z)
                    if below_id == 0 or below_id == 4: return # Needs solid block below
                    upper_id = self.get_block(target_x, target_y + 1, target_z)
                    if upper_id != 0 and upper_id != 4: return # Needs air
                    
                    import math
                    yaw = self.camera.yaw % 360
                    if yaw < 45 or yaw >= 315: d = 2 # player looks East -> door faces West
                    elif yaw < 135: d = 3            # player looks South -> door faces North
                    elif yaw < 225: d = 0            # player looks West -> door faces East
                    else: d = 1                      # player looks North -> door faces South
                        
                    lower_data = d
                    upper_data = 8
                    
                    left_block = right_block = 0
                    if d == 0:   # East (player West)
                        left_block = self.get_block(target_x, target_y, target_z + 1)
                        right_block = self.get_block(target_x, target_y, target_z - 1)
                    elif d == 1: # South (player North)
                        left_block = self.get_block(target_x + 1, target_y, target_z)
                        right_block = self.get_block(target_x - 1, target_y, target_z)
                    elif d == 2: # West (player East)
                        left_block = self.get_block(target_x, target_y, target_z - 1)
                        right_block = self.get_block(target_x, target_y, target_z + 1)
                    elif d == 3: # North (player South)
                        left_block = self.get_block(target_x - 1, target_y, target_z)
                        right_block = self.get_block(target_x + 1, target_y, target_z)
                        
                    if left_block == new_block_id:
                        upper_data = 9
                    elif right_block == new_block_id:
                        upper_data = 8
                    
                    self.inventory_counts[self.selected_slot] -= 1
                    if self.inventory_counts[self.selected_slot] <= 0:
                        self.inventory_blocks[self.selected_slot] = 0
                        self.selected_block_id = 0
                        
                    self.set_block(target_x, target_y, target_z, new_block_id, lower_data)
                    self.set_block(target_x, target_y + 1, target_z, new_block_id, upper_data)
                    
                    if hasattr(self, 'sound_system'):
                        sound_enum = self.sound_system.get_step_sound(new_block_id)
                        self.sound_system.play(sound_enum, volume=0.5)
                    return
                    
                elif is_stairs(new_block_id) or is_slab(new_block_id):
                    hx, hy, hz = self.targeted_block if hasattr(self, 'targeted_block') and self.targeted_block else (target_x, target_y - 1, target_z) # Fallback
                    
                    if is_slab(new_block_id):
                        t_id, t_data = self.get_block_info(hx, hy, hz)
                        if t_id == new_block_id:
                            if (t_data == 0 and target_y == hy + 1) or (t_data == 4 and target_y == hy - 1):
                                self.set_block(hx, hy, hz, SLAB_TO_FULL.get(new_block_id, 1), 0)
                                return

                    # Determine hit fraction
                    hit_y_frac = 0.0
                    if target_y == hy + 1:
                        hit_y_frac = 0.0 # hit top face -> place normal
                    elif target_y == hy - 1:
                        hit_y_frac = 1.0 # hit bottom face -> place upside down
                    else:
                        # hit side face
                        dx, dy, dz = direction
                        if target_x == hx - 1: t = (hx - eye_pos[0]) / (dx + 1e-8)
                        elif target_x == hx + 1: t = (hx + 1 - eye_pos[0]) / (dx + 1e-8)
                        elif target_z == hz - 1: t = (hz - eye_pos[2]) / (dz + 1e-8)
                        elif target_z == hz + 1: t = (hz + 1 - eye_pos[2]) / (dz + 1e-8)
                        else: t = 0
                        hit_y_frac = (eye_pos[1] + t * dy) - hy
                        
                    upside_down = 4 if hit_y_frac > 0.5 else 0
                    
                    if is_stairs(new_block_id):
                        look_dx, look_dz = direction[0], direction[2]
                        if abs(look_dx) > abs(look_dz):
                            d = 0 if look_dx > 0 else 1
                        else:
                            d = 2 if look_dz > 0 else 3
                        data = d | upside_down
                    elif is_slab(new_block_id):
                        data = upside_down
                
                self.set_block(target_x, target_y, target_z, new_block_id, data)
                
                # Play block place sound
                if hasattr(self, 'sound_system'):
                    sound_enum = self.sound_system.get_step_sound(new_block_id)
                    self.sound_system.play(sound_enum, volume=0.5)
            

    def on_mouse_press(self, x, y, button, modifiers):
        if getattr(self, 'inventory_open', False) or getattr(self, 'crafting_open', False):
            if hasattr(self, '_handle_inventory_click'):
                self._handle_inventory_click(x, y, button)
            return

        if button in (mouse.LEFT, mouse.RIGHT):
            self.mouse_held[button] = True
            self.swing_time = 0.01 # Swing animasyonunu tetikle
            
            if button == mouse.RIGHT:
                self._handle_mouse_action(button)
                self.mouse_action_cooldown = 0.20 # 4 tick (0.2s) cooldown
            elif button == mouse.LEFT:
                self._handle_mouse_action(button) # Only handles entity attack now
                # Start block breaking raycast
                eye_pos = self.player.get_eye_position()
                direction = self.camera.get_front()
                hx, hy, hz, px, py, pz = raycast(eye_pos, direction, self.get_block)
                if hx is not None:
                    self.breaking_pos = (hx, hy, hz)
                    self.breaking_progress = 0.0

    def on_mouse_release(self, x, y, button, modifiers):
        if button in (mouse.LEFT, mouse.RIGHT):
            self.mouse_held[button] = False
            if button == mouse.LEFT:
                self.breaking_pos = None
                self.breaking_progress = 0.0

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_pos = (x, y)
        if getattr(self, 'inventory_open', False) or getattr(self, 'crafting_open', False): return

        self.camera.yaw += dx * self.camera.sensitivity
        self.camera.pitch += dy * self.camera.sensitivity
        self.camera.pitch = max(-89.0, min(89.0, self.camera.pitch))
        

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.mouse_pos = (x, y)
        if getattr(self, 'inventory_open', False) or getattr(self, 'crafting_open', False): return

        self.camera.yaw += dx * self.camera.sensitivity
        self.camera.pitch += dy * self.camera.sensitivity
        self.camera.pitch = max(-89.0, min(89.0, self.camera.pitch))
    

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.set_exclusive_mouse(False)
            self.close()
        elif symbol == key.F5:
            if hasattr(self, 'camera'):
                self.camera.third_person_mode = (self.camera.third_person_mode + 1) % 3
                modes = ["First Person", "Third Person Back", "Third Person Front"]
                if self.debug_mode:
                    print(f"[CAMERA] Switched to {modes[self.camera.third_person_mode]}")
        elif symbol == key.Q:
            if self.selected_block_id > 0 and self.inventory_counts[self.selected_slot] > 0:
                self.inventory_counts[self.selected_slot] -= 1
                
                # Calculate throw velocity based on camera
                dir_x, dir_y, dir_z = self.camera.get_front()
                
                eye = self.player.get_eye_position()
                
                # Spawn slightly in front
                self.spawn_item_entity(
                    self.selected_block_id, 
                    eye[0] + dir_x * 0.5, 
                    eye[1] - 0.2, 
                    eye[2] + dir_z * 0.5,
                    xd=dir_x * 0.3,
                    yd=dir_y * 0.3 + 0.1,
                    zd=dir_z * 0.3
                )
                
                if self.inventory_counts[self.selected_slot] <= 0:
                    self.inventory_blocks[self.selected_slot] = 0
                    self.selected_block_id = 0
                    
        elif symbol == key.O:
            # Gamemode 1 (Creative Mode) with Pagination
            self.player.is_flying = True
            from world.terrain import BLOCK_REGISTRY
            
            # Initialize creative page counter
            self.creative_page = getattr(self, 'creative_page', 0)
            
            # Extract valid block IDs
            valid_blocks = [data["id"] for name, data in BLOCK_REGISTRY.items() if data["id"] > 0]
            
            # Calculate start index
            start_idx = self.creative_page * len(self.inventory_blocks)
            if start_idx >= len(valid_blocks):
                self.creative_page = 0
                start_idx = 0
                
            page_blocks = valid_blocks[start_idx:start_idx + len(self.inventory_blocks)]
            
            # Fill inventory
            for i in range(len(self.inventory_blocks)):
                if i < len(page_blocks):
                    self.inventory_blocks[i] = page_blocks[i]
                    self.inventory_counts[i] = 64
                else:
                    self.inventory_blocks[i] = 0
                    self.inventory_counts[i] = 0
            
            self.creative_page += 1
            self.selected_block_id = self.inventory_blocks[self.selected_slot]
            if self.debug_mode:
                print(f"[PLAYER] Gamemode 1 Activated. Inventory filled (Page {self.creative_page}).")
                    
        elif symbol == key.TAB:
            self.player.is_flying = not self.player.is_flying
            mode = "ENABLED" if self.player.is_flying else "DISABLED"
            if self.debug_mode:
                print(f"[PLAYER] Flight Mode: {mode}")
        elif symbol == key._1:
            self.selected_slot = 0
            self.selected_block_id = self.inventory_blocks[0]
            if self.debug_mode:
                print("[PLAYER] Selected Block: STONE")
        elif symbol == key._2:
            self.selected_slot = 1
            self.selected_block_id = self.inventory_blocks[1]
            if self.debug_mode:
                print("[PLAYER] Selected Block: GRASS")
        elif symbol == key._3:
            self.selected_slot = 2
            self.selected_block_id = self.inventory_blocks[2]
            if self.debug_mode:
                print("[PLAYER] Selected Block: GLASS")
        elif symbol == key._4:
            self.selected_slot = 3
            self.selected_block_id = self.inventory_blocks[3]
            if self.debug_mode:
                print("[PLAYER] Selected Block: LEAVES")
        elif symbol == key._5:
            self.selected_slot = 4
            self.selected_block_id = self.inventory_blocks[4]
            if self.debug_mode:
                print("[PLAYER] Selected Block: WATER")
        elif symbol == key._6:
            self.selected_slot = 5
            self.selected_block_id = self.inventory_blocks[5]
            if self.debug_mode:
                print("[PLAYER] Selected Block: CACTUS")
        elif symbol == key._7:
            self.selected_slot = 6
            self.selected_block_id = self.inventory_blocks[6]
        elif symbol == key._8:
            self.selected_slot = 7
            self.selected_block_id = self.inventory_blocks[7]
        elif symbol == key._9:
            self.selected_slot = 8
            self.selected_block_id = self.inventory_blocks[8]
            

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        # scroll_y: +1 for scrolling up, -1 for scrolling down
        # We have 9 selectable slots (0-8)
        direction = -1 if scroll_y > 0 else 1
        self.selected_slot = (self.selected_slot + direction) % 9
        self.selected_block_id = self.inventory_blocks[self.selected_slot]
        
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
    
