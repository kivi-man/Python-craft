import math
import time
import random
from pyglet.window import key
from pyglet.window import mouse
import pyglet
from core.math_utils import normalize_vec, cross_vec
from core.raycast import raycast
from world.terrain import BLOCK_HARDNESS_ARRAY, PORKCHOP_RAW, get_break_time
from world.terrain import CHUNK_SIZE

class PlayerUpdateMixin:
    def _update_player_actions(self, dt):
        if getattr(self, 'swing_time', 0.0) > 0.0:
            self.swing_time += dt * 5.0
            if self.swing_time >= 1.0:
                self.swing_time = 0.0
                
        # Bobbing speed depends on walking state
        is_moving = False
        dx, dz = 0.0, 0.0
        front = self.camera.get_front()
        flat_front = normalize_vec([front[0], 0, front[2]])
        right = normalize_vec(cross_vec(flat_front, [0, 1, 0]))
        if self.keys[key.W]: dx += flat_front[0]; dz += flat_front[2]
        if self.keys[key.S]: dx -= flat_front[0]; dz -= flat_front[2]
        if self.keys[key.A]: dx -= right[0]; dz -= right[2]
        if self.keys[key.D]: dx += right[0]; dz += right[2]
        if math.sqrt(dx*dx + dz*dz) > 0.001:
            is_moving = True
            
        if is_moving and not self.player.is_flying:
            self.bob_time += dt * 10.0
        else:
            self.bob_time += dt * 1.5 # Slow breathing bobbing animation
            
        if not getattr(self, 'inventory_open', False):
            # Mouse hold actions (continuous block break/place)
            if self.mouse_action_cooldown > 0.0:
                self.mouse_action_cooldown -= dt
                
            if self.mouse_action_cooldown <= 0.005:
                if self.mouse_held[mouse.LEFT]:
                    self.swing_time = 0.01
                    self._handle_mouse_action(mouse.LEFT)
                    self.mouse_action_cooldown = 0.20
                elif self.mouse_held[mouse.RIGHT]:
                    if self.selected_block_id == PORKCHOP_RAW:
                        pass # Handled below for continuous eating
                    else:
                        self.swing_time = 0.01
                        self._handle_mouse_action(mouse.RIGHT)
                        self.mouse_action_cooldown = 0.20
                    
            # Continuous Eating Progress
            if self.mouse_held[mouse.RIGHT]:
                if self.selected_block_id == PORKCHOP_RAW and self.player.hunger < 20.0:
                    self.eating_progress += dt
                    
                    # Play random eating sound every 0.3 seconds
                    if self.eating_progress - self.last_eat_sound > 0.3:
                        self.last_eat_sound = self.eating_progress
                        self.sound_system.play("eSoundType_EATING", x=self.player.x, y=self.player.y, z=self.player.z, volume=0.5, pitch=random.uniform(0.9, 1.1))
                        
                    # Bob hand slightly while eating
                    self.swing_time += dt * 3.0
                    
                    if self.eating_progress >= 1.6: # Takes 1.6 seconds to eat
                        self.eating_progress = 0.0
                        self.player.hunger = min(20.0, getattr(self.player, 'hunger', 20.0) + 4.0) # Restores 4 hunger points (2 shanks)
                        self.inventory_counts[self.selected_slot] -= 1
                        if self.inventory_counts[self.selected_slot] <= 0:
                            self.inventory_blocks[self.selected_slot] = 0
                            self.selected_block_id = 0
                else:
                    self.eating_progress = 0.0
            else:
                self.eating_progress = 0.0
                    
            # Always update targeted block for highlighting
            eye_pos = self.player.get_eye_position()
            direction = self.camera.get_front()
            if eye_pos != self.last_raycast_eye_pos or direction != self.last_raycast_dir:
                self.last_raycast_eye_pos = eye_pos
                self.last_raycast_dir = direction
                self._camera_matrix_dirty = True
                hx, hy, hz, px, py, pz = raycast(eye_pos, direction, self.get_block)
                if hx is not None:
                    self.targeted_block = (hx, hy, hz)
                else:
                    self.targeted_block = None

            # Continuous Block Breaking Progress
            if self.mouse_held[mouse.LEFT]:
                if self.targeted_block is not None:
                    hx, hy, hz = self.targeted_block
                    current_target = (hx, hy, hz)
                    if getattr(self, 'breaking_pos', None) == current_target:
                        broken_id = self.get_block(hx, hy, hz)
                        if broken_id > 0:
                            hardness = BLOCK_HARDNESS_ARRAY[broken_id]
                            if hardness >= 0:
                                self.breaking_progress += dt
                                if random.random() < 0.25:
                                    self.spawn_crack_particles(hx, hy, hz, broken_id, 1)
                                    # Play continuous digging sound
                                    if time.perf_counter() - self.last_dig_sound_time > 0.2:
                                        self.last_dig_sound_time = time.perf_counter()
                                        sound_enum = self.sound_system.get_dig_sound(broken_id)
                                        self.sound_system.play(sound_enum, x=hx+0.5, y=hy+0.5, z=hz+0.5, volume=0.3, pitch=random.uniform(0.9, 1.1))
                                    
                                required_time = get_break_time(broken_id, self.selected_block_id)
                                if self.breaking_progress >= required_time:
                                    broken_id, broken_data = self.get_block_info(hx, hy, hz)
                                    self.spawn_destruction_particles(hx, hy, hz, broken_id)
                                    self.spawn_item_entity(broken_id, hx + 0.5, hy + 0.5, hz + 0.5)
                                    self.set_block(hx, hy, hz, 0)
                                    
                                    from core.special_blocks import is_door
                                    if is_door(broken_id):
                                        if (broken_data & 8) != 0:
                                            lower_id, _ = self.get_block_info(hx, hy - 1, hz)
                                            if lower_id == broken_id:
                                                self.set_block(hx, hy - 1, hz, 0)
                                        else:
                                            upper_id, _ = self.get_block_info(hx, hy + 1, hz)
                                            if upper_id == broken_id:
                                                self.set_block(hx, hy + 1, hz, 0)
                                    
                                    upper_id, upper_data = self.get_block_info(hx, hy + 1, hz)
                                    if is_door(upper_id) and (upper_data & 8) == 0:
                                        self.set_block(hx, hy + 1, hz, 0)
                                        self.spawn_item_entity(upper_id, hx + 0.5, hy + 1.5, hz + 0.5)
                                        uu_id, _ = self.get_block_info(hx, hy + 2, hz)
                                        if uu_id == upper_id:
                                            self.set_block(hx, hy + 2, hz, 0)
                                    
                                    # Play break sound
                                    sound_enum = self.sound_system.get_dig_sound(broken_id)
                                    self.sound_system.play(sound_enum, x=hx+0.5, y=hy+0.5, z=hz+0.5, volume=0.5)
                                    
                                    self.breaking_pos = None
                                    self.breaking_progress = 0.0
                                    self.mouse_action_cooldown = 0.20
                    else:
                        self.breaking_pos = current_target
                        self.breaking_progress = 0.0
                else:
                    self.breaking_pos = None
                    self.breaking_progress = 0.0
        else:
            self.breaking_pos = None
            self.breaking_progress = 0.0
        

    def _update_player_movement_and_physics(self, dt):
        front = self.camera.get_front()
        flat_front = normalize_vec([front[0], 0, front[2]])
        right = normalize_vec(cross_vec(flat_front, [0, 1, 0]))
        
        dx, dz = 0.0, 0.0
        if self.keys[key.W]: dx += flat_front[0]; dz += flat_front[2]
        if self.keys[key.S]: dx -= flat_front[0]; dz -= flat_front[2]
        if self.keys[key.A]: dx -= right[0]; dz -= right[2]
        if self.keys[key.D]: dx += right[0]; dz += right[2]
            
        length = math.sqrt(dx*dx + dz*dz)
        if length > 0.001:
            dx /= length
            dz /= length
            
        jump = self.keys[key.SPACE]
        crouch = self.keys[key.LSHIFT]
        sprint = self.keys[key.LCTRL] or getattr(self, 'is_sprinting_w', False)
        
        # Prevent player from falling if inside an unloaded chunk
        pcx = int(math.floor(self.player.x / CHUNK_SIZE))
        pcz = int(math.floor(self.player.z / CHUNK_SIZE))
        
        t_player_start = time.perf_counter()
        if (pcx, pcz) in self.world_chunks:
            old_x, old_z = self.player.x, self.player.z
            old_vy = self.player.vy
            was_on_ground = getattr(self.player, 'on_ground', False)
            was_in_water = getattr(self.player, 'in_water', False)
            
            self.player.update(dt, dx, dz, jump, crouch, sprint, self.get_block_info)
            
            is_on_ground = getattr(self.player, 'on_ground', False)
            is_in_water = getattr(self.player, 'in_water', False)
            
            # Fall Damage Sound
            if is_on_ground and not was_on_ground and not is_in_water:
                if old_vy < -25.0:
                    self.sound_system.play("eSoundType_DAMAGE_FALL_BIG", x=self.player.x, y=self.player.y, z=self.player.z, volume=0.8)
                    self.sound_system.play("eSoundType_DAMAGE_HURT", x=self.player.x, y=self.player.y, z=self.player.z, volume=0.8)
                elif old_vy < -15.0:
                    self.sound_system.play("eSoundType_DAMAGE_FALL_SMALL", x=self.player.x, y=self.player.y, z=self.player.z, volume=0.5)
                    self.sound_system.play("eSoundType_DAMAGE_HURT", x=self.player.x, y=self.player.y, z=self.player.z, volume=0.5)
            
            # Splash Sound
            if is_in_water and not was_in_water and old_vy < -10.0:
                self.sound_system.play("eSoundType_RANDOM_SPLASH", x=self.player.x, y=self.player.y, z=self.player.z, volume=0.6)
                
            # Accumulate walking distance regardless of being on ground (for bunny hopping)
            dx_moved = self.player.x - old_x
            dz_moved = self.player.z - old_z
            dist = math.sqrt(dx_moved*dx_moved + dz_moved*dz_moved)
            self.distance_walked += dist
            
            if is_on_ground or is_in_water:
                step_threshold = 1.0 if sprint else 1.5
                
                # Force a landing step sound if landing from a small jump
                if is_on_ground and not was_on_ground and not is_in_water:
                    if old_vy >= -15.0:
                        self.distance_walked = step_threshold + 0.1
                        
                if self.distance_walked > step_threshold:
                    self.distance_walked = 0.0
                    if is_in_water:
                        self.sound_system.play("eSoundType_LIQUID_SWIM", x=self.player.x, y=self.player.y, z=self.player.z, volume=0.3)
                    else:
                        block_below = self.get_block(self.player.x, self.player.y - 0.1, self.player.z)
                        if block_below > 0:
                            sound_enum = self.sound_system.get_step_sound(block_below)
                            self.sound_system.play(sound_enum, x=self.player.x, y=self.player.y, z=self.player.z, volume=0.3)
        else:
            self.player.vy = 0.0 # Reset gravity accumulation
        t_player_end = time.perf_counter()

    def _update_audio_listener(self):
        try:
            listener = pyglet.media.get_audio_driver().get_listener()
            listener.position = (self.camera.x, self.camera.y, self.camera.z)
            fx, fy, fz = self.camera.get_front()
            listener.forward_orientation = (fx, fy, fz)
            listener.up_orientation = (0, 1, 0)
        except Exception as e:
            pass # Failsafe if audio driver is missing
        

