import time
from core.entities.pig import Pig
from renderer.pig_renderer import PigRenderer
from core.entities.item_entity import ItemEntity
from renderer.item_renderer import ItemRenderer
from world.spawner import NaturalSpawner

class EntityMixin:
    def _init_entities(self):
        self.entities = []
        self.block_particles = []
        self.particle_renderer = None
        self.renderers = {
            Pig: PigRenderer(),
            ItemEntity: ItemRenderer()
        }
        self.tick_timer = 0.0
        self.spawner = NaturalSpawner(self)

    def _update_entities(self, dt):
        self.tick_timer += dt
        while self.tick_timer >= 0.05: # 20 TPS
            self.tick_timer -= 0.05
            self.spawner.tick()
            
            # Particle cleanup - filter instead of remove() which is O(n)
            if self.block_particles:
                alive_particles = []
                for p in self.block_particles:
                    if p.removed:
                        continue
                    p.tick(self.get_block)
                    alive_particles.append(p)
                self.block_particles = alive_particles
                
            # Entity cleanup - filter instead of remove() which is O(n)
            alive_entities = []
            item_entities = []
            
            sim_dist = getattr(self, 'simulation_distance', 4) * 16.0
            
            for entity in self.entities:
                if entity.removed:
                    continue
                    
                # Simulation Distance Check
                dx = entity.x - self.player.x
                dz = entity.z - self.player.z
                dist_sq = dx*dx + dz*dz
                
                if dist_sq <= sim_dist * sim_dist:
                    entity.tick(self.get_block)
                else:
                    # Keep frozen entities from interpolating their last movement tick
                    entity.xo, entity.yo, entity.zo = entity.x, entity.y, entity.z
                    entity.xRotO = entity.xRot
                    entity.yRotO = entity.yRot
                    if hasattr(entity, 'yHeadRot'):
                        entity.yHeadRotO = entity.yHeadRot
                    if hasattr(entity, 'yBodyRot'):
                        entity.yBodyRotO = entity.yBodyRot
                    if hasattr(entity, 'walk_anim_pos'):
                        entity.walk_anim_pos_o = entity.walk_anim_pos
                    if hasattr(entity, 'walk_anim_speed'):
                        entity.walk_anim_speed_o = entity.walk_anim_speed
                
                alive_entities.append(entity)
                
                if isinstance(entity, ItemEntity):
                    if entity.pickup_delay <= 0:
                        dx = self.player.x - entity.x
                        dy = self.player.y - entity.y
                        dz = self.player.z - entity.z
                        if dx*dx + dy*dy + dz*dz < 2.0:
                            while entity.count > 0:
                                if self.add_to_inventory(entity.block_id):
                                    entity.count -= 1
                                else:
                                    break
                            if entity.count == 0:
                                entity.remove()
                                
                    if not entity.removed:
                        item_entities.append(entity)
            self.entities = alive_entities
            
            # Spatial hash ile item merge - O(n) ortalama
            if item_entities:
                grid = {}
                for item in item_entities:
                    if item.removed:
                        continue
                    key = (int(item.x), int(item.y), int(item.z))
                    if key not in grid:
                        grid[key] = []
                    grid[key].append(item)
                
                for key, items in grid.items():
                    for i in range(len(items)):
                        if items[i].removed:
                            continue
                        for j in range(i + 1, len(items)):
                            if items[j].removed:
                                continue
                            dx = items[j].x - items[i].x
                            dy = items[j].y - items[i].y
                            dz = items[j].z - items[i].z
                            if dx*dx + dy*dy + dz*dz < 1.0:
                                items[i].merge_with(items[j])
                
    def _render_entities(self, camera_view, u_view_loc, u_tint_color_loc):
        partial_tick = self.tick_timer / 0.05
        
        if getattr(self, 'particle_renderer', None) is None:
            from renderer.particle_renderer import ParticleRenderer
            self.particle_renderer = ParticleRenderer()
            
        self.particle_renderer.render_particles(self.block_particles, self, partial_tick)
        
        # Render Player in Third Person
        if hasattr(self, 'camera') and self.camera.third_person_mode > 0:
            if not hasattr(self, 'player_renderer'):
                from renderer.player_renderer import PlayerRenderer
                self.player_renderer = PlayerRenderer()
            # Calculate walk_speed based on player velocity, or simplified
            speed = (abs(self.player.vx) + abs(self.player.vz)) * 0.5
            walk_pos = getattr(self, 'distance_walked', 0.0) * 1.5
            swinging = getattr(self, 'swing_time', 0.0)
            
            # Set required rotation attributes for entity renderer
            # If model looks inverted left/right, flip the yaw calculation. 
            yaw = self.camera.yaw - 90
            
            if not hasattr(self.player, 'yHeadRot'):
                self.player.yHeadRot = yaw
                self.player.yHeadRotO = yaw
                self.player.yBodyRot = yaw
                self.player.yBodyRotO = yaw
                
            self.player.yHeadRotO = self.player.yHeadRot
            self.player.yHeadRot = yaw
            
            self.player.yBodyRotO = self.player.yBodyRot
            
            moving = abs(self.player.vx) > 0.1 or abs(self.player.vz) > 0.1
            
            # Slowly turn body towards head
            diff = self.player.yHeadRot - self.player.yBodyRot
            while diff < -180.0: diff += 360.0
            while diff >= 180.0: diff -= 360.0
            
            if moving:
                # When moving, the body mostly faces where you look
                self.player.yBodyRot += diff * 0.3
            else:
                # When standing still, body only turns if head turns too far
                if diff > 45.0: self.player.yBodyRot += (diff - 45.0) * 0.2
                if diff < -45.0: self.player.yBodyRot += (diff + 45.0) * 0.2
                
            # Give player entity required attributes for renderer
            self.player.swing_progress = swinging
            
            # Invert pitch to fix looking up/down
            self.player_renderer.render(self.player, self.player.x, self.player.y, self.player.z, yaw, -self.camera.pitch, partial_tick, camera_view, u_view_loc, u_tint_color_loc)
            
            # Render held block in 3rd person
            if hasattr(self, 'hand_block_vaos') and self.selected_block_id in self.hand_block_vaos:
                if hasattr(self.player_renderer.model.arm0, 'last_final_matrix'):
                    import numpy as np
                    import math
                    from pyglet.gl import glUniformMatrix4fv, GL_FALSE, GLfloat, glBindVertexArray, glDrawArrays, GL_TRIANGLES, glBindTexture, GL_TEXTURE_2D_ARRAY
                    
                    arm_mat = self.player_renderer.model.arm0.last_final_matrix
                    scale_factor = 0.0625
                    
                    # Minecraft translates the held item slightly off-center of the hand
                    # Center of hand X=-1, Y=8, local Z forward is negative
                    t_hand = np.array([
                        [1, 0, 0, -1.0 * scale_factor],
                        [0, 1, 0, 9.0 * scale_factor],
                        [0, 0, 1, -2.0 * scale_factor],
                        [0, 0, 0, 1]
                    ], dtype=np.float32)
                    
                    # Block VAO goes from 0 to 1. Center it around (0.5, 0.5, 0.5)
                    t_center = np.array([
                        [1, 0, 0, -0.5],
                        [0, 1, 0, -0.5],
                        [0, 0, 1, -0.5],
                        [0, 0, 0, 1]
                    ], dtype=np.float32)
                    
                    # Block scale
                    # CRITICAL: EntityRenderer applies -1 scale to X and Y. We must invert them back 
                    # so the block texture is not rendered upside down/mirrored!
                    s = 0.35
                    s_mat = np.array([
                        [-s, 0, 0, 0],
                        [0, -s, 0, 0],
                        [0, 0, s, 0],
                        [0, 0, 0, 1]
                    ], dtype=np.float32)
                    
                    # Rotate block so it looks nice in hand
                    ry_rad = math.radians(45)
                    cy, sy = math.cos(ry_rad), math.sin(ry_rad)
                    ry_mat = np.array([
                        [cy, 0, sy, 0],
                        [0, 1, 0, 0],
                        [-sy, 0, cy, 0],
                        [0, 0, 0, 1]
                    ], dtype=np.float32)
                    
                    rx_rad = math.radians(10)
                    cx_val, sx_val = math.cos(rx_rad), math.sin(rx_rad)
                    rx_mat = np.array([
                        [1, 0, 0, 0],
                        [0, cx_val, -sx_val, 0],
                        [0, sx_val, cx_val, 0],
                        [0, 0, 0, 1]
                    ], dtype=np.float32)
                    
                    block_mat = arm_mat @ t_hand @ ry_mat @ rx_mat @ s_mat @ t_center
                    flat_mat = block_mat.T.flatten()
                    
                    # CRITICAL: Re-bind world texture atlas, because player texture is currently bound!
                    glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_id)
                    
                    glUniformMatrix4fv(u_view_loc, 1, GL_FALSE, (GLfloat * 16)(*flat_mat))
                    
                    vao_info = self.hand_block_vaos[self.selected_block_id]
                    glBindVertexArray(vao_info[0])
                    glDrawArrays(GL_TRIANGLES, 0, vao_info[2])
                    glBindVertexArray(0)

        for entity in self.entities:
            renderer = self.renderers.get(type(entity))
            if renderer:
                x = entity.xo + (entity.x - entity.xo) * partial_tick
                y = entity.yo + (entity.y - entity.yo) * partial_tick
                z = entity.zo + (entity.z - entity.zo) * partial_tick
                if isinstance(entity, ItemEntity):
                    renderer.render(entity, x, y, z, entity.yRot, entity.xRot, partial_tick, camera_view, u_view_loc, u_tint_color_loc, self)
                else:
                    renderer.render(entity, x, y, z, entity.yHeadRot, entity.xRot, partial_tick, camera_view, u_view_loc, u_tint_color_loc)
                
    def spawn_item_entity(self, block_id, x, y, z, xd=None, yd=None, zd=None):
        item = ItemEntity(block_id, x, y, z, xd, yd, zd)
        self.entities.append(item)

    def spawn_destruction_particles(self, x, y, z, block_id):
        if block_id == 0: return
        from core.entities.particle_entity import ParticleEntity
        SD = 4
        for xx in range(SD):
            for yy in range(SD):
                for zz in range(SD):
                    xp = x + (xx + 0.5) / SD
                    yp = y + (yy + 0.5) / SD
                    zp = z + (zz + 0.5) / SD
                    
                    xa = xp - x - 0.5
                    ya = yp - y - 0.5
                    za = zp - z - 0.5
                    
                    p = ParticleEntity(block_id, xp, yp, zp, xa, ya, za)
                    self.block_particles.append(p)
                    
        if len(self.block_particles) > 1500:
            self.block_particles = self.block_particles[-1000:]

    def spawn_crack_particles(self, x, y, z, block_id, face):
        if block_id == 0: return
        import random
        from core.entities.particle_entity import ParticleEntity
        r = 0.1
        xp = x + random.random() * (1.0 - r*2) + r
        yp = y + random.random() * (1.0 - r*2) + r
        zp = z + random.random() * (1.0 - r*2) + r
        
        if face == 0: yp = y - r         # BOTTOM
        elif face == 1: yp = y + 1.0 + r # TOP
        elif face == 2: zp = z - r       # FRONT
        elif face == 3: zp = z + 1.0 + r # BACK
        elif face == 4: xp = x - r       # LEFT
        elif face == 5: xp = x + 1.0 + r # RIGHT
        else: yp = y + 1.0 + r           # DEFAULT TO TOP
        
        p = ParticleEntity(block_id, xp, yp, zp, 0.0, 0.0, 0.0)
        p.set_power(0.2).set_scale(0.6)
        self.block_particles.append(p)

    def spawn_pig(self, x, y, z):
        self.spawn_entity(Pig, x, y, z)

    def spawn_entity(self, mob_class, x, y, z):
        mob = mob_class(x, y, z)
        mob.set_level(self) # EntityMixin acts as the level interface
        self.entities.append(mob)
        print(f"Spawned {mob_class.__name__} at {x:.2f}, {y:.2f}, {z:.2f}")
        
    def get_nearest_player(self, x, y, z, radius):
        dx = self.player.x - x
        dy = self.player.y - y
        dz = self.player.z - z
        if (dx*dx + dy*dy + dz*dz) <= radius*radius:
            return self.player
        return None
