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
            
            for p in self.block_particles[:]:
                if p.removed:
                    self.block_particles.remove(p)
                    continue
                p.tick(self.get_block)
                
            for entity in self.entities[:]:
                if entity.removed:
                    self.entities.remove(entity)
                    continue
                entity.tick(self.get_block)
                
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
                        for other in self.entities:
                            if other != entity and isinstance(other, ItemEntity):
                                dx = other.x - entity.x
                                dy = other.y - entity.y
                                dz = other.z - entity.z
                                if dx*dx + dy*dy + dz*dz < 1.0:
                                    entity.merge_with(other)
                
    def _render_entities(self, camera_view, u_view_loc, u_tint_color_loc):
        partial_tick = self.tick_timer / 0.05
        
        if getattr(self, 'particle_renderer', None) is None:
            from renderer.particle_renderer import ParticleRenderer
            self.particle_renderer = ParticleRenderer()
            
        self.particle_renderer.render_particles(self.block_particles, self, partial_tick)
        
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
