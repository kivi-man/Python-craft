import time
from core.entities.pig import Pig
from renderer.pig_renderer import PigRenderer
from world.spawner import NaturalSpawner

class EntityMixin:
    def _init_entities(self):
        self.entities = []
        self.renderers = {
            Pig: PigRenderer()
        }
        self.tick_timer = 0.0
        self.spawner = NaturalSpawner(self)

    def _update_entities(self, dt):
        self.tick_timer += dt
        while self.tick_timer >= 0.05: # 20 TPS
            self.tick_timer -= 0.05
            self.spawner.tick()
            for entity in self.entities[:]:
                if entity.removed:
                    self.entities.remove(entity)
                    continue
                entity.tick(self.get_block)
                
    def _render_entities(self, camera_view, u_view_loc, u_tint_color_loc):
        partial_tick = self.tick_timer / 0.05
        for entity in self.entities:
            renderer = self.renderers.get(type(entity))
            if renderer:
                x = entity.xo + (entity.x - entity.xo) * partial_tick
                y = entity.yo + (entity.y - entity.yo) * partial_tick
                z = entity.zo + (entity.z - entity.zo) * partial_tick
                renderer.render(entity, x, y, z, entity.yHeadRot, entity.xRot, partial_tick, camera_view, u_view_loc, u_tint_color_loc)
                
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
