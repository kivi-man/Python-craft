import random
import math
from world.mc_biomes import BIOME_SPAWN_LISTS, get_biome, PLAINS
from world.mc_terrain import CHUNK_SIZE, CHUNK_HEIGHT
from world.terrain import GRASS, DIRT, STONE, SAND, SNOW, MYCELIUM
from core.entities.pig import Pig

# Map string mob types to their python classes.
# Missing ones are mapped to None and will be skipped.
MOB_CLASSES = {
    "Pig": Pig,
    "Sheep": Pig, # Temporary mapping for testing
    "Cow": Pig, # Temporary
    "Chicken": Pig, # Temporary
    "Spider": None,
    "Zombie": None,
    "Skeleton": None,
    "Creeper": None,
    "Slime": None,
    "Enderman": None,
    "Squid": None,
    "Wolf": None,
    "MushroomCow": Pig,
    "Ghast": None,
    "PigZombie": None,
    "LavaSlime": None,
}

class NaturalSpawner:
    def __init__(self, level):
        self.level = level
        self.spawn_ticks = 0
        self.mob_cap = {
            "friendlies": 15,
            "enemies": 70,
            "water_friendlies": 5
        }

    def count_mobs(self):
        # In a real impl, we'd classify them. Since we only have Pigs, they are friendlies.
        counts = {"friendlies": 0, "enemies": 0, "water_friendlies": 0}
        for e in self.level.entities:
            # We assume all current entities are friendlies. 
            # When enemies are added, check type(e) to categorize.
            counts["friendlies"] += 1
        return counts

    def get_top_solid_block(self, cx, cz, x_mod, z_mod):
        chunk = self.level.world_chunks.get((cx, cz))
        if chunk is None:
            return -1, 0
            
        # Iterate downwards to find the highest non-air block
        for y in range(CHUNK_HEIGHT - 1, -1, -1):
            block_id = chunk[x_mod, y, z_mod]
            if block_id != 0: # 0 is AIR
                return y, block_id
        return -1, 0

    def get_biome_at(self, wx, wz):
        # Simplistic biome retrieval based on how mc_terrain handles it.
        # Pythoncraft uses get_biome(temperature, downfall). 
        # For our spawner, if we can't easily query noise, we assume PLAINS as fallback.
        # But we can try to query the chunk data if we stored biome maps.
        # Since Pythoncraft's Biome Map isn't exposed easily per block, we default to PLAINS.
        # (A full implementation would cache Biome ID per X/Z column).
        return PLAINS

    def tick(self):
        self.spawn_ticks += 1
        if self.spawn_ticks % 20 != 0: # Run once per second (20 ticks)
            return

        # Check mob caps
        counts = self.count_mobs()
        
        # Only process loaded chunks
        loaded_chunks = list(self.level.world_chunks.keys())
        if not loaded_chunks:
            return
            
        # Try a few random spawn attempts
        for _ in range(5):
            if counts["friendlies"] >= self.mob_cap["friendlies"]:
                break
                
            # Pick a random chunk
            cx, cz = random.choice(loaded_chunks)
            
            # Pick a random block within the chunk
            x_mod = random.randint(0, CHUNK_SIZE - 1)
            z_mod = random.randint(0, CHUNK_SIZE - 1)
            
            # Find the surface
            y, block_id = self.get_top_solid_block(cx, cz, x_mod, z_mod)
            if y < 0:
                continue
                
            wx = cx * CHUNK_SIZE + x_mod
            wz = cz * CHUNK_SIZE + z_mod
            
            # Check if valid spawn surface for friendlies
            # C++ Biome.cpp natural spawning requires grass for most friendlies
            if block_id not in (GRASS, MYCELIUM):
                continue
                
            # Get biome and its spawn list
            b_id = self.get_biome_at(wx, wz)
            spawn_lists = BIOME_SPAWN_LISTS.get(b_id)
            if not spawn_lists:
                continue
                
            friendlies_list = spawn_lists["friendlies"]
            if not friendlies_list:
                continue
                
            # Weighted random selection
            total_weight = sum(item.weight for item in friendlies_list)
            if total_weight <= 0:
                continue
                
            rand_val = random.randint(0, total_weight - 1)
            selected_data = None
            current_weight = 0
            for item in friendlies_list:
                current_weight += item.weight
                if rand_val < current_weight:
                    selected_data = item
                    break
                    
            if not selected_data:
                continue
                
            mob_class = MOB_CLASSES.get(selected_data.mob_type)
            if not mob_class:
                continue # Skip if not implemented yet
                
            # Spawn a pack
            pack_size = random.randint(selected_data.min_count, selected_data.max_count)
            for _ in range(pack_size):
                if counts["friendlies"] >= self.mob_cap["friendlies"]:
                    break
                    
                # Add some spread to the pack spawning
                spread_x = wx + random.uniform(-4, 4)
                spread_z = wz + random.uniform(-4, 4)
                
                # Check height at spread location
                spread_cx = int(math.floor(spread_x / CHUNK_SIZE))
                spread_cz = int(math.floor(spread_z / CHUNK_SIZE))
                spread_x_mod = int(math.floor(spread_x)) % CHUNK_SIZE
                spread_z_mod = int(math.floor(spread_z)) % CHUNK_SIZE
                
                spread_y, spread_block = self.get_top_solid_block(spread_cx, spread_cz, spread_x_mod, spread_z_mod)
                if spread_y > 0 and spread_block in (GRASS, MYCELIUM):
                    # Spawn the mob (y+1 because it spawns ON the block)
                    self.level.spawn_entity(mob_class, spread_x, spread_y + 1, spread_z)
                    counts["friendlies"] += 1
