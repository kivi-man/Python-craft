import os
import json
import numpy as np
from nbt import nbt, region
import zlib
import threading

DB_LOCK = threading.Lock()
CHUNK_SIZE = 16
CHUNK_HEIGHT = 256

WORLD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "world_data")
REGION_DIR = os.path.join(WORLD_DIR, "region")
PLAYER_DIR = os.path.join(WORLD_DIR, "playerdata")
ENTITY_DIR = os.path.join(WORLD_DIR, "entities")

import time

def ensure_level_dat():
    level_dat_path = os.path.join(WORLD_DIR, "level.dat")
    if not os.path.exists(level_dat_path):
        root = nbt.NBTFile()
        data = nbt.TAG_Compound(name="Data")
        root.tags.append(data)
        
        data.tags.append(nbt.TAG_Int(name="version", value=19133))
        data.tags.append(nbt.TAG_String(name="LevelName", value="PythonCraft World"))
        data.tags.append(nbt.TAG_String(name="generatorName", value="default"))
        data.tags.append(nbt.TAG_Int(name="GameType", value=0))
        data.tags.append(nbt.TAG_Int(name="SpawnX", value=0))
        data.tags.append(nbt.TAG_Int(name="SpawnY", value=64))
        data.tags.append(nbt.TAG_Int(name="SpawnZ", value=0))
        data.tags.append(nbt.TAG_Long(name="RandomSeed", value=123456789))
        data.tags.append(nbt.TAG_Long(name="Time", value=0))
        data.tags.append(nbt.TAG_Long(name="DayTime", value=0))
        data.tags.append(nbt.TAG_Long(name="lastPlayed", value=int(time.time() * 1000)))
        
        root.write_file(level_dat_path)

INTERNAL_TO_MC_ID = np.zeros(256, dtype=np.uint8)
INTERNAL_TO_MC_META = np.zeros(256, dtype=np.uint8)

_mapping = {
    1: (1, 0),    # STONE
    2: (3, 0),    # DIRT
    3: (2, 0),    # GRASS
    4: (9, 0),    # WATER
    5: (12, 0),   # SAND
    6: (80, 0),   # SNOW
    7: (7, 0),    # BEDROCK
    8: (13, 0),   # GRAVEL
    9: (24, 0),   # SANDSTONE
    10: (110, 0), # MYCELIUM
    11: (17, 0),  # WOOD (Oak)
    12: (18, 0),  # LEAVES (Oak)
    13: (81, 0),  # CACTUS
    14: (17, 2),  # BIRCH_WOOD
    15: (17, 1),  # SPRUCE_WOOD
    16: (18, 2),  # BIRCH_LEAVES
    17: (18, 1),  # SPRUCE_LEAVES
    20: (20, 0),  # GLASS
    21: (21, 0),  # LAPIS_ORE
    22: (11, 0),  # LAVA
    31: (31, 1),  # TALLGRASS
    37: (37, 0),  # DANDELION
    38: (38, 0),  # ROSE
    40: (14, 0),  # GOLD_ORE
    41: (15, 0),  # IRON_ORE
    42: (16, 0),  # COAL_ORE
    56: (56, 0),  # DIAMOND_ORE
    73: (73, 0),  # REDSTONE_ORE
    79: (79, 0),  # ICE
    129: (129, 0),# EMERALD_ORE
    175: (175, 2),# DOUBLE_GRASS_BTM
    176: (175, 10),# DOUBLE_GRASS_TOP
    177: (175, 4),# DOUBLE_ROSE_BTM
    178: (175, 10),# DOUBLE_ROSE_TOP
    43: (4, 0), # COBBLESTONE
    44: (5, 0), # PLANKS_OAK
    45: (5, 1), # PLANKS_SPRUCE
    46: (5, 2), # PLANKS_BIRCH
    47: (5, 3), # PLANKS_JUNGLE
    48: (5, 4), # PLANKS_ACACIA
    49: (5, 5), # PLANKS_DARK_OAK
    50: (19, 0), # SPONGE
    51: (22, 0), # LAPIS_BLOCK
    52: (41, 0), # GOLD_BLOCK
    53: (42, 0), # IRON_BLOCK
    54: (45, 0), # BRICKS
    55: (46, 0), # TNT
    57: (47, 0), # BOOKSHELF
    58: (48, 0), # MOSSY_COBBLESTONE
    59: (49, 0), # OBSIDIAN
    60: (57, 0), # DIAMOND_BLOCK
    61: (82, 0), # CLAY_BLOCK
    62: (84, 0), # JUKEBOX
    63: (87, 0), # NETHERRACK
    64: (88, 0), # SOUL_SAND
    65: (89, 0), # GLOWSTONE
    66: (98, 0), # STONEBRICK
    67: (98, 1), # STONEBRICK_MOSSY
    68: (98, 2), # STONEBRICK_CRACKED
    69: (98, 3), # STONEBRICK_CARVED
    70: (103, 0), # MELON_BLOCK
    71: (112, 0), # NETHER_BRICK
    72: (121, 0), # END_STONE
    74: (133, 0), # EMERALD_BLOCK_SOLID
    75: (152, 0), # REDSTONE_BLOCK
    76: (155, 0), # QUARTZ_BLOCK
    77: (155, 1), # QUARTZ_CHISELED
    78: (155, 2), # QUARTZ_PILLAR
    80: (173, 0), # COAL_BLOCK_SOLID
    81: (174, 0), # PACKED_ICE
    82: (12, 1), # RED_SAND
    83: (3, 2), # PODZOL
    84: (3, 1), # COARSE_DIRT
    85: (1, 5), # ANDESITE
    86: (1, 6), # ANDESITE_POLISHED
    87: (1, 3), # DIORITE
    88: (1, 4), # DIORITE_POLISHED
    89: (1, 1), # GRANITE
    90: (1, 2), # GRANITE_POLISHED
    91: (168, 0), # PRISMARINE
    92: (168, 1), # PRISMARINE_BRICKS
    93: (168, 2), # PRISMARINE_DARK
    94: (169, 0), # SEA_LANTERN
    95: (165, 0), # SLIME_BLOCK
    96: (170, 0), # HAY_BLOCK
    116: (58, 0), # CRAFTING_TABLE
    100: (35, 0), # WOOL_WHITE
    120: (95, 0), # GLASS_WHITE
    140: (159, 0), # STAINED_CLAY_WHITE
    101: (35, 1), # WOOL_ORANGE
    121: (95, 1), # GLASS_ORANGE
    141: (159, 1), # STAINED_CLAY_ORANGE
    102: (35, 2), # WOOL_MAGENTA
    122: (95, 2), # GLASS_MAGENTA
    142: (159, 2), # STAINED_CLAY_MAGENTA
    103: (35, 3), # WOOL_LIGHT_BLUE
    123: (95, 3), # GLASS_LIGHT_BLUE
    143: (159, 3), # STAINED_CLAY_LIGHT_BLUE
    104: (35, 4), # WOOL_YELLOW
    124: (95, 4), # GLASS_YELLOW
    144: (159, 4), # STAINED_CLAY_YELLOW
    105: (35, 5), # WOOL_LIME
    125: (95, 5), # GLASS_LIME
    145: (159, 5), # STAINED_CLAY_LIME
    106: (35, 6), # WOOL_PINK
    126: (95, 6), # GLASS_PINK
    146: (159, 6), # STAINED_CLAY_PINK
    107: (35, 7), # WOOL_GRAY
    127: (95, 7), # GLASS_GRAY
    147: (159, 7), # STAINED_CLAY_GRAY
    108: (35, 8), # WOOL_SILVER
    128: (95, 8), # GLASS_SILVER
    148: (159, 8), # STAINED_CLAY_SILVER
    109: (35, 9), # WOOL_CYAN
    129: (95, 9), # GLASS_CYAN
    149: (159, 9), # STAINED_CLAY_CYAN
    110: (35, 10), # WOOL_PURPLE
    130: (95, 10), # GLASS_PURPLE
    150: (159, 10), # STAINED_CLAY_PURPLE
    111: (35, 11), # WOOL_BLUE
    131: (95, 11), # GLASS_BLUE
    151: (159, 11), # STAINED_CLAY_BLUE
    112: (35, 12), # WOOL_BROWN
    132: (95, 12), # GLASS_BROWN
    152: (159, 12), # STAINED_CLAY_BROWN
    113: (35, 13), # WOOL_GREEN
    133: (95, 13), # GLASS_GREEN
    153: (159, 13), # STAINED_CLAY_GREEN
    114: (35, 14), # WOOL_RED
    134: (95, 14), # GLASS_RED
    154: (159, 14), # STAINED_CLAY_RED
    115: (35, 15), # WOOL_BLACK
    135: (95, 15), # GLASS_BLACK
    155: (159, 15), # STAINED_CLAY_BLACK
    160: (172, 0), # HARDENED_CLAY
    200: (53, 0), # OAK_STAIRS
    201: (67, 0), # COBBLESTONE_STAIRS
    202: (126, 0), # WOODEN_SLAB (OAK)
    203: (44, 0), # STONE_SLAB
    204: (134, 0), # SPRUCE_STAIRS
    205: (135, 0), # BIRCH_STAIRS
    206: (136, 0), # JUNGLE_STAIRS
    207: (163, 0), # ACACIA_STAIRS
    208: (164, 0), # DARK_OAK_STAIRS
    209: (108, 0), # BRICK_STAIRS
    210: (109, 0), # STONE_BRICK_STAIRS
    211: (114, 0), # NETHER_BRICK_STAIRS
    212: (128, 0), # SANDSTONE_STAIRS
    213: (156, 0), # QUARTZ_STAIRS
    214: (126, 1), # SPRUCE_SLAB
    215: (126, 2), # BIRCH_SLAB
    216: (126, 3), # JUNGLE_SLAB
    217: (126, 4), # ACACIA_SLAB
    218: (126, 5), # DARK_OAK_SLAB
    219: (44, 4), # BRICK_SLAB
    220: (44, 5), # STONE_BRICK_SLAB
    221: (44, 6), # NETHER_BRICK_SLAB
    222: (44, 1), # SANDSTONE_SLAB
    223: (44, 7), # QUARTZ_SLAB
    224: (44, 3), # COBBLESTONE_SLAB
    162: (64, 0), # WOOD_DOOR
    163: (71, 0), # IRON_DOOR
}

for internal_id, (mc_id, mc_meta) in _mapping.items():
    INTERNAL_TO_MC_ID[internal_id] = mc_id
    INTERNAL_TO_MC_META[internal_id] = mc_meta

MC_TO_INTERNAL = np.zeros((256, 16), dtype=np.uint8)
for internal_id, (mc_id, mc_meta) in _mapping.items():
    MC_TO_INTERNAL[mc_id, mc_meta] = internal_id

# Fallback metadata to primary block type if exact meta not found
for internal_id, (mc_id, mc_meta) in _mapping.items():
    for m in range(16):
        if MC_TO_INTERNAL[mc_id, m] == 0:
            MC_TO_INTERNAL[mc_id, m] = internal_id

def init_db():
    os.makedirs(REGION_DIR, exist_ok=True)
    os.makedirs(PLAYER_DIR, exist_ok=True)
    os.makedirs(ENTITY_DIR, exist_ok=True)
    try:
        ensure_level_dat()
    except Exception as e:
        print(f"Error creating level.dat: {e}")

init_db()

def _get_region_file(cx, cz):
    rx = cx >> 5
    rz = cz >> 5
    filename = os.path.join(REGION_DIR, f"r.{rx}.{rz}.mca")
    if not os.path.exists(filename):
        open(filename, 'wb').close()
    return region.RegionFile(filename)

def pack_nibble_array(arr):
    # arr is a 1D numpy array of shape (N,) containing values 0-15
    # Pack 2 values into 1 byte: lower nibble is even index, upper nibble is odd index
    even = arr[0::2]
    odd = arr[1::2]
    return (even | (odd << 4)).astype(np.uint8).tobytes()

def unpack_nibble_array(byte_data, size):
    arr = np.frombuffer(byte_data, dtype=np.uint8)
    even = arr & 0x0F
    odd = (arr >> 4) & 0x0F
    res = np.empty(size, dtype=np.uint8)
    res[0::2] = even
    res[1::2] = odd
    return res

def save_chunk(cx, cz, blocks, data, lights):
    """Saves a chunk to Anvil MCA file format.
    blocks: numpy array (16, 256, 16) uint8
    data: numpy array (16, 256, 16) uint8
    lights: numpy array (16, 256, 16) uint8
    """
    root = nbt.NBTFile()
    level = nbt.TAG_Compound(name="Level")
    root.tags.append(level)
    
    level.tags.append(nbt.TAG_Int(name="xPos", value=cx))
    level.tags.append(nbt.TAG_Int(name="zPos", value=cz))
    level.tags.append(nbt.TAG_Long(name="LastUpdate", value=0))
    level.tags.append(nbt.TAG_Byte(name="LightPopulated", value=1))
    level.tags.append(nbt.TAG_Byte(name="TerrainPopulated", value=1))
    level.tags.append(nbt.TAG_Byte(name="V", value=1))
    
    biomes_tag = nbt.TAG_Byte_Array(name="Biomes")
    biomes_tag.value = bytearray(256)
    level.tags.append(biomes_tag)
    
    height_tag = nbt.TAG_Int_Array(name="HeightMap")
    height_tag.value = np.zeros(256, dtype=np.int32).tolist() # nbt library expects list of ints or int array
    level.tags.append(height_tag)
    
    sections = nbt.TAG_List(name="Sections", type=nbt.TAG_Compound)
    level.tags.append(sections)
    
    for y_sec in range(16):
        y_start = y_sec * 16
        y_end = y_start + 16
        
        # Minecraft uses [y, z, x]
        sec_blocks_internal = blocks[:, y_start:y_end, :].transpose((1, 2, 0)).flatten()
        sec_data_internal = data[:, y_start:y_end, :].transpose((1, 2, 0)).flatten()
        sec_lights = lights[:, y_start:y_end, :].transpose((1, 2, 0)).flatten()
        
        if not np.any(sec_blocks_internal):
            continue
            
        sec_blocks_mc = INTERNAL_TO_MC_ID[sec_blocks_internal]
        # Use our passed 'data' array instead of INTERNAL_TO_MC_META mapping
        # but fallback to INTERNAL_TO_MC_META if data is 0 for backwards compatibility with blocks that need specific meta
        fallback_meta = INTERNAL_TO_MC_META[sec_blocks_internal]
        sec_meta_mc = np.where(sec_data_internal > 0, sec_data_internal, fallback_meta)
            
        sec_tag = nbt.TAG_Compound()
        sec_tag.tags.append(nbt.TAG_Byte(name="Y", value=y_sec))
        
        blocks_tag = nbt.TAG_Byte_Array(name="Blocks")
        blocks_tag.value = bytearray(sec_blocks_mc.tobytes())
        sec_tag.tags.append(blocks_tag)
        
        data_tag = nbt.TAG_Byte_Array(name="Data")
        data_tag.value = bytearray(pack_nibble_array(sec_meta_mc))
        sec_tag.tags.append(data_tag)
        
        block_light_bytes = pack_nibble_array(sec_lights)
        
        sec_tag.tags.append(nbt.TAG_Byte_Array(name="BlockLight"))
        sec_tag["BlockLight"].value = bytearray(block_light_bytes)
        
        sec_tag.tags.append(nbt.TAG_Byte_Array(name="SkyLight"))
        sec_tag["SkyLight"].value = bytearray(block_light_bytes)
        
        sections.tags.append(sec_tag)
        
    level.tags.append(nbt.TAG_List(name="Entities", type=nbt.TAG_Compound))
    level.tags.append(nbt.TAG_List(name="TileEntities", type=nbt.TAG_Compound))
    
    try:
        with DB_LOCK:
            rfile = _get_region_file(cx, cz)
            rx = cx % 32
            rz = cz % 32
            rfile.write_chunk(rx, rz, root)
            rfile.close()
    except Exception as e:
        print(f"Error saving chunk {cx}, {cz}: {e}")

def load_chunk(cx, cz):
    """Loads a chunk from Anvil MCA file format. Returns (blocks, data, lights) or None."""
    try:
        rx = cx >> 5
        rz = cz >> 5
        filename = os.path.join(REGION_DIR, f"r.{rx}.{rz}.mca")
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            return None
            
        with DB_LOCK:
            rfile = region.RegionFile(filename)
            rx_chunk = cx % 32
            rz_chunk = cz % 32
            
            try:
                chunk_nbt = rfile.get_chunk(rx_chunk, rz_chunk)
            except region.InconceivedChunk:
                rfile.close()
                return None
            except Exception:
                rfile.close()
                return None
                
            rfile.close()
        
        if not chunk_nbt or "Level" not in chunk_nbt:
            return None
            
        level = chunk_nbt["Level"]
        
        blocks = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
        data = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
        lights = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
        
        if "Sections" in level:
            for sec_tag in level["Sections"]:
                y_sec = sec_tag["Y"].value
                if y_sec < 0 or y_sec >= 16:
                    continue
                    
                y_start = y_sec * 16
                y_end = y_start + 16
                
                sec_blocks_flat = np.frombuffer(sec_tag["Blocks"].value, dtype=np.uint8)
                if "Data" in sec_tag:
                    sec_meta_flat = unpack_nibble_array(sec_tag["Data"].value, 4096)
                else:
                    sec_meta_flat = np.zeros(4096, dtype=np.uint8)
                    
                sec_blocks_internal = MC_TO_INTERNAL[sec_blocks_flat, sec_meta_flat]
                
                sec_blocks = sec_blocks_internal.reshape((16, 16, 16)).transpose((2, 0, 1))
                blocks[:, y_start:y_end, :] = sec_blocks
                
                # We store the raw meta flat into our data array so custom special blocks get their meta back
                sec_data = sec_meta_flat.reshape((16, 16, 16)).transpose((2, 0, 1))
                data[:, y_start:y_end, :] = sec_data
                
                sec_lights = np.zeros((16, 16, 16), dtype=np.uint8)
                if "BlockLight" in sec_tag:
                    bl_flat = unpack_nibble_array(sec_tag["BlockLight"].value, 4096)
                    sec_lights = np.maximum(sec_lights, bl_flat.reshape((16, 16, 16)).transpose((2, 0, 1)))
                if "SkyLight" in sec_tag:
                    sl_flat = unpack_nibble_array(sec_tag["SkyLight"].value, 4096)
                    sec_lights = np.maximum(sec_lights, sl_flat.reshape((16, 16, 16)).transpose((2, 0, 1)))
                
                lights[:, y_start:y_end, :] = sec_lights
                    
        return blocks, data, lights
    except Exception as e:
        print(f"Error loading chunk {cx}, {cz}: {e}")
        return None

def save_chunk_entities(cx, cz, entities_json):
    filename = os.path.join(ENTITY_DIR, f"{cx}_{cz}.json")
    with open(filename, 'w') as f:
        f.write(entities_json)

def load_chunk_entities(cx, cz):
    filename = os.path.join(ENTITY_DIR, f"{cx}_{cz}.json")
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return f.read()
    return None

def save_player_data(player_id, json_data):
    filename = os.path.join(PLAYER_DIR, f"{player_id}.json")
    with open(filename, 'w') as f:
        f.write(json_data)

def load_player_data(player_id):
    filename = os.path.join(PLAYER_DIR, f"{player_id}.json")
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return f.read()
    return None
