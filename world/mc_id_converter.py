import numpy as np

# Minecraft 1.7 Block definitions: (mc_id, mc_meta, tint_type)
# tint_type: 0=none, 1=grass, 2=foliage, 3=water
MC_1_7_BLOCKS = {
    "AIR": (0, 0, 0),
    "STONE": (1, 0, 0),
    "GRASS": (2, 0, 1),
    "DIRT": (3, 0, 0),
    "COBBLESTONE": (4, 0, 0),
    "PLANKS_OAK": (5, 0, 0),
    "PLANKS_SPRUCE": (5, 1, 0),
    "PLANKS_BIRCH": (5, 2, 0),
    "PLANKS_JUNGLE": (5, 3, 0),
    "PLANKS_ACACIA": (5, 4, 0),
    "PLANKS_DARK_OAK": (5, 5, 0),
    "BEDROCK": (7, 0, 0),
    "WATER": (9, 0, 3), # 8 and 9 are water, we use 9 (stationary)
    "LAVA": (11, 0, 0),
    "SAND": (12, 0, 0),
    "RED_SAND": (12, 1, 0),
    "GRAVEL": (13, 0, 0),
    "GOLD_ORE": (14, 0, 0),
    "IRON_ORE": (15, 0, 0),
    "COAL_ORE": (16, 0, 0),
    "WOOD": (17, 0, 0), # OAK
    "SPRUCE_WOOD": (17, 1, 0),
    "BIRCH_WOOD": (17, 2, 0),
    "JUNGLE_WOOD": (17, 3, 0),
    "LEAVES": (18, 0, 2), # OAK LEAVES
    "SPRUCE_LEAVES": (18, 1, 2),
    "BIRCH_LEAVES": (18, 2, 2),
    "JUNGLE_LEAVES": (18, 3, 2),
    "SPONGE": (19, 0, 0),
    "GLASS": (20, 0, 0),
    "LAPIS_ORE": (21, 0, 0),
    "LAPIS_BLOCK": (22, 0, 0),
    "SANDSTONE": (24, 0, 0),
    "TALLGRASS": (31, 1, 1),
    "DEADBUSH": (32, 0, 0),
    "WOOL_WHITE": (35, 0, 0),
    "WOOL_ORANGE": (35, 1, 0),
    "WOOL_MAGENTA": (35, 2, 0),
    "WOOL_LIGHT_BLUE": (35, 3, 0),
    "WOOL_YELLOW": (35, 4, 0),
    "WOOL_LIME": (35, 5, 0),
    "WOOL_PINK": (35, 6, 0),
    "WOOL_GRAY": (35, 7, 0),
    "WOOL_SILVER": (35, 8, 0),
    "WOOL_CYAN": (35, 9, 0),
    "WOOL_PURPLE": (35, 10, 0),
    "WOOL_BLUE": (35, 11, 0),
    "WOOL_BROWN": (35, 12, 0),
    "WOOL_GREEN": (35, 13, 0),
    "WOOL_RED": (35, 14, 0),
    "WOOL_BLACK": (35, 15, 0),
    "DANDELION": (37, 0, 0),
    "ROSE": (38, 0, 0),
    "MUSHROOM_BROWN": (39, 0, 0),
    "MUSHROOM_RED": (40, 0, 0), 
    "GOLD_BLOCK": (41, 0, 0),
    "IRON_BLOCK": (42, 0, 0),
    "STONE_SLAB": (44, 0, 0),
    "SANDSTONE_SLAB": (44, 1, 0),
    "WOODEN_SLAB": (126, 0, 0), 
    "COBBLESTONE_SLAB": (44, 3, 0),
    "BRICK_SLAB": (44, 4, 0),
    "STONE_BRICK_SLAB": (44, 5, 0),
    "NETHER_BRICK_SLAB": (44, 6, 0),
    "QUARTZ_SLAB": (44, 7, 0),
    "BRICKS": (45, 0, 0),
    "TNT": (46, 0, 0),
    "BOOKSHELF": (47, 0, 0),
    "MOSSY_COBBLESTONE": (48, 0, 0),
    "OBSIDIAN": (49, 0, 0),
    "OAK_STAIRS": (53, 0, 0),
    "DIAMOND_ORE": (56, 0, 0),
    "DIAMOND_BLOCK": (57, 0, 0),
    "CRAFTING_TABLE": (58, 0, 0),
    "WOOD_DOOR": (64, 0, 0),
    "COBBLESTONE_STAIRS": (67, 0, 0),
    "IRON_DOOR": (71, 0, 0),
    "REDSTONE_ORE": (73, 0, 0),
    "SNOW_LAYER": (78, 0, 0),
    "ICE": (79, 0, 0),
    "SNOW": (80, 0, 0), 
    "CACTUS": (81, 0, 0),
    "CLAY_BLOCK": (82, 0, 0),
    "REEDS": (83, 0, 0),
    "JUKEBOX": (84, 0, 0),
    "NETHERRACK": (87, 0, 0),
    "SOUL_SAND": (88, 0, 0),
    "GLOWSTONE": (89, 0, 0),
    "GLASS_WHITE": (95, 0, 0),
    "GLASS_ORANGE": (95, 1, 0),
    "GLASS_MAGENTA": (95, 2, 0),
    "GLASS_LIGHT_BLUE": (95, 3, 0),
    "GLASS_YELLOW": (95, 4, 0),
    "GLASS_LIME": (95, 5, 0),
    "GLASS_PINK": (95, 6, 0),
    "GLASS_GRAY": (95, 7, 0),
    "GLASS_SILVER": (95, 8, 0),
    "GLASS_CYAN": (95, 9, 0),
    "GLASS_PURPLE": (95, 10, 0),
    "GLASS_BLUE": (95, 11, 0),
    "GLASS_BROWN": (95, 12, 0),
    "GLASS_GREEN": (95, 13, 0),
    "GLASS_RED": (95, 14, 0),
    "GLASS_BLACK": (95, 15, 0),
    "STONEBRICK": (98, 0, 0),
    "STONEBRICK_MOSSY": (98, 1, 0),
    "STONEBRICK_CRACKED": (98, 2, 0),
    "STONEBRICK_CARVED": (98, 3, 0),
    "MELON_BLOCK": (103, 0, 0),
    "VINE": (106, 0, 2),
    "BRICK_STAIRS": (108, 0, 0),
    "STONE_BRICK_STAIRS": (109, 0, 0),
    "MYCELIUM": (110, 0, 0),
    "WATERLILY": (111, 0, 2),
    "NETHER_BRICK": (112, 0, 0),
    "NETHER_BRICK_STAIRS": (114, 0, 0),
    "END_STONE": (121, 0, 0),
    "SPRUCE_SLAB": (126, 1, 0),
    "BIRCH_SLAB": (126, 2, 0),
    "JUNGLE_SLAB": (126, 3, 0),
    "ACACIA_SLAB": (126, 4, 0),
    "DARK_OAK_SLAB": (126, 5, 0),
    "SANDSTONE_STAIRS": (128, 0, 0),
    "EMERALD_ORE": (129, 0, 0),
    "EMERALD_BLOCK_SOLID": (133, 0, 0),
    "SPRUCE_STAIRS": (134, 0, 0),
    "BIRCH_STAIRS": (135, 0, 0),
    "JUNGLE_STAIRS": (136, 0, 0),
    "REDSTONE_BLOCK": (152, 0, 0),
    "QUARTZ_BLOCK": (155, 0, 0),
    "QUARTZ_CHISELED": (155, 1, 0),
    "QUARTZ_PILLAR": (155, 2, 0),
    "QUARTZ_STAIRS": (156, 0, 0),
    "STAINED_CLAY_WHITE": (159, 0, 0),
    "STAINED_CLAY_ORANGE": (159, 1, 0),
    "STAINED_CLAY_MAGENTA": (159, 2, 0),
    "STAINED_CLAY_LIGHT_BLUE": (159, 3, 0),
    "STAINED_CLAY_YELLOW": (159, 4, 0),
    "STAINED_CLAY_LIME": (159, 5, 0),
    "STAINED_CLAY_PINK": (159, 6, 0),
    "STAINED_CLAY_GRAY": (159, 7, 0),
    "STAINED_CLAY_SILVER": (159, 8, 0),
    "STAINED_CLAY_CYAN": (159, 9, 0),
    "STAINED_CLAY_PURPLE": (159, 10, 0),
    "STAINED_CLAY_BLUE": (159, 11, 0),
    "STAINED_CLAY_BROWN": (159, 12, 0),
    "STAINED_CLAY_GREEN": (159, 13, 0),
    "STAINED_CLAY_RED": (159, 14, 0),
    "STAINED_CLAY_BLACK": (159, 15, 0),
    "HARDENED_CLAY": (172, 0, 0), 
    "ACACIA_LEAVES": (161, 0, 2),
    "DARK_OAK_LEAVES": (161, 1, 2),
    "ACACIA_WOOD": (162, 0, 0),
    "DARK_OAK_WOOD": (162, 1, 0),
    "ACACIA_STAIRS": (163, 0, 0),
    "DARK_OAK_STAIRS": (164, 0, 0),
    "SLIME_BLOCK": (165, 0, 0),
    "PRISMARINE": (168, 0, 0),
    "PRISMARINE_BRICKS": (168, 1, 0),
    "PRISMARINE_DARK": (168, 2, 0),
    "SEA_LANTERN": (169, 0, 0),
    "HAY_BLOCK": (170, 0, 0),
    "COAL_BLOCK_SOLID": (173, 0, 0),
    "PACKED_ICE": (174, 0, 0),
    "DOUBLE_GRASS_BTM": (175, 2, 1), 
    "DOUBLE_GRASS_TOP": (175, 10, 1), 
    "DOUBLE_ROSE_BTM": (175, 4, 0), 
    "DOUBLE_ROSE_TOP": (175, 14, 0), 
    
    # Missing/misc block properties added directly to map logic later
    "PODZOL": (3, 2, 0),
    "COARSE_DIRT": (3, 1, 0),
    "ANDESITE": (1, 5, 0),
    "ANDESITE_POLISHED": (1, 6, 0),
    "DIORITE": (1, 3, 0),
    "DIORITE_POLISHED": (1, 4, 0),
    "GRANITE": (1, 1, 0),
    "GRANITE_POLISHED": (1, 2, 0),
    "PUMPKIN": (86, 0, 0), 
}

MC_1_7_ITEMS = {
    "STICK": 280,
    "DIAMOND": 264,
    "IRON_INGOT": 265,
    "GOLD_INGOT": 266,
    "COAL": 263,
    "CHARCOAL": 263, 
    "BOWL": 281,
    "MUSHROOM_STEW": 282,
    
    "WOODEN_SWORD": 268,
    "WOODEN_SHOVEL": 269,
    "WOODEN_PICKAXE": 270,
    "WOODEN_AXE": 271,
    "WOODEN_HOE": 290,

    "STONE_SWORD": 272,
    "STONE_SHOVEL": 273,
    "STONE_PICKAXE": 274,
    "STONE_AXE": 275,
    "STONE_HOE": 291,

    "IRON_SWORD": 267,
    "IRON_SHOVEL": 256,
    "IRON_PICKAXE": 257,
    "IRON_AXE": 258,
    "IRON_HOE": 292,

    "DIAMOND_SWORD": 276,
    "DIAMOND_SHOVEL": 277,
    "DIAMOND_PICKAXE": 278,
    "DIAMOND_AXE": 279,
    "DIAMOND_HOE": 293,

    "GOLD_SWORD": 283,
    "GOLD_SHOVEL": 284,
    "GOLD_PICKAXE": 285,
    "GOLD_AXE": 286,
    "GOLD_HOE": 294,
    "PORKCHOP_RAW": 319,
}

INTERNAL_TO_MC_ID = np.zeros(256, dtype=np.uint8)
INTERNAL_TO_MC_META = np.zeros(256, dtype=np.uint8)
MC_TO_INTERNAL = np.zeros((256, 16), dtype=np.uint8)

def get_block_registry():
    registry = {}
    
    # Assign base blocks (meta == 0) to their exact MC ID
    for name, (mc_id, mc_meta, tint_type) in MC_1_7_BLOCKS.items():
        if mc_meta == 0:
            internal_id = mc_id
            registry[name] = internal_id
            INTERNAL_TO_MC_ID[internal_id] = mc_id
            INTERNAL_TO_MC_META[internal_id] = 0
            MC_TO_INTERNAL[mc_id, 0] = internal_id
            
    # Assign meta variants to free IDs starting from 176
    
    # Find next free ID by checking what has been used
    used_ids = set()
    for name, (mc_id, mc_meta, tint_type) in MC_1_7_BLOCKS.items():
        if mc_meta == 0:
            used_ids.add(mc_id)
            
    def get_free_id(used_set):
        for i in range(1, 256):
            if i not in used_set:
                used_set.add(i)
                return i
        raise Exception('Out of block IDs!')

    for name, (mc_id, mc_meta, tint_type) in MC_1_7_BLOCKS.items():
        if mc_meta > 0:
            internal_id = get_free_id(used_ids)
            registry[name] = internal_id
            INTERNAL_TO_MC_ID[internal_id] = mc_id
            INTERNAL_TO_MC_META[internal_id] = mc_meta
            MC_TO_INTERNAL[mc_id, mc_meta] = internal_id
            
    # Fallback missing metas to meta 0
    for i in range(256):
        if MC_TO_INTERNAL[i, 0] != 0:
            for m in range(1, 16):
                if MC_TO_INTERNAL[i, m] == 0:
                    MC_TO_INTERNAL[i, m] = MC_TO_INTERNAL[i, 0]
                    
    # Register items
    for name, mc_id in MC_1_7_ITEMS.items():
        registry[name] = mc_id
        
    return registry

def get_tint_array(registry):
    arr = np.zeros(256, dtype=np.uint8)
    for name, internal_id in registry.items():
        if name in MC_1_7_BLOCKS:
            _, _, tint_type = MC_1_7_BLOCKS[name]
            if internal_id < 256:
                arr[internal_id] = tint_type
    return arr

INTERNAL_NAMES_MAP = get_block_registry()
BLOCK_TINT_ARRAY = get_tint_array(INTERNAL_NAMES_MAP)
