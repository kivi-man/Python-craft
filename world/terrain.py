"""
PythonCraft - World Terrain Generator (Numba JIT Optimized)
Tüm noise hesaplamaları ve döngüler Numba ile makine koduna derlenir.
İlk çalıştırmada ~2sn derleme süresi olur, sonraki çağrılar C hızında çalışır.
"""
import numpy as np
from numba import njit, prange
from core.world_db import load_chunk

# Blok Kayıt Sistemi (Block Registry)
BLOCK_REGISTRY = {
    "AIR": {"id": 0, "texture": None, "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "STONE": {"id": 1, "texture": "stone.png", "color": (0.45, 0.45, 0.45), "transparent": False, "light": 0, "hardness": 1.5, "tool_type": "PICKAXE"},
    "DIRT": {"id": 2, "texture": "dirt.png", "color": (0.40, 0.26, 0.13), "transparent": False, "light": 0, "hardness": 0.5, "tool_type": "SHOVEL"},
    "GRASS": {"id": 2, "texture": {"top": "grass_top.png", "bottom": "dirt.png", "side": "grass_side.png"}, "color": (0.6, 0.8, 0.4), "transparent": False, "light": 0, "hardness": 0.6, "tool_type": "SHOVEL"},
    "WATER": {"id": 4, "texture": "water.png", "color": (0.20, 0.40, 0.75), "transparent": True, "light": 0, "hardness": -1.0, "tool_type": "NONE"},
    "SAND": {"id": 5, "texture": "sand.png", "color": (0.85, 0.80, 0.55), "transparent": False, "light": 0, "hardness": 0.5, "tool_type": "SHOVEL"},
    "SNOW": {"id": 6, "texture": "snow.png", "color": (0.95, 0.95, 0.95), "transparent": False, "light": 0, "hardness": 0.2, "tool_type": "SHOVEL"},
    "BEDROCK": {"id": 7, "texture": "bedrock.png", "color": (0.20, 0.20, 0.20), "transparent": False, "light": 0, "hardness": -1.0, "tool_type": "NONE"},
    "ICE": {"id": 79, "texture": "ice.png", "color": (0.60, 0.80, 1.00), "transparent": True, "light": 0, "hardness": 0.5, "tool_type": "PICKAXE"},
    "GRAVEL": {"id": 8, "texture": "gravel.png", "color": (0.50, 0.50, 0.50), "transparent": False, "light": 0, "hardness": 0.6, "tool_type": "SHOVEL"},
    "SANDSTONE": {"id": 9, "texture": "sandstone.png", "color": (0.80, 0.75, 0.50), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "PICKAXE"},
    "MYCELIUM": {"id": 10, "texture": {"top": "mycelium_top.png", "bottom": "dirt.png", "side": "mycelium_side.png"}, "color": (0.45, 0.35, 0.40), "transparent": False, "light": 0, "hardness": 0.6, "tool_type": "SHOVEL"},
    "WOOD": {"id": 11, "texture": {"top": "log_oak_top.png", "bottom": "log_oak_top.png", "side": "log_oak.png"}, "color": (0.40, 0.30, 0.15), "transparent": False, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "LEAVES": {"id": 12, "texture": "leaves_oak.png", "color": (0.15, 0.50, 0.15), "transparent": True, "light": 0, "hardness": 0.2, "tool_type": "NONE"},
    "CACTUS": {"id": 13, "texture": {"top": "cactus_top.png", "bottom": "cactus_bottom.png", "side": "cactus_side.png"}, "color": (0.10, 0.60, 0.20), "transparent": True, "light": 0, "hardness": 0.4, "tool_type": "NONE"},
    "BIRCH_WOOD": {"id": 14, "texture": {"top": "log_birch_top.png", "bottom": "log_birch_top.png", "side": "log_birch.png"}, "color": (0.90, 0.90, 0.85), "transparent": False, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "SPRUCE_WOOD": {"id": 15, "texture": {"top": "log_spruce_top.png", "bottom": "log_spruce_top.png", "side": "log_spruce.png"}, "color": (0.30, 0.20, 0.10), "transparent": False, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "BIRCH_LEAVES": {"id": 16, "texture": "leaves_birch.png", "color": (0.25, 0.55, 0.25), "transparent": True, "light": 0, "hardness": 0.2, "tool_type": "NONE"},
    "SPRUCE_LEAVES": {"id": 17, "texture": "leaves_spruce.png", "color": (0.10, 0.35, 0.15), "transparent": True, "light": 0, "hardness": 0.2, "tool_type": "NONE"},
    "GLASS": {"id": 20, "texture": "glass.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.3, "tool_type": "NONE"},
    "LAVA": {"id": 22, "texture": "lava_still.png", "color": (0.90, 0.40, 0.10), "transparent": False, "light": 15, "hardness": -1.0, "tool_type": "NONE"},
    
    # Ores and new blocks
    "GOLD_ORE": {"id": 40, "texture": "gold_ore.png", "color": (1.0, 1.0, 1.0), "transparent": False, "light": 0, "hardness": 3.0, "tool_type": "PICKAXE"},
    "IRON_ORE": {"id": 41, "texture": "iron_ore.png", "color": (1.0, 1.0, 1.0), "transparent": False, "light": 0, "hardness": 3.0, "tool_type": "PICKAXE"},
    "COAL_ORE": {"id": 42, "texture": "coal_ore.png", "color": (1.0, 1.0, 1.0), "transparent": False, "light": 0, "hardness": 3.0, "tool_type": "PICKAXE"},
    "LAPIS_ORE": {"id": 21, "texture": "lapis_ore.png", "color": (1.0, 1.0, 1.0), "transparent": False, "light": 0, "hardness": 3.0, "tool_type": "PICKAXE"},
    "DIAMOND_ORE": {"id": 56, "texture": "diamond_ore.png", "color": (1.0, 1.0, 1.0), "transparent": False, "light": 0, "hardness": 3.0, "tool_type": "PICKAXE"},
    "REDSTONE_ORE": {"id": 73, "texture": "redstone_ore.png", "color": (1.0, 1.0, 1.0), "transparent": False, "light": 0, "hardness": 3.0, "tool_type": "PICKAXE"},
    "EMERALD_ORE": {"id": 129, "texture": "emerald_ore.png", "color": (1.0, 1.0, 1.0), "transparent": False, "light": 0, "hardness": 3.0, "tool_type": "PICKAXE"},
    
    "TALLGRASS": {"id": 31, "texture": "tallgrass.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "DANDELION": {"id": 37, "texture": "flower_dandelion.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "ROSE": {"id": 38, "texture": "flower_rose.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "DOUBLE_GRASS_BTM": {"id": 175, "texture": "double_plant_grass_bottom.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "DOUBLE_GRASS_TOP": {"id": 176, "texture": "double_plant_grass_top.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "DOUBLE_ROSE_BTM": {"id": 177, "texture": "double_plant_rose_bottom.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "DOUBLE_ROSE_TOP": {"id": 178, "texture": "double_plant_rose_top.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "PORKCHOP_RAW": {"id": 1000, "texture": "porkchop_raw.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "COBBLESTONE": {"id": 43, "texture": "cobblestone.png", "color": (0.5, 0.5, 0.5), "transparent": False, "light": 0, "hardness": 2.0, "tool_type": "PICKAXE"},
    "PLANKS_OAK": {"id": 44, "texture": "planks_oak.png", "color": (0.6, 0.5, 0.3), "transparent": False, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "PLANKS_SPRUCE": {"id": 45, "texture": "planks_spruce.png", "color": (0.4, 0.3, 0.2), "transparent": False, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "PLANKS_BIRCH": {"id": 46, "texture": "planks_birch.png", "color": (0.8, 0.8, 0.6), "transparent": False, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "PLANKS_JUNGLE": {"id": 47, "texture": "planks_jungle.png", "color": (0.7, 0.5, 0.4), "transparent": False, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "PLANKS_ACACIA": {"id": 48, "texture": "planks_acacia.png", "color": (0.8, 0.4, 0.3), "transparent": False, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "PLANKS_DARK_OAK": {"id": 49, "texture": "planks_big_oak.png", "color": (0.3, 0.2, 0.1), "transparent": False, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "SPONGE": {"id": 50, "texture": "sponge.png", "color": (0.9, 0.9, 0.3), "transparent": False, "light": 0, "hardness": 0.6, "tool_type": "NONE"},
    "LAPIS_BLOCK": {"id": 51, "texture": "lapis_block.png", "color": (0.1, 0.3, 0.8), "transparent": False, "light": 0, "hardness": 3.0, "tool_type": "PICKAXE"},
    "GOLD_BLOCK": {"id": 52, "texture": "gold_block.png", "color": (0.9, 0.8, 0.2), "transparent": False, "light": 0, "hardness": 3.0, "tool_type": "PICKAXE"},
    "IRON_BLOCK": {"id": 53, "texture": "iron_block.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 5.0, "tool_type": "PICKAXE"},
    "BRICKS": {"id": 54, "texture": "bricks.png", "color": (0.7, 0.3, 0.2), "transparent": False, "light": 0, "hardness": 2.0, "tool_type": "PICKAXE"},
    "TNT": {"id": 55, "texture": {"top": "tnt_top.png", "bottom": "tnt_bottom.png", "side": "tnt_side.png"}, "color": (0.8, 0.2, 0.2), "transparent": False, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "BOOKSHELF": {"id": 57, "texture": {"top": "planks_oak.png", "bottom": "planks_oak.png", "side": "bookshelf.png"}, "color": (0.6, 0.4, 0.2), "transparent": False, "light": 0, "hardness": 1.5, "tool_type": "AXE"},
    "MOSSY_COBBLESTONE": {"id": 58, "texture": "cobblestone_mossy.png", "color": (0.4, 0.5, 0.4), "transparent": False, "light": 0, "hardness": 2.0, "tool_type": "PICKAXE"},
    "OBSIDIAN": {"id": 59, "texture": "obsidian.png", "color": (0.1, 0.0, 0.2), "transparent": False, "light": 0, "hardness": 50.0, "tool_type": "PICKAXE"},
    "DIAMOND_BLOCK": {"id": 60, "texture": "diamond_block.png", "color": (0.3, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 5.0, "tool_type": "PICKAXE"},
    "CLAY_BLOCK": {"id": 61, "texture": "clay.png", "color": (0.6, 0.6, 0.7), "transparent": False, "light": 0, "hardness": 0.6, "tool_type": "SHOVEL"},
    "JUKEBOX": {"id": 62, "texture": {"top": "jukebox_top.png", "bottom": "jukebox_side.png", "side": "jukebox_side.png"}, "color": (0.5, 0.3, 0.2), "transparent": False, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "NETHERRACK": {"id": 63, "texture": "netherrack.png", "color": (0.4, 0.1, 0.1), "transparent": False, "light": 0, "hardness": 0.4, "tool_type": "PICKAXE"},
    "SOUL_SAND": {"id": 64, "texture": "soul_sand.png", "color": (0.3, 0.2, 0.1), "transparent": False, "light": 0, "hardness": 0.5, "tool_type": "SHOVEL"},
    "GLOWSTONE": {"id": 65, "texture": "glowstone.png", "color": (0.8, 0.8, 0.4), "transparent": False, "light": 15, "hardness": 0.3, "tool_type": "NONE"},
    "STONEBRICK": {"id": 66, "texture": "stonebrick.png", "color": (0.5, 0.5, 0.5), "transparent": False, "light": 0, "hardness": 1.5, "tool_type": "PICKAXE"},
    "STONEBRICK_MOSSY": {"id": 67, "texture": "stonebrick_mossy.png", "color": (0.4, 0.5, 0.4), "transparent": False, "light": 0, "hardness": 1.5, "tool_type": "PICKAXE"},
    "STONEBRICK_CRACKED": {"id": 68, "texture": "stonebrick_cracked.png", "color": (0.5, 0.5, 0.5), "transparent": False, "light": 0, "hardness": 1.5, "tool_type": "PICKAXE"},
    "STONEBRICK_CARVED": {"id": 69, "texture": "stonebrick_carved.png", "color": (0.5, 0.5, 0.5), "transparent": False, "light": 0, "hardness": 1.5, "tool_type": "PICKAXE"},
    "MELON_BLOCK": {"id": 70, "texture": {"top": "melon_top.png", "bottom": "melon_top.png", "side": "melon_side.png"}, "color": (0.3, 0.6, 0.2), "transparent": False, "light": 0, "hardness": 1.0, "tool_type": "AXE"},
    "NETHER_BRICK": {"id": 71, "texture": "nether_brick.png", "color": (0.3, 0.1, 0.1), "transparent": False, "light": 0, "hardness": 2.0, "tool_type": "PICKAXE"},
    "END_STONE": {"id": 72, "texture": "end_stone.png", "color": (0.8, 0.8, 0.6), "transparent": False, "light": 0, "hardness": 3.0, "tool_type": "PICKAXE"},
    "EMERALD_BLOCK_SOLID": {"id": 74, "texture": "emerald_block.png", "color": (0.2, 0.8, 0.3), "transparent": False, "light": 0, "hardness": 5.0, "tool_type": "PICKAXE"},
    "REDSTONE_BLOCK": {"id": 75, "texture": "redstone_block.png", "color": (0.8, 0.1, 0.1), "transparent": False, "light": 0, "hardness": 5.0, "tool_type": "PICKAXE"},
    "QUARTZ_BLOCK": {"id": 76, "texture": {"top": "quartz_block_top.png", "bottom": "quartz_block_bottom.png", "side": "quartz_block_side.png"}, "color": (0.9, 0.9, 0.9), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "PICKAXE"},
    "QUARTZ_CHISELED": {"id": 77, "texture": {"top": "quartz_block_chiseled_top.png", "bottom": "quartz_block_chiseled_top.png", "side": "quartz_block_chiseled.png"}, "color": (0.9, 0.9, 0.9), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "PICKAXE"},
    "QUARTZ_PILLAR": {"id": 78, "texture": {"top": "quartz_block_lines_top.png", "bottom": "quartz_block_lines_top.png", "side": "quartz_block_lines.png"}, "color": (0.9, 0.9, 0.9), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "PICKAXE"},
    "COAL_BLOCK_SOLID": {"id": 80, "texture": "coal_block.png", "color": (0.1, 0.1, 0.1), "transparent": False, "light": 0, "hardness": 5.0, "tool_type": "PICKAXE"},
    "PACKED_ICE": {"id": 81, "texture": "ice_packed.png", "color": (0.6, 0.8, 1.0), "transparent": False, "light": 0, "hardness": 0.5, "tool_type": "PICKAXE"},
    "RED_SAND": {"id": 82, "texture": "red_sand.png", "color": (0.8, 0.4, 0.1), "transparent": False, "light": 0, "hardness": 0.5, "tool_type": "SHOVEL"},
    "PODZOL": {"id": 83, "texture": {"top": "dirt_podzol_top.png", "bottom": "dirt.png", "side": "dirt_podzol_side.png"}, "color": (0.4, 0.3, 0.2), "transparent": False, "light": 0, "hardness": 0.5, "tool_type": "SHOVEL"},
    "COARSE_DIRT": {"id": 84, "texture": "coarse_dirt.png", "color": (0.4, 0.3, 0.2), "transparent": False, "light": 0, "hardness": 0.5, "tool_type": "SHOVEL"},
    "ANDESITE": {"id": 85, "texture": "stone_andesite.png", "color": (0.5, 0.5, 0.5), "transparent": False, "light": 0, "hardness": 1.5, "tool_type": "PICKAXE"},
    "ANDESITE_POLISHED": {"id": 86, "texture": "stone_andesite_smooth.png", "color": (0.5, 0.5, 0.5), "transparent": False, "light": 0, "hardness": 1.5, "tool_type": "PICKAXE"},
    "DIORITE": {"id": 87, "texture": "stone_diorite.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 1.5, "tool_type": "PICKAXE"},
    "DIORITE_POLISHED": {"id": 88, "texture": "stone_diorite_smooth.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 1.5, "tool_type": "PICKAXE"},
    "GRANITE": {"id": 89, "texture": "stone_granite.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.5, "tool_type": "PICKAXE"},
    "GRANITE_POLISHED": {"id": 90, "texture": "stone_granite_smooth.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.5, "tool_type": "PICKAXE"},
    "PRISMARINE": {"id": 91, "texture": "prismarine_rough.png", "color": (0.3, 0.6, 0.6), "transparent": False, "light": 0, "hardness": 1.5, "tool_type": "PICKAXE"},
    "PRISMARINE_BRICKS": {"id": 92, "texture": "prismarine_bricks.png", "color": (0.3, 0.6, 0.6), "transparent": False, "light": 0, "hardness": 1.5, "tool_type": "PICKAXE"},
    "PRISMARINE_DARK": {"id": 93, "texture": "prismarine_dark.png", "color": (0.2, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.5, "tool_type": "PICKAXE"},
    "SEA_LANTERN": {"id": 94, "texture": "sea_lantern.png", "color": (0.8, 0.9, 0.9), "transparent": False, "light": 15, "hardness": 0.3, "tool_type": "NONE"},
    "SLIME_BLOCK": {"id": 95, "texture": "slime.png", "color": (0.4, 0.8, 0.4), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "HAY_BLOCK": {"id": 96, "texture": {"top": "hay_block_top.png", "bottom": "hay_block_top.png", "side": "hay_block_side.png"}, "color": (0.8, 0.8, 0.2), "transparent": False, "light": 0, "hardness": 0.5, "tool_type": "NONE"},
    "WOOL_WHITE": {"id": 100, "texture": "wool_colored_white.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "NONE"},
    "GLASS_WHITE": {"id": 120, "texture": "glass_white.png", "color": (0.8, 0.8, 0.8), "transparent": True, "light": 0, "hardness": 0.3, "tool_type": "NONE"},
    "STAINED_CLAY_WHITE": {"id": 140, "texture": "hardened_clay_stained_white.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.25, "tool_type": "PICKAXE"},
    "WOOL_ORANGE": {"id": 101, "texture": "wool_colored_orange.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "NONE"},
    "GLASS_ORANGE": {"id": 121, "texture": "glass_orange.png", "color": (0.8, 0.8, 0.8), "transparent": True, "light": 0, "hardness": 0.3, "tool_type": "NONE"},
    "STAINED_CLAY_ORANGE": {"id": 141, "texture": "hardened_clay_stained_orange.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.25, "tool_type": "PICKAXE"},
    "WOOL_MAGENTA": {"id": 102, "texture": "wool_colored_magenta.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "NONE"},
    "GLASS_MAGENTA": {"id": 122, "texture": "glass_magenta.png", "color": (0.8, 0.8, 0.8), "transparent": True, "light": 0, "hardness": 0.3, "tool_type": "NONE"},
    "STAINED_CLAY_MAGENTA": {"id": 142, "texture": "hardened_clay_stained_magenta.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.25, "tool_type": "PICKAXE"},
    "WOOL_LIGHT_BLUE": {"id": 103, "texture": "wool_colored_light_blue.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "NONE"},
    "GLASS_LIGHT_BLUE": {"id": 123, "texture": "glass_light_blue.png", "color": (0.8, 0.8, 0.8), "transparent": True, "light": 0, "hardness": 0.3, "tool_type": "NONE"},
    "STAINED_CLAY_LIGHT_BLUE": {"id": 143, "texture": "hardened_clay_stained_light_blue.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.25, "tool_type": "PICKAXE"},
    "WOOL_YELLOW": {"id": 104, "texture": "wool_colored_yellow.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "NONE"},
    "GLASS_YELLOW": {"id": 124, "texture": "glass_yellow.png", "color": (0.8, 0.8, 0.8), "transparent": True, "light": 0, "hardness": 0.3, "tool_type": "NONE"},
    "STAINED_CLAY_YELLOW": {"id": 144, "texture": "hardened_clay_stained_yellow.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.25, "tool_type": "PICKAXE"},
    "WOOL_LIME": {"id": 105, "texture": "wool_colored_lime.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "NONE"},
    "GLASS_LIME": {"id": 125, "texture": "glass_lime.png", "color": (0.8, 0.8, 0.8), "transparent": True, "light": 0, "hardness": 0.3, "tool_type": "NONE"},
    "STAINED_CLAY_LIME": {"id": 145, "texture": "hardened_clay_stained_lime.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.25, "tool_type": "PICKAXE"},
    "WOOL_PINK": {"id": 106, "texture": "wool_colored_pink.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "NONE"},
    "GLASS_PINK": {"id": 126, "texture": "glass_pink.png", "color": (0.8, 0.8, 0.8), "transparent": True, "light": 0, "hardness": 0.3, "tool_type": "NONE"},
    "STAINED_CLAY_PINK": {"id": 146, "texture": "hardened_clay_stained_pink.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.25, "tool_type": "PICKAXE"},
    "WOOL_GRAY": {"id": 107, "texture": "wool_colored_gray.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "NONE"},
    "GLASS_GRAY": {"id": 127, "texture": "glass_gray.png", "color": (0.8, 0.8, 0.8), "transparent": True, "light": 0, "hardness": 0.3, "tool_type": "NONE"},
    "STAINED_CLAY_GRAY": {"id": 147, "texture": "hardened_clay_stained_gray.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.25, "tool_type": "PICKAXE"},
    "WOOL_SILVER": {"id": 108, "texture": "wool_colored_silver.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "NONE"},
    "GLASS_SILVER": {"id": 128, "texture": "glass_silver.png", "color": (0.8, 0.8, 0.8), "transparent": True, "light": 0, "hardness": 0.3, "tool_type": "NONE"},
    "STAINED_CLAY_SILVER": {"id": 148, "texture": "hardened_clay_stained_silver.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.25, "tool_type": "PICKAXE"},
    "WOOL_CYAN": {"id": 109, "texture": "wool_colored_cyan.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "NONE"},
    "GLASS_CYAN": {"id": 225, "texture": "glass_cyan.png", "color": (0.8, 0.8, 0.8), "transparent": True, "light": 0, "hardness": 0.3, "tool_type": "NONE"},
    "STAINED_CLAY_CYAN": {"id": 149, "texture": "hardened_clay_stained_cyan.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.25, "tool_type": "PICKAXE"},
    "WOOL_PURPLE": {"id": 110, "texture": "wool_colored_purple.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "NONE"},
    "GLASS_PURPLE": {"id": 130, "texture": "glass_purple.png", "color": (0.8, 0.8, 0.8), "transparent": True, "light": 0, "hardness": 0.3, "tool_type": "NONE"},
    "STAINED_CLAY_PURPLE": {"id": 150, "texture": "hardened_clay_stained_purple.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.25, "tool_type": "PICKAXE"},
    "WOOL_BLUE": {"id": 111, "texture": "wool_colored_blue.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "NONE"},
    "GLASS_BLUE": {"id": 131, "texture": "glass_blue.png", "color": (0.8, 0.8, 0.8), "transparent": True, "light": 0, "hardness": 0.3, "tool_type": "NONE"},
    "STAINED_CLAY_BLUE": {"id": 151, "texture": "hardened_clay_stained_blue.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.25, "tool_type": "PICKAXE"},
    "WOOL_BROWN": {"id": 112, "texture": "wool_colored_brown.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "NONE"},
    "GLASS_BROWN": {"id": 132, "texture": "glass_brown.png", "color": (0.8, 0.8, 0.8), "transparent": True, "light": 0, "hardness": 0.3, "tool_type": "NONE"},
    "STAINED_CLAY_BROWN": {"id": 152, "texture": "hardened_clay_stained_brown.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.25, "tool_type": "PICKAXE"},
    "WOOL_GREEN": {"id": 113, "texture": "wool_colored_green.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "NONE"},
    "GLASS_GREEN": {"id": 133, "texture": "glass_green.png", "color": (0.8, 0.8, 0.8), "transparent": True, "light": 0, "hardness": 0.3, "tool_type": "NONE"},
    "STAINED_CLAY_GREEN": {"id": 153, "texture": "hardened_clay_stained_green.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.25, "tool_type": "PICKAXE"},
    "WOOL_RED": {"id": 114, "texture": "wool_colored_red.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "NONE"},
    "GLASS_RED": {"id": 134, "texture": "glass_red.png", "color": (0.8, 0.8, 0.8), "transparent": True, "light": 0, "hardness": 0.3, "tool_type": "NONE"},
    "STAINED_CLAY_RED": {"id": 154, "texture": "hardened_clay_stained_red.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.25, "tool_type": "PICKAXE"},
    "WOOL_BLACK": {"id": 115, "texture": "wool_colored_black.png", "color": (0.8, 0.8, 0.8), "transparent": False, "light": 0, "hardness": 0.8, "tool_type": "NONE"},
    "GLASS_BLACK": {"id": 135, "texture": "glass_black.png", "color": (0.8, 0.8, 0.8), "transparent": True, "light": 0, "hardness": 0.3, "tool_type": "NONE"},
    "STAINED_CLAY_BLACK": {"id": 155, "texture": "hardened_clay_stained_black.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.25, "tool_type": "PICKAXE"},
    "WOOD_DOOR": {"id": 162, "texture": {"top": "door_wood_upper.png", "bottom": "door_wood_lower.png", "side": "door_wood_lower.png"}, "color": (0.5, 0.3, 0.1), "transparent": True, "light": 0, "hardness": 3.0, "tool_type": "AXE"},
    "IRON_DOOR": {"id": 163, "texture": {"top": "door_iron_upper.png", "bottom": "door_iron_lower.png", "side": "door_iron_lower.png"}, "color": (0.8, 0.8, 0.8), "transparent": True, "light": 0, "hardness": 5.0, "tool_type": "PICKAXE"},
    "CRAFTING_TABLE": {"id": 116, "texture": {"top": "crafting_table_top.png", "bottom": "planks_oak.png", "side": "crafting_table_side.png"}, "color": (0.6, 0.5, 0.3), "transparent": False, "light": 0, "hardness": 2.5, "tool_type": "AXE"},
    "HARDENED_CLAY": {"id": 160, "texture": "hardened_clay.png", "color": (0.6, 0.4, 0.4), "transparent": False, "light": 0, "hardness": 1.25, "tool_type": "PICKAXE"},
    "OAK_STAIRS": {"id": 200, "texture": "planks_oak.png", "color": (0.6, 0.5, 0.3), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "COBBLESTONE_STAIRS": {"id": 201, "texture": "cobblestone.png", "color": (0.5, 0.5, 0.5), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "PICKAXE"},
    "WOODEN_SLAB": {"id": 202, "texture": "planks_oak.png", "color": (0.6, 0.5, 0.3), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "STONE_SLAB": {"id": 203, "texture": "stone_slab_top.png", "color": (0.5, 0.5, 0.5), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "PICKAXE"},
    "SPRUCE_STAIRS": {"id": 204, "texture": "planks_spruce.png", "color": (0.4, 0.3, 0.2), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "BIRCH_STAIRS": {"id": 205, "texture": "planks_birch.png", "color": (0.8, 0.8, 0.6), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "JUNGLE_STAIRS": {"id": 206, "texture": "planks_jungle.png", "color": (0.7, 0.5, 0.4), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "ACACIA_STAIRS": {"id": 207, "texture": "planks_acacia.png", "color": (0.8, 0.4, 0.3), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "DARK_OAK_STAIRS": {"id": 208, "texture": "planks_big_oak.png", "color": (0.3, 0.2, 0.1), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "BRICK_STAIRS": {"id": 209, "texture": "bricks.png", "color": (0.7, 0.3, 0.2), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "PICKAXE"},
    "STONE_BRICK_STAIRS": {"id": 210, "texture": "stonebrick.png", "color": (0.5, 0.5, 0.5), "transparent": True, "light": 0, "hardness": 1.5, "tool_type": "PICKAXE"},
    "NETHER_BRICK_STAIRS": {"id": 211, "texture": "nether_brick.png", "color": (0.3, 0.1, 0.1), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "PICKAXE"},
    "SANDSTONE_STAIRS": {"id": 212, "texture": "sandstone.png", "color": (0.80, 0.75, 0.50), "transparent": True, "light": 0, "hardness": 0.8, "tool_type": "PICKAXE"},
    "QUARTZ_STAIRS": {"id": 213, "texture": "quartz_block_side.png", "color": (0.9, 0.9, 0.9), "transparent": True, "light": 0, "hardness": 0.8, "tool_type": "PICKAXE"},
    "SPRUCE_SLAB": {"id": 214, "texture": "planks_spruce.png", "color": (0.4, 0.3, 0.2), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "BIRCH_SLAB": {"id": 215, "texture": "planks_birch.png", "color": (0.8, 0.8, 0.6), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "JUNGLE_SLAB": {"id": 216, "texture": "planks_jungle.png", "color": (0.7, 0.5, 0.4), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "ACACIA_SLAB": {"id": 217, "texture": "planks_acacia.png", "color": (0.8, 0.4, 0.3), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "DARK_OAK_SLAB": {"id": 218, "texture": "planks_big_oak.png", "color": (0.3, 0.2, 0.1), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "BRICK_SLAB": {"id": 219, "texture": "bricks.png", "color": (0.7, 0.3, 0.2), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "PICKAXE"},
    "STONE_BRICK_SLAB": {"id": 220, "texture": "stonebrick.png", "color": (0.5, 0.5, 0.5), "transparent": True, "light": 0, "hardness": 1.5, "tool_type": "PICKAXE"},
    "NETHER_BRICK_SLAB": {"id": 221, "texture": "nether_brick.png", "color": (0.3, 0.1, 0.1), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "PICKAXE"},
    "SANDSTONE_SLAB": {"id": 222, "texture": "sandstone.png", "color": (0.80, 0.75, 0.50), "transparent": True, "light": 0, "hardness": 0.8, "tool_type": "PICKAXE"},
    "QUARTZ_SLAB": {"id": 223, "texture": "quartz_block_side.png", "color": (0.9, 0.9, 0.9), "transparent": True, "light": 0, "hardness": 0.8, "tool_type": "PICKAXE"},
    "COBBLESTONE_SLAB": {"id": 224, "texture": "cobblestone.png", "color": (0.5, 0.5, 0.5), "transparent": True, "light": 0, "hardness": 2.0, "tool_type": "PICKAXE"},
    
    # Missing Biome and Decor Blocks
    "DEADBUSH": {"id": 32, "texture": "deadbush.png", "color": (0.6, 0.4, 0.2), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "MUSHROOM_BROWN": {"id": 39, "texture": "mushroom_brown.png", "color": (0.8, 0.7, 0.5), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "MUSHROOM_RED": {"id": 230, "texture": "mushroom_red.png", "color": (0.9, 0.2, 0.2), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "VINE": {"id": 231, "texture": "vine.png", "color": (0.2, 0.6, 0.2), "transparent": True, "light": 0, "hardness": 0.2, "tool_type": "NONE"},
    "REEDS": {"id": 232, "texture": "reeds.png", "color": (0.4, 0.8, 0.3), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "WATERLILY": {"id": 233, "texture": "waterlily.png", "color": (0.2, 0.7, 0.3), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "PUMPKIN": {"id": 234, "texture": {"top": "pumpkin_top.png", "bottom": "pumpkin_top.png", "side": "pumpkin_side.png"}, "color": (0.9, 0.5, 0.1), "transparent": False, "light": 0, "hardness": 1.0, "tool_type": "AXE"},
    "SNOW_LAYER": {"id": 235, "texture": "snow.png", "color": (0.95, 0.95, 0.95), "transparent": True, "light": 0, "hardness": 0.1, "tool_type": "SHOVEL"},
    "JUNGLE_WOOD": {"id": 236, "texture": {"top": "log_jungle_top.png", "bottom": "log_jungle_top.png", "side": "log_jungle.png"}, "color": (0.5, 0.4, 0.2), "transparent": False, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "ACACIA_WOOD": {"id": 237, "texture": {"top": "log_acacia_top.png", "bottom": "log_acacia_top.png", "side": "log_acacia.png"}, "color": (0.6, 0.5, 0.4), "transparent": False, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "DARK_OAK_WOOD": {"id": 238, "texture": {"top": "log_big_oak_top.png", "bottom": "log_big_oak_top.png", "side": "log_big_oak.png"}, "color": (0.3, 0.2, 0.1), "transparent": False, "light": 0, "hardness": 2.0, "tool_type": "AXE"},
    "JUNGLE_LEAVES": {"id": 239, "texture": "leaves_jungle.png", "color": (0.2, 0.6, 0.1), "transparent": True, "light": 0, "hardness": 0.2, "tool_type": "NONE"},
    "ACACIA_LEAVES": {"id": 240, "texture": "leaves_acacia.png", "color": (0.4, 0.7, 0.3), "transparent": True, "light": 0, "hardness": 0.2, "tool_type": "NONE"},
    "DARK_OAK_LEAVES": {"id": 241, "texture": "leaves_big_oak.png", "color": (0.1, 0.4, 0.1), "transparent": True, "light": 0, "hardness": 0.2, "tool_type": "NONE"},

    # Items
    "STICK": {"id": 1001, "texture": "stick.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "DIAMOND": {"id": 1002, "texture": "diamond.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "IRON_INGOT": {"id": 1003, "texture": "iron_ingot.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "GOLD_INGOT": {"id": 1004, "texture": "gold_ingot.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "COAL": {"id": 1005, "texture": "coal.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "CHARCOAL": {"id": 1006, "texture": "charcoal.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "BOWL": {"id": 1007, "texture": "bowl.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "MUSHROOM_STEW": {"id": 1008, "texture": "mushroom_stew.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    
    "WOODEN_SWORD": {"id": 1009, "texture": "wood_sword.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "WOODEN_SHOVEL": {"id": 1010, "texture": "wood_shovel.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "WOODEN_PICKAXE": {"id": 1011, "texture": "wood_pickaxe.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "WOODEN_AXE": {"id": 1012, "texture": "wood_axe.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "WOODEN_HOE": {"id": 1013, "texture": "wood_hoe.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},

    "STONE_SWORD": {"id": 1014, "texture": "stone_sword.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "STONE_SHOVEL": {"id": 1015, "texture": "stone_shovel.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "STONE_PICKAXE": {"id": 1016, "texture": "stone_pickaxe.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "STONE_AXE": {"id": 1017, "texture": "stone_axe.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "STONE_HOE": {"id": 1018, "texture": "stone_hoe.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},

    "IRON_SWORD": {"id": 1019, "texture": "iron_sword.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "IRON_SHOVEL": {"id": 1020, "texture": "iron_shovel.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "IRON_PICKAXE": {"id": 1021, "texture": "iron_pickaxe.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "IRON_AXE": {"id": 1022, "texture": "iron_axe.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "IRON_HOE": {"id": 1023, "texture": "iron_hoe.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},

    "DIAMOND_SWORD": {"id": 1024, "texture": "diamond_sword.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "DIAMOND_SHOVEL": {"id": 1025, "texture": "diamond_shovel.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "DIAMOND_PICKAXE": {"id": 1026, "texture": "diamond_pickaxe.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "DIAMOND_AXE": {"id": 1027, "texture": "diamond_axe.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "DIAMOND_HOE": {"id": 1028, "texture": "diamond_hoe.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},

    "GOLD_SWORD": {"id": 1029, "texture": "gold_sword.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "GOLD_SHOVEL": {"id": 1030, "texture": "gold_shovel.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "GOLD_PICKAXE": {"id": 1031, "texture": "gold_pickaxe.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "GOLD_AXE": {"id": 1032, "texture": "gold_axe.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
    "GOLD_HOE": {"id": 1033, "texture": "gold_hoe.png", "color": (1.0, 1.0, 1.0), "transparent": True, "light": 0, "hardness": 0.0, "tool_type": "NONE"},
}

# --- DYNAMICALLY ASSIGNED FROM MC_ID_CONVERTER ---
AIR = 0
STONE = 1
GRASS = 2
DIRT = 3
COBBLESTONE = 4
PLANKS_OAK = 5
BEDROCK = 7
WATER = 9
LAVA = 11
SAND = 12
GRAVEL = 13
GOLD_ORE = 14
IRON_ORE = 15
COAL_ORE = 16
WOOD = 17
LEAVES = 18
SPONGE = 19
GLASS = 20
LAPIS_ORE = 21
LAPIS_BLOCK = 22
SANDSTONE = 24
DEADBUSH = 32
WOOL_WHITE = 35
DANDELION = 37
ROSE = 38
MUSHROOM_BROWN = 39
MUSHROOM_RED = 40
GOLD_BLOCK = 41
IRON_BLOCK = 42
STONE_SLAB = 44
WOODEN_SLAB = 126
BRICKS = 45
TNT = 46
BOOKSHELF = 47
MOSSY_COBBLESTONE = 48
OBSIDIAN = 49
OAK_STAIRS = 53
DIAMOND_ORE = 56
DIAMOND_BLOCK = 57
CRAFTING_TABLE = 58
WOOD_DOOR = 64
COBBLESTONE_STAIRS = 67
IRON_DOOR = 71
REDSTONE_ORE = 73
SNOW_LAYER = 78
ICE = 79
SNOW = 80
CACTUS = 81
CLAY_BLOCK = 82
REEDS = 83
JUKEBOX = 84
NETHERRACK = 87
SOUL_SAND = 88
GLOWSTONE = 89
GLASS_WHITE = 95
STONEBRICK = 98
MELON_BLOCK = 103
VINE = 106
BRICK_STAIRS = 108
STONE_BRICK_STAIRS = 109
MYCELIUM = 110
WATERLILY = 111
NETHER_BRICK = 112
NETHER_BRICK_STAIRS = 114
END_STONE = 121
SANDSTONE_STAIRS = 128
EMERALD_ORE = 129
EMERALD_BLOCK_SOLID = 133
SPRUCE_STAIRS = 134
BIRCH_STAIRS = 135
JUNGLE_STAIRS = 136
REDSTONE_BLOCK = 152
QUARTZ_BLOCK = 155
QUARTZ_STAIRS = 156
STAINED_CLAY_WHITE = 159
HARDENED_CLAY = 172
ACACIA_LEAVES = 161
ACACIA_WOOD = 162
ACACIA_STAIRS = 163
DARK_OAK_STAIRS = 164
SLIME_BLOCK = 165
PRISMARINE = 168
SEA_LANTERN = 169
HAY_BLOCK = 170
COAL_BLOCK_SOLID = 173
PACKED_ICE = 174
PUMPKIN = 86
PLANKS_SPRUCE = 6
PLANKS_BIRCH = 8
PLANKS_JUNGLE = 10
PLANKS_ACACIA = 23
PLANKS_DARK_OAK = 25
RED_SAND = 26
SPRUCE_WOOD = 27
BIRCH_WOOD = 28
JUNGLE_WOOD = 29
SPRUCE_LEAVES = 30
BIRCH_LEAVES = 31
JUNGLE_LEAVES = 33
TALLGRASS = 34
WOOL_ORANGE = 36
WOOL_MAGENTA = 43
WOOL_LIGHT_BLUE = 50
WOOL_YELLOW = 51
WOOL_LIME = 52
WOOL_PINK = 54
WOOL_GRAY = 55
WOOL_SILVER = 59
WOOL_CYAN = 60
WOOL_PURPLE = 61
WOOL_BLUE = 62
WOOL_BROWN = 63
WOOL_GREEN = 65
WOOL_RED = 66
WOOL_BLACK = 68
SANDSTONE_SLAB = 69
COBBLESTONE_SLAB = 70
BRICK_SLAB = 72
STONE_BRICK_SLAB = 74
NETHER_BRICK_SLAB = 75
QUARTZ_SLAB = 76
GLASS_ORANGE = 77
GLASS_MAGENTA = 85
GLASS_LIGHT_BLUE = 90
GLASS_YELLOW = 91
GLASS_LIME = 92
GLASS_PINK = 93
GLASS_GRAY = 94
GLASS_SILVER = 96
GLASS_CYAN = 97
GLASS_PURPLE = 99
GLASS_BLUE = 100
GLASS_BROWN = 101
GLASS_GREEN = 102
GLASS_RED = 104
GLASS_BLACK = 105
STONEBRICK_MOSSY = 107
STONEBRICK_CRACKED = 113
STONEBRICK_CARVED = 115
SPRUCE_SLAB = 116
BIRCH_SLAB = 117
JUNGLE_SLAB = 118
ACACIA_SLAB = 119
DARK_OAK_SLAB = 120
QUARTZ_CHISELED = 122
QUARTZ_PILLAR = 123
STAINED_CLAY_ORANGE = 124
STAINED_CLAY_MAGENTA = 125
STAINED_CLAY_LIGHT_BLUE = 127
STAINED_CLAY_YELLOW = 130
STAINED_CLAY_LIME = 131
STAINED_CLAY_PINK = 132
STAINED_CLAY_GRAY = 137
STAINED_CLAY_SILVER = 138
STAINED_CLAY_CYAN = 139
STAINED_CLAY_PURPLE = 140
STAINED_CLAY_BLUE = 141
STAINED_CLAY_BROWN = 142
STAINED_CLAY_GREEN = 143
STAINED_CLAY_RED = 144
STAINED_CLAY_BLACK = 145
DARK_OAK_LEAVES = 146
DARK_OAK_WOOD = 147
PRISMARINE_BRICKS = 148
PRISMARINE_DARK = 149
DOUBLE_GRASS_BTM = 150
DOUBLE_GRASS_TOP = 151
DOUBLE_ROSE_BTM = 153
DOUBLE_ROSE_TOP = 154
PODZOL = 157
COARSE_DIRT = 158
ANDESITE = 160
ANDESITE_POLISHED = 166
DIORITE = 167
DIORITE_POLISHED = 171
GRANITE = 175
GRANITE_POLISHED = 176
STICK = 280
DIAMOND = 264
IRON_INGOT = 265
GOLD_INGOT = 266
COAL = 263
CHARCOAL = 263
BOWL = 281
MUSHROOM_STEW = 282
WOODEN_SWORD = 268
WOODEN_SHOVEL = 269
WOODEN_PICKAXE = 270
WOODEN_AXE = 271
WOODEN_HOE = 290
STONE_SWORD = 272
STONE_SHOVEL = 273
STONE_PICKAXE = 274
STONE_AXE = 275
STONE_HOE = 291
IRON_SWORD = 267
IRON_SHOVEL = 256
IRON_PICKAXE = 257
IRON_AXE = 258
IRON_HOE = 292
DIAMOND_SWORD = 276
DIAMOND_SHOVEL = 277
DIAMOND_PICKAXE = 278
DIAMOND_AXE = 279
DIAMOND_HOE = 293
GOLD_SWORD = 283
GOLD_SHOVEL = 284
GOLD_PICKAXE = 285
GOLD_AXE = 286
GOLD_HOE = 294
PORKCHOP_RAW = 319

from world.mc_id_converter import INTERNAL_NAMES_MAP, BLOCK_TINT_ARRAY

import sys
_mod = sys.modules[__name__]
for _k, _v in INTERNAL_NAMES_MAP.items():
    setattr(_mod, _k, _v)

# --- DYNAMICALLY OVERRIDE BLOCK_REGISTRY IDs ---
for name, block in BLOCK_REGISTRY.items():
    if name in INTERNAL_NAMES_MAP:
        block["id"] = INTERNAL_NAMES_MAP[name]

# Numba için hızlı erişim dizileri (max id 2048 instead of dynamically looking up)
BLOCK_OPAQUE_ARRAY = np.ones(2048, dtype=np.bool_)
BLOCK_LIGHT_EMISSION_ARRAY = np.zeros(2048, dtype=np.uint8)
BLOCK_COLORS_ARRAY = np.ones((2048, 3), dtype=np.float32)
BLOCK_HARDNESS_ARRAY = np.zeros(2048, dtype=np.float32)

# Tool Types mapping to integer for Numba:
# 0: NONE, 1: PICKAXE, 2: AXE, 3: SHOVEL
TOOL_TYPE_MAP = {"NONE": 0, "PICKAXE": 1, "AXE": 2, "SHOVEL": 3}
BLOCK_TOOL_ARRAY = np.zeros(2048, dtype=np.uint8)
BLOCK_MAX_STACK_ARRAY = np.full(2048, 64, dtype=np.uint8)

for name, data in BLOCK_REGISTRY.items():
    bid = data["id"]
    if bid < 2048:
        BLOCK_OPAQUE_ARRAY[bid] = not data["transparent"]
        BLOCK_LIGHT_EMISSION_ARRAY[bid] = data["light"]
        BLOCK_COLORS_ARRAY[bid] = data["color"]
        BLOCK_HARDNESS_ARRAY[bid] = data["hardness"]
        BLOCK_TOOL_ARRAY[bid] = TOOL_TYPE_MAP.get(data.get("tool_type", "NONE"), 0)
        
        if bid >= 1009 and bid <= 1033: # Tools
            BLOCK_MAX_STACK_ARRAY[bid] = 1
        else:
            BLOCK_MAX_STACK_ARRAY[bid] = data.get("max_stack", 64)

CHUNK_SIZE = 16
CHUNK_HEIGHT = 256
WATER_LEVEL = 20

# ─────────────── Numba JIT Perlin Noise ───────────────
# Saf Python ile yazılmış, Numba ile makine koduna derlenen noise.
# C kütüphanesine bağımlılık yok. Tüm döngüler CPU SIMD ile hızlanır.

# Permütasyon tablosu (sabit, Perlin'in orijinal tablosu)
_PERM = np.array([
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,
    69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,
    252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,
    171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,
    122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,
    63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,
    188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,
    38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,
    42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,
    43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,
    218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,
    145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,
    115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,
    141,128,195,78,66,215,61,156,180
], dtype=np.int32)

@njit(cache=True, nogil=True)
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

@njit(cache=True, nogil=True)
def _grad2d(h, x, y):
    h = h & 3
    if h == 0: return  x + y
    if h == 1: return -x + y
    if h == 2: return  x - y
    return -x - y

@njit(cache=True, nogil=True)
def _perlin2d(x, y, perm):
    """Tek nokta için 2D Perlin Noise (Numba JIT ile makine kodu)"""
    xi = int(np.floor(x)) & 255
    yi = int(np.floor(y)) & 255
    xf = x - np.floor(x)
    yf = y - np.floor(y)
    
    u = _fade(xf)
    v = _fade(yf)
    
    aa = perm[(perm[xi] + yi) & 255]
    ab = perm[(perm[xi] + yi + 1) & 255]
    ba = perm[(perm[(xi + 1) & 255] + yi) & 255]
    bb = perm[(perm[(xi + 1) & 255] + yi + 1) & 255]
    
    x1 = _grad2d(aa, xf, yf) * (1 - u) + _grad2d(ba, xf - 1, yf) * u
    x2 = _grad2d(ab, xf, yf - 1) * (1 - u) + _grad2d(bb, xf - 1, yf - 1) * u
    
    return x1 * (1 - v) + x2 * v

@njit(cache=True, nogil=True)
def _fbm2d(x, y, perm, octaves, persistence):
    """Fractal Brownian Motion — gerçekçi dağlar için katmanlı noise"""
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_val = 0.0
    
    for _ in range(octaves):
        total += _perlin2d(x * frequency, y * frequency, perm) * amplitude
        max_val += amplitude
        amplitude *= persistence
        frequency *= 2.0
    
    return total / max_val

# ─────────────── Numba JIT Chunk Üretici ───────────────

@njit(cache=True, nogil=True)
def _generate_chunk_data(cx, cz, perm):
    blocks = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
    light_map = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
    
    # 1. Arazi Üretimi
    for lx in range(CHUNK_SIZE):
        for lz in range(CHUNK_SIZE):
            wx = cx * CHUNK_SIZE + lx
            wz = cz * CHUNK_SIZE + lz
            
            h1 = _fbm2d(wx * 0.005, wz * 0.005, perm, 6, 0.5) * 32.0
            h2 = _fbm2d(wx * 0.02,  wz * 0.02,  perm, 3, 0.4) * 8.0
            height = int(28 + h1 + h2)
            if height < 1: height = 1
            if height >= CHUNK_HEIGHT: height = CHUNK_HEIGHT - 1
            
            for y in range(height):
                if y < height - 4:
                    blocks[lx, y, lz] = STONE
                elif y < height - 1:
                    blocks[lx, y, lz] = DIRT
                else:
                    if y <= WATER_LEVEL + 1:
                        blocks[lx, y, lz] = SAND
                    else:
                        blocks[lx, y, lz] = GRASS
            
            for y in range(height, WATER_LEVEL):
                if blocks[lx, y, lz] == AIR:
                    blocks[lx, y, lz] = WATER

@njit(cache=True, nogil=True)
def _calc_light_jit(blocks, light_map):
    # 1. Dikey Güneş Işığı (SkyLight)
    for x in range(CHUNK_SIZE):
        for z in range(CHUNK_SIZE):
            light = 15
            for y in range(CHUNK_HEIGHT - 1, -1, -1):
                b = blocks[x, y, z]
                if BLOCK_OPAQUE_ARRAY[b]:
                    light = 0
                light_map[x, y, z] = light

    # 2. Yatay Işık Yayılımı (BFS)
    queue_x = np.empty(CHUNK_SIZE * CHUNK_HEIGHT * CHUNK_SIZE, dtype=np.int32)
    queue_y = np.empty(CHUNK_SIZE * CHUNK_HEIGHT * CHUNK_SIZE, dtype=np.int32)
    queue_z = np.empty(CHUNK_SIZE * CHUNK_HEIGHT * CHUNK_SIZE, dtype=np.int32)
    head = 0
    tail = 0
    
    for x in range(CHUNK_SIZE):
        for y in range(CHUNK_HEIGHT):
            for z in range(CHUNK_SIZE):
                if light_map[x, y, z] == 15:
                    queue_x[tail] = x
                    queue_y[tail] = y
                    queue_z[tail] = z
                    tail += 1
                    
    dxs = np.array([1, -1, 0, 0, 0, 0], dtype=np.int32)
    dys = np.array([0, 0, 1, -1, 0, 0], dtype=np.int32)
    dzs = np.array([0, 0, 0, 0, 1, -1], dtype=np.int32)
    
    while head < tail:
        x = queue_x[head]
        y = queue_y[head]
        z = queue_z[head]
        head += 1
        
        L = light_map[x, y, z]
        if L <= 1: continue
        
        new_light = L - 1
        for i in range(6):
            nx = x + dxs[i]
            ny = y + dys[i]
            nz = z + dzs[i]
            
            if 0 <= nx < CHUNK_SIZE and 0 <= ny < CHUNK_HEIGHT and 0 <= nz < CHUNK_SIZE:
                if not BLOCK_OPAQUE_ARRAY[blocks[nx, ny, nz]]:
                    if light_map[nx, ny, nz] < new_light:
                        light_map[nx, ny, nz] = new_light
                        queue_x[tail] = nx
                        queue_y[tail] = ny
                        queue_z[tail] = nz
                        tail += 1
                        
@njit(cache=True, nogil=True)
def _generate_chunk_data(cx, cz, perm):
    blocks = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
    light_map = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
    
    # 1. Arazi Üretimi
    for lx in range(CHUNK_SIZE):
        for lz in range(CHUNK_SIZE):
            wx = cx * CHUNK_SIZE + lx
            wz = cz * CHUNK_SIZE + lz
            
            h1 = _fbm2d(wx * 0.005, wz * 0.005, perm, 6, 0.5) * 32.0
            h2 = _fbm2d(wx * 0.02,  wz * 0.02,  perm, 3, 0.4) * 8.0
            height = int(28 + h1 + h2)
            if height < 1: height = 1
            if height >= CHUNK_HEIGHT: height = CHUNK_HEIGHT - 1
            
            for y in range(height):
                if y < height - 4:
                    blocks[lx, y, lz] = STONE
                elif y < height - 1:
                    blocks[lx, y, lz] = DIRT
                else:
                    if y <= WATER_LEVEL + 1:
                        blocks[lx, y, lz] = SAND
                    else:
                        blocks[lx, y, lz] = GRASS
            
            for y in range(height, WATER_LEVEL):
                if blocks[lx, y, lz] == AIR:
                    blocks[lx, y, lz] = WATER

    _calc_light_jit(blocks, light_map)
    return blocks, light_map

_PERM_DOUBLE = np.concatenate([_PERM, _PERM])

def generate_chunk(cx, cz):
    return _generate_chunk_data(cx, cz, _PERM_DOUBLE)

def load_or_generate_chunk(cx, cz):
    data = load_chunk(cx, cz)
    if data is not None:
        # data is (blocks, lights)
        return data[0], data[1], np.zeros((0, 4), dtype=np.int32), np.full((16, 16), 1, dtype=np.int32), False
    blocks, lights, oob, chunk_biomes = generate_chunk(cx, cz)
    return blocks, lights, oob, chunk_biomes, True

def recalculate_chunk_light(blocks, light_map):
    _calc_light_jit(blocks, light_map)

def get_break_time(block_id, held_id):
    hardness = BLOCK_HARDNESS_ARRAY[block_id]
    if hardness < 0:
        return 999999.0
        
    base_time = hardness * 5.0
    
    if held_id == 0 or held_id < 1000:
        return base_time
        
    block_tool_type = BLOCK_TOOL_ARRAY[block_id]
    if block_tool_type == 0:
        return base_time
        
    # Determine held tool type and multiplier
    held_tool_type = 0
    multiplier = 1.0
    
    # 1: PICKAXE, 2: AXE, 3: SHOVEL
    if held_id in (270, 274, 257, 278, 285):
        held_tool_type = 1
    elif held_id in (271, 275, 258, 279, 286):
        held_tool_type = 2
    elif held_id in (269, 273, 256, 277, 284):
        held_tool_type = 3
        
    if held_tool_type > 0 and held_tool_type == block_tool_type:
        if held_id in (268, 269, 270, 271, 290): # Wood
            multiplier = 2.0
        elif held_id in (272, 273, 274, 275, 291): # Stone
            multiplier = 4.0
        elif held_id in (267, 256, 257, 258, 292): # Iron
            multiplier = 6.0
        elif held_id in (276, 277, 278, 279, 293): # Diamond
            multiplier = 8.0
        elif held_id in (283, 284, 285, 286, 294): # Gold
            multiplier = 12.0
            
        return base_time / multiplier
        
    return base_time
