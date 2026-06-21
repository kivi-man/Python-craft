import os

NEW_BLOCKS = [
    ("COBBLESTONE", 43, 'cobblestone.png', '(0.5, 0.5, 0.5)', 'False', '0', '2.0', '"PICKAXE"', 4, 0),
    ("PLANKS_OAK", 44, 'planks_oak.png', '(0.6, 0.5, 0.3)', 'False', '0', '2.0', '"AXE"', 5, 0),
    ("PLANKS_SPRUCE", 45, 'planks_spruce.png', '(0.4, 0.3, 0.2)', 'False', '0', '2.0', '"AXE"', 5, 1),
    ("PLANKS_BIRCH", 46, 'planks_birch.png', '(0.8, 0.8, 0.6)', 'False', '0', '2.0', '"AXE"', 5, 2),
    ("PLANKS_JUNGLE", 47, 'planks_jungle.png', '(0.7, 0.5, 0.4)', 'False', '0', '2.0', '"AXE"', 5, 3),
    ("PLANKS_ACACIA", 48, 'planks_acacia.png', '(0.8, 0.4, 0.3)', 'False', '0', '2.0', '"AXE"', 5, 4),
    ("PLANKS_DARK_OAK", 49, 'planks_big_oak.png', '(0.3, 0.2, 0.1)', 'False', '0', '2.0', '"AXE"', 5, 5),
    ("SPONGE", 50, 'sponge.png', '(0.9, 0.9, 0.3)', 'False', '0', '0.6', '"NONE"', 19, 0),
    ("LAPIS_BLOCK", 51, 'lapis_block.png', '(0.1, 0.3, 0.8)', 'False', '0', '3.0', '"PICKAXE"', 22, 0),
    ("GOLD_BLOCK", 52, 'gold_block.png', '(0.9, 0.8, 0.2)', 'False', '0', '3.0', '"PICKAXE"', 41, 0),
    ("IRON_BLOCK", 53, 'iron_block.png', '(0.8, 0.8, 0.8)', 'False', '0', '5.0', '"PICKAXE"', 42, 0),
    ("BRICKS", 54, 'bricks.png', '(0.7, 0.3, 0.2)', 'False', '0', '2.0', '"PICKAXE"', 45, 0),
    ("TNT", 55, '{"top": "tnt_top.png", "bottom": "tnt_bottom.png", "side": "tnt_side.png"}', '(0.8, 0.2, 0.2)', 'False', '0', '0.0', '"NONE"', 46, 0),
    ("BOOKSHELF", 57, '{"top": "planks_oak.png", "bottom": "planks_oak.png", "side": "bookshelf.png"}', '(0.6, 0.4, 0.2)', 'False', '0', '1.5', '"AXE"', 47, 0),
    ("MOSSY_COBBLESTONE", 58, 'cobblestone_mossy.png', '(0.4, 0.5, 0.4)', 'False', '0', '2.0', '"PICKAXE"', 48, 0),
    ("OBSIDIAN", 59, 'obsidian.png', '(0.1, 0.0, 0.2)', 'False', '0', '50.0', '"PICKAXE"', 49, 0),
    ("DIAMOND_BLOCK", 60, 'diamond_block.png', '(0.3, 0.8, 0.8)', 'False', '0', '5.0', '"PICKAXE"', 57, 0),
    ("CLAY_BLOCK", 61, 'clay.png', '(0.6, 0.6, 0.7)', 'False', '0', '0.6', '"SHOVEL"', 82, 0),
    ("JUKEBOX", 62, '{"top": "jukebox_top.png", "bottom": "jukebox_side.png", "side": "jukebox_side.png"}', '(0.5, 0.3, 0.2)', 'False', '0', '2.0', '"AXE"', 84, 0),
    ("NETHERRACK", 63, 'netherrack.png', '(0.4, 0.1, 0.1)', 'False', '0', '0.4', '"PICKAXE"', 87, 0),
    ("SOUL_SAND", 64, 'soul_sand.png', '(0.3, 0.2, 0.1)', 'False', '0', '0.5', '"SHOVEL"', 88, 0),
    ("GLOWSTONE", 65, 'glowstone.png', '(0.8, 0.8, 0.4)', 'False', '15', '0.3', '"NONE"', 89, 0),
    ("STONEBRICK", 66, 'stonebrick.png', '(0.5, 0.5, 0.5)', 'False', '0', '1.5', '"PICKAXE"', 98, 0),
    ("STONEBRICK_MOSSY", 67, 'stonebrick_mossy.png', '(0.4, 0.5, 0.4)', 'False', '0', '1.5', '"PICKAXE"', 98, 1),
    ("STONEBRICK_CRACKED", 68, 'stonebrick_cracked.png', '(0.5, 0.5, 0.5)', 'False', '0', '1.5', '"PICKAXE"', 98, 2),
    ("STONEBRICK_CARVED", 69, 'stonebrick_carved.png', '(0.5, 0.5, 0.5)', 'False', '0', '1.5', '"PICKAXE"', 98, 3),
    ("MELON_BLOCK", 70, '{"top": "melon_top.png", "bottom": "melon_top.png", "side": "melon_side.png"}', '(0.3, 0.6, 0.2)', 'False', '0', '1.0', '"AXE"', 103, 0),
    ("NETHER_BRICK", 71, 'nether_brick.png', '(0.3, 0.1, 0.1)', 'False', '0', '2.0', '"PICKAXE"', 112, 0),
    ("END_STONE", 72, 'end_stone.png', '(0.8, 0.8, 0.6)', 'False', '0', '3.0', '"PICKAXE"', 121, 0),
    ("EMERALD_BLOCK_SOLID", 74, 'emerald_block.png', '(0.2, 0.8, 0.3)', 'False', '0', '5.0', '"PICKAXE"', 133, 0),
    ("REDSTONE_BLOCK", 75, 'redstone_block.png', '(0.8, 0.1, 0.1)', 'False', '0', '5.0', '"PICKAXE"', 152, 0),
    ("QUARTZ_BLOCK", 76, '{"top": "quartz_block_top.png", "bottom": "quartz_block_bottom.png", "side": "quartz_block_side.png"}', '(0.9, 0.9, 0.9)', 'False', '0', '0.8', '"PICKAXE"', 155, 0),
    ("QUARTZ_CHISELED", 77, '{"top": "quartz_block_chiseled_top.png", "bottom": "quartz_block_chiseled_top.png", "side": "quartz_block_chiseled.png"}', '(0.9, 0.9, 0.9)', 'False', '0', '0.8', '"PICKAXE"', 155, 1),
    ("QUARTZ_PILLAR", 78, '{"top": "quartz_block_lines_top.png", "bottom": "quartz_block_lines_top.png", "side": "quartz_block_lines.png"}', '(0.9, 0.9, 0.9)', 'False', '0', '0.8', '"PICKAXE"', 155, 2),
    ("COAL_BLOCK_SOLID", 80, 'coal_block.png', '(0.1, 0.1, 0.1)', 'False', '0', '5.0', '"PICKAXE"', 173, 0),
    ("PACKED_ICE", 81, 'ice_packed.png', '(0.6, 0.8, 1.0)', 'False', '0', '0.5', '"PICKAXE"', 174, 0),
    ("RED_SAND", 82, 'red_sand.png', '(0.8, 0.4, 0.1)', 'False', '0', '0.5', '"SHOVEL"', 12, 1),
    ("PODZOL", 83, '{"top": "dirt_podzol_top.png", "bottom": "dirt.png", "side": "dirt_podzol_side.png"}', '(0.4, 0.3, 0.2)', 'False', '0', '0.5', '"SHOVEL"', 3, 2),
    ("COARSE_DIRT", 84, 'coarse_dirt.png', '(0.4, 0.3, 0.2)', 'False', '0', '0.5', '"SHOVEL"', 3, 1),
    ("ANDESITE", 85, 'stone_andesite.png', '(0.5, 0.5, 0.5)', 'False', '0', '1.5', '"PICKAXE"', 1, 5),
    ("ANDESITE_POLISHED", 86, 'stone_andesite_smooth.png', '(0.5, 0.5, 0.5)', 'False', '0', '1.5', '"PICKAXE"', 1, 6),
    ("DIORITE", 87, 'stone_diorite.png', '(0.8, 0.8, 0.8)', 'False', '0', '1.5', '"PICKAXE"', 1, 3),
    ("DIORITE_POLISHED", 88, 'stone_diorite_smooth.png', '(0.8, 0.8, 0.8)', 'False', '0', '1.5', '"PICKAXE"', 1, 4),
    ("GRANITE", 89, 'stone_granite.png', '(0.6, 0.4, 0.4)', 'False', '0', '1.5', '"PICKAXE"', 1, 1),
    ("GRANITE_POLISHED", 90, 'stone_granite_smooth.png', '(0.6, 0.4, 0.4)', 'False', '0', '1.5', '"PICKAXE"', 1, 2),
    ("PRISMARINE", 91, 'prismarine_rough.png', '(0.3, 0.6, 0.6)', 'False', '0', '1.5', '"PICKAXE"', 168, 0),
    ("PRISMARINE_BRICKS", 92, 'prismarine_bricks.png', '(0.3, 0.6, 0.6)', 'False', '0', '1.5', '"PICKAXE"', 168, 1),
    ("PRISMARINE_DARK", 93, 'prismarine_dark.png', '(0.2, 0.4, 0.4)', 'False', '0', '1.5', '"PICKAXE"', 168, 2),
    ("SEA_LANTERN", 94, 'sea_lantern.png', '(0.8, 0.9, 0.9)', 'False', '15', '0.3', '"NONE"', 169, 0),
    ("SLIME_BLOCK", 95, 'slime.png', '(0.4, 0.8, 0.4)', 'True', '0', '0.0', '"NONE"', 165, 0),
    ("HAY_BLOCK", 96, '{"top": "hay_block_top.png", "bottom": "hay_block_top.png", "side": "hay_block_side.png"}', '(0.8, 0.8, 0.2)', 'False', '0', '0.5', '"NONE"', 170, 0),
]

COLORS = ["white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray", "silver", "cyan", "purple", "blue", "brown", "green", "red", "black"]
for i, color in enumerate(COLORS):
    # Wool
    NEW_BLOCKS.append((f"WOOL_{color.upper()}", 100 + i, f"wool_colored_{color}.png", '(0.8, 0.8, 0.8)', 'False', '0', '0.8', '"NONE"', 35, i))
    # Glass
    NEW_BLOCKS.append((f"GLASS_{color.upper()}", 120 + i, f"glass_{color}.png", '(0.8, 0.8, 0.8)', 'True', '0', '0.3', '"NONE"', 95, i))
    # Stained Clay
    NEW_BLOCKS.append((f"STAINED_CLAY_{color.upper()}", 140 + i, f"hardened_clay_stained_{color}.png", '(0.6, 0.4, 0.4)', 'False', '0', '1.25', '"PICKAXE"', 159, i))

NEW_BLOCKS.append(("HARDENED_CLAY", 160, 'hardened_clay.png', '(0.6, 0.4, 0.4)', 'False', '0', '1.25', '"PICKAXE"', 172, 0))

import os

def patch_terrain():
    with open("world/terrain.py", "r", encoding='utf-8') as f:
        content = f.read()
    
    # 1. Add to BLOCK_REGISTRY
    registry_end = content.find("}") # Find end of BLOCK_REGISTRY dict
    # actually, PORKCHOP_RAW is the last
    porkchop_idx = content.find('"PORKCHOP_RAW"')
    if porkchop_idx != -1:
        insert_idx = content.find('}', porkchop_idx)
        
        insert_str = ""
        for b in NEW_BLOCKS:
            name, bid, tex, col, trans, light, hard, tool, mc_id, mc_meta = b
            if tex.startswith('{'):
                tex_val = tex
            else:
                tex_val = f'"{tex}"'
            insert_str += f'    "{name}": {{"id": {bid}, "texture": {tex_val}, "color": {col}, "transparent": {trans}, "light": {light}, "hardness": {hard}, "tool_type": {tool}}},\n'
        
        content = content[:insert_idx] + insert_str + content[insert_idx:]
    
    # 2. Add globals
    globals_idx = content.find("SNOW = 6")
    if globals_idx != -1:
        insert_idx = content.find('\n', globals_idx) + 1
        insert_str = ""
        for b in NEW_BLOCKS:
            insert_str += f'{b[0]} = {b[1]}\n'
        content = content[:insert_idx] + insert_str + content[insert_idx:]
        
    with open("world/terrain.py", "w", encoding='utf-8') as f:
        f.write(content)

def patch_world_db():
    with open("core/world_db.py", "r", encoding='utf-8') as f:
        content = f.read()
    
    # 1. Add to _mapping
    mapping_idx = content.find("178: (175, 10),# DOUBLE_ROSE_TOP")
    if mapping_idx != -1:
        insert_idx = content.find('\n', mapping_idx) + 1
        insert_str = ""
        for b in NEW_BLOCKS:
            insert_str += f'    {b[1]}: ({b[8]}, {b[9]}), # {b[0]}\n'
        content = content[:insert_idx] + insert_str + content[insert_idx:]
        
    with open("core/world_db.py", "w", encoding='utf-8') as f:
        f.write(content)

patch_terrain()
patch_world_db()
print("Done patching.")
