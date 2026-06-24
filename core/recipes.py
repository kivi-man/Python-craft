import json

class ShapedRecipe:
    def __init__(self, pattern, result_id, result_count):
        self.pattern = pattern
        self.height = len(pattern)
        self.width = max(len(row) for row in pattern) if self.height > 0 else 0
        self.result_id = result_id
        self.result_count = result_count

    def match(self, grid, grid_width, grid_height):
        # Extract bounding box of non-empty items from the grid
        min_x, max_x = grid_width, -1
        min_y, max_y = grid_height, -1
        
        for y in range(grid_height):
            for x in range(grid_width):
                if grid[y][x] > 0:
                    if x < min_x: min_x = x
                    if x > max_x: max_x = x
                    if y < min_y: min_y = y
                    if y > max_y: max_y = y
                    
        if min_x > max_x: # Empty grid
            return False
            
        bb_width = max_x - min_x + 1
        bb_height = max_y - min_y + 1
        
        if bb_width != self.width or bb_height != self.height:
            return False
            
        # Compare bounding box against recipe pattern
        for y in range(self.height):
            for x in range(self.width):
                grid_item = grid[min_y + y][min_x + x]
                recipe_item = self.pattern[y][x]
                if grid_item not in recipe_item:
                    return False
        return True

class ShapelessRecipe:
    def __init__(self, ingredients, result_id, result_count):
        self.ingredients = ingredients
        self.result_id = result_id
        self.result_count = result_count
        
    def match(self, grid, grid_width, grid_height):
        items = []
        for y in range(grid_height):
            for x in range(grid_width):
                if grid[y][x] > 0:
                    items.append(grid[y][x])
        if len(items) != len(self.ingredients):
            return False
            
        used = [False] * len(items)
        for ing in self.ingredients:
            found = False
            for i, it in enumerate(items):
                if not used[i] and it in ing:
                    used[i] = True
                    found = True
                    break
            if not found:
                return False
        return True

MC_TO_PC = {
    1: 1,      # Stone
    (1, 1): 89, # Granite
    (1, 2): 90, # Polished Granite
    (1, 3): 87, # Diorite
    (1, 4): 88, # Polished Diorite
    (1, 5): 85, # Andesite
    (1, 6): 86, # Polished Andesite
    
    2: 3,      # Grass
    3: 2,      # Dirt
    (3, 1): 84, # Coarse Dirt
    (3, 2): 83, # Podzol
    
    4: 43,     # Cobblestone
    
    5: 44,        # Planks (fallback Oak)
    (5, 0): 44,   # Oak Planks
    (5, 1): 45,   # Spruce Planks
    (5, 2): 46,   # Birch Planks
    (5, 3): 47,   # Jungle Planks
    (5, 4): 48,   # Acacia Planks
    (5, 5): 49,   # Dark Oak Planks
    
    12: 5,     # Sand
    (12, 1): 82, # Red Sand
    
    13: 8,     # Gravel
    14: 40,    # Gold Ore
    15: 41,    # Iron Ore
    16: 42,    # Coal Ore
    
    17: 11,       # Wood (fallback Oak)
    (17, 0): 11,  # Oak Log
    (17, 1): 15,  # Spruce Log
    (17, 2): 14,  # Birch Log
    (17, 3): 11,  # Jungle Log (Mapped to Oak for now)
    
    162: 11,      # Wood2 (fallback Oak)
    (162, 0): 11, # Acacia Log
    (162, 1): 11, # Dark Oak Log
    
    18: 12,       # Leaves (fallback Oak)
    (18, 0): 12,  # Oak Leaves
    (18, 1): 17,  # Spruce Leaves
    (18, 2): 16,  # Birch Leaves
    
    19: 50,    # Sponge
    20: 20,    # Glass
    21: 21,    # Lapis Ore
    22: 51,    # Lapis Block
    24: 9,     # Sandstone
    41: 52,    # Gold Block
    42: 53,    # Iron Block
    
    44: 203,      # Stone Slabs
    (44, 0): 203, # Stone Slab
    (44, 1): 222, # Sandstone Slab
    (44, 2): 202, # Wooden Slab (Old)
    (44, 3): 224, # Cobblestone Slab
    (44, 4): 219, # Brick Slab
    (44, 5): 220, # Stone Brick Slab
    (44, 6): 221, # Nether Brick Slab
    (44, 7): 223, # Quartz Slab
    
    45: 54,    # Bricks
    46: 55,    # TNT
    47: 57,    # Bookshelf
    48: 58,    # Mossy Cobble
    49: 59,    # Obsidian
    
    53: 200,   # Oak Stairs
    
    56: 56,    # Diamond Ore
    57: 60,    # Diamond Block
    58: 116,   # Crafting Table
    
    67: 201,   # Cobblestone Stairs
    
    82: 61,    # Clay
    84: 62,    # Jukebox
    87: 63,    # Netherrack
    88: 64,    # Soul Sand
    89: 65,    # Glowstone
    98: 66,    # Stone Bricks
    (98, 1): 67, # Mossy Stone Bricks
    (98, 2): 68, # Cracked Stone Bricks
    (98, 3): 69, # Chiseled Stone Bricks
    
    103: 70,   # Melon
    108: 209,  # Brick Stairs
    109: 210,  # Stone Brick Stairs
    112: 71,   # Nether Brick
    114: 211,  # Nether Brick Stairs
    121: 72,   # End Stone
    
    126: 202,      # Wooden Slabs
    (126, 0): 202, # Oak Slab
    (126, 1): 214, # Spruce Slab
    (126, 2): 215, # Birch Slab
    (126, 3): 216, # Jungle Slab
    (126, 4): 217, # Acacia Slab
    (126, 5): 218, # Dark Oak Slab
    
    128: 212,  # Sandstone Stairs
    129: 129,  # Emerald Ore
    133: 74,   # Emerald Block
    
    134: 204,  # Spruce Stairs
    135: 205,  # Birch Stairs
    136: 206,  # Jungle Stairs
    152: 75,   # Redstone Block
    155: 76,   # Quartz Block
    (155, 1): 77, # Chiseled Quartz
    (155, 2): 78, # Pillar Quartz
    
    156: 213,  # Quartz Stairs
    163: 207,  # Acacia Stairs
    164: 208,  # Dark Oak Stairs
    
    165: 95,   # Slime Block
    168: 91,   # Prismarine
    (168, 1): 92, # Prismarine Bricks
    (168, 2): 93, # Dark Prismarine
    169: 94,   # Sea Lantern
    
    # --- ITEMS & TOOLS ---
    256: 1020, # Iron Shovel
    257: 1021, # Iron Pickaxe
    258: 1022, # Iron Axe
    
    263: 1005, # Coal
    (263, 1): 1006, # Charcoal
    264: 1002, # Diamond
    265: 1003, # Iron Ingot
    266: 1004, # Gold Ingot
    267: 1019, # Iron Sword
    268: 1009, # Wooden Sword
    269: 1010, # Wooden Shovel
    270: 1011, # Wooden Pickaxe
    271: 1012, # Wooden Axe
    272: 1014, # Stone Sword
    273: 1015, # Stone Shovel
    274: 1016, # Stone Pickaxe
    275: 1017, # Stone Axe
    276: 1024, # Diamond Sword
    277: 1025, # Diamond Shovel
    278: 1026, # Diamond Pickaxe
    279: 1027, # Diamond Axe
    280: 1001, # Stick
    281: 1007, # Bowl
    282: 1008, # Mushroom Stew
    283: 1029, # Gold Sword
    284: 1030, # Gold Shovel
    285: 1031, # Gold Pickaxe
    286: 1032, # Gold Axe
    
    290: 1013, # Wooden Hoe
    291: 1018, # Stone Hoe
    292: 1023, # Iron Hoe
    293: 1028, # Diamond Hoe
    294: 1033, # Gold Hoe
    
    324: 162,  # Wood Door Item -> Wood Door Block
    330: 163,  # Iron Door Item -> Iron Door Block
    
    173: 80,   # Coal Block
    174: 81,   # Packed Ice
}

class RecipeManager:
    def __init__(self, path):
        self.recipes = []
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                self._parse(data)
        except Exception as e:
            print(f"Failed to load recipes from {path}: {e}")

    def _extract_id(self, item):
        mc_id = 0
        mc_meta = None
        if isinstance(item, int):
            mc_id = item
        elif isinstance(item, dict):
            mc_id = item.get("id", 0)
            if "metadata" in item:
                mc_meta = item.get("metadata")
        
        allowed = set()
        
        if mc_meta is not None:
            pc_id = MC_TO_PC.get((mc_id, mc_meta), MC_TO_PC.get(mc_id, 0))
            if pc_id > 0:
                allowed.add(pc_id)
        else:
            if mc_id in MC_TO_PC:
                allowed.add(MC_TO_PC[mc_id])
            for k, v in MC_TO_PC.items():
                if isinstance(k, tuple) and k[0] == mc_id:
                    allowed.add(v)
                    
        # Planks equivalency: if Oak Planks (44) are allowed, all planks are allowed
        if 44 in allowed:
            allowed.update((44, 45, 46, 47, 48, 49))
            
        # Coal equivalency: if Coal (1005) is allowed, Charcoal (1006) is also allowed
        if 1005 in allowed:
            allowed.add(1006)
            
        if not allowed:
            return (0,)
        return tuple(allowed)

    def _parse(self, data):
        for res_id_str, recipe_list in data.items():
            mc_res_id = int(res_id_str)
            
            for r in recipe_list:
                result_count = 1
                pc_res_id = MC_TO_PC.get(mc_res_id, 0)
                
                if "result" in r:
                    res_obj = r["result"]
                    if isinstance(res_obj, dict):
                        result_count = res_obj.get("count", 1)
                        if "id" in res_obj:
                            r_id = res_obj["id"]
                            r_meta = res_obj.get("metadata", 0)
                            if (r_id, r_meta) in MC_TO_PC:
                                pc_res_id = MC_TO_PC[(r_id, r_meta)]
                            else:
                                pc_res_id = MC_TO_PC.get(r_id, 0)
                            
                # If the result block doesn't exist in Pythoncraft, skip this recipe!
                if pc_res_id == 0:
                    continue
                            
                if "inShape" in r:
                    pattern = []
                    valid = True
                    for row in r["inShape"]:
                        pattern_row = []
                        for item in row:
                            mapped_id = self._extract_id(item)
                            # If an ingredient cannot be mapped, the recipe is invalid
                            if 0 in mapped_id and item is not None and str(item) != "0" and str(item) != "{}":
                                valid = False
                            pattern_row.append(mapped_id)
                        pattern.append(pattern_row)
                    if valid:
                        self.recipes.append(ShapedRecipe(pattern, pc_res_id, result_count))
                    
                elif "ingredients" in r:
                    ingredients = []
                    valid = True
                    for item in r["ingredients"]:
                        mapped_id = self._extract_id(item)
                        if 0 in mapped_id:
                            valid = False
                            break
                        ingredients.append(mapped_id)
                    if valid:
                        self.recipes.append(ShapelessRecipe(ingredients, pc_res_id, result_count))

    def match(self, grid, grid_width, grid_height):
        # grid is a 2D list: grid[y][x] = block_id
        for recipe in self.recipes:
            # Check grid size constraints
            if isinstance(recipe, ShapedRecipe):
                if recipe.width > grid_width or recipe.height > grid_height:
                    continue
            elif isinstance(recipe, ShapelessRecipe):
                if len(recipe.ingredients) > grid_width * grid_height:
                    continue
                    
            if recipe.match(grid, grid_width, grid_height):
                return recipe.result_id, recipe.result_count
                
        return 0, 0
