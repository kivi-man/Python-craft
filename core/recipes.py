import json
import os
from world.mc_id_converter import MC_TO_INTERNAL, INTERNAL_NAMES_MAP

class ShapedRecipe:
    def __init__(self, pattern, result_id, result_count):
        self.pattern = pattern
        self.height = len(pattern)
        self.width = len(pattern[0]) if self.height > 0 else 0
        self.result_id = result_id
        self.result_count = result_count

    def match(self, grid, grid_width, grid_height):
        # grid is 2D list of item IDs
        for y_offset in range(grid_height - self.height + 1):
            for x_offset in range(grid_width - self.width + 1):
                if self._check_match(grid, grid_width, grid_height, x_offset, y_offset, False):
                    return True
                if self._check_match(grid, grid_width, grid_height, x_offset, y_offset, True):
                    return True
        return False

    def _check_match(self, grid, grid_width, grid_height, x_offset, y_offset, mirror):
        for y in range(grid_height):
            for x in range(grid_width):
                py = y - y_offset
                px = x - x_offset
                if mirror:
                    px = self.width - 1 - px
                    
                target = set([-1])
                if 0 <= py < self.height and 0 <= px < self.width:
                    target = self.pattern[py][px]
                    
                grid_item = grid[y][x]
                if grid_item == 0:
                    if target != set([-1]):
                        return False
                else:
                    if target == set([-1]) or grid_item not in target:
                        return False
        return True

class ShapelessRecipe:
    def __init__(self, ingredients, result_id, result_count):
        self.ingredients = ingredients
        self.result_id = result_id
        self.result_count = result_count

    def match(self, grid, grid_width, grid_height):
        # Flatten grid
        items = []
        for row in grid:
            for item in row:
                if item != 0:
                    items.append(item)
                    
        if len(items) != len(self.ingredients):
            return False
            
        used = [False] * len(items)
        for target_set in self.ingredients:
            found = False
            for i, item in enumerate(items):
                if not used[i] and item in target_set:
                    used[i] = True
                    found = True
                    break
            if not found:
                return False
                
        return True

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
        
        if mc_id >= 256: # Item
            allowed.add(mc_id)
        else:
            if mc_meta is not None:
                internal_id = MC_TO_INTERNAL[mc_id, mc_meta]
                if internal_id > 0:
                    allowed.add(internal_id)
            else:
                for m in range(16):
                    internal_id = int(MC_TO_INTERNAL[mc_id, m])
                    if internal_id > 0:
                        allowed.add(internal_id)
                    
        # Planks equivalency
        if INTERNAL_NAMES_MAP.get("PLANKS_OAK", 5) in allowed:
            for name in ["PLANKS_OAK", "PLANKS_SPRUCE", "PLANKS_BIRCH", "PLANKS_JUNGLE", "PLANKS_ACACIA", "PLANKS_DARK_OAK"]:
                if name in INTERNAL_NAMES_MAP:
                    allowed.add(INTERNAL_NAMES_MAP[name])
            
        # Coal equivalency
        if 263 in allowed:
            allowed.add(263)
            
        if not allowed:
            return set([-1])
            
        return allowed

    def _parse(self, data):
        for res_id_str, recipe_list in data.items():
            mc_res_id = int(res_id_str)
            
            for r in recipe_list:
                result_count = 1
                pc_res_id = int(MC_TO_INTERNAL[mc_res_id, 0]) if mc_res_id < 256 else mc_res_id
                
                if "result" in r:
                    res_obj = r["result"]
                    if isinstance(res_obj, dict):
                        result_count = res_obj.get("count", 1)
                        if "id" in res_obj:
                            r_id = res_obj["id"]
                            r_meta = res_obj.get("metadata", 0)
                            if r_id < 256:
                                pc_res_id = int(MC_TO_INTERNAL[r_id, r_meta])
                            else:
                                pc_res_id = r_id
                            
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
