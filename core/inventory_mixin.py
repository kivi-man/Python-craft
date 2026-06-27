from pyglet.window import mouse
from world.terrain import BLOCK_MAX_STACK_ARRAY

class InventoryMixin:
    def _handle_inventory_click(self, x, y, button):
        
        clicked_slot = -1
        for i in range(55):
            sx, sy, sw, sh = self._get_slot_rect(i)
            if sx <= x <= sx + sw and sy <= y <= sy + sh:
                if getattr(self, 'crafting_open', False) and 40 <= i <= 44:
                    continue
                if getattr(self, 'inventory_open', False) and 45 <= i <= 54:
                    continue
                clicked_slot = i
                break
                
        if clicked_slot == -1:
            if button == mouse.LEFT and getattr(self, 'cursor_item_id', 0) > 0:
                for _ in range(self.cursor_item_count):
                    self.spawn_item_entity(self.cursor_item_id, self.player.x, self.player.y + 1.5, self.player.z)
                self.cursor_item_id = 0
                self.cursor_item_count = 0
            return
            
        if 36 <= clicked_slot <= 39:
            if getattr(self, 'cursor_item_id', 0) > 0:
                return
                
        if clicked_slot in (44, 54):
            slot_id = self.inventory_blocks[clicked_slot]
            if slot_id == 0: return
            
            cursor_id = getattr(self, 'cursor_item_id', 0)
            cursor_count = getattr(self, 'cursor_item_count', 0)
            slot_count = self.inventory_counts[clicked_slot]
            
            max_stack = BLOCK_MAX_STACK_ARRAY[slot_id] if slot_id > 0 else 64
            
            if cursor_id == 0 or (cursor_id == slot_id and cursor_count + slot_count <= max_stack):
                self.cursor_item_id = slot_id
                self.cursor_item_count = cursor_count + slot_count
                self.inventory_blocks[clicked_slot] = 0
                self.inventory_counts[clicked_slot] = 0
                
                # Consume grid
                grid_slots = range(45, 54) if clicked_slot == 54 else range(40, 44)
                for i in grid_slots:
                    if self.inventory_counts[i] > 0:
                        self.inventory_counts[i] -= 1
                        if self.inventory_counts[i] == 0:
                            self.inventory_blocks[i] = 0
                self._evaluate_crafting()
                
                if hasattr(self, 'sound_system'):
                    self.sound_system.play("eSoundType_RANDOM_CLICK", volume=0.5)
            return
                
        slot_id = self.inventory_blocks[clicked_slot]
        slot_count = self.inventory_counts[clicked_slot]
        
        # Play UI click sound
        if hasattr(self, 'sound_system'):
            self.sound_system.play("eSoundType_RANDOM_CLICK", volume=0.5)
        
        cursor_id = getattr(self, 'cursor_item_id', 0)
        cursor_count = getattr(self, 'cursor_item_count', 0)
        
        if button == mouse.LEFT:
            if cursor_id == 0:
                if slot_id > 0:
                    self.cursor_item_id = slot_id
                    self.cursor_item_count = slot_count
                    self.inventory_blocks[clicked_slot] = 0
                    self.inventory_counts[clicked_slot] = 0
            else:
                if slot_id == cursor_id:
                    max_stack = BLOCK_MAX_STACK_ARRAY[slot_id]
                    space = max_stack - slot_count
                    if space >= cursor_count:
                        self.inventory_counts[clicked_slot] += cursor_count
                        self.cursor_item_id = 0
                        self.cursor_item_count = 0
                    else:
                        self.inventory_counts[clicked_slot] = max_stack
                        self.cursor_item_count -= space
                else:
                    self.inventory_blocks[clicked_slot] = cursor_id
                    self.inventory_counts[clicked_slot] = cursor_count
                    self.cursor_item_id = slot_id
                    self.cursor_item_count = slot_count
        elif button == mouse.RIGHT:
            if cursor_id == 0:
                if slot_id > 0:
                    half = int(slot_count / 2)
                    rem = slot_count - half
                    if half > 0:
                        self.cursor_item_id = slot_id
                        self.cursor_item_count = half
                        self.inventory_counts[clicked_slot] = rem
                    else:
                        self.cursor_item_id = slot_id
                        self.cursor_item_count = 1
                        self.inventory_counts[clicked_slot] = 0
                        self.inventory_blocks[clicked_slot] = 0
            else:
                if slot_id == 0:
                    self.inventory_blocks[clicked_slot] = cursor_id
                    self.inventory_counts[clicked_slot] = 1
                    self.cursor_item_count -= 1
                    if self.cursor_item_count <= 0:
                        self.cursor_item_id = 0
                elif slot_id == cursor_id:
                    max_stack = BLOCK_MAX_STACK_ARRAY[slot_id]
                    if slot_count < max_stack:
                        self.inventory_counts[clicked_slot] += 1
                        self.cursor_item_count -= 1
                        if self.cursor_item_count <= 0:
                            self.cursor_item_id = 0
                else:
                    self.inventory_blocks[clicked_slot] = cursor_id
                    self.inventory_counts[clicked_slot] = cursor_count
                    self.cursor_item_id = slot_id
                    self.cursor_item_count = slot_count
                    
        # Synchronize selected block if hotbar is changed
        self.selected_block_id = self.inventory_blocks[self.selected_slot]
        self._evaluate_crafting()

    def _evaluate_crafting(self):
        if getattr(self, 'crafting_open', False):
            grid_width, grid_height = 3, 3
            start_slot = 45
            out_slot = 54
        elif getattr(self, 'inventory_open', False):
            grid_width, grid_height = 2, 2
            start_slot = 40
            out_slot = 44
        else:
            return
            
        grid = []
        for y in range(grid_height):
            row = []
            for x in range(grid_width):
                idx = start_slot + y * grid_width + x
                row.append(self.inventory_blocks[idx])
            grid.append(row)
            
        res_id, res_count = self.recipe_manager.match(grid, grid_width, grid_height)
        if res_id > 0:
            self.inventory_blocks[out_slot] = res_id
            self.inventory_counts[out_slot] = res_count
        else:
            self.inventory_blocks[out_slot] = 0
            self.inventory_counts[out_slot] = 0

    def _drop_inventory_excess(self):
        # Drop cursor item
        if getattr(self, 'cursor_item_count', 0) > 0 and getattr(self, 'cursor_item_id', 0) > 0:
            for _ in range(self.cursor_item_count):
                self.spawn_item_entity(self.cursor_item_id, self.player.x, self.player.y + 1.5, self.player.z)
            self.cursor_item_count = 0
            self.cursor_item_id = 0
            
        # Drop crafting grid items (slots 40, 41, 42, 43)
        for i in range(40, 44):
            if self.inventory_counts[i] > 0 and self.inventory_blocks[i] > 0:
                for _ in range(self.inventory_counts[i]):
                    self.spawn_item_entity(self.inventory_blocks[i], self.player.x, self.player.y + 1.5, self.player.z)
                self.inventory_counts[i] = 0
                self.inventory_blocks[i] = 0

        # Drop large crafting grid items (slots 45-53)
        for i in range(45, 54):
            if self.inventory_counts[i] > 0 and self.inventory_blocks[i] > 0:
                for _ in range(self.inventory_counts[i]):
                    self.spawn_item_entity(self.inventory_blocks[i], self.player.x, self.player.y + 1.5, self.player.z)
                self.inventory_counts[i] = 0
                self.inventory_blocks[i] = 0
