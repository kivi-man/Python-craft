import sys

content = open(r'c:\Users\ecrin\OneDrive\Desktop\pythoncraft\core\sound_system.py', encoding='utf-8').read()

# The erroneous block starts with '"eSoundType_RANDOM_BOW": "random.bow",' and ends with '}' right before def play
start_str = '"eSoundType_RANDOM_BOW": "random.bow",'
end_str = '"eSoundType_ITEM_ELYTRA_FLYING": "item.elytra.flying",\n        }'

# We only want to remove the SECOND occurrence of this block, which is incorrectly placed.
# Wait, let's just find the first occurrence AFTER 'def play'
play_idx = content.find('def play(self, sound_enum_name')

start_idx = content.find(start_str, play_idx)
end_idx = content.find(end_str, start_idx)

if start_idx != -1 and end_idx != -1:
    end_idx += len(end_str)
    
    # We replace the bad block with the proper except block
    restore_str = """                    
                self.active_sounds.append({'player': player, 'file_path': file_path})
            except Exception as e:
                print(f"Error playing sound {sound_enum_name} ({file_path}): {e}")
"""
    
    fixed_content = content[:start_idx] + restore_str + content[end_idx:]
    open(r'c:\Users\ecrin\OneDrive\Desktop\pythoncraft\core\sound_system.py', 'w', encoding='utf-8').write(fixed_content)
    print("Fixed syntax error")
else:
    print("Could not find the block to remove")
