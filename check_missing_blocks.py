import re

terrain_ids = set()
with open('world/terrain.py', 'r', encoding='utf-8') as f:
    for line in f:
        m = re.search(r'"id":\s*(\d+)', line)
        if m:
            terrain_ids.add(int(m.group(1)))

db_ids = set()
with open('core/world_db.py', 'r', encoding='utf-8') as f:
    for line in f:
        m = re.search(r'^\s*(\d+):\s*\(', line)
        if m:
            db_ids.add(int(m.group(1)))

missing = terrain_ids - db_ids
print("Missing:", sorted(list(missing)))
