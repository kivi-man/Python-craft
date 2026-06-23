import os
import re

with open("main.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
imports_to_add = set()

# Regex to match indented imports
import_re = re.compile(r'^[ \t]+(import [a-zA-Z0-9_\.]+)[ \t]*$')
from_import_re = re.compile(r'^[ \t]+(from [a-zA-Z0-9_\.]+ import [a-zA-Z0-9_\., ]+)[ \t]*$')

# Exception for pyglet.window import mouse, we don't want to conflict if it's already there or something, but let's just collect all.
for line in lines:
    m1 = import_re.match(line)
    m2 = from_import_re.match(line)
    
    if m1:
        imports_to_add.add(m1.group(1))
        # Keep empty line if needed or just skip it
        continue
    elif m2:
        imports_to_add.add(m2.group(1))
        continue
    else:
        new_lines.append(line)

# Now add collected imports right after the initial block of imports.
# Let's find the end of the first import block.
insert_idx = 0
for i, line in enumerate(new_lines):
    if line.startswith("from renderer.camera import Camera"):
        insert_idx = i + 1
        break

sorted_imports = sorted(list(imports_to_add))
import_lines = [imp + "\n" for imp in sorted_imports]

final_lines = new_lines[:insert_idx] + ["\n# --- MOVED IMPORTS ---\n"] + import_lines + ["# ---------------------\n\n"] + new_lines[insert_idx:]

with open("main.py", "w", encoding="utf-8") as f:
    f.writelines(final_lines)
print("Finished moving imports in main.py")
