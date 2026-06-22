import sys

content = open(r'c:\Users\ecrin\OneDrive\Desktop\pythoncraft\main.py', encoding='utf-8').read()

bad_str = """        # Update Sound System (Music and ambiance)
        if hasattr(self, 'sound_system'):
            self.sound_system.update_music(dt, dimension="OVERWORLD")
            
"""

if bad_str in content:
    content = content.replace(bad_str, '')
    
    # Now insert it at the very top of update(self, dt)
    update_str = '    def update(self, dt):'
    insert_str = """    def update(self, dt):
        # Update Sound System (Music and ambiance)
        if hasattr(self, 'sound_system'):
            self.sound_system.update_music(dt, dimension="OVERWORLD")
"""
    content = content.replace(update_str, insert_str)
    
    open(r'c:\Users\ecrin\OneDrive\Desktop\pythoncraft\main.py', 'w', encoding='utf-8').write(content)
    print('Fixed main.py')
else:
    print('Could not find bad string')
