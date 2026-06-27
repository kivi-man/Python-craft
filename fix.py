import os

with open('main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

out_lines = []
for line in lines:
    if line.startswith('def main():'):
        break
    out_lines.append(line)

main_block = """def main():
    print("=============================================")
    print("      PYTHONCRAFT ENGINE INITIATING...       ")
    print("=============================================")
    
    distance = 4
    sim_distance = 4
    fast_leaves = False
    debug_mode = False
    flat_mode = False
    console_mode = False
    force_legacy = False
    
    import sys
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "-fast":
                fast_leaves = True
                print("Fast mode enabled: Leaves are now opaque.")
            elif arg == "-debug":
                debug_mode = True
                print("Debug mode enabled: Performance metrics will be printed to console.")
            elif arg == "-flat":
                flat_mode = True
                print("Flat mode enabled: World will be generated as flat.")
            elif arg == "-sim" and i + 1 < len(args):
                try:
                    sim_distance = int(args[i+1])
                    print(f"Simulation Distance set to: {sim_distance}")
                except ValueError:
                    pass
                i += 1
            elif arg == "-console":
                console_mode = True
                print("Console mode enabled.")
            elif arg == "-legacy":
                force_legacy = True
                print("Legacy mode forced via command line.")
            else:
                try:
                    distance = int(arg)
                    print(f"User requested Render Distance: {distance} ({distance*2}x{distance*2} = {distance*distance*4} chunks)")
                except ValueError:
                    pass
            i += 1
                    
    from world.terrain import BLOCK_OPAQUE_ARRAY
    if fast_leaves:
        BLOCK_OPAQUE_ARRAY[12] = True
        BLOCK_OPAQUE_ARRAY[16] = True
        BLOCK_OPAQUE_ARRAY[17] = True
        
    gpu_mode = False
    config = None
    import pyglet
    if force_legacy:
        gpu_mode = False
        config = pyglet.gl.Config(depth_size=24, double_buffer=True)
        print("GPU_MODE [PASIF]: Force Legacy Mode aktif. OpenGL 3.3 kullanilacak.")
    else:
        try:
            pyglet.options['search_local_libs'] = True
            config_43 = pyglet.gl.Config(major_version=4, minor_version=3, forward_compatible=True, depth_size=24, double_buffer=True)
            test_win = pyglet.window.Window(width=1, height=1, config=config_43, visible=False)
            test_win.close()
            gpu_mode = True
            config = config_43
            print("GPU_MODE [AKTIF]: OpenGL 4.3 destekleniyor. Compute Shaders kullanilacak.")
        except Exception as e:
            gpu_mode = False
            config = pyglet.gl.Config(depth_size=24, double_buffer=True)
            print(f"GPU_MODE [PASIF]: OpenGL 4.3 desteklenmiyor. Legacy Mode kullanilacak. ({e})")
        
    engine = PythonCraftEngine(render_distance=distance, simulation_distance=sim_distance, fast_leaves=fast_leaves, debug_mode=debug_mode, flat_mode=flat_mode, console_mode=console_mode, config=config, gpu_mode=gpu_mode)
    pyglet.app.run()

if __name__ == '__main__':
    main()
"""

with open('main.py', 'w', encoding='utf-8') as f:
    f.write("".join(out_lines))
    f.write(main_block)

print("main.py fixed!")
