import pyglet
import cProfile
import pstats
from main import main

def exit_app(dt):
    print("Exiting app for profile...")
    pyglet.app.exit()

def run_profile():
    import sys
    sys.argv = ['main.py']
    
    pyglet.clock.schedule_once(exit_app, 10.0)
    
    cProfile.run('main()', 'profile.stats')
    
    p = pstats.Stats('profile.stats')
    print("\n--- Top 30 by Cumulative Time ---")
    p.sort_stats('cumulative').print_stats(30)
    print("\n--- Top 30 by Internal Time (tottime) ---")
    p.sort_stats('tottime').print_stats(30)

if __name__ == '__main__':
    run_profile()
