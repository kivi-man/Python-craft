import os
import pyglet

class SoundSystem:
    def __init__(self, sfx_dir):
        self.sfx_dir = sfx_dir
        self.sounds = {}
        self.active_sounds = [] # List of dicts: {'player': player, 'file_path': file_path}
        
        # C++ Engine Constants
        self.MAX_POLYPHONY = 64
        self.MAX_SAME_SOUNDS_PLAYING = 8
        
        # Music System Variables (C++ SoundEngine parity)
        import random
        # Initial delay 10 to 20 minutes (600 to 1200 seconds)
        self.music_delay_timer = random.uniform(600.0, 1200.0) 
        self.current_music_player = None
        self.current_music_dimension = None
        
        self.sound_names = {
            "eSoundType_MOB_CHICKEN_AMBIENT": "mob.chicken.say",
            "eSoundType_MOB_CHICKEN_HURT": "mob.chicken.hurt",
            "eSoundType_MOB_CHICKENPLOP": "mob.chicken.plop",
            "eSoundType_MOB_COW_AMBIENT": "mob.cow.say",
            "eSoundType_MOB_COW_HURT": "mob.cow.hurt",
            "eSoundType_MOB_PIG_AMBIENT": "mob.pig.say",
            "eSoundType_MOB_PIG_DEATH": "mob.pig.death",
            "eSoundType_MOB_SHEEP_AMBIENT": "mob.sheep.say",
            "eSoundType_MOB_WOLF_GROWL": "mob.wolf.growl",
            "eSoundType_MOB_WOLF_WHINE": "mob.wolf.whine",
            "eSoundType_MOB_WOLF_PANTING": "mob.wolf.panting",
            "eSoundType_MOB_WOLF_BARK": "mob.wolf.bark",
            "eSoundType_MOB_WOLF_HURT": "mob.wolf.hurt",
            "eSoundType_MOB_WOLF_DEATH": "mob.wolf.death",
            "eSoundType_MOB_WOLF_SHAKE": "mob.wolf.shake",
            "eSoundType_MOB_BLAZE_BREATHE": "mob.blaze.breathe",
            "eSoundType_MOB_BLAZE_HURT": "mob.blaze.hit",
            "eSoundType_MOB_BLAZE_DEATH": "mob.blaze.death",
            "eSoundType_MOB_GHAST_MOAN": "mob.ghast.moan",
            "eSoundType_MOB_GHAST_SCREAM": "mob.ghast.scream",
            "eSoundType_MOB_GHAST_DEATH": "mob.ghast.death",
            "eSoundType_MOB_GHAST_FIREBALL": "mob.ghast.fireball",
            "eSoundType_MOB_GHAST_CHARGE": "mob.ghast.charge",
            "eSoundType_MOB_ENDERMEN_IDLE": "mob.endermen.idle",
            "eSoundType_MOB_ENDERMEN_HIT": "mob.endermen.hit",
            "eSoundType_MOB_ENDERMEN_DEATH": "mob.endermen.death",
            "eSoundType_MOB_ENDERMEN_PORTAL": "mob.endermen.portal",
            "eSoundType_MOB_ZOMBIEPIG_AMBIENT": "mob.zombiepig.zpig",
            "eSoundType_MOB_ZOMBIEPIG_HURT": "mob.zombiepig.zpighurt",
            "eSoundType_MOB_ZOMBIEPIG_DEATH": "mob.zombiepig.zpigdeath",
            "eSoundType_MOB_ZOMBIEPIG_ZPIGANGRY": "mob.zombiepig.zpigangry",
            "eSoundType_MOB_SILVERFISH_AMBIENT": "mob.silverfish.say",
            "eSoundType_MOB_SILVERFISH_HURT": "mob.silverfish.hit",
            "eSoundType_MOB_SILVERFISH_DEATH": "mob.silverfish.kill",
            "eSoundType_MOB_SILVERFISH_STEP": "mob.silverfish.step",
            "eSoundType_MOB_SKELETON_AMBIENT": "mob.skeleton",
            "eSoundType_MOB_SKELETON_HURT": "mob.skeletonhurt",
            "eSoundType_MOB_SPIDER_AMBIENT": "mob.spider.say",
            "eSoundType_MOB_SPIDER_DEATH": "mob.spider.death",
            "eSoundType_MOB_SLIME": "mob.slime",
            "eSoundType_MOB_SLIME_ATTACK": "mob.slime.attack",
            "eSoundType_MOB_CREEPER_HURT": "mob.creeper.say",
            "eSoundType_MOB_CREEPER_DEATH": "mob.creeper.death",
            "eSoundType_MOB_ZOMBIE_AMBIENT": "mob.zombie.say",
            "eSoundType_MOB_ZOMBIE_HURT": "mob.zombie.hurt",
            "eSoundType_MOB_ZOMBIE_DEATH": "mob.zombie.death",
            "eSoundType_MOB_ZOMBIE_WOOD": "mob.zombie.wood",
            "eSoundType_MOB_ZOMBIE_WOOD_BREAK": "mob.zombie.woodbreak",
            "eSoundType_MOB_ZOMBIE_METAL": "mob.zombie.metal",
            "eSoundType_MOB_MAGMACUBE_BIG": "mob.magmacube.big",
            "eSoundType_MOB_MAGMACUBE_SMALL": "mob.magmacube.small",
            "eSoundType_MOB_CAT_PURR": "mob.cat.purr",
            "eSoundType_MOB_CAT_PURREOW": "mob.cat.purreow",
            "eSoundType_MOB_CAT_MEOW": "mob.cat.meow",
            "eSoundType_MOB_CAT_HITT": "mob.cat.hit",
            "eSoundType_RANDOM_BOW": "random.bow",
            "eSoundType_RANDOM_BOW_HIT": "random.bowhit",
            "eSoundType_RANDOM_EXPLODE": "random.explode",
            "eSoundType_RANDOM_FIZZ": "random.fizz",
            "eSoundType_RANDOM_POP": "random.pop",
            "eSoundType_RANDOM_FUSE": "random.fuse",
            "eSoundType_RANDOM_DRINK": "random.drink",
            "eSoundType_RANDOM_EAT": "random.eat",
            "eSoundType_RANDOM_BURP": "random.burp",
            "eSoundType_RANDOM_SPLASH": "random.splash",
            "eSoundType_RANDOM_CLICK": "random.click",
            "eSoundType_RANDOM_GLASS": "random.glass",
            "eSoundType_RANDOM_ORB": "random.orb",
            "eSoundType_RANDOM_BREAK": "random.break",
            "eSoundType_RANDOM_CHEST_OPEN": "random.chestopen",
            "eSoundType_RANDOM_CHEST_CLOSE": "random.chestclosed",
            "eSoundType_RANDOM_DOOR_OPEN": "random.door_open",
            "eSoundType_RANDOM_DOOR_CLOSE": "random.door_close",
            "eSoundType_AMBIENT_WEATHER_RAIN": "ambient.weather.rain",
            "eSoundType_AMBIENT_WEATHER_THUNDER": "ambient.weather.thunder",
            "eSoundType_CAVE_CAVE DON'T USE FOR XBOX 360!!!": "ambient.cave",
            "eSoundType_CAVE_CAVE2 - removed the two sounds that were at 192k in the first ambient cave event": "ambient.cave.cave2",
            "eSoundType_PORTAL_PORTAL": "portal.portal",
            "eSoundType_PORTAL_TRIGGER": "portal.trigger",
            "eSoundType_PORTAL_TRAVEL": "portal.travel",
            "eSoundType_FIRE_IGNITE": "fire.ignite",
            "eSoundType_FIRE_FIRE": "fire.fire",
            "eSoundType_DAMAGE_HURT": "damage.hurtflesh",
            "eSoundType_DAMAGE_FALL_SMALL": "damage.fallsmall",
            "eSoundType_DAMAGE_FALL_BIG": "damage.fallbig",
            "eSoundType_NOTE_HARP": "note.harp",
            "eSoundType_NOTE_BD": "note.bd",
            "eSoundType_NOTE_SNARE": "note.snare",
            "eSoundType_NOTE_HAT": "note.hat",
            "eSoundType_NOTE_BASSATTACK": "note.bassattack",
            "eSoundType_TILE_PISTON_IN": "tile.piston.in",
            "eSoundType_TILE_PISTON_OUT": "tile.piston.out",
            "eSoundType_LIQUID_WATER": "liquid.water",
            "eSoundType_LIQUID_LAVA_POP": "liquid.lavapop",
            "eSoundType_LIQUID_LAVA": "liquid.lava",
            "eSoundType_STEP_STONE": "step.stone",
            "eSoundType_STEP_WOOD": "step.wood",
            "eSoundType_STEP_GRAVEL": "step.gravel",
            "eSoundType_STEP_GRASS": "step.grass",
            "eSoundType_STEP_METAL": "step.metal",
            "eSoundType_STEP_CLOTH": "step.cloth",
            "eSoundType_STEP_SAND": "step.sand",
            "eSoundType_MOB_ENDERDRAGON_END": "mob.enderdragon.end",
            "eSoundType_MOB_ENDERDRAGON_GROWL": "mob.enderdragon.growl",
            "eSoundType_MOB_ENDERDRAGON_HIT": "mob.enderdragon.hit",
            "eSoundType_MOB_ENDERDRAGON_MOVE": "mob.enderdragon.wings",
            "eSoundType_MOB_IRONGOLEM_THROW": "mob.irongolem.throw",
            "eSoundType_MOB_IRONGOLEM_HIT": "mob.irongolem.hit",
            "eSoundType_MOB_IRONGOLEM_DEATH": "mob.irongolem.death",
            "eSoundType_MOB_IRONGOLEM_WALK": "mob.irongolem.walk",
            "eSoundType_DAMAGE_THORNS": "damage.thorns",
            "eSoundType_RANDOM_ANVIL_BREAK": "random.anvil_break",
            "eSoundType_RANDOM_ANVIL_LAND": "random.anvil_land",
            "eSoundType_EATING": "random.eat",
            "eSoundType_RANDOM_LEVELUP": "random.levelup",
            "eSoundType_MOB_VILLAGER_HAGGLE": "mob.villager.haggle",
            "eSoundType_MOB_VILLAGER_IDLE": "mob.villager.idle",
            "eSoundType_MOB_VILLAGER_HIT": "mob.villager.hit",
            "eSoundType_MOB_VILLAGER_DEATH": "mob.villager.death",
            "eSoundType_MOB_VILLAGER_YES": "mob.villager.yes",
            "eSoundType_MOB_VILLAGER_NO": "mob.villager.no",
            "eSoundType_MOB_ZOMBIE_INFECT": "mob.zombie.infect",
            "eSoundType_MOB_ZOMBIE_UNFECT": "mob.zombie.unfect",
            "eSoundType_MOB_ZOMBIE_REMEDY": "mob.zombie.remedy",
            "eSoundType_STEP_SNOW": "step.snow",
            "eSoundType_STEP_LADDER": "step.ladder",
            "eSoundType_DIG_CLOTH": "dig.cloth",
            "eSoundType_DIG_GRASS": "dig.grass",
            "eSoundType_DIG_GRAVEL": "dig.gravel",
            "eSoundType_DIG_SAND": "dig.sand",
            "eSoundType_DIG_SNOW": "dig.snow",
            "eSoundType_DIG_STONE": "dig.stone",
            "eSoundType_DIG_WOOD": "dig.wood",
            "eSoundType_FIREWORKS_LAUNCH": "fireworks.launch",
            "eSoundType_FIREWORKS_BLAST": "fireworks.blast",
            "eSoundType_FIREWORKS_BLAST_FAR": "fireworks.blast_far",
            "eSoundType_FIREWORKS_LARGE_BLAST": "fireworks.large_blast",
            "eSoundType_FIREWORKS_LARGE_BLAST_FAR": "fireworks.large_blast_far",
            "eSoundType_FIREWORKS_TWINKLE": "fireworks.twinkle",
            "eSoundType_FIREWORKS_TWINKLE_FAR": "fireworks.twinkle_far",
            "eSoundType_MOB_BAT_IDLE": "mob.bat.idle",
            "eSoundType_MOB_BAT_HURT": "mob.bat.hurt",
            "eSoundType_MOB_BAT_DEATH": "mob.bat.death",
            "eSoundType_MOB_BAT_TAKEOFF": "mob.bat.takeoff",
            "eSoundType_MOB_WITHER_SPAWN": "mob.wither.spawn",
            "eSoundType_MOB_WITHER_IDLE": "mob.wither.idle",
            "eSoundType_MOB_WITHER_HURT": "mob.wither.hurt",
            "eSoundType_MOB_WITHER_DEATH": "mob.wither.death",
            "eSoundType_MOB_WITHER_SHOOT": "mob.wither.shoot",
            "eSoundType_MOB_COW_STEP": "mob.cow.step",
            "eSoundType_MOB_CHICKEN_STEP": "mob.chicken.step",
            "eSoundType_MOB_PIG_STEP": "mob.pig.step",
            "eSoundType_MOB_ENDERMAN_STARE": "mob.enderman.stare",
            "eSoundType_MOB_ENDERMAN_SCREAM": "mob.enderman.scream",
            "eSoundType_MOB_SHEEP_SHEAR": "mob.sheep.shear",
            "eSoundType_MOB_SHEEP_STEP": "mob.sheep.step",
            "eSoundType_MOB_SKELETON_DEATH": "mob.skeleton.death",
            "eSoundType_MOB_SKELETON_STEP": "mob.skeleton.step",
            "eSoundType_MOB_SPIDER_STEP": "mob.spider.step",
            "eSoundType_MOB_WOLF_STEP": "mob.wolf.step",
            "eSoundType_MOB_ZOMBIE_STEP": "mob.zombie.step",
            "eSoundType_LIQUID_SWIM": "liquid.swim",
            "eSoundType_MOB_HORSE_LAND": "Mob.horse.land",
            "eSoundType_MOB_HORSE_ARMOR": "Mob.horse.armor",
            "eSoundType_MOB_HORSE_LEATHER": "Mob.horse.leather",
            "eSoundType_MOB_HORSE_ZOMBIE_DEATH": "Mob.horse.zombie.death",
            "eSoundType_MOB_HORSE_SKELETON_DEATH": "Mob.horse.skeleton.death",
            "eSoundType_MOB_HORSE_DONKEY_DEATH": "Mob.horse.donkey.death",
            "eSoundType_MOB_HORSE_DEATH": "Mob.horse.death",
            "eSoundType_MOB_HORSE_ZOMBIE_HIT": "Mob.horse.zombie.hit",
            "eSoundType_MOB_HORSE_SKELETON_HIT": "Mob.horse.skeleton.hit",
            "eSoundType_MOB_HORSE_DONKEY_HIT": "Mob.horse.donkey.hit",
            "eSoundType_MOB_HORSE_HIT": "Mob.horse.hit",
            "eSoundType_MOB_HORSE_ZOMBIE_IDLE": "Mob.horse.zombie.idle",
            "eSoundType_MOB_HORSE_SKELETON_IDLE": "Mob.horse.skeleton.idle",
            "eSoundType_MOB_HORSE_DONKEY_IDLE": "Mob.horse.donkey.idle",
            "eSoundType_MOB_HORSE_IDLE": "Mob.horse.idle",
            "eSoundType_MOB_HORSE_DONKEY_ANGRY": "Mob.horse.donkey.angry",
            "eSoundType_MOB_HORSE_ANGRY": "Mob.horse.angry",
            "eSoundType_MOB_HORSE_GALLOP": "Mob.horse.gallop",
            "eSoundType_MOB_HORSE_BREATHE": "Mob.horse.breathe",
            "eSoundType_MOB_HORSE_WOOD": "Mob.horse.wood",
            "eSoundType_MOB_HORSE_SOFT": "Mob.horse.soft",
            "eSoundType_MOB_HORSE_JUMP": "Mob.horse.jump",
            "eSoundType_MOB_WITCH_IDLE": "mob.witch.ambient",
            "eSoundType_MOB_WITCH_HURT": "mob.witch.hurt",
            "eSoundType_MOB_WITCH_DEATH": "mob.witch.death",
            "eSoundType_MOB_SLIME_BIG": "mob.slime.big",
            "eSoundType_MOB_SLIME_SMALL": "mob.slime.small",
            "eSoundType_EATING						<--- missing": "eating",
            "eSoundType_RANDOM_LEVELUP": "random.levelup",
            "eSoundType_DAMAGE_CRITICAL": "damage.critical",
            "eSoundType_ITEM_ELYTRA_FLYING": "item.elytra.flying",
        }

    def update_music(self, dt, dimension="OVERWORLD"):
        import random
        import glob
        
        # Check if music is currently playing
        if self.current_music_player is not None:
            if self.current_music_player.playing:
                # If dimension changed while playing, stop music immediately (C++ logic)
                if self.current_music_dimension != dimension:
                    self.current_music_player.pause()
                    self.current_music_player.delete()
                    self.current_music_player = None
                    self.current_music_dimension = None
                    # Delay before starting new dimension music
                    self.music_delay_timer = random.uniform(5.0, 10.0)
                return
            else:
                # Music finished naturally
                self.current_music_player.delete()
                self.current_music_player = None
                self.current_music_dimension = None
                # Set next 10-20 minute delay
                self.music_delay_timer = random.uniform(600.0, 1200.0)
                
        # If no music playing, tick the timer
        self.music_delay_timer -= dt
        if self.music_delay_timer <= 0:
            # Time to play music!
            music_folder = os.path.join(self.sfx_dir, "music", dimension.lower())
            
            # If standard overworld, Minecraft usually keeps it in "music/game" or "music"
            if dimension == "OVERWORLD":
                music_folder = os.path.join(self.sfx_dir, "music")
                
            if not os.path.exists(music_folder):
                # Fail gracefully if folder doesn't exist, wait 1 minute before checking again
                self.music_delay_timer = 60.0
                return
                
            # Grab all .ogg or .wav files in the directory
            tracks = glob.glob(os.path.join(music_folder, "*.ogg")) + glob.glob(os.path.join(music_folder, "*.wav"))
            if not tracks:
                self.music_delay_timer = 60.0
                return
                
            # Pick a random track
            track_path = random.choice(tracks)
            
            try:
                # We don't cache music tracks to save RAM (streaming=True)
                source = pyglet.media.load(track_path, streaming=True)
                self.current_music_player = source.play()
                self.current_music_player.volume = 0.5 # Music default volume
                self.current_music_dimension = dimension
            except Exception as e:
                print(f"Error playing music track {track_path}: {e}")
                self.music_delay_timer = 60.0

    def play(self, sound_enum_name, x=None, y=None, z=None, volume=1.0, pitch=1.0):
        sound_name = self.sound_names.get(sound_enum_name)
        if not sound_name: return
        
        base_path = os.path.join(self.sfx_dir, sound_name.replace(".", "/"))
        file_path = base_path + ".ogg"
        
        if not os.path.exists(file_path):
            file_path = base_path + ".wav"
            
        # If no exact match, look for numbered variants (e.g. stone1.ogg, stone2.ogg)
        if not os.path.exists(file_path):
            import random
            if not hasattr(self, 'glob_cache'):
                self.glob_cache = {}
            if base_path not in self.glob_cache:
                import glob
                self.glob_cache[base_path] = glob.glob(base_path + "[0-9]*.ogg") + glob.glob(base_path + "[0-9]*.wav")
            
            variants = self.glob_cache[base_path]
            if variants:
                file_path = random.choice(variants)
                
        if os.path.exists(file_path):
            try:
                # Clean up finished players to prevent memory leaks
                self.active_sounds = [snd for snd in self.active_sounds if snd['player'].playing]
                
                # C++ MAX_POLYPHONY check
                if len(self.active_sounds) >= self.MAX_POLYPHONY:
                    return
                    
                # C++ MAX_SAME_SOUNDS_PLAYING check
                same_sound_count = sum(1 for snd in self.active_sounds if snd['file_path'] == file_path)
                if same_sound_count >= self.MAX_SAME_SOUNDS_PLAYING:
                    return
                
                # We cache loaded sounds based on the final file path to avoid reloading
                if file_path not in self.sounds:
                    self.sounds[file_path] = pyglet.media.load(file_path, streaming=False)
                    
                # Manual 16-block cut-off check before creating the player
                if x is not None and y is not None and z is not None:
                    try:
                        import math
                        listener = pyglet.media.get_audio_driver().get_listener()
                        lx, ly, lz = listener.position
                        dist = math.sqrt((x - lx)**2 + (y - ly)**2 + (z - lz)**2)
                        if dist > 16.0:
                            return # Do not play sounds that are too far away
                    except Exception:
                        pass
                        
                player = self.sounds[file_path].play()
                player.volume = volume
                player.pitch = pitch
                
                # Positional 3D audio setup if coords are given
                if x is not None and y is not None and z is not None:
                    player.position = (x, y, z)
                    
                self.active_sounds.append({'player': player, 'file_path': file_path})
            except Exception as e:
                print(f"Error playing sound {sound_enum_name} ({file_path}): {e}")


    def get_dig_sound(self, block_id):
        # Maps basic block IDs to dig sounds
        from world.terrain import BLOCK_REGISTRY
        # Fallback to a basic sound
        sound = "eSoundType_DIG_STONE"
        
        # Simple heuristic based on block name
        for name, info in BLOCK_REGISTRY.items():
            if info["id"] == block_id:
                name_l = name.lower()
                if "wood" in name_l or "plank" in name_l or "log" in name_l:
                    sound = "eSoundType_DIG_WOOD"
                elif "dirt" in name_l or "mycelium" in name_l:
                    sound = "eSoundType_DIG_GRAVEL"
                elif "grass" in name_l or "leaves" in name_l or "sponge" in name_l or "rose" in name_l or "dandelion" in name_l or "flower" in name_l:
                    sound = "eSoundType_DIG_GRASS"
                elif "sand" in name_l and "stone" not in name_l:
                    sound = "eSoundType_DIG_SAND"
                elif "snow" in name_l:
                    sound = "eSoundType_DIG_SNOW"
                elif "glass" in name_l:
                    sound = "eSoundType_DIG_STONE" # Use stone if no glass sound
                elif "cactus" in name_l or "wool" in name_l or "cloth" in name_l:
                    sound = "eSoundType_DIG_CLOTH"
                break
                
        return sound

    def get_step_sound(self, block_id):
        from world.terrain import BLOCK_REGISTRY
        sound = "eSoundType_STEP_STONE"
        
        for name, info in BLOCK_REGISTRY.items():
            if info["id"] == block_id:
                name_l = name.lower()
                if "wood" in name_l or "plank" in name_l or "log" in name_l:
                    sound = "eSoundType_STEP_WOOD"
                elif "dirt" in name_l or "mycelium" in name_l:
                    sound = "eSoundType_STEP_GRAVEL"
                elif "grass" in name_l or "leaves" in name_l or "sponge" in name_l or "rose" in name_l or "dandelion" in name_l or "flower" in name_l:
                    sound = "eSoundType_STEP_GRASS"
                elif "sand" in name_l and "stone" not in name_l:
                    sound = "eSoundType_STEP_SAND"
                elif "snow" in name_l:
                    sound = "eSoundType_STEP_SNOW"
                elif "glass" in name_l:
                    sound = "eSoundType_STEP_STONE"
                elif "cactus" in name_l or "wool" in name_l or "cloth" in name_l:
                    sound = "eSoundType_STEP_CLOTH"
                break
                
        return sound
