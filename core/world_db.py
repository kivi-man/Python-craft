import os
import json
import numpy as np
from nbt import nbt, region
import zlib
import threading

DB_LOCK = threading.Lock()
CHUNK_SIZE = 16
CHUNK_HEIGHT = 256

WORLD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "world_data")
REGION_DIR = os.path.join(WORLD_DIR, "region")
PLAYER_DIR = os.path.join(WORLD_DIR, "playerdata")
ENTITY_DIR = os.path.join(WORLD_DIR, "entities")

import time

def ensure_level_dat():
    level_dat_path = os.path.join(WORLD_DIR, "level.dat")
    if not os.path.exists(level_dat_path):
        root = nbt.NBTFile()
        data = nbt.TAG_Compound(name="Data")
        root.tags.append(data)
        
        data.tags.append(nbt.TAG_Int(name="version", value=19133))
        data.tags.append(nbt.TAG_String(name="LevelName", value="PythonCraft World"))
        data.tags.append(nbt.TAG_String(name="generatorName", value="default"))
        data.tags.append(nbt.TAG_Int(name="GameType", value=0))
        data.tags.append(nbt.TAG_Int(name="SpawnX", value=0))
        data.tags.append(nbt.TAG_Int(name="SpawnY", value=64))
        data.tags.append(nbt.TAG_Int(name="SpawnZ", value=0))
        data.tags.append(nbt.TAG_Long(name="RandomSeed", value=123456789))
        data.tags.append(nbt.TAG_Long(name="Time", value=0))
        data.tags.append(nbt.TAG_Long(name="DayTime", value=0))
        data.tags.append(nbt.TAG_Long(name="lastPlayed", value=int(time.time() * 1000)))
        
        root.write_file(level_dat_path)
from world.mc_id_converter import INTERNAL_TO_MC_ID, INTERNAL_TO_MC_META, MC_TO_INTERNAL

def init_db():
    os.makedirs(REGION_DIR, exist_ok=True)
    os.makedirs(PLAYER_DIR, exist_ok=True)
    os.makedirs(ENTITY_DIR, exist_ok=True)
    try:
        ensure_level_dat()
    except Exception as e:
        print(f"Error creating level.dat: {e}")

init_db()

def _get_region_file(cx, cz):
    rx = cx >> 5
    rz = cz >> 5
    filename = os.path.join(REGION_DIR, f"r.{rx}.{rz}.mca")
    if not os.path.exists(filename):
        open(filename, 'wb').close()
    return region.RegionFile(filename)

def pack_nibble_array(arr):
    # arr is a 1D numpy array of shape (N,) containing values 0-15
    # Pack 2 values into 1 byte: lower nibble is even index, upper nibble is odd index
    even = arr[0::2]
    odd = arr[1::2]
    return (even | (odd << 4)).astype(np.uint8).tobytes()

def unpack_nibble_array(byte_data, size):
    arr = np.frombuffer(byte_data, dtype=np.uint8)
    even = arr & 0x0F
    odd = (arr >> 4) & 0x0F
    res = np.empty(size, dtype=np.uint8)
    res[0::2] = even
    res[1::2] = odd
    return res

def save_chunk(cx, cz, blocks, data, lights):
    """Saves a chunk to Anvil MCA file format.
    blocks: numpy array (16, 256, 16) uint8
    data: numpy array (16, 256, 16) uint8
    lights: numpy array (16, 256, 16) uint8
    """
    root = nbt.NBTFile()
    level = nbt.TAG_Compound(name="Level")
    root.tags.append(level)
    
    level.tags.append(nbt.TAG_Int(name="xPos", value=cx))
    level.tags.append(nbt.TAG_Int(name="zPos", value=cz))
    level.tags.append(nbt.TAG_Long(name="LastUpdate", value=0))
    level.tags.append(nbt.TAG_Byte(name="LightPopulated", value=1))
    level.tags.append(nbt.TAG_Byte(name="TerrainPopulated", value=1))
    level.tags.append(nbt.TAG_Byte(name="V", value=1))
    
    biomes_tag = nbt.TAG_Byte_Array(name="Biomes")
    biomes_tag.value = bytearray(256)
    level.tags.append(biomes_tag)
    
    height_tag = nbt.TAG_Int_Array(name="HeightMap")
    height_tag.value = np.zeros(256, dtype=np.int32).tolist() # nbt library expects list of ints or int array
    level.tags.append(height_tag)
    
    sections = nbt.TAG_List(name="Sections", type=nbt.TAG_Compound)
    level.tags.append(sections)
    
    for y_sec in range(16):
        y_start = y_sec * 16
        y_end = y_start + 16
        
        # Minecraft uses [y, z, x]
        sec_blocks_internal = blocks[:, y_start:y_end, :].transpose((1, 2, 0)).flatten()
        sec_data_internal = data[:, y_start:y_end, :].transpose((1, 2, 0)).flatten()
        sec_lights = lights[:, y_start:y_end, :].transpose((1, 2, 0)).flatten()
        
        if not np.any(sec_blocks_internal):
            continue
            
        sec_blocks_mc = INTERNAL_TO_MC_ID[sec_blocks_internal]
        # Use our passed 'data' array instead of INTERNAL_TO_MC_META mapping
        # but fallback to INTERNAL_TO_MC_META if data is 0 for backwards compatibility with blocks that need specific meta
        fallback_meta = INTERNAL_TO_MC_META[sec_blocks_internal]
        sec_meta_mc = np.where(sec_data_internal > 0, sec_data_internal, fallback_meta)
            
        sec_tag = nbt.TAG_Compound()
        sec_tag.tags.append(nbt.TAG_Byte(name="Y", value=y_sec))
        
        blocks_tag = nbt.TAG_Byte_Array(name="Blocks")
        blocks_tag.value = bytearray(sec_blocks_mc.tobytes())
        sec_tag.tags.append(blocks_tag)
        
        data_tag = nbt.TAG_Byte_Array(name="Data")
        data_tag.value = bytearray(pack_nibble_array(sec_meta_mc))
        sec_tag.tags.append(data_tag)
        
        block_light_bytes = pack_nibble_array(sec_lights)
        
        sec_tag.tags.append(nbt.TAG_Byte_Array(name="BlockLight"))
        sec_tag["BlockLight"].value = bytearray(block_light_bytes)
        
        sec_tag.tags.append(nbt.TAG_Byte_Array(name="SkyLight"))
        sec_tag["SkyLight"].value = bytearray(block_light_bytes)
        
        sections.tags.append(sec_tag)
        
    level.tags.append(nbt.TAG_List(name="Entities", type=nbt.TAG_Compound))
    level.tags.append(nbt.TAG_List(name="TileEntities", type=nbt.TAG_Compound))
    
    try:
        with DB_LOCK:
            rfile = _get_region_file(cx, cz)
            rx = cx % 32
            rz = cz % 32
            rfile.write_chunk(rx, rz, root)
            rfile.close()
    except Exception as e:
        print(f"Error saving chunk {cx}, {cz}: {e}")

def load_chunk(cx, cz):
    """Loads a chunk from Anvil MCA file format. Returns (blocks, data, lights) or None."""
    try:
        rx = cx >> 5
        rz = cz >> 5
        filename = os.path.join(REGION_DIR, f"r.{rx}.{rz}.mca")
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            return None
            
        with DB_LOCK:
            rfile = region.RegionFile(filename)
            rx_chunk = cx % 32
            rz_chunk = cz % 32
            
            try:
                chunk_nbt = rfile.get_chunk(rx_chunk, rz_chunk)
            except region.InconceivedChunk:
                rfile.close()
                return None
            except Exception:
                rfile.close()
                return None
                
            rfile.close()
        
        if not chunk_nbt or "Level" not in chunk_nbt:
            return None
            
        level = chunk_nbt["Level"]
        
        blocks = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
        data = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
        lights = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
        
        if "Sections" in level:
            for sec_tag in level["Sections"]:
                y_sec = sec_tag["Y"].value
                if y_sec < 0 or y_sec >= 16:
                    continue
                    
                y_start = y_sec * 16
                y_end = y_start + 16
                
                sec_blocks_flat = np.frombuffer(sec_tag["Blocks"].value, dtype=np.uint8)
                if "Data" in sec_tag:
                    sec_meta_flat = unpack_nibble_array(sec_tag["Data"].value, 4096)
                else:
                    sec_meta_flat = np.zeros(4096, dtype=np.uint8)
                    
                sec_blocks_internal = MC_TO_INTERNAL[sec_blocks_flat, sec_meta_flat]
                
                sec_blocks = sec_blocks_internal.reshape((16, 16, 16)).transpose((2, 0, 1))
                blocks[:, y_start:y_end, :] = sec_blocks
                
                # We store the raw meta flat into our data array so custom special blocks get their meta back
                sec_data = sec_meta_flat.reshape((16, 16, 16)).transpose((2, 0, 1))
                data[:, y_start:y_end, :] = sec_data
                
                sec_lights = np.zeros((16, 16, 16), dtype=np.uint8)
                if "BlockLight" in sec_tag:
                    bl_flat = unpack_nibble_array(sec_tag["BlockLight"].value, 4096)
                    sec_lights = np.maximum(sec_lights, bl_flat.reshape((16, 16, 16)).transpose((2, 0, 1)))
                if "SkyLight" in sec_tag:
                    sl_flat = unpack_nibble_array(sec_tag["SkyLight"].value, 4096)
                    sec_lights = np.maximum(sec_lights, sl_flat.reshape((16, 16, 16)).transpose((2, 0, 1)))
                
                lights[:, y_start:y_end, :] = sec_lights
                    
        return blocks, data, lights
    except Exception as e:
        print(f"Error loading chunk {cx}, {cz}: {e}")
        return None

def save_chunk_entities(cx, cz, entities_json):
    filename = os.path.join(ENTITY_DIR, f"{cx}_{cz}.json")
    with open(filename, 'w') as f:
        f.write(entities_json)

def load_chunk_entities(cx, cz):
    filename = os.path.join(ENTITY_DIR, f"{cx}_{cz}.json")
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return f.read()
    return None

def save_player_data(player_id, json_data):
    filename = os.path.join(PLAYER_DIR, f"{player_id}.json")
    with open(filename, 'w') as f:
        f.write(json_data)

def load_player_data(player_id):
    filename = os.path.join(PLAYER_DIR, f"{player_id}.json")
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return f.read()
    return None

def save_level_dat(player_pos, player_rot, inventory_blocks, inventory_counts):
    level_dat_path = os.path.join(WORLD_DIR, "level.dat")
    try:
        if os.path.exists(level_dat_path):
            root = nbt.NBTFile(level_dat_path, 'rb')
        else:
            root = nbt.NBTFile()
            root.tags.append(nbt.TAG_Compound(name="Data"))
            
        data = root["Data"]
        
        if "Player" not in data:
            data.tags.append(nbt.TAG_Compound(name="Player"))
            
        player_tag = data["Player"]
        
        # Pos
        if "Pos" in player_tag:
            del player_tag["Pos"]
        pos_list = nbt.TAG_List(name="Pos", type=nbt.TAG_Double)
        pos_list.tags.extend([nbt.TAG_Double(val) for val in player_pos])
        player_tag.tags.append(pos_list)
        
        # Rotation
        if "Rotation" in player_tag:
            del player_tag["Rotation"]
        rot_list = nbt.TAG_List(name="Rotation", type=nbt.TAG_Float)
        rot_list.tags.extend([nbt.TAG_Float(val) for val in player_rot])
        player_tag.tags.append(rot_list)
        
        # Inventory
        if "Inventory" in player_tag:
            del player_tag["Inventory"]
        inv_list = nbt.TAG_List(name="Inventory", type=nbt.TAG_Compound)
        
        for slot in range(min(36, len(inventory_blocks))):
            internal_id = inventory_blocks[slot]
            count = inventory_counts[slot]
            if count > 0 and internal_id > 0:
                item_tag = nbt.TAG_Compound()
                mc_id = INTERNAL_TO_MC_ID[internal_id]
                mc_meta = INTERNAL_TO_MC_META[internal_id]
                
                item_tag.tags.append(nbt.TAG_Byte(name="Slot", value=slot))
                item_tag.tags.append(nbt.TAG_Short(name="id", value=int(mc_id)))
                item_tag.tags.append(nbt.TAG_Byte(name="Count", value=int(count)))
                item_tag.tags.append(nbt.TAG_Short(name="Damage", value=int(mc_meta)))
                inv_list.tags.append(item_tag)
                
        player_tag.tags.append(inv_list)
        
        root.write_file(level_dat_path)
    except Exception as e:
        print(f"Error saving level.dat: {e}")

def load_level_dat():
    level_dat_path = os.path.join(WORLD_DIR, "level.dat")
    if not os.path.exists(level_dat_path):
        return None
        
    try:
        root = nbt.NBTFile(level_dat_path, 'rb')
        if "Data" in root and "Player" in root["Data"]:
            player_tag = root["Data"]["Player"]
            
            result = {}
            if "Pos" in player_tag:
                result["pos"] = [val.value for val in player_tag["Pos"].tags]
            if "Rotation" in player_tag:
                result["rot"] = [val.value for val in player_tag["Rotation"].tags]
                
            if "Inventory" in player_tag:
                inv_blocks = [0] * 55
                inv_counts = [0] * 55
                for item in player_tag["Inventory"].tags:
                    slot = item["Slot"].value
                    if slot >= 0 and slot < 55:
                        # nbtlib or nbt might parse short or string id. If we use numeric for 1.7.10
                        if hasattr(item["id"], 'value'):
                            mc_id = item["id"].value
                        else:
                            mc_id = int(item["id"])
                        mc_meta = item["Damage"].value if "Damage" in item else 0
                        count = item["Count"].value
                        
                        internal_id = MC_TO_INTERNAL[mc_id, mc_meta]
                        inv_blocks[slot] = int(internal_id)
                        inv_counts[slot] = int(count)
                result["inventory_blocks"] = inv_blocks
                result["inventory_counts"] = inv_counts
                
            return result
    except Exception as e:
        print(f"Error loading level.dat: {e}")
    return None
