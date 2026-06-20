import sqlite3
import zlib
import numpy as np
import os
CHUNK_SIZE = 16
CHUNK_HEIGHT = 128

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "world.db")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Chunks Table: cx, cz are primary keys
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                cx INTEGER,
                cz INTEGER,
                blocks BLOB,
                lights BLOB,
                PRIMARY KEY (cx, cz)
            )
        ''')
        
        # Chunk Entities Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunk_entities (
                cx INTEGER,
                cz INTEGER,
                entities_json TEXT,
                PRIMARY KEY (cx, cz)
            )
        ''')
        
        # Players Table: for future inventory and stats
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                id TEXT PRIMARY KEY,
                data TEXT
            )
        ''')
        
        conn.commit()

# Initialize upon import
init_db()

def save_chunk(cx, cz, blocks, lights):
    """Saves a chunk to the database using zlib compression."""
    # Compress blocks
    blocks_bytes = blocks.tobytes()
    blocks_compressed = zlib.compress(blocks_bytes, level=1)
    
    # Compress lights
    lights_bytes = lights.tobytes()
    lights_compressed = zlib.compress(lights_bytes, level=1)
    
    with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO chunks (cx, cz, blocks, lights)
            VALUES (?, ?, ?, ?)
        ''', (cx, cz, blocks_compressed, lights_compressed))
        conn.commit()

def load_chunk(cx, cz):
    """Loads a chunk from the database. Returns (blocks, lights) or None."""
    with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT blocks, lights FROM chunks WHERE cx = ? AND cz = ?', (cx, cz))
        row = cursor.fetchone()
        
    if row is None:
        return None
        
    try:
        blocks_bytes = zlib.decompress(row[0])
        blocks = np.frombuffer(blocks_bytes, dtype=np.uint8).reshape((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE))
        blocks = blocks.copy()
        
        lights_bytes = zlib.decompress(row[1])
        lights = np.frombuffer(lights_bytes, dtype=np.uint8).reshape((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE))
        lights = lights.copy()
        
        return blocks, lights
    except ValueError:
        # Eski dünya yüksekliği veya bozuk chunk varsa silip yeniden oluşturmasını sağla
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE cx=? AND cz=?", (cx, cz))
            conn.commit()
        return None

def save_chunk_entities(cx, cz, entities_json):
    with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO chunk_entities (cx, cz, entities_json)
            VALUES (?, ?, ?)
        ''', (cx, cz, entities_json))
        conn.commit()

def load_chunk_entities(cx, cz):
    with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT entities_json FROM chunk_entities WHERE cx = ? AND cz = ?', (cx, cz))
        row = cursor.fetchone()
    if row is None:
        return None
    return row[0]

def save_player_data(player_id, json_data):
    """Save player data (inventory, position, health, etc.)"""
    with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO players (id, data)
            VALUES (?, ?)
        ''', (player_id, json_data))
        conn.commit()

def load_player_data(player_id):
    """Load player data (inventory, position, health, etc.)"""
    with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT data FROM players WHERE id = ?', (player_id,))
        row = cursor.fetchone()
    
    if row is None:
        return None
    return row[0]
