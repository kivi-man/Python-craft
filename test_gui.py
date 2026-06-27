from PIL import Image
import numpy as np
import time

def orig_func():
    dest = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    dest_pix = dest.load()
    
    top_img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
    left_img = Image.new("RGBA", (16, 16), (0, 255, 0, 255))
    right_img = Image.new("RGBA", (16, 16), (0, 0, 255, 255))
    
    top_pix = top_img.load()
    left_pix = left_img.load()
    right_pix = right_img.load()
    
    for y in range(64):
        for x in range(64):
            # Top Face
            if (x + 2*y >= 32) and (x - 2*y <= 32) and (x - 2*y >= -32) and (x + 2*y <= 96) and (y < 32):
                x_prime = (x - 32) / 32.0
                y_prime = (y - 16) / 16.0
                u = int(8 * (x_prime + y_prime + 1))
                v = int(8 * (-x_prime + y_prime + 1))
                u = max(0, min(15, u))
                v = max(0, min(15, v))
                dest_pix[x, y] = top_pix[u, v]
            # Left Face (Shaded 0.6)
            elif (x >= 0) and (x < 32) and (y >= 16 + x/2) and (y < 48 + x/2):
                u = int(x / 2)
                v = int((y - (16 + x/2.0)) / 2)
                u = max(0, min(15, u))
                v = max(0, min(15, v))
                r, g, b, a = left_pix[u, v]
                dest_pix[x, y] = (int(r * 0.6), int(g * 0.6), int(b * 0.6), a)
            # Right Face (Shaded 0.8)
            elif (x >= 32) and (x < 64) and (y >= 48 - x/2) and (y < 80 - x/2):
                u = int((x - 32) / 2)
                v = int((y - (48 - x/2.0)) / 2)
                u = max(0, min(15, u))
                v = max(0, min(15, v))
                r, g, b, a = right_pix[u, v]
                dest_pix[x, y] = (int(r * 0.8), int(g * 0.8), int(b * 0.8), a)
                
    return np.array(dest)

def new_func():
    top_img = np.zeros((16, 16, 4), dtype=np.float32)
    top_img[:, :] = [255, 0, 0, 255]
    left_img = np.zeros((16, 16, 4), dtype=np.float32)
    left_img[:, :] = [0, 255, 0, 255]
    right_img = np.zeros((16, 16, 4), dtype=np.float32)
    right_img[:, :] = [0, 0, 255, 255]
    
    dest = np.zeros((64, 64, 4), dtype=np.uint8)
    
    Y, X = np.ogrid[0:64, 0:64]
    
    # Top Face mask
    top_mask = (X + 2*Y >= 32) & (X - 2*Y <= 32) & (X - 2*Y >= -32) & (X + 2*Y <= 96) & (Y < 32)
    # Left Face mask
    left_mask = (X >= 0) & (X < 32) & (Y >= 16 + X/2.0) & (Y < 48 + X/2.0)
    # Right Face mask
    right_mask = (X >= 32) & (X < 64) & (Y >= 48 - X/2.0) & (Y < 80 - X/2.0)
    
    # We must eliminate overlaps if original code has `elif`.
    # original code: if top, elif left, elif right
    left_mask = left_mask & ~top_mask
    right_mask = right_mask & ~top_mask & ~left_mask
    
    # Top Mapping
    x_prime = (X[top_mask] - 32) / 32.0
    y_prime = (Y[top_mask] - 16) / 16.0
    u_top = np.clip(np.floor(8 * (x_prime + y_prime + 1)).astype(int), 0, 15)
    v_top = np.clip(np.floor(8 * (-x_prime + y_prime + 1)).astype(int), 0, 15)
    dest[top_mask] = top_img[v_top, u_top]
    
    # Left Mapping
    u_left = np.clip(np.floor(X[left_mask] / 2.0).astype(int), 0, 15)
    v_left = np.clip(np.floor((Y[left_mask] - (16 + X[left_mask]/2.0)) / 2.0).astype(int), 0, 15)
    left_vals = left_img[v_left, u_left].copy()
    left_vals[:, 0:3] *= 0.6
    dest[left_mask] = left_vals.astype(np.uint8)
    
    # Right Mapping
    u_right = np.clip(np.floor((X[right_mask] - 32) / 2.0).astype(int), 0, 15)
    v_right = np.clip(np.floor((Y[right_mask] - (48 - X[right_mask]/2.0)) / 2.0).astype(int), 0, 15)
    right_vals = right_img[v_right, u_right].copy()
    right_vals[:, 0:3] *= 0.8
    dest[right_mask] = right_vals.astype(np.uint8)
    
    return dest

t0 = time.time()
orig = orig_func()
t1 = time.time()
print(f"Orig: {t1 - t0}")

t0 = time.time()
new = new_func()
t1 = time.time()
print(f"New: {t1 - t0}")

diff = np.abs(orig.astype(int) - new.astype(int)).sum()
print(f"Diff: {diff}")
