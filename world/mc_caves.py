import numpy as np
from numba import njit
import math
from world.mc_noise import JavaRandom
from world.mc_biomes import get_biome_properties
from world.terrain import AIR, STONE, DIRT, GRASS, WATER, LAVA, CHUNK_HEIGHT

@njit(cache=True)
def _cave_add_tunnel(seed, cx, cz, blocks, xC_in, yC_in, zC_in, thick_in, yR_in, xR_in, step_in, dist_in, yS_in, biome_grid):
    # Stack arrays for simulated recursion
    s_seed = np.zeros(32, dtype=np.int64)
    s_xC = np.zeros(32, dtype=np.float64)
    s_yC = np.zeros(32, dtype=np.float64)
    s_zC = np.zeros(32, dtype=np.float64)
    s_thick = np.zeros(32, dtype=np.float64)
    s_yR = np.zeros(32, dtype=np.float64)
    s_xR = np.zeros(32, dtype=np.float64)
    s_step = np.zeros(32, dtype=np.int32)
    s_dist = np.zeros(32, dtype=np.int32)
    s_yS = np.zeros(32, dtype=np.float64)
    
    # Push initial
    s_seed[0] = seed; s_xC[0] = xC_in; s_yC[0] = yC_in; s_zC[0] = zC_in
    s_thick[0] = thick_in; s_yR[0] = yR_in; s_xR[0] = xR_in
    s_step[0] = step_in; s_dist[0] = dist_in; s_yS[0] = yS_in
    stack_size = 1
    
    xMid = cx * 16.0 + 8.0
    zMid = cz * 16.0 + 8.0
    
    while stack_size > 0:
        stack_size -= 1
        c_seed = s_seed[stack_size]
        xCave = s_xC[stack_size]
        yCave = s_yC[stack_size]
        zCave = s_zC[stack_size]
        thickness = s_thick[stack_size]
        yRot = s_yR[stack_size]
        xRot = s_xR[stack_size]
        step = s_step[stack_size]
        dist = s_dist[stack_size]
        yScale = s_yS[stack_size]
        
        yRota = 0.0
        xRota = 0.0
        random = JavaRandom(c_seed)
        
        if dist <= 0:
            max_d = 8 * 16 - 16
            dist = max_d - random.nextInt(max_d // 4)
            
        singleStep = False
        if step == -1:
            step = dist // 2
            singleStep = True
            
        splitPoint = random.nextInt(dist // 2) + dist // 4
        steep = (random.nextInt(6) == 0)
        
        for st in range(step, dist):
            rad = 1.5 + (math.sin(st * math.pi / dist) * thickness) * 1.0
            yRad = rad * yScale
            
            xc = math.cos(xRot)
            xs = math.sin(xRot)
            xCave += math.cos(yRot) * xc
            yCave += xs
            zCave += math.sin(yRot) * xc
            
            if steep:
                xRot *= 0.92
            else:
                xRot *= 0.7
                
            xRot += xRota * 0.1
            yRot += yRota * 0.1
            
            xRota *= 0.90
            yRota *= 0.75
            xRota += (random.nextFloat() - random.nextFloat()) * random.nextFloat() * 2.0
            yRota += (random.nextFloat() - random.nextFloat()) * random.nextFloat() * 4.0
            
            if not singleStep and st == splitPoint and thickness > 1.0 and dist > 0:
                if stack_size + 2 <= 32:
                    # Right branch
                    s_seed[stack_size] = random.nextLong(); s_xC[stack_size] = xCave; s_yC[stack_size] = yCave; s_zC[stack_size] = zCave
                    s_thick[stack_size] = random.nextFloat() * 0.5 + 0.5; s_yR[stack_size] = yRot + math.pi / 2.0; s_xR[stack_size] = xRot / 3.0
                    s_step[stack_size] = st; s_dist[stack_size] = dist; s_yS[stack_size] = 1.0
                    stack_size += 1
                    
                    # Left branch
                    s_seed[stack_size] = random.nextLong(); s_xC[stack_size] = xCave; s_yC[stack_size] = yCave; s_zC[stack_size] = zCave
                    s_thick[stack_size] = random.nextFloat() * 0.5 + 0.5; s_yR[stack_size] = yRot - math.pi / 2.0; s_xR[stack_size] = xRot / 3.0
                    s_step[stack_size] = st; s_dist[stack_size] = dist; s_yS[stack_size] = 1.0
                    stack_size += 1
                break
                
            if not singleStep and random.nextInt(4) == 0:
                continue
                
            xd_dist = xCave - xMid
            zd_dist = zCave - zMid
            remaining = dist - st
            rr = (thickness + 2.0) + 16.0
            if xd_dist*xd_dist + zd_dist*zd_dist - remaining*remaining > rr*rr:
                break
                
            if (xCave < xMid - 16.0 - rad * 2.0 or zCave < zMid - 16.0 - rad * 2.0 or
                xCave > xMid + 16.0 + rad * 2.0 or zCave > zMid + 16.0 + rad * 2.0):
                continue
                
            x0 = int(math.floor(xCave - rad)) - cx * 16 - 1
            x1 = int(math.floor(xCave + rad)) - cx * 16 + 1
            y0 = int(math.floor(yCave - yRad)) - 1
            y1 = int(math.floor(yCave + yRad)) + 1
            z0 = int(math.floor(zCave - rad)) - cz * 16 - 1
            z1 = int(math.floor(zCave + rad)) - cz * 16 + 1
            
            x0 = max(x0, 0); x1 = min(x1, 16)
            y0 = max(y0, 1); y1 = min(y1, CHUNK_HEIGHT - 8)
            z0 = max(z0, 0); z1 = min(z1, 16)
            
            detectedWater = False
            for xx in range(x0, x1):
                if detectedWater: break
                for zz in range(z0, z1):
                    if detectedWater: break
                    # Optimize: jump over inner blocks
                    yy = y1
                    while yy >= y0:
                        if blocks[xx, yy, zz] == WATER:
                            detectedWater = True
                            break
                        if yy != y0 and xx != x0 and xx != x1 - 1 and zz != z0 and zz != z1 - 1:
                            yy = y0 + 1
                        yy -= 1
                            
            if detectedWater:
                continue
                
            for xx in range(x0, x1):
                xd = ((xx + cx * 16 + 0.5) - xCave) / rad
                for zz in range(z0, z1):
                    zd = ((zz + cz * 16 + 0.5) - zCave) / rad
                    
                    hasGrass = False
                    if xd*xd + zd*zd < 1.0:
                        for yy in range(y1, y0 - 1, -1):
                            yd = (yy + 0.5 - yCave) / yRad
                            if yd > -0.7 and xd*xd + yd*yd + zd*zd < 1.0:
                                b = blocks[xx, yy, zz]
                                if b == GRASS:
                                    hasGrass = True
                                    
                                if b == STONE or b == DIRT or b == GRASS:
                                    if yy < 10:
                                        blocks[xx, yy, zz] = LAVA
                                    else:
                                        blocks[xx, yy, zz] = AIR
                                        if hasGrass and yy > 0 and blocks[xx, yy - 1, zz] == DIRT:
                                            b_id = biome_grid[(xx // 4) + 2, (zz // 4) + 2]
                                            _, _, top_block, _ = get_biome_properties(b_id)
                                            blocks[xx, yy - 1, zz] = top_block
                                            
            if singleStep:
                break

@njit(cache=True)
def _cave_add_feature(random, x, z, cx, cz, blocks, biome_grid):
    caves = random.nextInt(random.nextInt(random.nextInt(40) + 1) + 1)
    if random.nextInt(15) != 0:
        caves = 0
        
    for cave in range(caves):
        xCave = x * 16.0 + random.nextInt(16)
        yCave = random.nextInt(random.nextInt(CHUNK_HEIGHT - 8) + 8)
        zCave = z * 16.0 + random.nextInt(16)
        
        tunnels = 1
        if random.nextInt(4) == 0:
            _cave_add_tunnel(random.nextLong(), cx, cz, blocks, xCave, yCave, zCave, 
                             1.0 + random.nextFloat() * 6.0, 0.0, 0.0, -1, -1, 0.5, biome_grid)
            tunnels += random.nextInt(4)
            
        for i in range(tunnels):
            yRot = random.nextFloat() * math.pi * 2.0
            xRot = ((random.nextFloat() - 0.5) * 2.0) / 8.0
            thickness = random.nextFloat() * 2.0 + random.nextFloat()
            if random.nextInt(10) == 0:
                thickness *= random.nextFloat() * random.nextFloat() * 3.0 + 1.0
                
            _cave_add_tunnel(random.nextLong(), cx, cz, blocks, xCave, yCave, zCave,
                             thickness, yRot, xRot, 0, 0, 1.0, biome_grid)

@njit(cache=True)
def carve_caves(cx, cz, blocks, world_seed, biome_grid):
    radius = 8
    random = JavaRandom(world_seed)
    
    xScale = random.nextLong()
    zScale = random.nextLong()
    
    for x in range(cx - radius, cx + radius + 1):
        for z in range(cz - radius, cz + radius + 1):
            xx = x * xScale
            zz = z * zScale
            seed = (xx ^ zz ^ world_seed) & 0xFFFFFFFFFFFFFFFF
            random.setSeed(seed)
            _cave_add_feature(random, x, z, cx, cz, blocks, biome_grid)

@njit(cache=True)
def _canyon_add_tunnel(seed, cx, cz, blocks, xCave, yCave, zCave, thickness, yRot, xRot, step, dist, yScale, biome_grid):
    xMid = cx * 16.0 + 8.0
    zMid = cz * 16.0 + 8.0
    
    yRota = 0.0
    xRota = 0.0
    random = JavaRandom(seed)
    
    if dist <= 0:
        max_d = 8 * 16 - 16
        dist = max_d - random.nextInt(max_d // 4)
        
    rs = np.zeros(128, dtype=np.float64)
    f = 1.0
    for i in range(128):
        if i == 0 or random.nextInt(3) == 0:
            f = 1.0 + (random.nextFloat() * random.nextFloat()) * 1.0
        rs[i] = f * f
        
    for st in range(step, dist):
        rad = 1.5 + (math.sin(st * math.pi / dist) * thickness) * 1.0
        yRad = rad * yScale
        
        rad *= (random.nextFloat() * 0.25 + 0.75)
        yRad *= (random.nextFloat() * 0.25 + 0.75)
        
        xc = math.cos(xRot)
        xs = math.sin(xRot)
        xCave += math.cos(yRot) * xc
        yCave += xs
        zCave += math.sin(yRot) * xc
        
        xRot *= 0.7
        
        xRot += xRota * 0.05
        yRot += yRota * 0.05
        
        xRota *= 0.80
        yRota *= 0.50
        xRota += (random.nextFloat() - random.nextFloat()) * random.nextFloat() * 2.0
        yRota += (random.nextFloat() - random.nextFloat()) * random.nextFloat() * 4.0
        
        xd_dist = xCave - xMid
        zd_dist = zCave - zMid
        remaining = dist - st
        rr = (thickness + 2.0) + 16.0
        if xd_dist*xd_dist + zd_dist*zd_dist - remaining*remaining > rr*rr:
            break
            
        if (xCave < xMid - 16.0 - rad * 2.0 or zCave < zMid - 16.0 - rad * 2.0 or
            xCave > xMid + 16.0 + rad * 2.0 or zCave > zMid + 16.0 + rad * 2.0):
            continue
            
        x0 = int(math.floor(xCave - rad)) - cx * 16 - 1
        x1 = int(math.floor(xCave + rad)) - cx * 16 + 1
        y0 = int(math.floor(yCave - yRad)) - 1
        y1 = int(math.floor(yCave + yRad)) + 1
        z0 = int(math.floor(zCave - rad)) - cz * 16 - 1
        z1 = int(math.floor(zCave + rad)) - cz * 16 + 1
        
        x0 = max(x0, 0); x1 = min(x1, 16)
        y0 = max(y0, 1); y1 = min(y1, CHUNK_HEIGHT - 8)
        z0 = max(z0, 0); z1 = min(z1, 16)
        
        detectedWater = False
        for xx in range(x0, x1):
            if detectedWater: break
            for zz in range(z0, z1):
                if detectedWater: break
                yy = y1
                while yy >= y0:
                    if blocks[xx, yy, zz] == WATER:
                        detectedWater = True
                        break
                    if yy != y0 and xx != x0 and xx != x1 - 1 and zz != z0 and zz != z1 - 1:
                        yy = y0 + 1
                    yy -= 1
                        
        if detectedWater:
            continue
            
        for xx in range(x0, x1):
            xd = ((xx + cx * 16 + 0.5) - xCave) / rad
            for zz in range(z0, z1):
                zd = ((zz + cz * 16 + 0.5) - zCave) / rad
                
                hasGrass = False
                if xd*xd + zd*zd < 1.0:
                    for yy in range(y1, y0 - 1, -1):
                        yd = (yy + 0.5 - yCave) / yRad
                        
                        if (xd*xd + zd*zd) * rs[yy] + (yd*yd / 6.0) < 1.0:
                            b = blocks[xx, yy, zz]
                            if b == GRASS:
                                hasGrass = True
                                
                            if b == STONE or b == DIRT or b == GRASS:
                                if yy < 10:
                                    blocks[xx, yy, zz] = LAVA
                                else:
                                    blocks[xx, yy, zz] = AIR
                                    if hasGrass and yy > 0 and blocks[xx, yy - 1, zz] == DIRT:
                                        b_id = biome_grid[(xx // 4) + 2, (zz // 4) + 2]
                                        _, _, top_block, _ = get_biome_properties(b_id)
                                        blocks[xx, yy - 1, zz] = top_block

@njit(cache=True)
def _canyon_add_feature(random, x, z, cx, cz, blocks, biome_grid):
    if random.nextInt(50) != 0:
        return
        
    xCave = x * 16.0 + random.nextInt(16)
    yCave = random.nextInt(random.nextInt(40) + 8) + 20
    zCave = z * 16.0 + random.nextInt(16)
    
    yRot = random.nextFloat() * math.pi * 2.0
    xRot = ((random.nextFloat() - 0.5) * 2.0) / 8.0
    thickness = (random.nextFloat() * 2.0 + random.nextFloat()) * 2.0
    
    _canyon_add_tunnel(random.nextLong(), cx, cz, blocks, xCave, yCave, zCave,
                       thickness, yRot, xRot, 0, 0, 3.0, biome_grid)

@njit(cache=True)
def carve_canyons(cx, cz, blocks, world_seed, biome_grid):
    radius = 8
    random = JavaRandom(world_seed)
    
    xScale = random.nextLong()
    zScale = random.nextLong()
    
    for x in range(cx - radius, cx + radius + 1):
        for z in range(cz - radius, cz + radius + 1):
            xx = x * xScale
            zz = z * zScale
            seed = (xx ^ zz ^ world_seed) & 0xFFFFFFFFFFFFFFFF
            random.setSeed(seed)
            _canyon_add_feature(random, x, z, cx, cz, blocks, biome_grid)
