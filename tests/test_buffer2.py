import numpy as np
from world.mc_terrain import get_noise_buffer, _LPERLIN1, _LPERLIN2, _PERLIN1, _DEPTH, _SCALE
from world.mc_noise import JavaRandom, PerlinNoise

r = JavaRandom(12345)
lp1 = PerlinNoise(r, 16)
lp2 = PerlinNoise(r, 16)
p1 = PerlinNoise(r, 8)
depth = PerlinNoise(r, 16)
scale = PerlinNoise(r, 10)

# Simulate one point in get_noise_buffer
cx, cz = 0, 0
xx, zz = 0, 0
wx = cx * 4 + xx
wz = cz * 4 + zz

s = 684.412
hs = 684.412

sc = scale.getValue2D(wx * 1.121, wz * 1.121) * 0.5 + 0.5
d = depth.getValue2D(wx * 200.0, wz * 200.0) * 0.5 + 0.5

sss = sc * 0.9 + 0.1
ddd = (d * 4 - 1) / 8.0

print(f"sc: {sc}, d: {d}, sss: {sss}, ddd: {ddd}")

yy = 0
wy = yy
d_tmp = ddd
s_factor = sss

rdepth = (depth.getValue2D(wx, wz) / 8000.0)
rdepth = rdepth * 3.0 - 2.0
if rdepth < 0:
    rdepth /= 2.0
    if rdepth < -1.0: rdepth = -1.0
    rdepth /= 1.4
    rdepth /= 2.0
else:
    if rdepth > 1.0: rdepth = 1.0
    rdepth /= 8.0

d_tmp += rdepth * 0.2
d_tmp = d_tmp * 17 / 16.0
yCenter = (17 / 2.0) + (d_tmp * 4.0)

yOffs = (yy - yCenter) * 12.0 * 128.0 / 256.0 / s_factor
print(f"yCenter: {yCenter}, yOffs before clip: {yOffs}")
if yOffs < 0: yOffs *= 4.0

bb = lp1.getValue(wx * s, wy * hs, wz * s)
cc = lp2.getValue(wx * s, wy * hs, wz * s)
v = (p1.getValue(wx * s / 80.0, wy * hs / 160.0, wz * s / 80.0) / 10.0 + 1.0) / 2.0

print(f"bb (raw): {bb}, cc (raw): {cc}, v: {v}")
bb /= 512.0
cc /= 512.0

if v < 0.0: val = bb
elif v > 1.0: val = cc
else: val = bb + (cc - bb) * v

print(f"val before offs: {val}")
val -= yOffs
print(f"final val: {val}")
