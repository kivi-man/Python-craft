import numpy as np
from numba.experimental import jitclass
from numba import int64, int32, float64, boolean

java_random_spec = [
    ('seed', int64),
    ('haveNextNextGaussian', boolean),
    ('nextNextGaussian', float64)
]

@jitclass(java_random_spec)
class JavaRandom:
    def __init__(self, seed: int):
        self.setSeed(seed)
        
    def setSeed(self, s: int):
        self.seed = (s ^ 0x5DEECE66D) & ((1 << 48) - 1)
        self.haveNextNextGaussian = False
        
    def next(self, bits: int) -> int:
        self.seed = (self.seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        res = self.seed >> (48 - bits)
        if res >= 2147483648:
            res -= 4294967296
        return res
        
    def nextDouble(self) -> float:
        v1 = self.next(26)
        if v1 < 0: v1 += 4294967296
        v2 = self.next(27)
        if v2 < 0: v2 += 4294967296
        return ((v1 << 27) + v2) / float(1 << 53)
        
    def nextInt(self, n: int) -> int:
        if (n & -n) == n:
            v = self.next(31)
            if v < 0: v += 4294967296
            return int((v * n) >> 31)
            
        while True:
            bits = self.next(31)
            if bits < 0: bits += 4294967296
            val = bits % n
            if bits - val + (n - 1) >= 0:
                return val

improved_noise_spec = [
    ('p', int32[:]),
    ('xo', float64),
    ('yo', float64),
    ('zo', float64),
]

@jitclass(improved_noise_spec)
class ImprovedNoise:
    def __init__(self, random):
        self.p = np.zeros(512, dtype=np.int32)
        self.xo = random.nextDouble() * 256.0
        self.yo = random.nextDouble() * 256.0
        self.zo = random.nextDouble() * 256.0
        
        for i in range(256):
            self.p[i] = i
            
        for i in range(256):
            j = random.nextInt(256 - i) + i
            tmp = self.p[i]
            self.p[i] = self.p[j]
            self.p[j] = tmp
            self.p[i + 256] = self.p[i]
            
    def lerp(self, t, a, b):
        return a + t * (b - a)

    def grad2(self, hash_val, x, z):
        h = hash_val & 15
        u = (1 - ((h & 8) >> 3)) * x
        v = 0.0 if h < 4 else (x if h == 12 or h == 14 else z)
        res = (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
        return res

    def grad(self, hash_val, x, y, z):
        h = hash_val & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h == 12 or h == 14 else z)
        res = (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
        return res

    def noise(self, _x, _y, _z):
        x = _x + self.xo
        y = _y + self.yo
        z = _z + self.zo

        xf = int(x)
        yf = int(y)
        zf = int(z)

        if x < xf: xf -= 1
        if y < yf: yf -= 1
        if z < zf: zf -= 1

        X = xf & 255
        Y = yf & 255
        Z = zf & 255

        x -= xf
        y -= yf
        z -= zf

        u = x * x * x * (x * (x * 6 - 15) + 10)
        v = y * y * y * (y * (y * 6 - 15) + 10)
        w = z * z * z * (z * (z * 6 - 15) + 10)

        A = self.p[X] + Y
        AA = self.p[A] + Z
        AB = self.p[A + 1] + Z
        B = self.p[X + 1] + Y
        BA = self.p[B] + Z
        BB = self.p[B + 1] + Z

        return self.lerp(w, self.lerp(v, self.lerp(u, self.grad(self.p[AA], x, y, z),
                self.grad(self.p[BA], x - 1, y, z)),
                self.lerp(u, self.grad(self.p[AB], x, y - 1, z),
                        self.grad(self.p[BB], x - 1, y - 1, z))),
                self.lerp(v, self.lerp(u, self.grad(self.p[AA + 1], x, y, z - 1),
                        self.grad(self.p[BA + 1], x - 1, y, z - 1)),
                        self.lerp(u, self.grad(self.p[AB + 1], x, y - 1, z - 1), self.grad(self.p[BB + 1], x - 1, y - 1, z - 1))))


perlin_noise_spec = [
    ('levels', int32),
    ('p', int32[:, :]),
    ('xo', float64[:]),
    ('yo', float64[:]),
    ('zo', float64[:]),
]

@jitclass(perlin_noise_spec)
class PerlinNoise:
    def __init__(self, random, levels):
        self.levels = levels
        self.p = np.zeros((levels, 512), dtype=np.int32)
        self.xo = np.zeros(levels, dtype=np.float64)
        self.yo = np.zeros(levels, dtype=np.float64)
        self.zo = np.zeros(levels, dtype=np.float64)
        
        for i in range(levels):
            self.xo[i] = random.nextDouble() * 256.0
            self.yo[i] = random.nextDouble() * 256.0
            self.zo[i] = random.nextDouble() * 256.0
            
            for j in range(256):
                self.p[i, j] = j
                
            for j in range(256):
                k = random.nextInt(256 - j) + j
                tmp = self.p[i, j]
                self.p[i, j] = self.p[i, k]
                self.p[i, k] = tmp
                self.p[i, j + 256] = self.p[i, j]

    def lerp(self, t, a, b):
        return a + t * (b - a)

    def grad(self, hash_val, x, y, z):
        h = hash_val & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h == 12 or h == 14 else z)
        res = (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
        return res

    def noise_3d(self, level, _x, _y, _z):
        x = _x + self.xo[level]
        y = _y + self.yo[level]
        z = _z + self.zo[level]

        xf = int(x)
        yf = int(y)
        zf = int(z)

        if x < xf: xf -= 1
        if y < yf: yf -= 1
        if z < zf: zf -= 1

        X = xf & 255
        Y = yf & 255
        Z = zf & 255

        x -= xf
        y -= yf
        z -= zf

        u = x * x * x * (x * (x * 6 - 15) + 10)
        v = y * y * y * (y * (y * 6 - 15) + 10)
        w = z * z * z * (z * (z * 6 - 15) + 10)

        A = self.p[level, X] + Y
        AA = self.p[level, A] + Z
        AB = self.p[level, A + 1] + Z
        B = self.p[level, X + 1] + Y
        BA = self.p[level, B] + Z
        BB = self.p[level, B + 1] + Z

        return self.lerp(w, self.lerp(v, self.lerp(u, self.grad(self.p[level, AA], x, y, z),
                self.grad(self.p[level, BA], x - 1, y, z)),
                self.lerp(u, self.grad(self.p[level, AB], x, y - 1, z),
                        self.grad(self.p[level, BB], x - 1, y - 1, z))),
                self.lerp(v, self.lerp(u, self.grad(self.p[level, AA + 1], x, y, z - 1),
                        self.grad(self.p[level, BA + 1], x - 1, y, z - 1)),
                        self.lerp(u, self.grad(self.p[level, AB + 1], x, y - 1, z - 1), self.grad(self.p[level, BB + 1], x - 1, y - 1, z - 1))))

    def noise_2d(self, level, _x, _y):
        return self.noise_3d(level, _x, _y, 0.0)

    def getValue(self, x, y, z):
        value = 0.0
        power = 1.0
        for i in range(self.levels):
            value += self.noise_3d(i, x * power, y * power, z * power) / power
            power /= 2.0
        return value

    def getValue2D(self, x, y):
        value = 0.0
        power = 1.0
        for i in range(self.levels):
            value += self.noise_2d(i, x * power, y * power) / power
            power /= 2.0
        return value
