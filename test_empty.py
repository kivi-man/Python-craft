import os
from nbt import region

filename = "test_empty.mca"
open(filename, 'wb').close()

try:
    r = region.RegionFile(filename)
    print("RegionFile opened successfully. Size:", os.path.getsize(filename))
except Exception as e:
    print("Exception opening RegionFile:", repr(e))
