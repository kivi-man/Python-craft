from core.entities.entity import AABB, Entity
def mock_get_block(x, y, z):
    if y <= 78: return 2
    return 0

e = Entity(61.5, 81.5, 6.5)
e.set_size(0.9, 0.9)
for i in range(30):
    e.dx = 0.15
    e.tick(mock_get_block)
    print(f"y: {e.y:.5f}, dy: {e.dy:.5f}, dx: {e.dx:.5f}, x: {e.x:.5f}, on_ground: {e.on_ground}")
