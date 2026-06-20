class Model:
    def render(self, entity, walk_pos, walk_speed, alive_ticks, head_yaw, head_pitch, scale, parent_matrix, u_view_loc):
        self.setup_anim(walk_pos, walk_speed, alive_ticks, head_yaw, head_pitch, scale)
        
    def setup_anim(self, walk_pos, walk_speed, alive_ticks, head_yaw, head_pitch, scale):
        pass
