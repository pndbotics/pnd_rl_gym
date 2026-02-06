import struct


class KeyMap:
    A = 0
    B = 1
    X = 2
    Y = 3
    LB = 4
    RB = 5
    select = 6
    start = 7
    home = 8
    lo = 9
    ro = 10



class RemoteController:


    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.lt = 0
        self.rt = 0
        self.xx = 0
        self.yy = 0
        self.button = [0] * 10

        self.dead_area = 5000
        self.max_value = 32767
        self.ly_dir = -1.0
        self.lx_dir = 1.0
        self.rx_dir = -1.0
        self.max_speed_x = 1.0
        self.min_speed_x = -1.0
        self.max_speed_y = 1.0
        self.min_speed_y = -1.0
        self.max_speed_yaw = 1.0
        self.min_speed_yaw = -1.0

    def set(self, data):
        # wireless_remote
        for i in range(10):
            self.button[i] = data [i + 8]
        self.lx = data[0]
        self.ly = data[1]
        self.rx = data[2]
        self.ry = data[3]
        self.lt = data[4]
        self.rt = data[5]
        self.xx = data[6]
        self.yy = data[7]
        # print(f"lx: {self.lx}, ly: {self.ly}, rx: {self.rx}, ry: {self.ry}, lt: {self.lt}, rt: {self.rt}, xx: {self.xx}, yy: {self.yy}, buttons: {self.button}")
    
    def get_walk_x_direction_speed(self):
        """
        获取行走时的X方向速度（基于lt扳机值和ly摇杆值）
        
        返回:
            如果lt < 1000，根据ly摇杆值计算归一化速度
            如果lt >= 1000，返回固定值0.70
        """
        lt_value = self.lt
        if lt_value < 1000:
            x_value = self.ly
            abs_x_value = abs(x_value)
            if (abs_x_value > self.dead_area) and (abs_x_value <= self.max_value):
                if x_value > 0:
                    return self.ly_dir * self.max_speed_x * ((abs_x_value - self.dead_area) / (self.max_value - self.dead_area))
                else:
                    return self.ly_dir * self.min_speed_x * ((abs_x_value - self.dead_area) / (self.max_value - self.dead_area))
            
            # sim
            elif(abs_x_value <= 1):
                return self.ly * self.ly_dir
            
            else:
                return 0.0
        else:
            return 0.70
    
    def get_walk_y_direction_speed(self):
        """
        获取行走时的Y方向速度（基于lx摇杆值）
        
        返回:
            根据lx摇杆值计算归一化速度，如果不在有效范围内则返回0.0
        """
        y_value = self.lx
        abs_y_value = abs(y_value)
        if (abs_y_value > self.dead_area) and (abs_y_value <= self.max_value):
            if y_value > 0:
                return self.lx_dir * self.max_speed_y * ((abs_y_value - self.dead_area) / (self.max_value - self.dead_area))
            else:
                return self.lx_dir * self.min_speed_y * ((abs_y_value - self.dead_area) / (self.max_value - self.dead_area))
        else:
            return 0.0
    
    def get_walk_yaw_direction_speed(self):
        """
        获取行走时的Yaw方向速度（基于rx摇杆值）
        
        返回:
            根据rx摇杆值计算归一化速度，如果不在有效范围内则返回0.0
        """
        yaw_value = self.rx
        abs_yaw_value = abs(yaw_value)
        if (abs_yaw_value > self.dead_area) and (abs_yaw_value <= self.max_value):
            if yaw_value > 0:
                return self.rx_dir * self.max_speed_yaw * ((abs_yaw_value - self.dead_area) / (self.max_value - self.dead_area))
            else:
                return self.rx_dir * self.min_speed_yaw * ((abs_yaw_value - self.dead_area) / (self.max_value - self.dead_area))
        else:
            return 0.0