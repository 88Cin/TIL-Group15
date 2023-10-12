import numpy as np

class Car:
    
    def __init__(self, size: float, xs: np.array = [], vs: np.array = [], a_s: np.array = []) -> None:
        self.vs = vs
        self.xs = xs
        self.a_s = a_s
        self.size = size
        

class IntelDriverModel:
    
    def __init__(self, param: dict) -> None:
        ## time_step, v0, s0, T, a, b, sigma = 4
        self.v0 = param['v0']  # the velocity the vehicle would drive at in free traffic
        self.s0 = param['s0'] # a minimum desired net distance.
        self.T = param['T'] # the minimum possible time to the vehicle in front
        self.a = param['a'] # the maximum vehicle acceleration
        self.b = param['b'] # a positive number (comfortable breaking deceleration)
        self.sigma = param['sigma'] # exponent factor, usually set to 4

    def get_desired_s(self, v_this, delta_v_i):
        
        desired_s = v_this * self.T + v_this * delta_v_i / (2 * np.sqrt(self.a * self.b))
        return self.s0 + max(0, desired_s)
    def get_new_a(self, v_this, v_front, s_this):
        
        delta_v_i = v_this - v_front
        return self.a * (1 - np.power(v_this / self.v0, self.sigma) 
                            - np.power((self.get_desired_s(v_this, delta_v_i) / s_this), 2))
    
    def update(self, front_car: Car, this_car: Car, time_step: float):
        
        assert this_car is not None
        
        N = len(front_car.xs)
        
        for i in range(1, N):
            this_v_now = this_car.vs[i - 1]
            front_v_now = front_car.vs[i - 1]
            this_s_now = this_car.xs[i - 1]
            front_s_now = front_car.xs[i - 1]
            
            gap_now = front_s_now - this_s_now - front_car.size  # the distance is head's distance
            this_v_a = self.get_new_a(this_v_now, front_v_now, gap_now)
            # update
            if this_v_now + this_v_a * time_step < 0:
                this_car.vs.append(0)
            else:
                this_car.vs.append(this_car.vs[i - 1] + time_step * this_v_a)
            this_car.xs.append(this_car.xs[i - 1] + this_car.vs[i - 1] * time_step + 1 / 2 * this_v_a * pow(time_step, 2))
            this_car.a_s.append(this_v_a)
            