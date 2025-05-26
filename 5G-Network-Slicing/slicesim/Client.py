import random
import numpy as np
from .utils import distance

class Client:
    def __init__(self, id, x, y, slice_type, env, mobility=0.5, base_station=None):
        """Initialize a client
        
        Args:
            id (int): Client ID
            x (float): X coordinate
            y (float): Y coordinate
            slice_type (str): Type of slice ('eMBB', 'URLLC', 'mMTC', 'voice')
            env (simpy.Environment): Simulation environment
            mobility (float): Mobility factor (0-1)
            base_station (BaseStation): Initial base station
        """
        self.id = id
        self.x = x
        self.y = y
        self.slice_type = slice_type
        self.env = env
        self.mobility = mobility
        self.base_station = base_station
        self.connected = True
        self.active = True
        
        # For compatibility with original code
        self.pk = id
        
        # Traffic generation parameters based on slice type
        if slice_type == "eMBB":
            self.usage_mean = 15e6  # 15 Mbps average
            self.usage_var = 5e6    # 5 Mbps variance
            self.usage_freq = 0.8   # High usage frequency
        elif slice_type == "URLLC":
            self.usage_mean = 2e6   # 2 Mbps average
            self.usage_var = 0.5e6  # Low variance (consistent)
            self.usage_freq = 0.6   # Medium frequency
        elif slice_type == "mMTC":
            self.usage_mean = 0.5e6 # 0.5 Mbps average
            self.usage_var = 0.2e6  # Low variance
            self.usage_freq = 0.3   # Low frequency (intermittent)
        else:  # voice
            self.usage_mean = 0.1e6 # 0.1 Mbps average
            self.usage_var = 0.05e6 # Very low variance
            self.usage_freq = 0.5   # Medium frequency
    
    def generate_traffic(self):
        """Generate traffic according to a random process"""
        while True:
            # Wait some time between traffic generations
            wait_time = random.uniform(0.5, 2.0)
            yield self.env.timeout(wait_time)
            
            # Check if client should generate traffic
            if random.random() < self.usage_freq and self.connected and self.active:
                # Generate traffic amount
                amount = max(0, np.random.normal(self.usage_mean, self.usage_var))
                
                # Get appropriate slice from base station
                slice_obj = self.get_slice()
                
                if slice_obj and slice_obj.capacity.level >= amount:
                    try:
                        # Consume resources
                        yield slice_obj.capacity.get(amount)
                        
                        # Hold resources for some time
                        hold_time = random.uniform(0.5, 1.5)
                        yield self.env.timeout(hold_time)
                        
                        # Release resources
                        yield slice_obj.capacity.put(amount)
                    except Exception as e:
                        print(f"Error generating traffic for client {self.id}: {e}")
            
            # Move client
            if random.random() < self.mobility:
                self.move()
    
    def move(self):
        """Move the client randomly"""
        # Generate random movement
        movement_scale = 30.0 * self.mobility  # Scale movement based on mobility factor
        dx = random.uniform(-movement_scale, movement_scale)
        dy = random.uniform(-movement_scale, movement_scale)
        
        # Update position
        self.x += dx
        self.y += dy
        
        # Check if still in coverage
        if self.base_station:
            center = self.base_station.coverage.center
            radius = self.base_station.coverage.radius
            
            dist = distance((self.x, self.y), center)
            
            # If out of coverage, disconnect
            if dist > radius:
                self.connected = False
                # Could search for a new base station here
    
    def connect_to_base_station(self, base_station):
        """Connect to a base station"""
        self.base_station = base_station
        self.connected = True
    
    def get_slice(self):
        """Get the appropriate slice from the base station"""
        if not self.base_station:
            return None
            
        # Find matching slice type
        for s in self.base_station.slices:
            if s.name == self.slice_type:
                return s
        
        # If no matching slice found, return first slice as fallback
        if self.base_station.slices:
            return self.base_station.slices[0]
        
        return None
