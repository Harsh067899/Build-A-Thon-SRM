import math
import numpy as np

def distance(a, b):
    """Calculate Euclidean distance between two points"""
    return math.sqrt(sum((i-j)**2 for i,j in zip(a, b)))

def format_bps(size, pos=None, return_float=False):
    """Format bits per second with appropriate unit prefix"""
    # https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb
    power, n = 1000, 0
    power_labels = {0 : '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size >= power:
        size /= power
        n += 1
    if return_float:
        return f'{size:.3f} {power_labels[n]}bps'
    return f'{size:.0f} {power_labels[n]}bps'

class KDTree:
    """Simple KD-Tree implementation for nearest neighbor search"""
    
    def __init__(self, points):
        """Initialize KD-Tree with points
        
        Args:
            points (list): List of points [(x1, y1), (x2, y2), ...]
        """
        self.points = points
        self.n = len(points)
        
        # Convert to numpy array for faster operations
        if self.n > 0:
            self.points_array = np.array(points)
        else:
            self.points_array = np.array([])
    
    def query_nearest(self, query_point):
        """Find index of nearest point to query_point
        
        Args:
            query_point (tuple): Query point (x, y)
            
        Returns:
            int: Index of nearest point in the original points list
        """
        if self.n == 0:
            return None
            
        # Calculate distances to all points
        query_array = np.array(query_point)
        if len(query_array.shape) == 1:
            # Single point query
            distances = np.sqrt(np.sum((self.points_array - query_array)**2, axis=1))
            return np.argmin(distances)
        else:
            # Multiple point query
            results = []
            for q in query_array:
                distances = np.sqrt(np.sum((self.points_array - q)**2, axis=1))
                results.append(np.argmin(distances))
            return results
    
    def query_radius(self, query_point, radius):
        """Find indices of points within radius of query_point
        
        Args:
            query_point (tuple): Query point (x, y)
            radius (float): Search radius
            
        Returns:
            list: Indices of points within radius
        """
        if self.n == 0:
            return []
            
        # Calculate distances to all points
        query_array = np.array(query_point)
        distances = np.sqrt(np.sum((self.points_array - query_array)**2, axis=1))
        
        # Return indices of points within radius
        return np.where(distances <= radius)[0].tolist()
        
    # Add static methods for backward compatibility
    last_run_time = 0
    
    @staticmethod
    def run(clients, base_stations, run_at, assign=True):
        """For compatibility with original code"""
        if run_at == KDTree.last_run_time:
            return
        KDTree.last_run_time = run_at
        
        c_coor = [(c.x, c.y) for c in clients]
        bs_coor = [p.coverage.center for p in base_stations]

        # Create tree and query for nearest points
        tree = KDTree(bs_coor)
        
        for i, client in enumerate(clients):
            # Find nearest base station
            nearest_idx = tree.query_nearest(c_coor[i])
            
            if nearest_idx is not None:
                nearest_bs = base_stations[nearest_idx]
                d = distance(c_coor[i], nearest_bs.coverage.center)
                
                # Assign client to base station if within coverage
                if assign and d <= nearest_bs.coverage.radius:
                    client.base_station = nearest_bs
                
                # For compatibility with original code
                client.closest_base_stations = [(d, nearest_bs)]

# Initial connections using k-d tree
def kdtree(clients, base_stations):
    """Connect clients to nearest base stations using KDTree"""
    c_coor = [(c.x, c.y) for c in clients]
    bs_coor = [p.coverage.center for p in base_stations]

    tree = KDTree(bs_coor)
    
    for i, client in enumerate(clients):
        # Find nearest base station
        nearest_idx = tree.query_nearest(c_coor[i])
        
        if nearest_idx is not None:
            nearest_bs = base_stations[nearest_idx]
            d = distance(c_coor[i], nearest_bs.coverage.center)
            
            # Assign client to base station if within coverage
            if d <= nearest_bs.coverage.radius:
                client.base_station = nearest_bs
