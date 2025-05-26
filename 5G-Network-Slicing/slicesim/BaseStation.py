class BaseStation:
    def __init__(self, pk, coverage, capacity_bandwidth=1000, slices=None):
        self.pk = pk
        self.coverage = coverage
        self.capacity_bandwidth = capacity_bandwidth
        self.slices = [] if slices is None else slices
        self.id = pk  # Add id attribute for compatibility
        print(self)

    def __str__(self):
        return f'BS_{self.pk:<2}\t cov:{self.coverage}\t with cap {self.capacity_bandwidth:<5}'
        
    def add_slice(self, slice_obj):
        """Add a slice to this base station"""
        self.slices.append(slice_obj)
        print(f"Added {slice_obj.name} slice to BS_{self.pk}")

