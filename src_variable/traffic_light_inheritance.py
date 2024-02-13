import torch # Is all of torch needed?

# Defining base class

class TrafficLight:

    def __init__(self):
        pass

# Subclasses:
    
class SingleTrafficLight(TrafficLight):
    # Connecting two non-overlapping sets of roads by a single traffic light 

    def __init__(self, position, color):
        pass

class CoupledTrafficLight(TrafficLight):
    # Connecting two non-overlapping sets of roads by a coupled traffic light
    # This traffic light has two alternating states, where some incoming roads are 
    # active, and the rest are inactive, before switching to the other state

    def __init__(self, position, color):
        pass

class TrafficLightROW(CoupledTrafficLight):
    # Similar to the coupled traffic light state, but with the exception that
    # some of the acitve roads have right of way over the others
    # Still only two alternating states

    def __init__(self, position, color):
        pass

class GeneralTrafficLight(TrafficLight):
    # A traffic light with multiple alternating states, where some incoming roads
    # are active in each state, and the rest are inactive
    # Also has the option of having ROW

    def __init__(self, position, color):
        pass