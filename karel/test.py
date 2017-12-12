import os
from __init__ import Karel

BASE_PATH = os.path.realpath(__file__)

karel = Karel(world_path=os.path.join(BASE_PATH, "simple.map"))
karel.move()
