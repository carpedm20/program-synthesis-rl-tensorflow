import os
from __init__ import Karel, draw2d

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

karel = Karel(world_path=os.path.join(BASE_PATH, "simple.map"))
print(karel.state)
karel.draw()
import ipdb; ipdb.set_trace() 

karel.move()
karel.draw()
print(karel.state)
