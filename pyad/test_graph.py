import reverse_mode as rev
import numpy as np

# Simple test suite explaining how Tensor works
#Initialization

x = rev.Tensor(0.5)
y = rev.Tensor(4.2)
z = rev.Tensor(3)
f = x * y**3 + rev.sin(x) - rev.logistic(z)

#set df seed
f.backward()

rev_g = rev.rev_graph()
rev_g.plot_graph([x,y,z])