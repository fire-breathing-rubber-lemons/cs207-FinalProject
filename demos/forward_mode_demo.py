import pyad.forward_mode as fwd

x = fwd.Variable('x', 1)
y = fwd.Variable('y', 2)
z = fwd.Variable('z', 3)

def test_fun(x, y, z):
	return fwd.exp(fwd.cos(x) + fwd.sin(y))**z

result = test_fun(x,y,z)
print('function value: ', result.value)
print('derivative w.r.t. x: ', result.d['x'])
print('derivative w.r.t. y: ', result.d['y'])
print('derivative w.r.t. z: ', result.d['z'])
