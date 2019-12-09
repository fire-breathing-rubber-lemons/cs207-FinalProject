import pyad.reverse_mode as rev

x = rev.Tensor(0.5)
y = rev.Tensor(4.2)
z = rev.Tensor(3)
f = x * y**3 + rev.sin(x) - rev.logistic(z)

# set df seed
f.backward()

print('function value: ', f.value)
print('gradient w.r.t. x: ', x.grad)
print('gradient w.r.t. x: ', y.grad)
print('gradient w.r.t. x: ', z.grad)
