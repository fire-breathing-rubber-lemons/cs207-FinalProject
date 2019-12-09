import numpy as np
import pyad.forward_mode as fwd


def simple_test():
    print('Running a simple test')
    x = fwd.Variable('x', 0.5)
    y = fwd.Variable('y', 4.2)
    z = fwd.Variable('z', 3)
    f = x * y**3 + fwd.sin(x) - fwd.logistic(-x + fwd.cos(z) ** 2)

    print('function value: ', f.value)
    print('gradient w.r.t. x: ', f.d['x'])
    print('gradient w.r.t. y: ', f.d['y'])
    print('gradient w.r.t. z: ', f.d['z'])
    print('\n')


def newtons_method_test():
    def newtons_method(f, init, max_iters=1000):
        x = fwd.Variable('x', init)
        old_res = None

        for i in range(max_iters):
            res = f(x)
            if np.sum(res.d['x']) == 0:
                break

            x = x - res.value / res.d['x']
            if old_res is not None and abs(res.value - old_res.value) < 1e-10:
                break
            old_res = res

        return x

    def f(x):
        return fwd.exp(x[0] ** 2) - fwd.log(1 + fwd.cos(x[1]) ** 2) - 86

    print('Running a test of newtons method')
    root = newtons_method(f, [1, 2])
    print('root =', root.value)
    print('f(root) =', f(root).value)
    print('\n')


if __name__ == '__main__':
    simple_test()
    newtons_method_test()
