import numpy as np
import pyad.reverse_mode as rev


def simple_test():
    print('Running a simple test')
    x = rev.Tensor(0.5)
    y = rev.Tensor(4.2)
    z = rev.Tensor(3)
    f = x * y**3 + rev.sin(x) - rev.logistic(-x + rev.cos(z) ** 2)

    # set df seed
    f.backward()

    print('function value:', f.value)
    print('gradient w.r.t. x:', x.grad)
    print('gradient w.r.t. x:', y.grad)
    print('gradient w.r.t. x:', z.grad)
    print('\n')


def newtons_method_test():
    def newtons_method(f, init, max_iters=1000):
        x = rev.Tensor(init)
        old_res = None

        for i in range(max_iters):
            res = f(x)
            res.backward()

            if np.sum(x.grad) == 0:
                break

            x = x - res.value / x.grad
            if old_res is not None and abs(res.value - old_res.value) < 1e-10:
                break
            old_res = res
            x.reset_grad()

        return x

    def f(x):
        return rev.exp(x[0] ** 2) - rev.log(1 + rev.cos(x[1]) ** 2) - 86

    print('Running a test of newtons method')
    root = newtons_method(f, [1, 2])
    print('root =', root.value)
    print('f(root) =', f(root).value)
    print('\n')


if __name__ == '__main__':
    simple_test()
    newtons_method_test()
