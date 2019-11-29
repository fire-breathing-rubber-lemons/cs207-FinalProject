import numpy as np
import pyad


def newtons_method(f, init, max_iters=1000):
    x = init
    old_res = f(pyad.var('x', x))

    for i in range(max_iters):
        x = x - old_res.value / old_res.d['x']
        res = f(pyad.var('x', x))
        if abs(res.value - old_res.value) < 1e-10:
            break
        old_res = res

    return x


def test_newtons_method():
    # a simple polynomial with three roots
    def f(x):
        return (x - 18) * (x + 50) * (x - 207)

    res = newtons_method(f, 100)
    assert f(res) < 1e-10

    # quintic function with no exact solution
    def f(x):
        return x**5 - x - 1

    res = newtons_method(f, 100)
    assert f(res) < 1e-10

    # a function with infinitely many roots
    def f(x):
        return pyad.sin(pyad.exp(x))

    res = newtons_method(f, 100)
    assert f(res).value < 1e-10


def gauss_newton(f, init, max_rounds=100, max_diff=1e-12):
    prev_x = x = init

    print()
    for i in range(max_rounds):
        res = f(pyad.var('x', x))
        J = res.d['x']
        print('Jacobian\n=========')
        print(J)
        print()
        x = x - np.linalg.pinv(J.T @ J) @ J.T @ res.value
        if np.linalg.norm(x - prev_x) < max_diff:
            break
        prev_x = x
    return x


def test_gauss_newton():
    # return
    # raise Exception

    def norm(x, d=2):
        return sum(v**d for v in x) ** (1/d)

    def f(d):
        x = norm(d, 4) - 1
        y = norm([3*d[0] - 1*d[1], 1*d[0] + 0*d[1]], 4) - 1
        return pyad.stack([x, y])

    init_points = [
        [1, 1], [-1, 1], [1, -1], [-1, -1]
    ]

    print('Solutions Points to f(d) = (0, 0)')
    print('=================================')
    for init in init_points:
        sol = gauss_newton(f, init)
        sol_str = '(' + ', '.join(map(str, sol.round(6))) + ')'
        print(f'f{sol_str:<22} = {f(sol).value.round(12)}')
    print('\n')
