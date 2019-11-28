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
