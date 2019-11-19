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


def my_func(x):
    return (x - 12345678910) ** 2


if __name__ == '__main__':
    print(newtons_method(my_func, 100))
