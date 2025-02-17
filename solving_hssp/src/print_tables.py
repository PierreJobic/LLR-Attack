from sage.all import Integer


def table(L):
    for v in L:
        k = len(v)
        for i in range(k - 1):
            if isinstance(v[i], int):
                print("~", v[i], "~&"),
            else:
                print("~ %.1f" % v[i], "~&"),

        print("~", v[k - 1], "~\\\\")


def table_s(L):
    for v in L:
        k = len(v)
        for i in range(k - 1):
            if isinstance(v[i], (int, Integer)):
                print("~", v[i], "~&"),
            elif v[i] < 180:
                print("~ %.1f" % v[i], " s ~&"),
            else:
                print("~ %d" % round(v[i] / 60), " min ~&"),
        if isinstance(v[i], (int, Integer)):
            print("~", v[-1], "~\\\\")
        elif v[-1] < 180:
            print("~ %.1f" % v[-1], " s ~\\\\")
        else:
            print("~ %d" % round(v[-1] / 60), " min ~\\\\")
