import time

import numpy as np

from sage.all import Matrix, ZZ, matrix, vector, Integers, ones_matrix, copy

from . import multi


def is_cl(a, b, atol=0.01, rtol=1):
    return abs(a - b) <= (atol + rtol * abs(b))


def allpmones(v):
    return len([vj for vj in v if vj in [-1, 0, 1]]) == len(v)


def recoverBinary(M5, kappa=-1):
    lv = [allones(vi) for vi in M5 if allones(vi)]
    # print(f"{len(lv)=}")
    n = M5.nrows()
    m = M5.ncols()
    for v in lv:
        for i in range(n):
            nv = allones(M5[i] - v)
            if nv and nv not in lv:
                lv.append(nv)
            nv = allones(M5[i] + v)
            if nv and nv not in lv:
                lv.append(nv)
        if len(lv) >= 10 * n:
            raise ValueError(f"Too many vectors found ({len(lv)}>=10*n={m*n})")
    return Matrix(lv)


def allones(v):
    if len([vj for vj in v if vj in [0, 1]]) == len(v):
        return v
    if len([vj for vj in v if vj in [0, -1]]) == len(v):
        return -v
    return None


def Step2_BKZ_binary(ke, n, m, x0, X, b, a):
    # if n>170: return
    beta = 2
    tbk = time.time()
    while beta < n:
        print(beta)
        if beta == 2:
            M5 = ke.LLL()
            M5 = M5[:n]  # this is for the affine case
        else:
            M5 = M5.BKZ(block_size=beta)

        # we succeed if we only get vectors with {-1,0,1} components
        cl = len([True for v in M5 if allpmones(v)])
        if cl == n:
            print("DONE")
            break

        if beta == 2:
            beta = 10
        else:
            beta += 10

    print("BKZ beta=%d: %.1f" % (beta, time.time() - tbk)),
    t2 = time.time()
    MB = recoverBinary(M5)
    print("  Recovery: %.1f" % (time.time() - t2)),
    print("  Number of recovered vector=", MB.nrows()),
    nfound = len([True for MBi in MB if MBi in X.T])
    print("  NFound=", nfound),

    # NS=MB.T
    # b=X*a=NS*ra
    # invNSn=matrix(Integers(x0),NS[:n]).inverse()
    # ra=invNSn*b[:n]
    # nrafound=len([True for rai in ra if rai in a])
    # print "  Coefs of a found=",nrafound,"out of",n,
    print("  Total step2: %.1f" % (time.time() - tbk)),

    return MB, beta


def allbounded(v, B):
    return len([vj for vj in v if vj >= -B and vj <= B]) == len(v)


def allBounded(v, B):
    if len([vj for vj in v if vj >= 0 and vj <= B]) == len(v):
        return v
    if len([vj for vj in v if vj <= 0 and vj >= -B]) == len(v):
        return -v
    return None


def recoverBounded(M5, B):
    lv = [allBounded(vi, B) for vi in M5 if allBounded(vi, B)]
    n = M5.nrows()
    for v in lv:
        for i in range(n):
            nv = allBounded(M5[i] - v, B)
            if nv and nv not in lv:
                lv.append(nv)
            nv = allBounded(M5[i] + v, B)
            if nv and nv not in lv:
                lv.append(nv)
    return Matrix(lv)


# from fpylll import BKZ


def Step2_BKZ(ke, B, n, m, x0, X, b, a, kappa, original=False):
    # if n>170: return
    beta = 2
    tbk = time.time()
    while beta < n:
        print(beta)
        if beta == 2:
            M5 = ke.LLL()
            M5 = M5[:n]  # this is for the affine case
        else:
            #      M5=M5.BKZ(block_size=beta, strategies=BKZ.DEFAULT_STRATEGY, flags=BKZ.AUTO_ABORT|BKZ.GH_BND)
            M5 = M5.BKZ(block_size=beta)

        # we succeed if we only get vectors with {-1,0,1} components, for kappa>0 we relax this condition
        # to all except one vector
        # if len([True for v in M5 if allpmones(v)])==n: break
        cl = len([True for v in M5 if allbounded(v, B)])
        if cl == n:
            break
        # if kappa>0 and cl==n-1: break

        if beta == 2:
            beta = 10
        else:
            beta += 10
    for v in M5:
        if not allbounded(v, B):
            print(f"unbaalanced v: {v}")
    print("BKZ beta=%d: %.1f" % (beta, time.time() - tbk)),
    t2 = time.time()
    if B == 1:
        MB = recoverBinary(M5, kappa)
    else:
        MB = recoverBounded(M5, B)
    print("  Recovery: %.1f" % (time.time() - t2)),
    print("  Number of recovered vector=", MB.nrows()),
    nfound = len([True for MBi in MB if MBi in X.T])
    print("  NFound=", nfound),

    # NS=MB.T
    # b=X*a=NS*ra
    # invNSn=matrix(Integers(x0),NS[:n]).inverse()
    # ra=invNSn*b[:n]
    # nrafound=len([True for rai in ra if rai in a])
    # print "  Coefs of a found=",nrafound,"out of",n,
    print("  Total BKZ: %.1f" % (time.time() - tbk)),

    return MB, beta


def ns_original(H, MO):
    B = 1
    # m=int(max(2*n,16*log(n,2)))
    kappa = H.kappa
    n = H.n
    m = H.m

    bb = 1 / 2
    unbalanced = (abs(n * bb - kappa) / n) > 0.2
    print("unbalanced: ", unbalanced)

    t2 = time.time()
    NSo, beta = Step2_BKZ(matrix(ZZ, MO), B, H.n, H.m, H.x0, H.X, H.b, H.a, H.kappa, original=True)
    tt2 = time.time() - t2

    # return NSo

    print(f"NSo.dimensions()={NSo.dimensions()}")
    assert NSo.nrows() >= n - 1, "error Step2--not enough vector found"

    if kappa > 0 and B == 1 and unbalanced:
        print("    case 1:")
        li2 = NSo.rows()
        if len(li2) < n:
            ones = vector([1 for i in range(m)])
            if kappa > n / 2:
                missing = (n - kappa) * ones - sum(li2)
                NS = matrix(ZZ, [ones - x for x in li2 + [missing]])
            else:
                missing = (kappa) * ones - sum(li2)
                NS = matrix(ZZ, li2 + [missing])
        else:
            NS = matrix(ZZ, li2)
        print(NS.dimensions())
        Y = NS.T
        assert Y.rank() == n, "rank<n! extra binary vector to handle"  # we have to fix this.
    tt2 = time.time() - t2

    textra0 = time.time()
    if kappa != -1 and B == 1 and not unbalanced:
        print("    case 2:")
        ones = vector([1 for i in range(m)])
        li = NSo.rows()
        for NSi in NSo:
            if ones - NSi not in li:
                li.append(ones - NSi)
        li2 = []
        for r in li:
            if ones - r not in li2:
                li2.append(r)
        if len(li2) < n and ones not in li2:
            li2 = [ones] + li2
        assert len(li2) == n

        NS = matrix(ZZ, li2)
        thereisones = ones in li2
        print("ones in NS: ", thereisones)
        # ts = time.time()

        if thereisones:
            wm = sum([xi for xi in Y if xi != ones])
            k0 = max(wm)
            wm = k0 * ones - wm
            jone = (Y.rows()).index(ones)
            Y[jone] = wm
        # assert v1 == 0 # TODO: check this
        if k0 == n - kappa:
            for j in range(n):
                Y[j] = multi.ned(Y[j])

        nyfound = len([True for y in Y if y in H.X.T])
        print("  NFound=", nyfound, "out of", n),

        Y = Y.T
    elif not unbalanced or kappa == -1:
        print("    case 3:")
        Y = NSo.T
        nfound = len([True for NSi in NSo if NSi in H.X.T])
        print("  NFound=", nfound, "out of", n),

    # print("Y: ", Y)
    # return Y
    invNSn = matrix(Integers(H.x0), Y[:n]).inverse()
    ra = invNSn * H.b[:n]
    nrafound = len([True for rai in ra if rai in H.a])

    print("  Total step2: %.1f" % (time.time() - t2)),
    print("  Coefs of a found=", nrafound, "out of", n)

    if kappa == n / 2 and nyfound == 0:
        print("\n-->Reverse")
        nyfound = len([True for y in Y.T if ones - y in H.X.T])
        print("  NFound=", nyfound, "out of", n),
        OO = ones_matrix(ZZ, *Y.dimensions())
        Y = OO - Y
        invNSn = matrix(Integers(H.x0), Y[:n]).inverse()
        ra = invNSn * H.b[:n]
        nrafound = len([True for rai in ra if rai in H.a])
        print("  Coefs of a found=", nrafound, "out of", n)

    textra = time.time() - textra0

    return beta, tt2, nrafound, textra, ra


def ns(H, MO, B=1):
    # m=int(max(2*n,16*log(n,2)))
    kappa = H.kappa
    n = H.n
    m = H.m

    t2 = time.time()
    NSo, beta = Step2_BKZ(matrix(ZZ, MO), B, H.n, H.m, H.x0, H.X, H.b, H.a, H.kappa)
    tt2 = time.time() - t2

    print(NSo.dimensions())
    assert NSo.nrows() >= n - 1, "error Step2"

    bb = B / (B + 1)
    unbalanced = abs(n * bb - kappa) / n > 0.2
    print(unbalanced)

    if kappa > 0 and B == 1 and unbalanced:
        li2 = NSo.rows()
        if len(li2) < n:
            ones = vector([1 for i in range(m)])
            if kappa > n / 2:
                missing = (n - kappa) * ones - sum(li2)
                NS = matrix(ZZ, [ones - x for x in li2 + [missing]])
            else:
                missing = (kappa) * ones - sum(li2)
                NS = matrix(ZZ, li2 + [missing])
        else:
            NS = matrix(ZZ, li2)
        print(NS.dimensions())
        Y = NS.T
        assert Y.rank() == n, "rank<n! extra binary vector to handle"  # we have to fix this.
    tt2 = time.time() - t2

    textra0 = time.time()
    if kappa != -1 and B == 1 and not unbalanced:
        ones = vector([1 for i in range(m)])
        li = NSo.rows()
        for NSi in NSo:
            if ones - NSi not in li:
                li.append(ones - NSi)
        li2 = []
        for r in li:
            if ones - r not in li2:
                li2.append(r)
        if len(li2) < n and ones not in li2:
            li2 = [ones] + li2
        assert len(li2) == n

        NS = matrix(ZZ, li2)
        thereisones = ones in li2
        print("ones in NS: ", thereisones)
        ts = time.time()

        e = vector([1] * n)
        # if kappa<=n//2:
        Y = matrix(ZZ, NS[:n])
        # print ones in Y.rows()
        # else: Y=matrix(ZZ,NS[n:2*n])
        Y = copy(Y)
        v0 = np.var(e * Y)
        i = 0
        if thereisones:
            var_w = kappa * (n - kappa) / n**2
            cv0 = is_cl(v0, var_w)
        else:
            var_w = 0
            cv0 = True
        while i < n**2 and not cv0:
            j = i % n
            # print j
            if Y[j] == ones:
                i += 1
                print(j)
                continue
            Y[j] = multi.ned(Y[j])
            v1 = np.var(e * Y)
            # print v0, v1
            if is_cl(v1, var_w) and thereisones:
                break
            if v1 == 0 and not thereisones:
                break
            if v1 < v0:
                v0 = v1
            else:
                Y[j] = multi.ned(Y[j])
            i = i + 1
        print(v0, v1)
        print("Switching rounds", i, "time= %.1f" % (time.time() - ts))
        if thereisones:
            wm = sum([xi for xi in Y if xi != ones])
            k0 = max(wm)
            wm = k0 * ones - wm
            jone = (Y.rows()).index(ones)
            Y[jone] = wm
            v1 = np.var(e * Y)
        assert v1 == 0
        if k0 == n - kappa:
            for j in range(n):
                Y[j] = multi.ned(Y[j])

        nyfound = len([True for y in Y if y in H.X.T])
        print("  NFound=", nyfound, "out of", n),

        Y = Y.T
    elif not unbalanced or kappa == -1:
        Y = NSo.T
        nfound = len([True for NSi in NSo if NSi in H.X.T])
        print("  NFound=", nfound, "out of", n),

    invNSn = matrix(Integers(H.x0), Y[:n]).inverse()
    ra = invNSn * H.b[:n]
    nrafound = len([True for rai in ra if rai in H.a])

    print("  Total step2: %.1f" % (time.time() - t2)),
    print("  Coefs of a found=", nrafound, "out of", n)

    if kappa == n / 2 and nyfound == 0:
        print("\n-->Reverse")
        nyfound = len([True for y in Y.T if ones - y in H.X.T])
        print("  NFound=", nyfound, "out of", n),
        OO = ones_matrix(ZZ, *Y.dimensions())
        Y = OO - Y
        invNSn = matrix(Integers(H.x0), Y[:n]).inverse()
        ra = invNSn * H.b[:n]
        nrafound = len([True for rai in ra if rai in H.a])
        print("  Coefs of a found=", nrafound, "out of", n)

    textra = time.time() - textra0

    return beta, tt2, nrafound, textra
