import time

from sage.all import matrix, ZZ, Integers, round

import numpy as np
from .fica import ICA

from . import ortho_attack


def Sage2np(MO, n, m):
    MOn = np.matrix(MO)
    return MOn


def runICA(MOn, B=1):
    t1 = time.time()
    A_, S = ICA(MOn, B)
    S_ = np.dot(np.linalg.inv(A_), MOn)
    print("time Ica_S: ", time.time() - t1),
    S2 = matrix(ZZ, MOn.shape[0], MOn.shape[1], round(S_))
    return S2


def runICA_A(MOn, B=1, n=16, kappa=-1):
    t1 = time.time()
    A_, S = ICA(MOn, B, n, kappa)
    print("time Ica_A: ", time.time() - t1),
    A2 = matrix(ZZ, MOn.shape[0], MOn.shape[0], round(A_))
    return A2


def statistical(MO, n, m, x0, X, a, b, kappa, B=1, variant=None):

    if variant is None:
        if n <= 200:
            variant = "roundA"
        else:
            variant = "roundX"

    print("Step 2-ICA: ", variant)

    t2 = time.time()
    # print "matNbits=",matNbits(MO),
    tlll = time.time()
    MO = MO.LLL()
    print(" time LLL=", time.time() - tlll, "mathNbits=", ortho_attack.matNbits(MO)),

    MOn = Sage2np(MO, n, m)

    if variant == "roundA":
        A2 = runICA_A(MOn, B, n, kappa)
        try:
            S2 = A2.inverse() * MO
            print("mathNbits A=", ortho_attack.matNbits(A2)),
        except ZeroDivisionError:
            return 0, 0, 0
    elif variant == "roundX":
        S2 = runICA(MOn, B)
        print("mathNbits X=", ortho_attack.matNbits(X)),
    else:
        raise NameError("Variant algorithm non acceptable")
    tica = time.time() - t2
    print(" cputime ICA %.2f" % tica),

    tc = time.time()
    Y = X.T
    nfound = 0
    for i in range(n):
        for j in range(n):
            if S2[i, :n] == Y[j, :n] and S2[i] == Y[j]:
                nfound += 1
    t = time.time() - tc
    print("  NFound=", nfound, "out of", n, "check= %.2f" % t)

    if nfound < n:
        print()
        print()
        return tica, 0, nfound

    NS = S2.T

    tcoff = time.time()
    # b=X*a=NS*ra
    invNSn = matrix(Integers(x0), NS[:n]).inverse()
    ra = invNSn * b[:n]
    tcf = time.time() - tcoff

    nrafound = len([True for rai in ra if rai in a])
    print("  Coefs of a found=", nrafound, "out of", n, " time= %.2f" % tcf)

    tS2 = tcf + tica
    print("  Total step2: %.1f" % tS2),

    return tica, tS2, nrafound
