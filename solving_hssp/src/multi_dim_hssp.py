import time

from sage.all import log, matrix, ZZ, mod, identity_matrix, Matrix, Integers

# load("hssp.sage")

from . import building
from . import ortho_attack
from . import ns


# atm works only for k=-1
class multi_hssp:
    def __init__(self, n, ll, kappa=-1):
        self.n = n
        self.ll = ll
        self.kappa = kappa

    def gen_instance(self, m=0):  # m use to specify dimension of the sample
        if m == 0 and self.n % 2 == 1:
            m = self.n * (self.n + 3) / 2  # n is odd
        elif m == 0 and self.n % 2 == 0:
            m = self.n * (self.n + 4) / 2  # n is even
        self.m = int(m)

        print("n=", self.n, "l=", self.ll, "m=", m),
        if self.kappa > -1:
            print("kappa=", self.kappa),
        iota = 0.035
        self.nx0 = int(2 * iota * self.n**2 + self.n * log(self.n, 2))  # this is lower bound for log(Q)
        print("nx0=", self.nx0)
        # genParams returns the modulus x0, the weights vector a, the matrix x, the  sample vector b
        self.x0, self.A, self.X, self.B = genParams_mat(self.n, self.m, self.nx0, self.ll)


# This is the the function to perform the attacks.
# H is the instance to be attacked and alg is the algorithm to use :
#       **if alg='default' or alg='multi' runs the multivariate attack
#       **if alg='ns_original' runs the original Nguyen-Stern attack
#       **if alg='ns' runs the Nguyen-Stern attack with the improved orthogonal lattice attack
#       **if alg='statistical' runs the heuristic statistical attack


# def hssp_attack(H):
#     n = H.n
#     MO, tt1, tt10, tt1O = Step1_Mat(H.n, H.x0, H.A, H.X, H.B, H.m)
#     MB, beta = Step2_BK_mat(MO, H.n, H.m, H.X, kappa=-1)
#     return None


def genParams_mat(n=10, m=20, nx0=100, ll=10):
    # print "Generation of x0",
    x0 = building.genpseudoprime(nx0)

    # We generate the alpha_i's
    A = matrix(ZZ, n, ll)
    for i in range(n):
        for j in range(ll):
            A[i, j] = mod(ZZ.random_element(x0), x0)

    # The matrix X has m rows and must be of rank n
    while True:
        X = Matrix(ZZ, m, n)
        for i in range(m):
            for j in range(n):
                X[i, j] = ZZ.random_element(2)
        print(X.rank())
        if X.rank() == n:
            break
    print(X.density().n())

    # We generate an instance of the HSSP: b=X*A
    B = X * A % x0
    return x0, A, X, B


def Step1_Mat(n, x0, A, X, B, m):

    M = orthoLattice_mat(B, x0)

    print("Step 1"),
    t = time.time()
    M2 = M.LLL()
    tt10 = time.time() - t
    print("LLL step1: %.1f" % time.time() - t),

    assert sum([vi == 0 and 1 or 0 for vi in M2 * X]) == m - n

    MOrtho = M2[: m - n]

    # print
    # for i in range(m-n+1):
    #  print i,N(log(M2[i:i+1].norm(),2)),N(log(m^(n/(2*(m-n)))*sqrt((m-n)/17),2)+iota*m+nx0/(m-n))
    # #N(log(sqrt((m-n)*n)*(m/2)^(m/(2*(m-n))),2)+iota*m)

    print("  log(Height,2)=", int(log(MOrtho.height(), 2))),

    t2 = time.time()
    ke = ortho_attack.kernelLLL(MOrtho)
    tt1O = time.time() - t2
    print("  Kernel: %.1f" % time.time() - t2),
    tt1 = time.time() - t
    print("  Total step1: %.1f" % tt1)

    return ke, tt1, tt10, tt1O


# from fpylll import BKZ


def Step2_BK_mat(ke, n, m, X, kappa=-1):
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
        cl = len([True for v in M5 if ns.allbounded(v, 1)])
        if cl == n:
            break
        # if kappa>0 and cl==n-1: break

        if beta == 2:
            beta = 10
        else:
            beta += 10
    for v in M5:
        if not ns.allbounded(v, 1):
            print(v)
    print("BKZ beta=%d: %.1f" % (beta, time.time() - tbk)),
    t2 = time.time()
    MB = ns.recoverBinary(M5, kappa)
    print("  Recovery: %.1f" % time.time() - t2),
    print("  Number of recovered vector=", MB.nrows()),
    nfound = len([True for MBi in MB if MBi in X.T])
    print("  NFound=", nfound),

    # NS = MB.T

    # b=X*a=NS*ra
    # invNSn=matrix(Integers(x0),NS[:n]).inverse()
    # ra=invNSn*b[:n]
    # nrafound=len([True for rai in ra if rai in a])
    # print "  Coefs of a found=",nrafound,"out of",n,
    print("  Total BKZ: %.1f" % time.time() - tbk),

    return MB, beta


def orthoLattice_mat(H, x0):
    m, ll = H.dimensions()
    M = identity_matrix(ZZ, m)

    M[:ll, :ll] = x0 * M[:ll, :ll]
    H0i = Matrix(Integers(x0), H[:ll, :ll]).inverse()
    M[ll:m, 0:ll] = -H[ll:m, :] * H0i

    return M
