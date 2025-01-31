# This is the source code for "The Hidden Lattice Problem"
# The implementation is in SageMath 9.2, which can be downloaded from https://www.sagemath.org
# Content:
# implementation of our adaptation of the orthogonal lattice attack following the algorithm of
#  Nguyen-Stern (Algorithm I)
# implementation of our alternative algorithm based on the dual lattice (Algorithm II)
# various functions to run tests for related problems DHLP, NHLP

import time

import numpy as np

from sage.all import matrix, IntegerModRing, gcd, ZZ, QQ, ceil, vector, log, product, sqrt, next_prime, copy, Integer
from sage.all import kernel, random_matrix, is_prime, GF, pi, exp
from sage.modules.free_module_integer import IntegerLattice

# from sage.matrix.matrix_integer_dense_saturation import p_saturation
# from time import time

# we implement our adapted orthogonal lattice attack following the algorithm of Nguyen-Stern
# we implement our alternative algorithm


# A: preliminary functions
def modNear(a, b):
    # Input: integers a,b
    # Output: the central residue of a modulo b
    res = a % b
    if res > b / 2:
        return res - b
    return res


def ortho_lattice_mod(B, N):
    # Input: a basis matrix B (in ZZ**rxm, r<m) of rank r lattice in ZZ**m (vectors in rows), N a natural integer
    # Output: a basis matrix of size mxm of the lattice of vectors orthogonal to the rows of B modulo N
    # the algorithm assumes submatrix of B invertible modulo N
    r = B.nrows()
    m = B.ncols()
    B1 = B[0:r, 0 : m - r]
    B2 = B[0:r, m - r : m]
    if gcd(B2.determinant(), N) == 1:
        B1 = matrix(IntegerModRing(N), B1)
        B2 = matrix(IntegerModRing(N), B2)
        mat = matrix(ZZ, m, m)
        mat[0 : m - r, 0 : m - r] = matrix.identity(m - r)
        mat[m - r : m, 0 : m - r] = -matrix(ZZ, B2.inverse() * B1)
        mat[m - r : m, m - r : m] = N * matrix.identity(r)
    else:
        print("no inverse")
    if gcd(B2.determinant(), N) == 1:
        return mat.transpose()
    else:
        return False


def ortho_lattice(B):
    # Input: Basis matrix B(rxm) of rank r<m lattice in ZZ**m
    # Output: Basis matrix of size m-rxm of ortho lattice
    r = B.nrows()
    m = B.ncols()
    prodB = Integer(1)
    for j in range(r):
        prodB = prodB * (B.row(j).norm(2))
    # print(prodB)
    c = ceil((2) ** ((m - 1) / (2) + (m - r) * (m - r - 1) / (4)) * (prodB))
    BOrtho = (c * B.transpose()).augment(matrix.identity(m))
    RBOrtho = BOrtho.LLL()
    return RBOrtho[0 : m - r, r : m + r]


def span_lattice_mod(B, N):
    # Input: Basis matrix B(rxm) of rank r lattice in ZZ**m, a natural integer N
    # Output: Basis matrix of size (mxm) of lattice of span of B modulo N
    r = B.nrows()
    m = B.ncols()
    B1 = B[0:r, 0 : m - r]
    B2 = B[0:r, m - r : m]
    if gcd(B2.determinant(), N) == 1:
        B1 = matrix(IntegerModRing(N), B1)
        B2 = matrix(IntegerModRing(N), B2)
        B0 = matrix(ZZ, B2.inverse() * B1)
        mat = matrix(ZZ, m, m)
        mat[0 : m - r, 0 : m - r] = N * matrix.identity(m - r)
        mat[m - r : m, 0 : m - r] = matrix(ZZ, B0)
        mat[m - r : m, m - r : m] = matrix.identity(r)
    else:
        print("no inverse")
    if gcd(B2.determinant(), N) == 1:
        return mat
    else:
        return False


def completion(L):
    # Input: a lattice L
    # Output: the completion of L, computed by intersection
    B = L.basis_matrix()
    m = B.ncols()
    return B.row_space(QQ).intersection(ZZ**m)


def completion_using_orthogonal(L):
    # Input: a lattice L
    # Output: the completion of L, computed by double orthogonal
    LO = ortho_lattice(L.basis_matrix())
    LOO = ortho_lattice(LO)
    return LOO.row_space(ZZ)


def size_basis(B):
    # Input: a basis matrix B of some lattice of rank n
    # Output: an approximation of the size of the basis
    BB = np.array(B).astype(float)
    # print("norms of the basis vectors:")
    # for b in sorted([np.linalg.norm(b) ** 2 for b in BB]):
    #     print(b)
    return sqrt(1 / len(BB)) * sqrt(sum([np.linalg.norm(b) ** 2 for b in BB]))
    return (sqrt(1 / B.nrows()) * sqrt(sum([(b.norm(2)) ** 2 for b in B.rows()]))).n()


def local_completion(mat, p):
    # Input: a matrix mat and a prime p
    # output: the completion at p of the matrix mat; interpreted as the p-completion of the lattice with basis
    # the rows of mat
    comp_mat = copy(mat)
    comp_mat_p = matrix(GF(p), mat)
    ker_p = kernel(comp_mat_p)
    while ker_p.dimension() != 0:
        ker_p_basis = ker_p.basis_matrix().echelon_form()
        v = ker_p_basis[0]
        # lift the vector to Z
        lift_v = matrix([modNear(ZZ(vi), p) for vi in list(v)])
        # apply the vector to M
        z = lift_v * comp_mat
        # replace the ith row of M by x=1/p*z
        x = z / p
        # replace i-th row of mat by x
        i = list(v).index(1)
        comp_mat[i, :] = vector(x)
        # reduce mod p again
        comp_mat_p = matrix(GF(p), comp_mat)
        ker_p = kernel(comp_mat_p)
    return comp_mat


def gen_hlp(n, m, r, bound, N):
    # Input: parameters n,m,r,bound,N where
    # n is the rank of the hidden lattice
    # m is the dimension of the hidden lattice
    # r is the rank of the public lattice
    # bound is an upper bound on the absolute value of the entries of a basis of the hidden lattice
    # N is the public modulus
    # Output: generation of a hidden lattice problem with the given parameters:
    # a hidden lattice L with size upper bounded by size_L (and small basis X)
    # a lattice basis B such that the lattice M spanned by B lies in L modulo N
    X = matrix(ZZ, n, m)
    for i in range(n):
        for j in range(m):
            X[i, j] = ZZ.random_element(-bound, bound)
    size_L = size_basis(X)
    params = (n, m, r, size_L, N)
    L = IntegerLattice(X)
    A = matrix(ZZ, r, n)
    for i in range(r):
        for j in range(n):
            A[i, j] = ZZ.random_element(N)
    B = (A * X) % N
    return (params, X, L, B)


# B: Algorithm I
def heuristic_logN_I(h):
    # Input: a hidden lattice problem h
    # Output: heuristic lower bound on logN for Algorithm I based on Gaussian Heuristic
    (n, m, r, size_L, _) = h[0]
    # X, L, B = h[1], h[2], h[3]

    hermite = 0.03
    alpha = n / (m - n)
    mu = size_L

    log_modulus = ceil(n / r * (alpha + 1) * log(mu, 2) + m * n / r * hermite + n / (2 * r) * log(m - n, 2))
    return log_modulus


def ortho_algo(h):
    # Input: a hidden lattice problem h
    # Output: a basis candidate for completion(L) using the N-orthogonal lattice, and True/False according to whether
    # the hidden lattice is revealed successfully or not
    (n, m, _, size_L, N) = h[0]
    _, L, B = h[1], h[2], h[3]
    print("(n,m,r,size_L,N)", h[0])
    print("log(size_L)", log(size_L, 2).n())
    print("log(N)", log(N, 2).n())

    # Step 1: compute a reduced basis of the lattice of vectors orthogonal to B modulo N
    orthoB = ortho_lattice_mod(B, N)
    if not orthoB:
        print("sample a new problem")
        check = False
    else:
        # compute reduced basis of orthoB
        t_1 = time.time()
        orthoBLLL = orthoB.LLL()
        # extract first m-n vectors and compute basis of orthogonal lattice
        ext = orthoBLLL[0 : m - n]
        print("timing step 1:", (time.time() - t_1))

        # Step 2: compute a reduced basis of the ZZ-orthogonal to the extracted vectors
        t_2 = time.time()
        ortho_ext = ortho_lattice(ext)
        print("timing step 2:", (time.time() - t_2))

        # Step 3: verify the solution and compute the size of the output
        t_3 = time.time()
        # recovered lattice
        reclat = ortho_ext.row_space(ZZ)
        check = reclat == L
        print("recovered lattice is equal to L:", check)
        print("log(size) output basis:", log(size_basis(ortho_ext), 2).n())
        print("timing step 3:", (time.time() - t_3))

        print("total timing:", (time.time() - t_1))
    return ext, ortho_ext, check


def ortho_algo_compute_N1(h):
    # Input: a hidden lattice problem h
    # Output: computation (and timing thereof) of a basis of N_I in the orthogonal lattice algorithm (Algorithm I)
    (n, m, r, size_L, N) = h[0]
    _, L, B = h[1], h[2], h[3]
    print("(n,m,r,size_L,N)", h[0])
    print("log(size_L)", log(size_L, 2).n())
    print("log(N)", log(N, 2).n())

    # Step 1: compute a reduced basis of the lattice of vectors orthogonal to B modulo N
    orthoB = ortho_lattice_mod(B, N)
    if not orthoB:
        print("sample a new problem")
    else:
        # compute reduced basis of orthoB
        L_ortho = ortho_lattice(L.basis_matrix()).row_space(ZZ)
        t_1 = time.time()
        orthoBLLL = orthoB.LLL()
        # extract first m-n vectors and compute basis of orthogonal lattice
        ext = orthoBLLL[0 : m - n]
        print("timing step 1:", (time.time() - t_1))
        print("ext is sublattice of L_ortho:", ext.row_space(ZZ).is_submodule(L_ortho))


# C: Algorithm II
def heuristic_logN_II(h):
    # Input: a hidden lattice problem h
    # Output: heuristic lower bound on logN for Algorithm II based on Gaussian Heuristic
    (n, m, r, size_L, _) = h[0]
    # X, L, B = h[1], h[2], h[3]

    hermite = 0.03
    alpha = n / (m - n)
    mu = size_L

    log_modulus = ceil(n / r * (alpha + 1) * log(mu, 2) + (alpha + 1) * m * n / r * hermite)
    return log_modulus


def span_algo(h):
    # Input: a hidden lattice problem h
    # Output: a basis candidate for completion(L) using the N-congruence lattice, and True/False according to whether
    # the hidden lattice is revealed successfully or not
    (n, _, _, size_L, N) = h[0]
    _, L, B = h[1], h[2], h[3]
    print("(n,m,r,size_L,N)", h[0])
    print("log(size_L)", log(size_L, 2).n())
    print("N", N)
    print("log(N)", log(N, 2).n())

    # Step 1: compute a reduced basis of the lattice in the Z-span of B modulo N
    spanB = span_lattice_mod(B, N)
    if (not spanB) or (not is_prime(N)):
        print("sample a new problem using prime N")
        check = False
    else:
        # compute reduced basis of spanB
        s_1 = time.time()
        spanBLLL = spanB.LLL()
        # extract first n vectors
        ext = spanBLLL[0:n]
        print("timing step 1:", (time.time() - s_1))

        # Step 2: compute completion of the lattice spanned by the extracted vectors
        s_2 = time.time()
        # comp_ext=ext.saturation()                   #using build-in Sage completion function
        # comp_ext=completion_using_orthogonal(ext)   #using double orthogonal complement calculation
        # comp_ext=p_saturation(ext,N)                #using build-in Sage p-completion function
        comp_ext = local_completion(ext, N)  # using our own local_completion function
        print("timing step 2:", (time.time() - s_2))
        # print('index',ext.index_in_saturation().factor())  #factors the index

        # Step 3: verify the solution and compute/reduce the size of the output
        s_3 = time.time()
        reclat = comp_ext.row_space(ZZ)
        check = reclat == L
        print("recovered lattice is equal to L:", check)

        print("log(size) output basis before LLL:", log(size_basis(comp_ext), 2).n())
        print("output basis is LLL reduced:", comp_ext.is_LLL_reduced())
        comp_ext_LLL = comp_ext.LLL()
        print("log(size) output basis after LLL:", log(size_basis(comp_ext_LLL), 2).n())
        print("timing step 3:", (time.time() - s_3))

        print("total timing:", (time.time() - s_1))
    return check


def span_algo_compute_N2(h):
    # Input: a hidden lattice problem h
    # Output: computation (and timing thereof) of a basis of N_II in the congruence lattice algorithm (Algorithm II)
    (n, _, _, size_L, N) = h[0]
    _, L, B = h[1], h[2], h[3]
    print("(n,m,r,size_L,N)", h[0])
    print("log(size_L)", log(size_L, 2).n())
    print("log(N)", log(N, 2).n())

    # Step 1: compute a reduced basis of the lattice in the Z-span of B modulo N
    spanB = span_lattice_mod(B, N)
    if not spanB:
        print("sample a new problem")
    else:
        # compute reduced basis of spanB
        t_1 = time.time()
        spanBLLL = spanB.LLL()

        # extract first n vectors and compute completion
        ext = spanBLLL[0:n]
        print("timing step 1:", (time.time() - t_1))
        print("ext is sublattice of comp_L:", ext.row_space(ZZ).is_submodule(completion(L)))


# D: Variants of the HLP


# D.1: hidden lattice with noise (NHLP)
def gen_noisy_hlp(n, m, r, bound, noise_bound, N):
    # Input: parameters n,m,r,bound,noise_bound,N where
    # n is the rank of the hidden lattice
    # m is the dimension of the hidden lattice
    # r is the rank of the public lattice
    # bound is an upper bound on the absolute value of the entries of a basis of the hidden lattice
    # noise_bound is an upper bound on the absolute value of the entries of the noise vectors
    # N is the public modulus
    # Output: generation of a noisy hidden lattice problem with the given parameters:
    # a hidden lattice L with size upper bounded by size_L (and small basis X)
    # a lattice basis B such that the lattice M spanned by B lies in L modulo N, up to additional noise vectors

    X = matrix(ZZ, n, m)
    for i in range(n):
        for j in range(m):
            X[i, j] = ZZ.random_element(-bound, bound)
    size_L = size_basis(X)
    L = IntegerLattice(X)
    A = matrix(ZZ, r, n)
    for i in range(r):
        for j in range(n):
            A[i, j] = ZZ.random_element(N)
    Z = matrix(ZZ, r, m)
    for i in range(r):
        for j in range(m):
            Z[i, j] = ZZ.random_element(-noise_bound, noise_bound)
    size_noise = max([z.norm() for z in Z.rows()]).n()
    params = (n, m, r, size_L, size_noise, N)
    B = (A * X + Z) % N
    return (params, X, L, B, Z, A)


def algo_noisy(nh):
    # Input: a noisy hidden lattice problem nh
    # Output: a basis candidate for completion(L) following Algorithm I (or Algorithm II, commented) and using:
    # The embedding approach and resolution of linear system technique to distinguish hidden vectors and noise vectors
    # note: to run Algorithm II instead of Algorithm I, uncomment and comment corresponding code
    (n, m, r, size_L, size_noise, N) = nh[0]
    _, L, B, _ = nh[1], nh[2], nh[3], nh[4]
    print("(n,m,r,size_L,size_noise,N)", nh[0])
    print("log(size_L)", log(size_L, 2).n())
    print("log(size_noise)", log(size_noise, 2).n())
    print("log(N)", log(N, 2).n())

    print("Run LLL in embedding dimension m+r =", m + r)
    augment_B = B.augment(matrix.identity(r))

    # to run Algorithm I on augment_B (the lattice basis 'hid' is the candidate basis for the hidden lattice)
    orthoBN = ortho_lattice_mod(matrix(augment_B), N).LLL()

    print("first m-n vectors span the (extracted) lattice N1")
    ext = orthoBN[0 : m - n, :]

    print("compute the orthogonal complement of N1")
    ortho_ext = ortho_lattice(ext)
    hid = ortho_ext

    """
    #uncomment this section and comment previous section in order to run Algorithm II instead of Algorithm I
    #to run Algorithm II on augment_B (the lattice basis 'hid' is the candidate basis for the hidden lattice)

    spanBN=span_lattice_mod(matrix(augment_B),N)
    print('first n+r vectors span the (extracted) lattice N2')
    ext=spanBN.LLL()[0:n+r,:]
    print('compute the completion complement of N2')
    comp_ext=completion(ext.row_space(ZZ)).basis_matrix().LLL()
    hid=comp_ext
    """

    print("test A: a basis for the hidden lattice is on the top left nxm submatrix")
    rec_lat = hid[0:n, 0:m]
    print("   recovered lattice is equal to L:", L == rec_lat.row_space(ZZ))

    print("test B: following the linear system approach")
    U = hid[0 : n + r, m : m + r]
    ker_U = U.left_kernel().basis_matrix()
    out = ker_U * hid
    rec = out[:, 0:m]
    # print(rec)  #prints a basis of the computed lattice
    print("   recovered lattice is equal to L:", L == rec.row_space(ZZ))

    return ext, rec_lat, rec, hid


# D.2: decisonal HLP (DHLP)
def gaps_ortho_vs_random(n, m, r, bound, N):
    # Input: parameters (n,m,r,bound,N) describing a HLP
    # Output: comparison of HLP and non-HLP instances
    # test 1 (HLP instances):
    # creation of 10 random instances of HLP's with parameters (n,m,r,bound,N)
    # for each instance: comparison of the size of the hidden lattice L with the ratio of norms in LLL reduced basis
    # (following Algorithm I)
    # test 2 (non-HLP instances):
    # creation of 10 random instances of HLP's with parameters (n,m,r,bound,N)
    # for each instance: comparison of the size predicted by Gaussian Heuristic (GH) with the ratio of norms in LLL
    # reduced basis (following Algorithm I)
    print("test 1: gaps in HLP instances")
    for _ in range(10):
        h = gen_hlp(n, m, r, bound, N)
        (n, m, r, _, N) = h[0]
        _, B = h[2], h[3]
        print("   (n,m,r,size_L,N)", h[0])
        orthoB = ortho_lattice_mod(B, N)
        orthoBLLL = orthoB.LLL(delta=3 / 4)

        prodfirst = product([u.norm(2) for u in orthoBLLL[0 : m - n]])
        prodlast = product([u.norm(2) for u in orthoBLLL[m - n : m]])

        print("   size_L and ratio", (h[0][3].n(), (prodlast / prodfirst).n()))

    print("test2: gaps in non-HLP instances")
    print(
        "   heuristic bound under (GH)", (sqrt(m / (2 * pi * exp(1))) ** (2 * n - m) * N ** (r * (2 * n - m) / m)).n()
    )
    for ell in range(10):
        B = matrix(ZZ, random_matrix(IntegerModRing(N), r, m))
        orthoB = ortho_lattice_mod(B, N)
        orthoBLLL = orthoB.LLL(delta=3 / 4)

        prodfirst = product([u.norm(2) for u in orthoBLLL[0 : m - n]])
        prodlast = product([u.norm(2) for u in orthoBLLL[m - n : m]])
        print("   ratio", (prodlast / prodfirst).n())


# To run this function, run for example:
# print(gaps_ortho_vs_random(15,100,2,2**10,next_prime(2**100)))


# E: truncation method for HLPs of large dimension
def ortho_algo_truncated(h):
    # Input: a hidden lattice problem h
    # Output: a solution to h computed by multiple (parallelizable) lattice reductions on truncated lattice bases
    (n, m, _, size_L, N) = h[0]
    _, L, M = h[1], h[2], h[3]
    print("(n,m,r,size_L,N)", h[0])
    print("log(size_L)", log(size_L, 2).n())
    print("log(N)", log(N, 2).n())
    assert m % n == 0 % n

    print("a. Full lattice reduction:")
    check = ortho_algo(h)
    print("output ortho algo:", check)

    print("---")

    print("b. Truncated lattice reduction:")
    ell = 2  # we work with dimension ell*n=2*n;
    b = m / (ell * n) - 1
    print("   starting dimension m = ", m)
    print("   number of parallel truncations = ", b + 1)
    print("   new dimension = ", ell * n)

    M0 = M[:, 0 : ell * n]
    mat_M = [M0.augment(M[:, (j + 1) * ell * n : (j + 2) * ell * n]) for j in range(b)]
    # print('mat_M')
    # for j in range(b):
    #   print(j)
    #   print(mat_M[j])
    #   print('--')

    print("Running Algorithm I on all truncated HLPs")
    # uncomment instructions to print intermediate computations
    OM_LLL = [ortho_lattice_mod(mat_Mj, N).LLL() for mat_Mj in mat_M]
    ext = [OMj_LLL[0 : 2 * ell * n - n] for OMj_LLL in OM_LLL]
    rec = [ortho_lattice(ext_j) for ext_j in ext]

    mat_C0 = [rec_j[:, 0 : ell * n] for rec_j in rec]
    # print('matrices C0j')
    # for j in range(b):
    #   print('j=',j)
    #   print(mat_C0[j])
    #   print('--')

    mat_Cj = [rec_j[:, ell * n : 2 * ell * n] for rec_j in rec]
    # print('matrices Cj')
    # for j in range(b):
    #   print('j=',j)
    #   print(mat_Cj[j])
    #   print('--')

    mat_Qj = [C0j.solve_left(mat_C0[0]) for C0j in mat_C0]
    # print('list of matrices mat_Qj')
    # for j in range(b):
    #   print('j=',j)
    #   print(mat_Qj[j])
    #   print('--')

    out = matrix(ZZ, n, m)
    out[:, 0 : ell * n] = mat_C0[0]
    for j in range(b):
        #   print('j=',j)
        out[:, (j + 1) * ell * n : (j + 2) * ell * n] = mat_Qj[j] * mat_Cj[j]
    # print('out')
    # print(out)
    print("hidden lattice is revealed:", out.row_space(ZZ) == L)
    # print(out.LLL())
    # print(L)


# To run this function, run for example:
# h=gen_hlp(8,64,3,2**3,next_prime(2**80))
# print(ortho_algo_truncated(h))

# F: TESTS


def outputs():
    # Run various test on HLP and NHLP
    # ---HLP's
    h1 = gen_hlp(5, 20, 2, 2**10, next_prime(2**120))
    # h1=gen_hlp(10,100,5,2**15,next_prime(2**50))
    # h2=gen_hlp(20,100,5,2**15,next_prime(2**100))
    # h3=gen_hlp(40,100,5,2**15,next_prime(2**250))
    # h4=gen_hlp(80,100,5,2**15,next_prime(2**1400))
    # h5=gen_hlp(90,100,5,2**15,next_prime(2**3100))

    # h1=gen_hlp(150,200,60,2**10,next_prime(2**140))
    # h1=gen_hlp(150,200,110,2**10,next_prime(2**90))
    # h1=gen_hlp(180,200,175,2**10,next_prime(2**140))
    # h1=gen_hlp(100,300,80,2**10,next_prime(2**75))
    # h1=gen_hlp(200,300,150,2**10,next_prime(2**75))
    # h1=gen_hlp(160,320,80,2**10,next_prime(2**80))
    # h1=gen_hlp(260,400,200,2**10,next_prime(2**75))
    # h1=gen_hlp(250,300,230,2**20,next_prime(2**160)) s
    # h1=gen_hlp(50,100,5,2**10,next_prime(2**275))
    # h1=gen_hlp(75,150,5,2**15,next_prime(2**585))
    # h1=gen_hlp(100,200,10,2**15,next_prime(2**400))
    # h1=gen_hlp(150,300,30,2**15,next_prime(2**250))
    # h1=gen_hlp(200,400,50,2**5,next_prime(2**168))

    # ---NHLP's
    # h1=gen_noisy_hlp(15,50,3,2**45,2**20,next_prime(2**500))
    # h1=gen_noisy_hlp(15,50,3,2**10,2**20,next_prime(2**500))

    for h in [h1]:
        print("heursitic lower bound I for log(N):", heuristic_logN_I(h))
        print("heursitic lower bound II for log(N):", heuristic_logN_II(h))

        print("Algorithm I for HLP")
        print(ortho_algo(h))
        # print(ortho_algo_compute_N1(h))

        print("------------")

        print("Algorithm II for HLP")
        print(span_algo(h))
        # print(span_algo_compute_N2(h))

    #   print('------------')

    #   print('Algorithm for NHLP')
    #   print(algo_noisy(h))


# print(outputs())
