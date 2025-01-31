import numpy as np

from .src.fica import ICA

from sage.all import matrix, ZZ, QQ, GF, Matrix, identity_matrix, gcd, IntegerModRing, round, Integer

from .src import hidden_lattice
from .src import ns


# #### ##### ##### ##### ##### UTILS ##### ##### ##### ##### #### #
def fullKernelMatrix(ke3, n, K):
    ke23 = Matrix(K, ke3.nrows(), n * n)
    ind = n
    for i in range(n):
        for j in range(i, n):
            ke23[:, i * n + j] = ke3[:, ind]
            ke23[:, j * n + i] = ke3[:, ind]
            ind += 1
    return ke23


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


def Sage2np(MO, n, m):
    MOn = np.matrix(MO)
    return MOn


def runICA(MOn, B=1):
    # t1 = time.time()
    A_, S = ICA(MOn, B)
    S_ = np.dot(np.linalg.inv(A_), MOn)
    # print("time Ica_S: ", time.time() - t1),
    S2 = matrix(ZZ, MOn.shape[0], MOn.shape[1], round(S_))
    return S2


def runICA_A(MOn, B=1, n=16, kappa=-1):
    # t1 = time.time()
    A_, S = ICA(MOn, B, n, kappa)
    # print("time Ica_A: ", time.time() - t1),
    A2 = matrix(ZZ, MOn.shape[0], MOn.shape[0], round(A_))
    return A2


def matNbits(M):
    return max([M[i, j].nbits() for i in range(M.nrows()) for j in range(M.ncols())])


# #### ##### ##### ##### ##### HIDDEN LATTICE PROBLEM ##### ##### ##### ##### #### #
def recover_hidden_noisy_lattice_algo_I(B, n, m, N, verbose=True):
    # Input: a noisy hidden lattice problem nh
    # Output: a basis candidate for completion(L) following Algorithm I (or Algorithm II, commented) and using:
    # The embedding approach and resolution of linear system technique to distinguish hidden vectors and noise vectors
    # note: to run Algorithm II instead of Algorithm I, uncomment and comment corresponding code

    r = 1
    augment_B = B.augment(matrix.identity(r))
    orthoBN = hidden_lattice.ortho_lattice_mod(matrix(augment_B), N).LLL()
    ext = orthoBN[0 : m - n, :]
    ortho_ext = hidden_lattice.ortho_lattice(ext)
    hid = ortho_ext
    rec_lat = hid[0:n, 0:m]
    U = hid[0 : n + r, m : m + r]
    ker_U = U.left_kernel().basis_matrix()
    ker_U_ortho = hidden_lattice.ortho_lattice(ker_U)
    rec_noisy = ker_U_ortho * hid
    out = ker_U * hid
    rec = out[:, 0:m]
    return ext, rec_lat, rec, hid, rec_noisy


def recover_hidden_noisy_lattice_algo_II(B, n, m, N, verbose=True):
    # Input: a noisy hidden lattice problem nh
    # Output: a basis candidate for completion(L) following Algorithm I (or Algorithm II, commented) and using:
    # The embedding approach and resolution of linear system technique to distinguish hidden vectors and noise vectors
    # note: to run Algorithm II instead of Algorithm I, uncomment and comment corresponding code

    r = 1
    augment_B = B.augment(matrix.identity(r))
    # orthoBN = hidden_lattice.ortho_lattice_mod(matrix(augment_B), N).LLL()
    # ext = orthoBN[0 : m - n, :]
    # ortho_ext = hidden_lattice.ortho_lattice(ext)
    # hid = ortho_ext

    spanBN = span_lattice_mod(matrix(augment_B), N)
    if verbose:
        print("first n+r vectors span the (extracted) lattice N2")
    ext = spanBN.LLL()[0 : n + r, :]
    if verbose:
        print("compute the completion complement of N2")
    comp_ext = completion(ext.row_space(ZZ)).basis_matrix().LLL()
    hid = comp_ext

    rec_lat = hid[0:n, 0:m]
    U = hid[0 : n + r, m : m + r]
    ker_U = U.left_kernel().basis_matrix()
    ker_U_ortho = hidden_lattice.ortho_lattice(ker_U)
    rec_noisy = ker_U_ortho * hid
    out = ker_U * hid
    rec = out[:, 0:m]
    return ext, rec_lat, rec, hid, rec_noisy


# #### ##### ##### ##### ##### NGUYEN STERN ATTACK ##### ##### ##### ##### #### #
def Step2_BKZ(ke, B, n, kappa=-1, verbose=True):
    # if n>170: return
    beta = 2
    while beta < n:
        # print(f"beta={beta}")
        if beta == 2:
            M5 = ke.LLL()
            M5 = M5[:n]  # this is for the affine case
        else:
            #      M5=M5.BKZ(block_size=beta, strategies=BKZ.DEFAULT_STRATEGY, flags=BKZ.AUTO_ABORT|BKZ.GH_BND)
            M5 = M5.BKZ(block_size=beta)

        # we succeed if we only get vectors with {-1,0,1} components, for kappa>0 we relax this condition
        # to all except one vector
        # if len([True for v in M5 if allpmones(v)])==n: break
        cl = len([True for v in M5 if ns.allbounded(v, B)])
        if cl == n:
            break
        # if kappa>0 and cl==n-1: break

        if beta == 2:
            beta = 10
        else:
            beta += 10
    for v in M5:
        if verbose and not ns.allbounded(v, B):
            print(f"unbalanced v: {v}")

    MB = ns.recoverBinary(M5, kappa)
    if verbose:
        print("  Number of recovered vector=", MB.nrows()),
    return MB


def ns_original(B, MO, rec_noisy, n, m, N, verbose=True):
    # m=int(max(2*n,16*log(n,2)))
    kappa = -1

    bb = 1 / 2
    unbalanced = (abs(n * bb - kappa) / n) > 0.2
    if verbose:
        print("unbalanced: ", unbalanced)

    NSo = Step2_BKZ(matrix(ZZ, MO), 1, n, kappa=kappa, verbose=verbose)

    if verbose:
        print(f"n={n}, m={m}, NSo={NSo.dimensions()}=={n}x{m}")
    if not NSo.nrows() >= n - 1:
        raise ValueError("Not enough vectors found")

    # if NSo.nrows() > n:
    #     print(f"NSo={NSo}")

    NS = NSo.T  # [:m, :n]
    NS_rank = NS.rank()

    if NS_rank < n:
        raise ValueError("Not enough INDEPENDANT vectors found")
    if verbose:
        print(f"NS.dimensions()={NS.dimensions()}=={m}x{n}")
        print(f"NS.rank()={NS.rank()}")
        print(f"size_basis(NS)={hidden_lattice.size_basis(NS)}")
        # print(f"NS={NS}")
    # print(f"NS={NS}")

    # n_new = NS.rank()
    # NS = NS[:, :n_new]
    # arg_idx = np.argsort([np.linalg.norm(b) ** 2 for b in NS.T])
    # NS = matrix(Integers(N), np.asarray(NS.T)[arg_idx][:n]).T

    # print(f"NS.dimensions()={NS.dimensions()}=={m}x{n}")
    # print(f"NS.rank()={NS.rank()}")
    # print(f"size_basis(NS)={hidden_lattice.size_basis(NS)}")
    # print(f"NS={NS}")

    # print(f"NS.dimensions()={NS.dimensions()}=={m}x{n}")
    # print(f"Y={Y}")

    # # Step 3 Bis
    # print(NS[:, :n].dimensions())
    # invNSn = matrix(ZZ, NS[:n, :n]).inverse()
    # extracted_B = vector(ZZ, [B[0, i] for i in range(n)])
    # ra = invNSn * extracted_B

    # return ra, NS

    # Step 3 Official
    m_prime = min(n + 75, m)
    # m_prime = m
    # N_prime = hidden_lattice.next_prime(Integer(np.max(B) * 2**64))
    N_prime = N
    B_vector = matrix(ZZ, B[0, :m_prime])
    NS_matrix = matrix(ZZ, NS[:m_prime])
    M_matrix = matrix(ZZ, np.eye(N=m_prime, M=m_prime)) * N_prime
    B_noise = (rec_noisy.T)[:m_prime]
    if verbose:
        print(f"B_vector={B_vector.dimensions()}")
        print(f"B_noise={B_noise.dimensions()}")
        print(f"NS_matrix={NS_matrix.dimensions()}")
        print(f"M_matrix={M_matrix.dimensions()}")

    Lattice_problem = (B_vector.T).augment(B_noise).augment(NS_matrix).augment(M_matrix)
    # Lattice_problem = (B_vector.T).augment(NS_matrix).augment(M_matrix)
    if verbose:
        print(f"Lattice_problem={Lattice_problem.dimensions()}=={m_prime}x{m_prime + n+2}")
    Lattice_ortho = hidden_lattice.ortho_lattice(Lattice_problem)
    if verbose:
        print(f"Lattice_ortho={Lattice_ortho.dimensions()}=={m_prime}x{m_prime + n}")
        print(f"Lattice_ortho={Lattice_ortho}")
    try:
        # print(np.abs(np.sum(np.asarray(Lattice_ortho)[:, n + 3 :], axis=1)) < 1e-3)
        # print(10 > np.asarray(Lattice_ortho)[:, 0] > 0.5)
        correct_idx = np.where(
            (np.absolute(np.asarray(Lattice_ortho)[:, 0]) > 0.5) * (np.absolute(np.asarray(Lattice_ortho)[:, 0]) < 10.5)
        )[0]
        if verbose:
            print(f"correct_idx={correct_idx}")
        if len(correct_idx) == 1:
            lattice_np = np.asarray(Lattice_ortho)[correct_idx].squeeze()
            ra = lattice_np
            # ra = (lattice_np[2 : n + 2] / (-lattice_np[0])).astype(np.int64)
        else:
            raise ZeroDivisionError
    except RuntimeWarning:
        if verbose:
            print(Lattice_ortho)
        return np.array(list(Lattice_ortho[0])), NS
    except ZeroDivisionError:
        if verbose:
            print(Lattice_ortho)
        return np.array(list(Lattice_ortho[0])), NS
    return ra, NS


# #### ##### ##### ##### ##### MULTIVARIATE ATTACK ##### ##### ##### ##### #### #
def eigen(B, MO, rec_noisy, n, m, N, verbose=True):
    if verbose:
        print("Step 2"),
    K = GF(3)
    xt23 = Matrix(
        K, [(-x).list() + [x[i] * x[j] * ((i == j) and 1 or 2) for i in range(n) for j in range(i, n)] for x in MO.T]
    )
    if verbose:
        print(f"xt23.rank()={xt23.rank()}=={int(n * (n + 1) / 2)}?")
    assert xt23.rank() == int(n * (n + 1) / 2)  # Otherwise the attack will not work

    ke3 = xt23.right_kernel().matrix()
    # print(f"ke3={ke3}")
    # ke3 = xt23.left_kernel().matrix()
    # ke3 = matrix(K, xt23.right_kernel().basis())

    assert xt23.nrows() == m
    assert xt23.ncols() == n * (n + 1) / 2 + n

    if verbose:
        print(f"dim(ker E)=ke3.dimensions()={ke3.dimensions()}=={n}x{int(n * (n + 1) / 2 + n)}?")
        print(f"xt23.dimensions()={xt23.dimensions()}=={m}x{int(n * (n + 1) / 2 + n)}?")  # E in the paper

    ke23 = fullKernelMatrix(ke3, n, K)

    if verbose:
        print(f"ke23.dimensions()={ke23.dimensions()}=={n}x{n * n}?")

    # assert ke23.nrows() == n
    # assert ke23.ncols() == n * n

    # print(f"ke23={ke23}")

    # We will compute the list of eigenvectors
    # We start with the full space.
    # We loop over the coordinates. This will split the eigenspaces.
    li = [Matrix(K, identity_matrix(n))]
    for j in range(n):  # We loop over the coordinates of the wi vectors.
        # print "j=",j
        M = ke23[:, j * n : (j + 1) * n]  # We select the submatrix corresponding to coordinate j
        li2 = []  # We initialize the next list
        for v in li:
            if v.nrows() == 1:  # We are done with this eigenvector
                li2.append(v)
            else:  # eigenspace of dimension >1
                # print("eigenspace of dim:", v.nrows()),
                # print(f"M.dimensions()={M.dimensions()}")
                v_m = v * M  # [: v.nrows()]  # .transpose()
                # print(f"v_m.dimensions()={v_m.dimensions()}")

                A = v.solve_left(
                    v_m
                )  # v*M=A*v. When we apply M on the right, this is equivalent to applying the matrix A.
                # The eigenvalues of matrix A correspond to the jth coordinates of the wi vectors in that
                # eigenspace
                for e, v2 in A.eigenspaces_left():  # We split the eigenspace according to the eigenvalues of A.
                    vv2 = v2.matrix()
                    # print "  eigenspace of dim:",(vv2*v).nrows()
                    li2.append(vv2 * v)  # The new eigenspaces

        li = li2

    NS = Matrix([v[0] for v in li]) * MO
    for i in range(n):
        if any(c == 2 for c in NS[i]):
            NS[i] = -NS[i]

    if verbose:
        print("  Number of recovered vectors:", NS.nrows()),

    # nfound = len([True for NSi in NS if NSi in X.T])
    # print("  NFound=", nfound, "out of", n),
    # print("NS=", NS.T),

    NS = NS.T

    if verbose:
        print(f"NS.dimensions()={NS.dimensions()}=={m}x{n}")
        print(f"NS.rank()={NS.rank()}")
        print(f"size_basis(NS)={hidden_lattice.size_basis(NS)}")

    # # Extract the first n linearly independent rows
    # YY = NS
    # ZZZ = matrix(ZZ, YY[0])
    # current_rank = 1
    # extract_idx_rows = [0]
    # for i in range(1, m):
    #     ZZZ_bis = ZZZ.stack(YY[i])
    #     if ZZZ_bis.rank() > current_rank:
    #         extract_idx_rows.append(i)
    #         current_rank = ZZZ_bis.rank()
    #         ZZZ = ZZZ_bis
    #     else:
    #         pass
    #         # print(f"ZZZ_bis.rank()={ZZZ_bis.rank()}")
    #     if current_rank == n:
    #         break
    # ZZZ = YY.matrix_from_rows(extract_idx_rows)

    # # b=X*a=NS*ra
    # invNSn = matrix(Integers(N), ZZZ).inverse()
    # extracted_B = vector(Integers(N), [b[0, i] for i in extract_idx_rows])
    # ra = invNSn * extracted_B
    # # nrafound = len([True for rai in ra if rai in a])

    # # print("  Total step2: %.1f" % tt2),
    # # print()

    # Step 3 Official
    m_prime = min(n + 25, m)
    # m_prime = m
    N_prime = hidden_lattice.next_prime(Integer(np.max(B) * 2**128))
    B_vector = matrix(ZZ, B[0, :m_prime])
    NS_matrix = matrix(ZZ, NS[:m_prime])
    M_matrix = matrix(ZZ, np.eye(N=m_prime, M=m_prime)) * N_prime
    B_noise = (rec_noisy.T)[:m_prime]
    if verbose:
        print(f"B_vector={B_vector.dimensions()}")
        print(f"B_noise={B_noise.dimensions()}")
        print(f"NS_matrix={NS_matrix.dimensions()}")
        print(f"M_matrix={M_matrix.dimensions()}")

    Lattice_problem = (B_vector.T).augment(B_noise).augment(NS_matrix).augment(M_matrix)
    if verbose:
        print(f"Lattice_problem={Lattice_problem.dimensions()}=={m_prime}x{m_prime + n+2}")
    Lattice_ortho = hidden_lattice.ortho_lattice(Lattice_problem)
    if verbose:
        print(f"Lattice_ortho={Lattice_ortho.dimensions()}=={m_prime}x{m_prime + n}")
        print(f"Lattice_ortho={Lattice_ortho}")
    try:
        correct_idx = np.where(np.abs(np.sum(np.asarray(Lattice_ortho)[:, n + 2 :], axis=1)) < 1e-3)[0]
        if verbose:
            print(f"correct_idx={correct_idx}")
        if len(correct_idx) == 1:
            lattice_np = np.asarray(Lattice_ortho)[correct_idx].squeeze()
            ra = (lattice_np[2 : n + 2] / (-lattice_np[0])).astype(np.int64)
        else:
            raise ZeroDivisionError
    except RuntimeWarning:
        if verbose:
            print(Lattice_ortho)
        return np.array(list(Lattice_ortho[0])), NS
    except ZeroDivisionError:
        if verbose:
            print(Lattice_ortho)
        return np.array(list(Lattice_ortho[0])), NS
    return ra, NS


# #### ##### ##### ##### ##### STATISTICAL ATTACK ##### ##### ##### ##### #### #
def statistical(B, MO, rec_noisy, n, m, N, variant=None, verbose=True):
    params_B = 1
    kappa = -1
    if variant is None:
        if n <= 200:
            variant = "roundA"
        else:
            variant = "roundX"

    if verbose:
        print("Step 2-ICA: ", variant)
        print("matNbits MO=", matNbits(MO))

    MO = MO.LLL()
    MOn = Sage2np(MO, n, m)

    if variant == "roundA":
        A2 = runICA_A(MOn, params_B, n, kappa)
        if verbose:
            print(f"A2.dimensions()={A2.dimensions()}=={n}x{n}")
            print(f"A2.rank()={A2.rank()}")
        S2 = A2.inverse() * MO
        if verbose:
            print("mathNbits A=", matNbits(A2)),
    elif variant == "roundX":
        S2 = runICA(MOn, params_B)
    else:
        raise NameError("Variant algorithm non acceptable")

    NS = S2.T

    if verbose:
        print(f"NS.dimensions()={NS.dimensions()}=={m}x{n}")
        print(f"NS.rank()={NS.rank()}")
        print(f"size_basis(NS)={hidden_lattice.size_basis(NS)}")

    # # Extract the first n linearly independent rows
    # YY = NS
    # ZZZ = matrix(ZZ, YY[0])
    # current_rank = 1
    # extract_idx_rows = [0]
    # for i in range(1, m):
    #     ZZZ_bis = ZZZ.stack(YY[i])
    #     if ZZZ_bis.rank() > current_rank:
    #         extract_idx_rows.append(i)
    #         current_rank = ZZZ_bis.rank()
    #         ZZZ = ZZZ_bis
    #     else:
    #         pass
    #         print(f"extracted i = {i}, not linearly independant")
    #         return 0, NS
    #     if current_rank == n:
    #         break
    # ZZZ = YY.matrix_from_rows(extract_idx_rows)

    # invNSn = matrix(Integers(N), ZZZ).inverse()
    # extracted_B = vector(Integers(N), [b[0, i] for i in extract_idx_rows])
    # ra = invNSn * extracted_B
    # return ra, NS

    # Step 3 Official
    m_prime = min(n + 25, m)
    # m_prime = m
    N_prime = hidden_lattice.next_prime(Integer(np.max(B) * 2**128))
    B_vector = matrix(ZZ, B[0, :m_prime])
    NS_matrix = matrix(ZZ, NS[:m_prime])
    M_matrix = matrix(ZZ, np.eye(N=m_prime, M=m_prime)) * N_prime
    B_noise = (rec_noisy.T)[:m_prime]
    if verbose:
        print(f"B_vector={B_vector.dimensions()}")
        print(f"B_noise={B_noise.dimensions()}")
        print(f"NS_matrix={NS_matrix.dimensions()}")
        print(f"M_matrix={M_matrix.dimensions()}")

    Lattice_problem = (B_vector.T).augment(B_noise).augment(NS_matrix).augment(M_matrix)
    if verbose:
        print(f"Lattice_problem={Lattice_problem.dimensions()}=={m_prime}x{m_prime + n+2}")
    Lattice_ortho = hidden_lattice.ortho_lattice(Lattice_problem)
    if verbose:
        print(f"Lattice_ortho={Lattice_ortho.dimensions()}=={m_prime}x{m_prime + n}")
        print(f"Lattice_ortho={Lattice_ortho}")
    try:
        correct_idx = np.where(np.abs(np.sum(np.asarray(Lattice_ortho)[:, n + 2 :], axis=1)) < 1e-3)[0]
        if verbose:
            print(f"correct_idx={correct_idx}")
        if len(correct_idx) == 1:
            lattice_np = np.asarray(Lattice_ortho)[correct_idx].squeeze()
            ra = (lattice_np[2 : n + 2] / (-lattice_np[0])).astype(np.int64)
        else:
            raise ZeroDivisionError
    except RuntimeWarning:
        if verbose:
            print(Lattice_ortho)
        return np.array(list(Lattice_ortho[0])), NS
    except ZeroDivisionError:
        if verbose:
            print(Lattice_ortho)
        return np.array(list(Lattice_ortho[0])), NS
    return ra, NS


# #### ##### ##### ##### ##### MAIN FUNCTION ##### ##### ##### ##### #### #
def gradient_attack(P, n, m, N, ndigits=None, alg="default"):
    # P = matrix(np.int64(np.asarray(P).reshape(1, m)))
    # P = round(P, ndigits=ndigits)

    # ns original
    if alg == "ns_original":
        print("Nguyen-Stern (Original) Attack")
        ext, rec_lat, rec, hid = recover_hidden_noisy_lattice_algo_I(P, n, m, N)
        ra = ns_original(P, rec, n, m, N)
        return ra

    # if alg == "ns":
    #     print("\nNguyen-Stern (Improved) Attack")
    #     ra = ns_(P, rec)
    #     return ra

    if alg in ["default", "multi"]:
        assert m > (n**2 + n) / 2, "m too small"
        ext, rec_lat, rec, hid = recover_hidden_noisy_lattice_algo_II(P, n, m, N)
        ra = eigen(P, rec, n, m, N)
        return ra
