import os
# set env variables to limit numpy thread usage to 1 in order
# to avoid oversubscription.
# Current Machine uses OpenBLAS but all threading env. variables
# were assigned to 1 as a precaution.
# Vars need to be defined before importing numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from mpi4py import MPI
import numpy as np
import time
import argparse
import csv


# Obtain the size and rank of the communicator
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--matrix", type=int, default=100,
                    help="square matrix size (Default: 100)")
parser.add_argument("-i", "--iter", type=int, default=10000,
                    help="number of Iterations for Power Method (Default: 10000)")
parser.add_argument("-v", "--verbose", type=int, default=2,
                    help="verbose (0, 1, 2) (Default: 2)")
parser.add_argument("-np", "--numpy", type=int, default=1,
                    help="use Numpy (default: 1)")
parser.add_argument("-s", "--save", type=str, default=None,
                    help="save to file (default: None)")

args = parser.parse_args()
ndim = args.matrix
n_iter = args.iter
verbose = args.verbose
save_file = args.save
use_np = bool(args.numpy)


def generate_matrix(dim):
    """Generate a matrix with eigenvalues between 1 and 10."""
    from scipy.stats import ortho_group
    from scipy.sparse import spdiags
    a = ortho_group.rvs(dim, random_state=0)
    b = np.linspace(1., 10., dim)
    return a @ spdiags(b, 0, dim, dim) @ a.T


def matvec_local(mat, vec, use_numpy=False):
    """Perform a matrix-vector multiplication."""
    if use_numpy:
        return mat @ vec
    else:
        matvec = np.zeros(mat.shape[0], dtype=float)
        for row in range(mat.shape[0]):
            for col in range(mat.shape[1]):
                matvec[row] += mat[row, col] * vec[col]
        return matvec


def norm_local(vec, use_numpy=False):
    """Calculate the norm of a vector"""
    if use_numpy:
        norm = np.linalg.norm(vec)
    else:
        sum_squared = 0
        for row in range(len(vec)):
            sum_squared += vec[row]**2
        norm = sum_squared**0.5
    return norm
# -----


if ndim % size != 0:
    raise RuntimeError("Matrix must be divisible by the number of processes")
rows_per_proc = ndim // size

if rank == 0:
    if verbose > 1:
        print(f"Processes:\t{size:,}")
        print(f"Matrix Size:\t{ndim:,}")
        print(f"Iterations:\t{n_iter:,}")

    # Matrix and vector b_k defined inside rank 0
    full_matrix = generate_matrix(ndim)

    # Runtime starts after the declaration of the matrix
    start_time = time.time()
    b_k = np.ones(ndim, dtype=float)

    # Matrix is divided by the number of existing processes
    send_matrix = np.array_split(full_matrix, size)
    send_matrix = np.array(send_matrix, dtype=float)
    recv_matvec = np.zeros(ndim, dtype=float)
else:
    # Placeholder / Buffer
    send_matrix = None
    b_k = np.zeros(ndim, dtype=float)
    recv_matvec = None
chunk_matrix = np.zeros((rows_per_proc, ndim), dtype=float)

# Each chunk of the matrix is scattered to a process
comm.Scatter(send_matrix, chunk_matrix, root=0)

for _ in range(n_iter):
    # B_k vector is broadcasted to every process
    comm.Bcast(b_k, root=0)
    # Chunk-of-Each-Matrix and Vector operation is performed on each process
    send_matvec = matvec_local(chunk_matrix, b_k, use_numpy=use_np)
    # Matvec result gathered back into a single vector in Rank 0
    comm.Gather(send_matvec, recv_matvec, root=0)
    if rank == 0:
        # Norm and new B_k are calculated only on Rank 0
        norm = norm_local(recv_matvec, use_numpy=use_np)
        b_k = recv_matvec / norm


if rank == 0:
    # Rank 0 displays output info. and metrics.
    runtime = time.time() - start_time
    est_eigvec_rsp = f'> Est. eigenvector:  {b_k[0:5]}... (Total length: {len(b_k):,})'
    eigenvector = b_k
    eigenvalue = np.dot(b_k, np.dot(full_matrix, b_k)) / np.dot(b_k, b_k)
    eigvals, eigvecs = np.linalg.eig(full_matrix)
    eigvecs = eigvecs[:, np.argmax(eigvals)]
    corr_eigvec_rsp = f'> Real eigenvector:  {eigvecs[0:5]}... (Total length: {len(eigvecs):,})'
    corr_eigval = np.max(eigvals)
    if verbose > 0:
        print(f"Rank {rank}:")
        print(f"- Total Runtime:{runtime:>9.2f} Secs.")
    if verbose > 1:
        print(f"> Est. eigenvalue:{eigenvalue:20}")
        print(est_eigvec_rsp)
        print(f"> Real eigenvalue:{corr_eigval:21}")
        print(corr_eigvec_rsp)
        err = (eigenvalue - corr_eigval)
        abs_err = abs(err)
        pct_err = (err / corr_eigval)
        if abs_err < 0.01:
            err = f'{err:.2e}'
            pct_err = f'{pct_err:.2e}%'
        print(f'- Error: {err} | % Error: {pct_err}')

    # Results can be saved inside a file if desired
    if save_file:
        filename = f'{save_file}.csv'
        file_exists = os.path.isfile(filename)

        with open(filename, 'a') as csvfile:
            header = ['Name', 'Processes', 'Cols', 'Iteration', 'Runtime',
                      'Est_Eigenvalue', 'Real_Eigenvalue', 'Error', 'Pct_Error']
            writer = csv.writer(csvfile, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

            if not file_exists:
                writer.writerow(header)

            writer.writerow([save_file, size, ndim, n_iter, runtime,
                             eigenvalue, corr_eigval, (eigenvalue - corr_eigval),
                             (eigenvalue - corr_eigval) / corr_eigval])
