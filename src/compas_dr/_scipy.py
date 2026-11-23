"""Typed interfaces for the specific SciPy operations used by DR."""

from typing import cast

from scipy.linalg import norm  # pyright: ignore[reportUnknownVariableType]
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import dia_matrix
from scipy.sparse import diags  # pyright: ignore[reportUnknownVariableType]

from compas_dr.types import FloatArray


def dot_coordinates(matrix: csr_matrix | csc_matrix, coordinates: FloatArray) -> FloatArray:
    """Multiply a sparse matrix by a dense coordinate array."""
    return cast(FloatArray, matrix.dot(coordinates))  # pyright: ignore[reportUnknownMemberType]


def weighted_product(left: csr_matrix, weights: dia_matrix, right: csr_matrix) -> csc_matrix:
    """Form the sparse product left.T * weights * right."""
    return cast(csc_matrix, weighted_transpose(left, weights).dot(right))  # pyright: ignore[reportUnknownMemberType]


def weighted_transpose(matrix: csr_matrix, weights: dia_matrix) -> csc_matrix:
    """Multiply the transpose of a connectivity matrix by the weights."""
    return cast(csc_matrix, matrix.T.dot(weights))  # pyright: ignore[reportUnknownMemberType]


def squared_transpose(matrix: csr_matrix) -> csc_matrix:
    """Return a transposed copy with each nonzero entry squared."""
    transpose = cast(csc_matrix, matrix.transpose())  # pyright: ignore[reportUnknownMemberType]
    squared = cast(csc_matrix, transpose.copy())
    squared.data **= 2  # pyright: ignore[reportUnknownMemberType]
    return squared


def diagonal(values: FloatArray) -> dia_matrix:
    """Construct a sparse matrix with the given main diagonal."""
    return cast(dia_matrix, diags([values.flatten()], [0]))  # pyright: ignore[reportArgumentType]


def array_norm(values: FloatArray) -> float:
    """Return the Euclidean or Frobenius norm of a dense array."""
    return cast(float, norm(values))
