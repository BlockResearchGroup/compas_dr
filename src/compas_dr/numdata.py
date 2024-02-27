try:
    import numpy as np
    from compas.linalg import normrow
    from compas.matrices import connectivity_matrix
except ImportError:
    has_numpy = False
else:
    has_numpy = True

try:
    import numpy.typing as npt  # noqa: F401
except ImportError:
    pass

import compas.datastructures  # noqa: F401
import compas.geometry  # noqa: F401
from compas.data import Data


class InputData(Data):
    """Class representing input data for DR solvers.

    Parameters
    ----------
    vertices
    edges
    fixed
    loads
    qpre
    fpre
    lpre
    linit
    E
    radius

    Attributes
    ----------
    vertices
    edges
    fixed
    free
    loads
    qpre
    fpre
    lpre
    linit
    E
    radius
    q0
    l0
    v0
    r0
    C

    """

    @property
    def __data__(self):
        # type: () -> dict
        return {
            "vertices": self.vertices,
            "edges": self.edges,
            "fixed": self.fixed,
            "loads": self.loads,
            "qpre": self.qpre,
            "fpre": self.fpre,
            "lpre": self.lpre,
            "linit": self.linit,
            "E": self.E,
            "radius": self.radius,
        }

    @classmethod
    def __from_data__(cls, data):
        # type: (dict) -> InputData
        return super(InputData, cls).__from_data__(data)

    def __init__(
        self,
        vertices,  # type: list[list[float]]
        edges,  # type: list[tuple[int, int]]
        fixed,  # type: list[int]
        loads,  # type: list[list[float]]
        qpre,  # type: list[float]
        fpre=None,  # type: list[float] | None
        lpre=None,  # type: list[float] | None
        linit=None,  # type: list[float] | None
        E=None,  # type: list[float] | None
        radius=None,  # type: list[float] | None
    ):  # type: (...) -> None
        self._vertices = vertices
        self._vertices_array = None
        self._edges = edges
        self._edges_array = None
        self._fixed = fixed
        self._free = None
        self._loads = loads
        self._loads_array = None
        self._qpre = qpre
        self._qpre_array = None
        self._fpre = fpre
        self._fpre_array = None
        self._lpre = lpre
        self._lpre_array = None
        self._linit = linit
        self._linit_array = None
        self._E = E
        self._E_array = None
        self._radius = radius
        self._radius_array = None
        # (lazy) computed properties
        self._C = None

    @property
    def vertices(self):
        # type: () -> npt.ArrayLike
        if has_numpy:
            if self._vertices_array is None:
                self._vertices_array = np.asarray(
                    self._vertices,
                    dtype=np.float64,
                ).reshape((-1, 3))
            return self._vertices_array
        return self._vertices

    @property
    def edges(self):
        # type: () -> npt.ArrayLike
        if has_numpy:
            if self._edges_array is None:
                self._edges_array = np.asarray(
                    self._edges,
                    dtype=np.int32,
                ).reshape((-1, 2))
            return self._edges_array
        return self._edges

    @property
    def fixed(self):
        # type: () -> npt.ArrayLike
        return self._fixed

    @property
    def free(self):
        # type: () -> npt.ArrayLike
        if self._free is None:
            self._free = list(set(range(len(self._vertices))) - set(self._fixed))
        return self._free

    @property
    def loads(self):
        # type: () -> npt.ArrayLike
        if has_numpy:
            if self._loads_array is None:
                self._loads_array = np.asarray(
                    self._loads,
                    dtype=np.float64,
                ).reshape((-1, 3))
            return self._loads_array
        return self._loads

    @property
    def qpre(self):
        # type: () -> npt.ArrayLike
        if has_numpy:
            if self._qpre_array is None:
                self._qpre_array = np.asarray(
                    self._qpre,
                    dtype=np.float64,
                ).reshape((-1, 1))
            return self._qpre_array
        return self._qpre

    @property
    def fpre(self):
        # type: () -> npt.ArrayLike
        if self._fpre is None:
            self._fpre = [0.0] * len(self._edges)
        if has_numpy:
            if self._fpre_array is None:
                self._fpre_array = np.asarray(
                    self._fpre,
                    dtype=np.float64,
                ).reshape((-1, 1))
            return self._fpre_array
        return self._fpre

    @property
    def lpre(self):
        # type: () -> npt.ArrayLike
        if self._lpre is None:
            self._lpre = [0.0] * len(self._edges)
        if has_numpy:
            if self._lpre_array is None:
                self._lpre_array = np.asarray(
                    self._lpre,
                    dtype=np.float64,
                ).reshape((-1, 1))
            return self._lpre_array
        return self._lpre

    @property
    def linit(self):
        # type: () -> npt.ArrayLike
        if self._linit is None:
            self._linit = [0.0] * len(self._edges)
        if has_numpy:
            if self._linit_array is None:
                self._linit_array = np.asarray(
                    self._linit,
                    dtype=np.float64,
                ).reshape((-1, 1))
            return self._linit_array
        return self._linit

    @property
    def E(self):
        # type: () -> npt.ArrayLike
        if self._E is None:
            self._E = [0.0] * len(self._edges)
        if has_numpy:
            if self._E_array is None:
                self._E_array = np.asarray(
                    self._E,
                    dtype=np.float64,
                ).reshape((-1, 1))
            return self._E_array
        return self._E

    @property
    def radius(self):
        # type: () -> npt.ArrayLike
        if self._radius is None:
            self._radius = [0.0] * len(self._edges)
        if has_numpy:
            if self._radius_array is None:
                self._radius_array = np.asarray(
                    self._radius,
                    dtype=np.float64,
                ).reshape((-1, 1))
            return self._radius_array
        return self._radius

    # =============================================================================
    # Computed
    # =============================================================================

    @property
    def q0(self):
        if has_numpy:
            return np.ones((len(self._edges), 1), dtype=np.float64)
        return [1.0] * len(self._edges)

    @property
    def l0(self):
        if has_numpy:
            return normrow(self.C.dot(self.vertices))
        # return ...

    @property
    def v0(self):
        if has_numpy:
            return np.zeros((len(self._vertices), 3), dtype=np.float64)
        # return ...

    @property
    def r0(self):
        if has_numpy:
            return np.zeros((len(self._vertices), 3), dtype=np.float64)
        # return ...

    @property
    def C(self):
        # type: () -> npt.ArrayLike | None
        if has_numpy:
            if self._C is None:
                self._C = connectivity_matrix(self._edges, rtype="csr")
            return self._C
        # return ...

    # =============================================================================
    # Constructors
    # =============================================================================

    @classmethod
    def from_mesh(
        cls,
        mesh,  # type: compas.datastructures.Mesh
        fixed,  # type: list[int]
        loads,  # type: list[list[float]]
        qpre,  # type: list[float]
        fpre=None,  # type: list[float] | None
        lpre=None,  # type: list[float] | None
        linit=None,  # type: list[float] | None
        E=None,  # type: list[float] | None
        radius=None,  # type: list[float] | None
    ):  # type: (...) -> InputData
        vertex_index = {vertex: index for index, vertex in enumerate(mesh.vertices())}
        vertices = mesh.vertices_attributes("xyz")
        edges = [(vertex_index[u], vertex_index[v]) for u, v in mesh.edges()]

        return cls(
            vertices=vertices,
            edges=edges,
            fixed=fixed,
            loads=loads,
            qpre=qpre,
            fpre=fpre,
            lpre=lpre,
            linit=linit,
            E=E,
            radius=radius,
        )


class ResultData(Data):
    """Class representing the result of a calculation by one of the solvers.

    Parameters
    ----------
    xyz
    q
    forces
    lengths
    residuals

    Attributes
    ----------
    xyz
    q
    forces
    lengths
    residuals

    """

    @property
    def __data__(self):
        # type: () -> dict
        return {}

    @classmethod
    def __from_data__(cls, data):
        # type: (dict) -> ResultData
        return super(ResultData, cls).__from_data__(data)

    def __init__(self, xyz, q, forces, lengths, residuals):
        # type: (npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike) -> None
        self.xyz = xyz
        self.q = q
        self.forces = forces
        self.lengths = lengths
        self.residuals = residuals
