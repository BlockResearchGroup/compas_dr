from math import ceil

from numpy import float64
from numpy import seterr
from numpy import isinf
from numpy import isnan
from numpy import all
from numpy import array
from numpy import zeros
from numpy import ones
from numpy.linalg import norm
from scipy.sparse import diags

from compas.datastructures import Network
from compas.numerical import connectivity_matrix
from compas.numerical import normrow

from compas.geometry import cross_vectors
from compas.geometry import length_vector
from compas.geometry import length_vector_sqrd

PI = 3.14159

oldsettings = seterr(all='ignore')


__all__ = ['bend_splines']


def bend_splines(network, cables, splines, config=None):
    # --------------------------------------------------------------------------
    # initialise configuration options
    # --------------------------------------------------------------------------
    config = config if config else {}
    units = type('Units', (), dict())
    units.E = config.get('unit.E', 1e+9)
    units.radius = config.get('unit.radius', 1e-3)
    units.thickness = config.get('unit.thickness', 1e-3)
    # --------------------------------------------------------------------------
    # maps
    # --------------------------------------------------------------------------
    key_index = network.key_index()
    uv_index  = network.uv_index()
    # --------------------------------------------------------------------------
    # attribute lists
    # --------------------------------------------------------------------------
    num_v     = network.number_of_nodes()
    num_e     = network.number_of_edges()
    anchors   = list(network.nodes_where({'is_anchor': True}))
    fixed     = [key_index[key] for key in anchors]
    free      = list(set(range(num_v)) - set(fixed))
    xyz       = network.nodes_attributes('xyz')
    p         = network.nodes_attributes(['px', 'py', 'pz'])
    edges     = list(network.edges())
    edges     = [(key_index[u], key_index[v]) for u, v in edges]
    qpre      = network.edges_attribute('qpre')
    fpre      = network.edges_attribute('fpre')                                   # kN
    lpre      = network.edges_attribute('lpre')                                   # m
    linit     = network.edges_attribute('linit')                                  # m
    E         = network.edges_attribute('E')                                      # kN/mm2
    radius    = network.edges_attribute('radius')                                 # mm
    thickness = network.edges_attribute('thickness')                              # mm
    # --------------------------------------------------------------------------
    # attribute arrays
    # --------------------------------------------------------------------------
    xyz       = array(xyz, dtype=float64).reshape((-1, 3))                        # m
    p         = array(p, dtype=float64).reshape((-1, 3))                          # kN
    qpre      = array(qpre, dtype=float64).reshape((-1, 1))
    fpre      = array(fpre, dtype=float64).reshape((-1, 1))                       # kN
    lpre      = array(lpre, dtype=float64).reshape((-1, 1))                       # m
    linit     = array(linit, dtype=float64).reshape((-1, 1))                      # m
    E         = array(E, dtype=float64).reshape((-1, 1))                          # kN/mm2
    radius    = array(radius, dtype=float64).reshape((-1, 1))                     # mm
    thickness = array(thickness, dtype=float64).reshape((-1, 1))                  # mm
    # --------------------------------------------------------------------------
    # scaling
    # with respect to the base units
    # length: m
    # force: N
    # mass: kg
    # --------------------------------------------------------------------------
    E = E * units.E
    radius = radius * units.radius
    thickness = thickness * units.thickness
    # --------------------------------------------------------------------------
    # sectional properties
    # --------------------------------------------------------------------------
    A  = PI * (radius ** 2 - (radius - thickness) ** 2)                         # mm2
    I  = PI * (radius ** 4 - (radius - thickness) ** 4) / 4.                    # mm4
    EA = E * A                                                                  # kN
    EI = E * I                                                                  # kNmm2
    # --------------------------------------------------------------------------
    # overwrite cable force densities
    # --------------------------------------------------------------------------
    for cable in cables:
        for edge in cable['edges']:
            index = uv_index[edge]
            qpre[index, 0] = cable['qpre']
    # --------------------------------------------------------------------------
    # preprocess splines
    # --------------------------------------------------------------------------
    spline_nodes = []
    for spline in splines:
        spline['vi'] = [key_index[spline['start']]]
        spline['ei'] = []
        for u, v in spline['edges']:
            ui = key_index[u]
            vi = key_index[v]
            ei = uv_index[(u, v)]
            spline['ei'].append(ei)
            if spline['vi'][-1] == ui:
                spline['vi'].append(vi)
            else:
                spline['vi'].append(ui)
                edges[ei] = vi, ui
        spline_nodes += spline['vi']
    # --------------------------------------------------------------------------
    # nodes
    # --------------------------------------------------------------------------
    spline_nodes = list(set(spline_nodes))
    membrane_nodes = list(set(free) - set(spline_nodes))
    spline_nodes = list(set(free) & set(spline_nodes))
    # --------------------------------------------------------------------------
    # create the connectivity matrices
    # after spline edges have been aligned
    # --------------------------------------------------------------------------
    C     = connectivity_matrix(edges, 'csr')
    Ct    = C.transpose()
    Ci    = C[:, free]
    Cit   = Ci.transpose()
    Ct2   = Ct.copy()
    Ct2.data **= 2
    # --------------------------------------------------------------------------
    # make connectivity matrices for the splines
    # overwrite properties of the spline edges
    # set qpre, lpre, fpre to zero
    # --------------------------------------------------------------------------
    for spline in splines:
        spline['C']         = C[spline['ei'], :]
        spline['Ct']        = spline['C'].transpose()
        spline['Ci']        = spline['C'][:, free]
        spline['Cit']       = spline['Ci'].transpose()
        spline['E']         = spline['E'] * units.E
        spline['radius']    = spline['radius'] * units.radius
        spline['thickness'] = spline['thickness'] * units.thickness
        spline['A']         = PI * (spline['radius'] ** 2 - (spline['radius'] - spline['thickness']) ** 2)
        spline['I']         = PI * (spline['radius'] ** 4 - (spline['radius'] - spline['thickness']) ** 4) / 4.
        spline['EA']        = spline['E'] * spline['A']
        spline['EI']        = spline['E'] * spline['I']
        for i in spline['ei']:
            qpre[i, 0] = 0.0
            lpre[i, 0] = 0.0
            fpre[i, 0] = 0.0
            EA[i, 0]   = spline['EA']
            EI[i, 0]   = spline['EI']
    # --------------------------------------------------------------------------
    # if none of the initial lengths are set,
    # set the initial lengths to the current lengths
    # --------------------------------------------------------------------------
    if all(linit == 0):
        linit = normrow(C.dot(xyz))
    # --------------------------------------------------------------------------
    # solver parameters
    # --------------------------------------------------------------------------
    alpha = config.get('alpha', 10000)
    kmax  = config.get('kmax', 10000)
    kmax  = int(kmax)
    kdiv  = config.get('kdiv', 100)
    kdiv  = int(kdiv)
    dt    = 1.0
    cc    = 0.1
    ca    = (1 - cc * 0.5) / (1 + cc * 0.5)
    cb    = 0.5 * (1 + ca)
    tol1  = config.get('tol1', 1e-3)
    tol2  = config.get('tol2', 1e-2)
    tol3  = config.get('tol3', 1e-6)
    # --------------------------------------------------------------------------
    # initial values
    # q: force densities
    # f: edge forces
    # l: edge lengths
    # --------------------------------------------------------------------------
    q = ones((num_e, 1), dtype=float64)
    l = normrow(C.dot(xyz))
    f = q * l
    # --------------------------------------------------------------------------
    # initial values
    # v: velocities
    # r: residual forces
    # s: shear forces
    # m: bending moment vectors
    # --------------------------------------------------------------------------
    v = zeros((num_v, 3), dtype=float64)
    r = zeros((num_v, 3), dtype=float64)
    s = zeros((num_v, 3), dtype=float64)
    m = zeros((num_v, 3), dtype=float64)
    # --------------------------------------------------------------------------
    # bracket the iterations
    # --------------------------------------------------------------------------
    kmax = max(1, kmax // kdiv)
    # --------------------------------------------------------------------------
    # helper functions
    # --------------------------------------------------------------------------

    def fdensity():
        q_fpre = fpre / l
        q_lpre = f / lpre
        q_EA   = EA * (l - linit) / (linit * l)
        q_lpre[isinf(q_lpre)] = 0
        q_lpre[isnan(q_lpre)] = 0
        q_EA[isinf(q_EA)]     = 0
        q_EA[isnan(q_EA)]     = 0
        return q_fpre, q_lpre, q_EA

    def shear():
        s_ = 0
        if not splines:
            return s
        for spline in splines:
            v1  = spline['vi'][0]
            v2  = spline['vi'][1]
            b   = xyz[v2] - xyz[v1]
            lb2 = length_vector_sqrd(b)
            for i in range(len(spline['vi']) - 2):
                v1      = spline['vi'][i + 1]
                v2      = spline['vi'][i + 2]
                a       = -b
                b       = xyz[v2] - xyz[v1]
                axb     = array(cross_vectors(a, b))
                la2     = lb2
                lb2     = length_vector_sqrd(b)
                temp    = [la2 * b[_] - lb2 * a[_] for _ in range(3)]
                o       = 0.5 * array(cross_vectors(temp, axb)) / length_vector_sqrd(axb)
                lo      = length_vector(o)
                uo      = o / lo
                bending = spline['EI'] / lo
                if isnan(bending) or isinf(bending):
                    bending = 0
                mvec    = bending * uo
                m[v1]   = mvec
            # multiply the shear force with alpha
            # this scales up the shear force to allow it to compete with
            # the axial forces in the system
            # note that this results in fast convergence far from the target
            # but slow convergence towards the end...
            #
            # spline['C'].dot(m) => mvec difference vectors of spline edges
            # _ / l[spline['ei']] => mvec difference over length of spline edges
            # spline['Ct'].dot(_) => sum of mvec difference over length of spline edges at nodes
            s_ += alpha * spline['Ct'].dot(spline['C'].dot(m) / l[spline['ei']])
        return s_

    def rk4():
        def acceleration(t, v):
            # update shear forces based on the updated geometry!
            dx = v * t
            xyz[free] = xyz0[free] + dx[free]
            r[free] = p[free] + s[free] - D.dot(xyz)
            a = cb * r / mass
            return a
        K0 = dt * acceleration(0.0 * dt, v0)
        K1 = dt * acceleration(0.5 * dt, v0 + 0.5 * K0)
        K2 = dt * acceleration(0.5 * dt, v0 + 0.5 * K1)
        K3 = dt * acceleration(1.0 * dt, v0 + 1.0 * K2)
        dv = (1.0 * K0 + 2.0 * K1 + 2.0 * K2 + 1.0 * K3) / 6.0
        return dv
    # --------------------------------------------------------------------------
    # start iterating
    # --------------------------------------------------------------------------
    crit1 = 1000
    crit2 = 1000
    crit3 = 1000
    iterations = {'membrane': {}, 'spline': {}, 'displacements': {}}
    for i in range(kmax):
        if crit1 < tol1 and crit2 < tol2:
            if alpha == 1:
                break
            alpha = ceil(0.5 * alpha)
        if crit3 < tol3:
            if alpha == 1:
                break
            alpha = ceil(0.5 * alpha)
        for j in range(kdiv):
            k = i * kdiv + j
            print(k)
            q_fpre, q_lpre, q_EA = fdensity()
            q = qpre + q_fpre + q_lpre + q_EA
            Q = diags([q.ravel()], [0])
            D = Cit.dot(Q).dot(C)
            # relax
            mass      = 0.5 * dt ** 2 * Ct2.dot(qpre + q_fpre + q_lpre + EA / linit + 4 * EI / l ** 3)
            xyz0      = xyz.copy()
            v0        = ca * v.copy()
            dv        = rk4()
            v         = v0 + dv
            dx        = v * dt
            xyz[free] = xyz0[free] + dx[free]
            # update
            l = normrow(C.dot(xyz))
            f = q * l
            s = shear()
            r = p + s - Ct.dot(Q).dot(C).dot(xyz)
        # convergence
        crit1 = norm(r[membrane_nodes])
        crit2 = norm(r[spline_nodes])
        crit3 = norm(dx[free])
        # print k, crit1, crit2, crit3
        iterations['membrane'][str(k)] = crit1
        iterations['spline'][str(k)] = crit2
        iterations['displacements'][str(k)] = crit3
    # --------------------------------------------------------------------------
    # update
    # --------------------------------------------------------------------------
    for key, attr in network.nodes(True):
        index = key_index[key]
        attr['x']  = xyz[index, 0]
        attr['y']  = xyz[index, 1]
        attr['z']  = xyz[index, 2]
        attr['rx'] = r[index, 0]
        attr['ry'] = r[index, 1]
        attr['rz'] = r[index, 2]
        attr['sx'] = s[index, 0]
        attr['sy'] = s[index, 1]
        attr['sz'] = s[index, 2]
        attr['mx'] = m[index, 0]
        attr['my'] = m[index, 1]
        attr['mz'] = m[index, 2]
    for key, attr in network.edges(True):
        index = uv_index[key]
        attr['q'] = q[index, 0]
        attr['f'] = f[index, 0]
        attr['l'] = l[index, 0]
        attr['linit'] = linit[index, 0]
    return iterations


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':

    pass
