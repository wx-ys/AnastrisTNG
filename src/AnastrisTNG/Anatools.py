'''
Some useful functions
Orbit: fit the orbit, based on the observed pos,vel,t
ang_mom: vector
angle_between_vectors:
fit_krotmax:

'''

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize
from pynbody.analysis.angmom import calc_faceon_matrix


class Orbit:
    def __init__(self, pos, vel, t):
        '''
        input:
            pos, shape: Nx3, x,y,z
            vel, shape: Nx3, vx,vy,vz
            T, shape: 1xN, (t1,t2,t3...tn)
            sliceT, int or array,
        output:
            pos,
            vel
            T,
        #TODO a, defalt: None, ax,ay,az
        '''
        t = np.asarray(t)
        Targ = np.argsort(t)
        pos = np.asarray(pos)[Targ]
        vel = np.asarray(vel)[Targ]
        t = t[Targ]
        self.pos = pos
        self.vel = vel
        self.t = t
        self.x = pos.T[0]
        self.y = pos.T[1]
        self.z = pos.T[2]
        self.vx = vel.T[0]
        self.vy = vel.T[1]
        self.vz = vel.T[2]
        self._fit()
        self.tmax = np.max(self.t)
        self.tmin = np.min(self.t)

    def _fit(self):
        self.coefx = []
        self.coefy = []
        self.coefz = []
        for i in range(len(self.t) - 1):
            t1 = self.t[i]
            t2 = self.t[i + 1]

            x1 = self.x[i]
            x2 = self.x[i + 1]
            v1 = self.vx[i]
            v2 = self.vx[i + 1]
            A = np.array(
                [
                    [1, t1, t1**2, t1**3],
                    [1, t2, t2**2, t2**3],
                    [0, 1, 2 * t1, 3 * t1**2],
                    [0, 1, 2 * t2, 3 * t2**2],
                ]
            )
            B = np.array([x1, x2, v1, v2])
            coef = solve(A, B)
            self.coefx.append(coef)

            x1 = self.y[i]
            x2 = self.y[i + 1]
            v1 = self.vy[i]
            v2 = self.vy[i + 1]
            A = np.array(
                [
                    [1, t1, t1**2, t1**3],
                    [1, t2, t2**2, t2**3],
                    [0, 1, 2 * t1, 3 * t1**2],
                    [0, 1, 2 * t2, 3 * t2**2],
                ]
            )
            B = np.array([x1, x2, v1, v2])
            coef = solve(A, B)
            self.coefy.append(coef)

            x1 = self.z[i]
            x2 = self.z[i + 1]
            v1 = self.vz[i]
            v2 = self.vz[i + 1]
            A = np.array(
                [
                    [1, t1, t1**2, t1**3],
                    [1, t2, t2**2, t2**3],
                    [0, 1, 2 * t1, 3 * t1**2],
                    [0, 1, 2 * t2, 3 * t2**2],
                ]
            )
            B = np.array([x1, x2, v1, v2])
            coef = solve(A, B)
            self.coefz.append(coef)
        self.coefx = np.array(self.coefx)
        self.coefy = np.array(self.coefy)
        self.coefz = np.array(self.coefz)

    def get(self, t):
        if not hasattr(t, '__iter__'):
            t = [t]
        t = np.array(t)

        ti = np.searchsorted(self.t, t, side='left') - 1
        ti[t < self.tmin] = 0
        ti[t > self.tmax] = 0
        ti[ti == len(self.t) - 1] = len(self.t) - 2

        POS = np.array(
            [
                self.coefx[ti].T[0]
                + self.coefx[ti].T[1] * t
                + self.coefx[ti].T[2] * t**2
                + self.coefx[ti].T[3] * t**3,
                self.coefy[ti].T[0]
                + self.coefy[ti].T[1] * t
                + self.coefy[ti].T[2] * t**2
                + self.coefy[ti].T[3] * t**3,
                self.coefz[ti].T[0]
                + self.coefz[ti].T[1] * t
                + self.coefz[ti].T[2] * t**2
                + self.coefz[ti].T[3] * t**3,
            ]
        ).T

        VEL = np.array(
            [
                self.coefx[ti].T[1]
                + 2 * self.coefx[ti].T[2] * t
                + 3 * self.coefx[ti].T[3] * t**2,
                self.coefy[ti].T[1]
                + 2 * self.coefy[ti].T[2] * t
                + 3 * self.coefy[ti].T[3] * t**2,
                self.coefz[ti].T[1]
                + 2 * self.coefz[ti].T[2] * t
                + 3 * self.coefz[ti].T[3] * t**2,
            ]
        ).T
        POS[t < self.tmin] = np.array([0, 0, 0])
        POS[t > self.tmax] = np.array([0, 0, 0])
        VEL[t < self.tmin] = np.array([0, 0, 0])
        VEL[t > self.tmax] = np.array([0, 0, 0])
        return POS, VEL


# from pynbody.analysis.angmom.ang_mom_vec
def ang_mom(snap):
    angmom = (
        (snap['mass'].reshape((len(snap), 1)) * np.cross(snap['pos'], snap['vel']))
        .sum(axis=0)
        .view(np.ndarray)
    )
    return angmom


def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    cos_theta = dot_product / (v1_norm * v2_norm)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)


def get_krot(pos, vel, mass, Rota):
    pos_new = np.dot(Rota, pos.T)
    vel_new = np.dot(Rota, vel.T)
    rxy = np.sqrt((pos_new[0] ** 2 + pos_new[1] ** 2))
    vcxy = (pos_new[0] * vel_new[1] - pos_new[1] * vel_new[0]) / rxy
    Krot = np.array(
        np.sum((0.5 * mass * (vcxy**2))) / np.sum(mass * 0.5 * (vel_new**2).sum(axis=0))
    )
    return Krot


def _kroterr(initpa, *args):

    x, y, z = initpa
    if (x + y + z) == 0:
        z = 1
    Rc = np.array([x, y, z])
    Rota = calc_faceon_matrix(Rc)
    pos, vel, mass = args
    Krot = get_krot(pos, vel, mass, Rota)
    return 100 - Krot * 100


def fit_krotmax(pos, vel, mass, method='BFGS'):

    res = minimize(
        _kroterr,
        (0, 0.0, 1.0),
        (pos, vel, mass),
        method=method,
    )
    if res.success:
        result = {
            'krotmax': (1 - res.fun / 100),
            'krotvec': (res.x / np.linalg.norm(res.x)),
            'krotmat': calc_faceon_matrix(res.x),
            'angle': angle_between_vectors(res.x, np.array([0, 0, 1])),
        }
        return result
    else:
        print('Failed to fit maximum krot')
        return None

# a modified version from https://pynbody.readthedocs.io/latest/_modules/pynbody/analysis/halo.html#shape
def MoI_shape(sim, calpa: str = 'mass', nbins=1, rmin=None, rmax=None, bins='equal',
          ndim=3, max_iterations=10, tol=1e-3, justify=False, **kwargs):
    if (rmax == None): rmax = sim['r'].max()
    if (rmin == None): rmin = rmax / 1E3
    assert ndim in [2, 3]
    assert max_iterations > 0
    assert tol > 0
    assert rmin >= 0
    assert rmax > rmin
    assert nbins > 0
    if ndim == 2:
        assert np.sum((sim['rxy'] >= rmin) & (sim['rxy'] < rmax)) > nbins * 2
    elif ndim == 3:
        assert np.sum((sim['r'] >= rmin) & (sim['r'] < rmax)) > nbins * 2
    if bins not in ['equal', 'log', 'lin']: bins = 'equal'

    # Handy 90 degree rotation matrices:
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    Ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # -----------------------------FUNCTIONS-----------------------------
    sn = lambda r, N: np.append([r[i * int(len(r) / N):(1 + i) * int(len(r) / N)][0] \
                                 for i in range(N)], r[-1])

    # General equation for an ellipse/ellipsoid:
    def Ellipsoid(pos, a, R):
        x = np.dot(R.T, pos.T)
        return np.sum(np.divide(x.T, a) ** 2, axis=1)

    # Define moment of inertia tensor:
    def MoI(r, m, ndim=3):
        return np.array([[np.sum(m * r[:, i] * r[:, j]) for j in range(ndim)] for i in range(ndim)])

    # Calculate the shape in a single shell:
    def shell_shape(r, pos, mass, a, R, r_range, ndim=3):

        # Find contents of homoeoidal shell:
        mult = r_range / np.mean(a)
        in_shell = (r > min(a) * mult[0]) & (r < max(a) * mult[1])
        pos, mass = pos[in_shell], mass[in_shell]
        inner = Ellipsoid(pos, a * mult[0], R)
        outer = Ellipsoid(pos, a * mult[1], R)
        in_ellipse = (inner > 1) & (outer < 1)
        ellipse_pos, ellipse_mass = pos[in_ellipse], mass[in_ellipse]

        # End if there is no data in range:
        if not len(ellipse_mass):
            return a, R, np.sum(in_ellipse)

        # Calculate shape tensor & diagonalise:
        D = list(np.linalg.eigh(MoI(ellipse_pos, ellipse_mass, ndim) / np.sum(ellipse_mass)))

        # Rescale axis ratios to maintain constant ellipsoidal volume:
        R2 = np.array(D[1])
        a2 = np.sqrt(abs(D[0]) * ndim)
        div = (np.prod(a) / np.prod(a2)) ** (1 / float(ndim))
        a2 *= div

        return a2, R2, np.sum(in_ellipse)

    # Re-align rotation matrix:
    def realign(R, a, ndim):
        if ndim == 3:
            if a[0] > a[1] > a[2] < a[0]:
                pass  # abc
            elif a[0] > a[1] < a[2] < a[0]:
                R = np.dot(R, Rx)  # acb
            elif a[0] < a[1] > a[2] < a[0]:
                R = np.dot(R, Rz)  # bac
            elif a[0] < a[1] > a[2] > a[0]:
                R = np.dot(np.dot(R, Rx), Ry)  # bca
            elif a[0] > a[1] < a[2] > a[0]:
                R = np.dot(np.dot(R, Rx), Rz)  # cab
            elif a[0] < a[1] < a[2] > a[0]:
                R = np.dot(R, Ry)  # cba
        elif ndim == 2:
            if a[0] > a[1]:
                pass  # ab
            elif a[0] < a[1]:
                R = np.dot(R, Rz[:2, :2])  # ba
        return R

    # Calculate the angle between two vectors:
    def angle(a, b):
        return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # Flip x,y,z axes of R2 if they provide a better alignment with R1.
    def flip_axes(R1, R2):
        for i in range(len(R1)):
            if angle(R1[:, i], -R2[:, i]) < angle(R1[:, i], R2[:, i]):
                R2[:, i] *= -1
        return R2

    # -----------------------------FUNCTIONS-----------------------------

    # Set up binning:
    r = np.array(sim['r']) if ndim == 3 else np.array(sim['rxy'])
    pos = np.array(sim['pos'])[:, :ndim]
    mass = np.array(sim[calpa])

    if (bins == 'equal'):  # Bins contain equal number of particles
        full_bins = sn(np.sort(r[(r >= rmin) & (r <= rmax)]), nbins * 2)
        bin_edges = full_bins[0:nbins * 2 + 1:2]
        rbins = full_bins[1:nbins * 2 + 1:2]
    elif (bins == 'log'):  # Bins are logarithmically spaced
        bin_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
        rbins = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    elif (bins == 'lin'):  # Bins are linearly spaced
        bin_edges = np.linspace(rmin, rmax, nbins + 1)
        rbins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Initialise the shape arrays:
    rbins = rbins
    axis_lengths = np.zeros([nbins, ndim])
    N_in_bin = np.zeros(nbins).astype('int')
    rotations = [0] * nbins

    # Loop over all radial bins:
    for i in range(nbins):

        # Initial spherical shell:
        a = np.ones(ndim) * rbins[i]
        a2 = np.zeros(ndim)
        a2[0] = np.inf
        R = np.identity(ndim)

        # Iterate shape estimate until a convergence criterion is met:
        iteration_counter = 0
        while ((np.abs(a[1] / a[0] - np.sort(a2)[-2] / max(a2)) > tol) & \
               (np.abs(a[-1] / a[0] - min(a2) / max(a2)) > tol)) & \
                (iteration_counter < max_iterations):
            a2 = a.copy()
            a, R, N = shell_shape(r, pos, mass, a, R, bin_edges[[i, i + 1]], ndim)
            iteration_counter += 1

        # Adjust orientation to match axis ratio order:
        R = realign(R, a, ndim)

        # Ensure consistent coordinate system:
        if np.sign(np.linalg.det(R)) == -1:
            R[:, 1] *= -1

        # Update profile arrays:
        a = np.flip(np.sort(a))
        axis_lengths[i], rotations[i], N_in_bin[i] = a, R, N

    # Ensure the axis vectors point in a consistent direction:
    if justify:
        _, _, _, R_global = MoI_shape(sim, nbins=1, rmin=rmin, rmax=rmax, ndim=ndim)
        rotations = np.array([flip_axes(R_global, i) for i in rotations])
    rotations = np.squeeze(rotations)
    if len(rotations.shape)>2:
        angles = [np.degrees(angle(np.array([0,0,1]), np.dot(i,np.array([0,0,1])))) for i in rotations]
        abc_vec = [[np.dot(i,np.array([1,0,0])),np.dot(i,np.array([0,1,0])),np.dot(i,np.array([0,0,1]))] for i in rotations]
    else:
        angles = np.degrees(angle(np.array([0,0,1]), np.dot(rotations,np.array([0,0,1])))) 
        abc_vec = [np.dot(rotations,np.array([1,0,0])),np.dot(rotations,np.array([0,1,0])),np.dot(rotations,np.array([0,0,1]))]
    retdict={}
    retdict['rbins']=rbins
    retdict['N_in_bin']=N_in_bin
    retdict['abc']=np.squeeze(axis_lengths.T).T
    retdict['abc_vec']=abc_vec
    retdict['rotations']=rotations
    retdict['angles']=angles
    return retdict