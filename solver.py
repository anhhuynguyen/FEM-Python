import numpy as np
from scipy.sparse import lil_matrix, diags, coo_matrix
from scipy.sparse.linalg import spsolve, splu, eigsh
import stat


class FEM2D:

    def __init__(self, meshobj, constraints, forces, matprop, thickness=1, **kwargs):

        self.nodes = meshobj.nodes  # nodal coordinates
        self.elements = meshobj.elements  # node connectivity
        self.eltype = meshobj.eltype  # element type
        self.nod = meshobj.nod  # number of nodes per elements
        self.nelts = meshobj.nelts  # number of elements in the mesh
        self.nn = meshobj.nn  # number of nodes in the mesh
        self.thickness = thickness

        self.natural_boundary = forces
        self.essential_boundary = constraints
        self.matprop = matprop
        self.response = None

        self.neq = None  # number of degrees of freedom in the mesh
        self.ndof = None  # number of degrees of freedom per node
        self.D = None  # field variables (temperature, displacement)

    def __repr__(self):
        return f'Number of nodes: {self.nn}\nNumber of elements: {self.nelts}\n' \
            f'Number of dofs: {self.neq}\nElement type: {self.eltype}'

    def shape_der(self, *args, **kwargs):
        pass

    def shape_grad(self, *args, **kwargs):
        pass

    def shape_func(self, *args, **kwargs):
        pass

    def consistent_mass_matrix(self, *args):
        """ Compute the consistent mass matrix """
        pass

    def lumped_mass_matrix(self, *args):
        """ Comptute the lumped mass matrix """
        pass

    def assemble_stiffness(self):
        pass

    def assemble_mass_matrix(self, mass_type='lumped'):
        pass

    def assemble_force_vector(self, *args):
        pass

    def stiffness(self, D, x, nip):
        """
        Compute element stiffness matrix [k]
        :param nip: number of integration points
        :param D: conductivity coefficient matrix
        :param x: nodal coordinates of an element
        """
        k = np.zeros((self.nod * self.ndof, self.nod * self.ndof))
        if self.eltype in ['quad4', 'reduced-integration quad4']:
            gauss_points, weight = FEM2D.gauss(nip)
            if nip == 2:
                for zeta in gauss_points:
                    for eta in gauss_points:
                        grad_shape = self.shape_grad(zeta, eta)
                        detjac, jac = FEM2D.jacobian(grad_shape, x)
                        B = self.shape_der(detjac, jac, grad_shape)
                        k += (B.T @ D @ B) * detjac * self.thickness * weight ** 2
            elif nip == 1:
                grad_shape = self.shape_grad(gauss_points, gauss_points)
                detjac, jac = FEM2D.jacobian(grad_shape, x)
                B = self.shape_der(detjac, jac, grad_shape)
                k = (B.T @ D @ B) * detjac * self.thickness * weight ** 2
        elif self.eltype == 'tri3':
            B = self.shape_der(coords=x)
            k = B.T @ D @ B * self.thickness * stat.tri_area(*x)
        return k

    def steady_state_solver(self):
        # set problem type
        self.response = 'steady-state'
        # construnct the global stiffness matrix and the force/flux vector
        K = self.assemble_stiffness()
        F = self.assemble_force_vector()
        # incorporate the boundary condition
        for value, j in self.essential_boundary.items():
            K[j, j] *= 1e20
            F[j] = (K[j, j] * value).toarray()
        # solve for D, if using sparse:
        self.D = spsolve(K.tocsr(), F)[:, np.newaxis]

    @staticmethod
    def jacobian(grad_shape, x):
        # jacobian matrix J
        jac = grad_shape @ x
        # determinant of jacobian |J|
        detjac = jac[0, 0] * jac[1, 1] - jac[1, 0] * jac[0, 1]
        return detjac, jac

    @staticmethod
    def gauss(nip):
        """
        Compute gauss points and the corresponding weights
        :return: gauss points and weights
        """
        if nip == 1:
            return 0, 2
        elif nip == 2:
            return [-1 / np.sqrt(3), 1 / np.sqrt(3)], 1


class HeatTransfer(FEM2D):

    def __init__(self, meshobj, prescribed_temp, flux, heat_source, matprop, thickness=1, **kwargs):
        FEM2D.__init__(self, meshobj, prescribed_temp, flux, matprop, thickness)
        self.source = heat_source
        self.ndof = 1
        self.neq = self.nn * self.ndof
        self.D = np.zeros((self.neq, 1))
        self.mass_density = kwargs.get('density', 0)
        self.specific_heat = kwargs.get('specific_heat', 0)
        self.boundary_convection = kwargs.get('boundary_convection', {})
        self.ambient_temp = kwargs.get('ambient_temp', 0)
        self.timestep = kwargs.get('timestep', 1)  # timestep size for transisent analysis

    def lumped_mass_matrix(self, x):
        """
        Comptute the lumped mass matrix
        :param x: nodal coordinates of an element
        :return: the lumped mass matrix
        """
        return self.specific_heat * self.mass_density * self.thickness * stat.quad_area(*x) * 0.25 * np.ones(4)

    def consistent_mass_matrix(self, x, nip):
        """
        Compute the consitent mass matrix
        :param x: nodal coordinates of an element
        :param nip: number of Gauss points
        :return: the cnsistent mass matrix
        """
        m = np.zeros((self.nod, self.nod))
        gauss_points, weights = FEM2D.gauss(nip)

        for zeta in gauss_points:
            for eta in gauss_points:
                grad_shape = self.shape_grad(zeta, eta)
                detjac, jac = FEM2D.jacobian(grad_shape, x)
                N = self.shape_func(zeta, eta)
                m += N[:, np.newaxis] @ N[np.newaxis, :] * detjac * \
                    self.mass_density * self.specific_heat * self.thickness
        return m

    def implicit_transient_solver(self, beta, tspan=1, intial_temp=0, mass='consistent', eps=None):
        """
        Solve a trasient heat transfer problem employing the Newmark method or Beta method with implicit method
        :param beta: chosen parameter. Default to backward difference (beta = 1)
                      beta = 0.5: Crank-Nicolson, unconditionally stable
                      beta = 2/3: Galerkin, unconditionally stable
                      beta = 1  : Backward difference, unconditionally stable
        :param tspan: time span
        :param intial_temp: initial temperature
        :param mass: lumped mass or consistent mass matrix
        :param eps: stop when maximum change in temperature achieves eps

        """
        assert 0 < beta <= 1, "Choose beta = 0.5, 2/3 or 1"
        # set problem type
        self.response = 'transient'
        # get the time step size
        dt = self.timestep
        # set prescribed temperatures to the initial temperature vector
        T = np.zeros((self.neq, 1))
        T[:] = intial_temp
        # set up the global stiffness and mass matrices and convert them to csr
        K = self.assemble_stiffness().tocsr()
        M = self.assemble_mass_matrix(mass).tocsr()
        # evaluate the effective stiffness matrix Keff
        Keff = lil_matrix(1 / dt * M + beta * K)
        Ki = 1 / dt * M - (1 - beta) * K
        # assemble the load vector R
        R = self.assemble_force_vector()[:, np.newaxis]
        # solve for T
        t = 0
        with open(r'C:\Users\ANHHUY\Documents\Python\FEM2D\database\temp.txt', 'w') as file:
            np.savetxt(file, np.concatenate(([[t]], T.T), axis=1), delimiter='\t')
            while t < tspan:
                # update i, t
                t += dt
                # evaluate the effective loads Reff
                Reff = Ki @ T + (1 - beta) * R + beta * R
                Ke = Keff.copy()
                Re = Reff.copy()
                # impose boundary coditions
                for value, j in self.essential_boundary.items():
                    Ke[j, j] *= 1e20
                    Re[j] = (Ke[j, j] * value).toarray().T
                # solve the next T
                Told = T
                T = spsolve(Ke.tocsr(), Re)[:, np.newaxis]
                # compute change in temperature
                if eps is not None:
                    if np.max(abs(T - Told)) < eps:
                        break
                # write the current solution to file
                np.savetxt(file, np.concatenate(([[t]], T.T), axis=1), delimiter='\t')

    def explicit_transient_solver(self, tspan, initial_temp=0, eps=None):
        """
        Solve a trasient heat transfer problem employing the Newmark method or Beta method with explicit method
        beta = 0: Forward difference(or Euler), conditionally stable
        :param tspan: time span
        :param initial_temp: initial temperature
        :param eps: stop when maximum change in temperature achieves eps
        """

        # set problem type
        self.response = 'transient'
        # get the time step size
        dt = self.timestep
        # set prescribed temperatures to the initial temperature vector
        T = np.zeros((self.neq, 1))
        T[:] = initial_temp
        # set up the global stiffness and mass matrices
        K = self.assemble_stiffness().tocsr()
        M = self.assemble_mass_matrix('lumped').tocsr()
        # evaluate the effective stiffness matrix Keff
        Ki = 1 / dt * M - K
        # get diags from M
        M = M.diagonal()[:, np.newaxis]
        # assemble the load vector R
        R = self.assemble_force_vector()[:, np.newaxis]
        # solve for T
        t = 0
        with open(r'C:\Users\ANHHUY\Documents\Python\FEM2D\database\temp.txt', 'w') as file:
            np.savetxt(file, np.concatenate(([[t]], T.T), axis=1), delimiter='\t')
            while t < tspan:
                # update t
                t += dt
                # impose boundary condition
                for value, dofs in self.essential_boundary.items():
                    T[dofs] = value
                # solve for the next T
                Told = T
                T = (Ki @ T + R) * dt / M
                # compute error
                if eps is not None:
                    if np.max(abs(T - Told)) < eps:
                        break
                # write the current solution to file
                np.savetxt(file, np.concatenate(([[t]], T.T), axis=1), delimiter='\t')

    def assemble_stiffness(self):
        # global stiffness matrix
        K = lil_matrix((self.neq, self.neq))
        for element in self.elements:
            k = self.stiffness(self.matprop, self.nodes[element], 2).ravel()
            ii = 0
            for i in element:
                for j in element:
                    K[i, j] += k[ii]
                    ii += 1
        if self.boundary_convection:
            K = self.assemble_convection_matrix(K)
        return K

    def assemble_mass_matrix(self, mass_type='lumped'):
        # global mass matrix
        if mass_type == 'lumped':
            M = np.zeros(self.neq)
            for element in self.elements:
                M[element] += self.lumped_mass_matrix(self.nodes[element])
            return diags(M)
        else:
            M = lil_matrix((self.neq, self.neq))
            for element in self.elements:
                m = self.consistent_mass_matrix(self.nodes[element], 2).ravel()
                ii = 0
                for i in element:
                    for j in element:
                        M[i, j] += m[ii]
                        ii += 1
            return M

    def assemble_convection_matrix(self, K):
        for film_coeff, elements in self.boundary_convection.items():
            for element, edge in elements:
                h = self.convection_stiffness(film_coeff, self.elements[element], edge).ravel()
                ii = 0
                for i in self.elements[element]:
                    for j in self.elements[element]:
                        K[i, j] += h[ii]
                        ii += 1
        return K

    def assemble_force_vector(self):
        F = np.zeros(self.neq, dtype=float)
        if self.source:
            for val, elements in self.source.items():
                for element in elements:
                    fb = self.nodal_source_vector(self.elements[element], val)
                    F[self.elements[element]] += fb
        if self.natural_boundary:
            for val, items in self.natural_boundary.items():
                for element, edge in items:
                    fq = self.nodal_flux_vector(self.elements[element], edge, val)
                    F[self.elements[element]] += fq
        if self.boundary_convection:
            for film_coeff, items in self.boundary_convection.items():
                for element, edge in items:
                    fh = self.convection_load_vector(self.elements[element], edge, film_coeff)
                    F[self.elements[element]] += fh
        return F

    def convection_stiffness(self, film_coeff, element, i):
        """ Convection stiffness matrix for boundary elements """
        h = np.zeros((self.nod, self.nod))
        h[[i - 3, i - 4], [[i - 3, i - 4], [i - 4, i - 3]]] = [[2, 2], [1, 1]]  # Just for linear retangular elements
        if i == 0 or i == 2:
            length = abs((self.nodes[element][0][0] - self.nodes[element][1][0]))
        else:
            length = abs((self.nodes[element][1][1] - self.nodes[element][2][1]))
        return 1 / 6 * film_coeff * self.thickness * length * h

    def convection_load_vector(self, element, i, film_coeff):
        if i == 0 or i == 2:
            length = abs((self.nodes[element][0][0] - self.nodes[element][1][0]))
        else:
            length = abs((self.nodes[element][1][1] - self.nodes[element][2][1]))
        convection_load = np.zeros(self.nod)
        convection_load[[i - 4, i - 3]] = 1
        return 0.5 * film_coeff * self.ambient_temp * length * convection_load

    def nodal_source_vector(self, element, val):
        gauss_points, weight = FEM2D.gauss(2)
        heat_source = np.zeros(self.nod)
        for zeta in gauss_points:
            for eta in gauss_points:
                grad_shape = self.shape_grad(zeta, eta)
                detjac, _ = FEM2D.jacobian(grad_shape, self.nodes[element])
                shape = self.shape_func(zeta, eta)
                heat_source += weight * weight * shape * val * detjac
        return heat_source

    def nodal_flux_vector(self, element, edge, val):
        # gauss_points, weight = FEM2D.gauss(1)
        # if edge == 0 or edge == 2:
        #     zeta = gauss_points
        #     eta = 1 if edge == 2 else -1
        #     detjac = abs((self.nodes[element][0][0] - self.nodes[element][1][0])) * 0.5
        # else:
        #     eta = gauss_points
        #     zeta = 1 if edge == 1 else -1
        #     detjac = abs((self.nodes[element][1][1] - self.nodes[element][2][1])) * 0.5
        els = self.nodes[self.elements[element]]
        gs, weight = FEM2D.gauss(1)
        if edge == 0:
            zeta, eta = gs, -1
            detjac = 0.5 * np.linalg.norm(els[0] - els[1])
        elif edge == 1:
            zeta, eta = 1, gs
            detjac = 0.5 * np.linalg.norm(els[2] - els[1])
        elif edge == 2:
            zeta, eta = gs, 1
            detjac = 0.5 * np.linalg.norm(els[2] - els[3])
        else:
            zeta, eta = -1, gs
            detjac = 0.5 * np.linalg.norm(els[1] - els[3])
        shape = self.shape_func(zeta, eta)
        return weight * val * shape * detjac

    def get_flux(self, sol, pos='centroid', average=False):
        """
        Compute the flux vectors at each gauss point in an element or at the centroid of the element
        :return: an array of the flux vector.
        """

        if pos == 'centroid':
            q = np.zeros((self.nelts, 2))
            gs, weights = FEM2D.gauss(1)
            for i, element in enumerate(self.elements):
                grad_shape = self.shape_grad(gs, gs)
                detjac, jac = FEM2D.jacobian(grad_shape, self.nodes[element])
                B = self.shape_der(detjac, jac, grad_shape)
                q[i, :] = (-self.matprop @ B @ sol[element]).ravel()
            return q
        elif pos == 'intp':
            gs, weights = FEM2D.gauss(2)
            if average:
                N = np.array([
                    [(1 + np.sqrt(3) / 2), -0.5, (1 - np.sqrt(3) / 2), -0.5],
                    [-0.5, (1 + np.sqrt(3) / 2), -0.5, (1 - np.sqrt(3) / 2)],
                    [(1 - np.sqrt(3) / 2), -0.5, (1 + np.sqrt(3) / 2), -0.5],
                    [-0.5, (1 - np.sqrt(3) / 2), -0.5, (1 + np.sqrt(3) / 2)]
                ])
                f, npn = np.zeros((self.nn, 2)), np.zeros((self.nn, 1))
                q = np.zeros((self.nod, 2))
                for i, element in enumerate(self.elements):
                    for j, (zeta, eta) in enumerate(zip([gs[0], gs[1], gs[1], gs[0]], [gs[0], gs[0], gs[1], gs[1]])):
                        grad_shape = self.shape_grad(zeta, eta)
                        detjac, jac = FEM2D.jacobian(grad_shape, self.nodes[element])
                        B = self.shape_der(detjac, jac, grad_shape)
                        q[j] = (-self.matprop @ B @ sol[element]).ravel()
                    f[element, :] += N @ q
                    npn[element] += 1
                return f / npn
            else:
                q = np.zeros((self.nelts, self.nod, 2))
                for i, element in enumerate(self.elements):
                    for j, (zeta, eta) in enumerate(zip([gs[0], gs[1], gs[1], gs[0]], [gs[0], gs[0], gs[1], gs[1]])):
                        grad_shape = self.shape_grad(zeta, eta)
                        detjac, jac = FEM2D.jacobian(grad_shape, self.nodes[element])
                        B = self.shape_der(detjac, jac, grad_shape)
                        q[i, j, :] = (-self.matprop @ B @ sol[element]).ravel()
                return q

    def shape_func(self, zeta, eta):
        """
        Compute the value of shape functions at a point.
        :param zeta: local x coordinates of the elememt.
        :param eta: local y coordinates of the elememt.
        :return: the values of the 1x4 shape functions at a point.
        """

        if self.eltype == 'quad4':
            return np.array([
                0.25 * (1 - zeta) * (1 - eta),
                0.25 * (1 + zeta) * (1 - eta),
                0.25 * (1 + zeta) * (1 + eta),
                0.25 * (1 - zeta) * (1 + eta)
            ])

    def shape_der(self, detjac, jac, grad_shape):
        """
        Compute derivatives of the shape functions [B] with respect to global coordinates
        :param detjac: determinant of jacobian
        :param jac: jacobian matrix
        :param grad_shape: derivatives of the shape functions with respect to local coordinates
        :return: derivatives of the shape functions
        """

        if self.eltype == 'quad4':
            invjac = (1 / detjac) * np.array([[jac[1, 1], -jac[0, 1]], [-jac[1, 0], jac[0, 0]]])
            return invjac @ grad_shape

    def shape_grad(self, zeta, eta):
        """
         Compute derivatives of the shape functions with respect to local coordinates
         :param zeta: local x coordinates of the elememt.
         :param eta: local y coordinates of the elememt.
         :return: derivatives of the shape functions
         """

        if self.eltype == 'quad4':
            return 0.25 * np.array([
                [eta - 1, 1 - eta, 1 + eta, -eta - 1],
                [zeta - 1, -1 - zeta, 1 + zeta, 1 - zeta]
            ])


class PlaneStress(FEM2D):

    def __init__(self, meshobj, constraints, force, pressure, matprop, thickness=1, **kwargs):
        FEM2D.__init__(self, meshobj, constraints, force, matprop, thickness)

        self.ndof = 2
        self.eltdofs = self.ndof * self.nod
        self.neq = self.nn * self.ndof
        self.D = np.zeros((self.neq, 1))
        self.pressure = pressure
        self.mass_density = kwargs.get('mass_density', 0)
        E, nu = matprop
        if kwargs.get('model', 'plane stress') == 'plane stress':
            self.matprop = (E / (1 - nu ** 2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, 0.5 * (1 - nu)]])
        elif kwargs.get('model', 'plane stress') == 'plane strain':
            self.matprop = (E / ((1 + nu) * (1 - 2 * nu))) * \
                           np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, 0.5 * (1 - 2 * nu)]])

    def steady_state_solver(self):
        # set problem type
        self.response = 'steady-state'
        # construnct the global stiffness matrix and the force/flux vector
        K = self.assemble_stiffness()
        F = self.assemble_force_vector(self.natural_boundary, self.pressure)
        print(K.toarray())
        # incorporate the boundary condition
        for (dx, dy), j in self.essential_boundary:
            if dx is not None:
                K[2 * j, 2 * j] *= 1e20
                F[2 * j] = (K[2 * j, 2 * j] * dx).toarray().T
            if dy is not None:
                K[2 * j + 1, 2 * j + 1] *= 1e20
                F[2 * j + 1] = (K[2 * j + 1, 2 * j + 1] * dy).toarray().T
        print()
        print(K.toarray())
        print()
        print(F)
        # solve for D, if using sparse:
        self.D = spsolve(K.tocsr(), F)[:, np.newaxis]
        print(self.D)

    def assemble_stiffness(self):
        """ Assemble global stiffness matrix """
        # select number of inegration points
        nip = 2 if self.eltype == 'quad4' else 1
        # form 1D arrays to store the positions of each element and their values
        nk = self.eltdofs ** 2  # number of elements in the element stiffness
        Ig = np.zeros(nk * self.nelts)
        Jg = np.zeros(nk * self.nelts)
        Kg = np.zeros(nk * self.nelts)
        # loop over elements to fill Ig, Jg, Kf
        for j, element in enumerate(self.elements):
            k = self.stiffness(self.matprop, self.nodes[element], nip).ravel()
            # print(self.stiffness(self.matprop, self.nodes[element], nip))
            # print()
            dofs = np.array([[2 * i, 2 * i + 1] for i in element]).ravel()
            Ig[nk*j:nk*j+nk] = np.repeat(dofs, self.eltdofs)
            Jg[nk*j:nk*j+nk] = np.tile(dofs, self.eltdofs)
            Kg[nk*j:nk*j+nk] = k
        return coo_matrix((Kg, (Ig, Jg)), shape=(self.neq, self.neq)).tolil()

    def assemble_force_vector(self, natural_boundary, pressure):
        """
        force = [[(Fx, Fy), dofs], [...]]
        """
        F = np.zeros((self.neq, 1))
        if natural_boundary:
            for (fx, fy), dofs in natural_boundary:
                if fx:
                    F[2 * dofs] += fx
                if fy:
                    F[2 * dofs + 1] += fy
        if pressure:
            for p, lines in pressure:
                for element, line in np.array(lines):
                    fp = self.surface_forces(np.array(p), line, element)
                    dofs = np.array([[2 * i, 2 * i + 1] for i in self.elements[element]]).ravel()
                    F[dofs] += fp
        return F

    def surface_forces(self, p, line, element):
        els = self.nodes[self.elements[element]]
        if self.eltype in ['quad4', 'reduced-integration quad4']:
            gs, weight = FEM2D.gauss(1)
            if line == 0:
                zeta, eta = gs, -1
                detjac = 0.5 * np.linalg.norm(els[0] - els[1])
            elif line == 1:
                zeta, eta = 1, gs
                detjac = 0.5 * np.linalg.norm(els[2] - els[1])
            elif line == 2:
                zeta, eta = gs, 1
                detjac = 0.5 * np.linalg.norm(els[2] - els[3])
            else:
                zeta, eta = -1, gs
                detjac = 0.5 * np.linalg.norm(els[1] - els[3])
            N = self.shape_func(zeta, eta)
            return N.T @ p[:, np.newaxis] * detjac * self.thickness * weight
        elif self.eltype == 'tri3':
            if line == 0:
                zeta, eta = 1, 0
                length = np.linalg.norm(els[0] - els[1])
            elif line == 1:
                zeta, eta = 0.5, 0.5
                length = np.linalg.norm(els[0] - els[1])
            else:
                zeta, eta = 0, 1
                length = np.linalg.norm(els[0] - els[1])
            N = self.shape_func(zeta, eta)
            return N.T @ p[:, np.newaxis] * self.thickness * length

    def get_stress(self):
        """ Compute stress/strain"""
        sigma = np.zeros((self.nelts, 3, 1))  # stresses
        epsilon = np.zeros((self.nelts, 3, 1))  # strains
        D = self.matprop
        gs, w = FEM2D.gauss(1)

        for i, element in enumerate(self.elements):
            grad_shape = self.shape_grad(gs, gs)
            detjac, jac = FEM2D.jacobian(grad_shape, self.nodes[element])
            B = self.shape_der(detjac, jac, grad_shape)
            dofs = np.array([[2 * i, 2 * i + 1] for i in element]).ravel()
            epsilon[i] = B @ self.D[dofs]
            sigma[i] = D @ epsilon[i]

        return sigma, epsilon

    def shape_der(self, detjac=None, jac=None, grad_shape_local=None, coords=None):
        """
        Compute derivatives of the shape functions [B] with respect to global coordinates
        :param detjac: determinant of jacobian
        :param jac: jacobian matrix
        :param grad_shape_local: derivatives of the shape functions with respect to local coordinates
        :param coords: global coordinates of nodas.
        :return: derivatives of the shape functions
        """
        if self.eltype in ['quad4', 'reduced-integration quad4']:
            invjac = (1 / detjac) * np.array([[jac[1, 1], -jac[0, 1]], [-jac[1, 0], jac[0, 0]]])
            G = invjac @ grad_shape_local
            B = np.array([
                [G[0, 0], 0, G[0, 1], 0, G[0, 2], 0, G[0, 3], 0],
                [0, G[1, 0], 0, G[1, 1], 0, G[1, 2], 0, G[1, 3]],
                [G[1, 0], G[0, 0], G[1, 1], G[0, 1], G[1, 2], G[0, 2], G[1, 3], G[0, 3]]
            ])
            return B
        elif self.eltype == 'tri3':
            A = stat.tri_area(*coords)
            x, y = coords[:, 0], coords[:, 1]
            y23 = y[1] - y[2]
            y31 = y[2] - y[0]
            y12 = y[0] - y[1]
            x32 = x[2] - x[1]
            x13 = x[0] - x[2]
            x21 = x[1] - x[0]
            B = 1 / (2 * A) * np.array([
                [y23, 0, y31, 0, y12, 0],
                [0, x32, 0, x13, 0, x21],
                [x32, y23, x13, y31, x21, y12]
            ])
            return B

    def shape_grad(self, zeta, eta):
        """
        Compute derivatives of the shape functions with respect to local coordinates
        :param zeta: local x coordinates of the elememt.
        :param eta: local y coordinates of the elememt.
        :return: derivatives of the shape functions
        """

        if self.eltype in ['quad4', 'reduced-integration quad4']:
            return 0.25 * np.array([
                [eta - 1, 1 - eta, 1 + eta, -eta - 1],
                [zeta - 1, -1 - zeta, 1 + zeta, 1 - zeta]
            ])

    def shape_func(self, zeta, eta):
        """
        Compute the value of shape functions at a point.
        :param zeta: local x coordinates of the elememt.
        :param eta: local y coordinates of the elememt.
        :return: the values of the shape functions at a point.
        """

        if self.eltype in ['quad4', 'reduced-integration quad4']:
            return 0.25 * np.array([
                [(1 - zeta) * (1 - eta), 0, (1 + zeta) * (1 - eta), 0,
                 (1 + zeta) * (1 + eta), 0, (1 - zeta) * (1 + eta), 0],
                [0, (1 - zeta) * (1 - eta), 0, (1 + zeta) * (1 - eta),
                 0, (1 + zeta) * (1 + eta), 0, (1 - zeta) * (1 + eta)]
            ])
        elif self.eltype == 'tri3':
            return np.array([
                [(1 - zeta - eta), 0, zeta, 0, eta, 0],
                [0, (1 - zeta - eta), 0, zeta, 0, eta]
            ])

    def consistent_mass_matrix(self, x, nip):
        """
        Compute consistent mass matrix
        :param x: elemental coordinates
        :param nip: number of integration points
        :return: consistent mass matrix
        """

        m = np.zeros((self.eltdofs, self.eltdofs))
        gauss_points, weights = FEM2D.gauss(nip)

        for zeta in gauss_points:
            for eta in gauss_points:
                grad_shape = self.shape_grad(zeta, eta)
                detjac, jac = FEM2D.jacobian(grad_shape, x)
                N = self.shape_func(zeta, eta)
                m += N.T @ N * detjac * self.mass_density * self.thickness
        return m

    def lumped_mass_matrix(self, x):
        """
        Comptute the lumped mass matrix
        :param x: nodal coordinates of an element
        :return: the lumped mass matrix
        """
        return self.mass_density * self.thickness * stat.quad_area(*x) * 0.25 * np.ones(self.nod * self.ndof)

    def assemble_mass_matrix(self, mass_type='lumped'):
        # global mass matrix
        if mass_type == 'lumped':
            M = np.zeros(self.neq)
            for element in self.elements:
                dofs = np.array([[2 * i, 2 * i + 1] for i in element]).ravel()
                M[dofs] += self.lumped_mass_matrix(self.nodes[element])
            return diags(M)
        else:
            # form 1D arrays to store the positions of each element and their values
            nk = self.eltdofs ** 2  # number of elements in the element stiffness
            Ig = np.zeros(nk * self.nelts)
            Jg = np.zeros(nk * self.nelts)
            Mg = np.zeros(nk * self.nelts)
            # loop over elements to fill Ig, Jg, Kf
            for j, element in enumerate(self.elements):
                m = self.consistent_mass_matrix(self.nodes[element], 2).ravel()
                dofs = np.array([[2 * i, 2 * i + 1] for i in element]).ravel()
                Ig[nk * j:nk * j + nk] = np.repeat(dofs, self.eltdofs)
                Jg[nk * j:nk * j + nk] = np.tile(dofs, self.eltdofs)
                Mg[nk * j:nk * j + nk] = m
            return coo_matrix((Mg, (Ig, Jg)), shape=(self.neq, self.neq)).tolil()

    def Newmark(self, tspan, timestep=None, Uo=0, Udo=0, eps=None, gamma=0.55, a=0, b=0, n=2):
        """
        Solve structural dynamic problems using the Newmark method
        :param tspan: time span
        :param Uo: initial displacement
        :param Udo: initial velocity
        :param timestep: time step size
        :param eps: stopping criterion
        :param gamma: chosen parameter for Newmark method.
        :param a: coefficient of mass proportional damping
        :param b: coefficient of stiffness proportional damping
        :param n: the results are recorded every equal interval n
        """

        # check mass density declaration
        assert self.mass_density, "Declare material's mass density first."
        # Calculate Newmark paramters
        assert gamma >= 0.5, "Choose gamma greater than or equal to 0.5"
        beta = 0.25 * (gamma + 0.5) ** 2
        dt = timestep
        t = 0
        # form global stiffness K, global mass M and damping C
        K = self.assemble_stiffness().tocsr()
        M = self.assemble_mass_matrix('consistent').tocsr()
        C = a * M + b * K
        # assemble the initial global load vector
        R = self.assemble_force_vector(*self.update_loads(t))
        # initialize U, Ud, Udd at t=0
        U = np.ones((self.neq, 1)) * Uo
        Ud = np.ones((self.neq, 1)) * Udo
        Udd = spsolve(M, R - K @ U - C @ Ud)[:, np.newaxis]
        # calculate integration constants
        a0 = 1 / (beta * dt ** 2)
        a1 = 1 / (beta * dt)
        a2 = 1 / (2 * beta) - 1
        a3 = gamma / (beta * dt)
        a4 = gamma / beta - 1
        a5 = gamma / (2 * beta) - 1
        # calculate effective stiffness matrix
        Keff = lil_matrix(K + a0 * M)
        K = Keff.copy()
        # impose boundary conditions
        for (dx, dy), j in self.essential_boundary:
            if dx is not None:
                Keff[2 * j, 2 * j] *= 1e20
            if dy is not None:
                Keff[2 * j + 1, 2 * j + 1] *= 1e20
        # factorize Keff
        Keff = splu(Keff.tocsc())
        # solve for T at each timestep
        # t = 0
        it = 0
        with open(r'C:\Users\ANHHUY\Documents\Python\FEM2D\database\stress_dynamic.txt', 'w') as file:
            np.savetxt(file, np.concatenate(([[t]], U.T), axis=1), delimiter='\t')
            while t < tspan:
                # update t
                t += dt
                it += 1
                # update load vector at time t
                R = self.assemble_force_vector(*self.update_loads(t))
                # calculate effective loads at time t
                Reff = R + M @ (a0 * U + a1 * Ud + a2 * Udd) + C @ (a3 * U + a4 * Ud + dt * a5 * Udd)
                # impose the boundary conditions
                for (dx, dy), j in self.essential_boundary:
                    if dx is not None:
                        Reff[2 * j] = (K[2 * j, 2 * j] * dx).toarray().T
                    if dy is not None:
                        Reff[2 * j + 1] = (K[2 * j + 1, 2 * j + 1] * dy).toarray().T
                # solve for the next U
                Uold = U
                Udd_old = Udd
                U = Keff.solve(Reff)
                # Compute Ud, Udd
                Udd = a0 * (U - Uold - dt * Ud) - a2 * Udd
                Ud = a3 * (U - Uold) - a4 * Ud - dt * a5 * Udd_old
                # compute error
                if eps is not None:
                    if max(abs((U - Uold) / Uold)) < eps:
                        break
                # write the current solution to file
                if not (it % n) or t >= tspan:
                    np.savetxt(file, np.concatenate(([[t]], U.T), axis=1), delimiter='\t')

    def normal_modes(self, k=10):
        """
        Compute natural frequencies and the corresponding mode shapes.
        :param k: k first modes (k < N) where N is the number of dofs
        :return: natural frquencies ans mode shapes
        """

        # Form global stiffness matrix K, and global mass matrix M
        K = self.assemble_stiffness()
        M = self.assemble_mass_matrix('consistent')
        # impose bounadry conditions if defined
        if self.essential_boundary:
            for (dx, dy), j in self.essential_boundary:
                if dx is not None:
                    K[2 * j, 2 * j] *= 1e20
                if dy is not None:
                    K[2 * j + 1, 2 * j + 1] *= 1e20
        # Compute eigensolutions
        W, D = eigsh(K.tocsc(), k, M.tocsc(), sigma=1e4)
        with open(r'C:\Users\ANHHUY\Documents\Python\FEM2D\database\normal_modes.txt', 'w') as file:
            np.savetxt(file, np.concatenate((W[:, np.newaxis], D.T), axis=1), delimiter='\t')
        return W, D

    def update_loads(self, t):
        natural_bc, pressure = [], []
        for (fx, fy), dofs in self.natural_boundary:
            if isinstance(fx, np.ndarray):
                fx = PlaneStress.interpolate_loads(fx[:, 0], fx[:, 1], t)
            if isinstance(fy, np.ndarray):
                fy = PlaneStress.interpolate_loads(fy[:, 0], fy[:, 1], t)
            natural_bc.append([(fx, fy), dofs])
        for (px, py), dofs in self.pressure:
            if isinstance(px, np.ndarray):
                px = PlaneStress.interpolate_loads(px[:, 0], px[:, 1], t)
            if isinstance(py, np.ndarray):
                py = PlaneStress.interpolate_loads(py[:, 0], py[:, 1], t)
            pressure.append(([(px, py), dofs]))
        return natural_bc, pressure

    @staticmethod
    def interpolate_loads(t, f, ti):
        if ti in t:
            fi = f[t == ti][0]
        else:
            i = t < ti
            ii = t > ti
            fi = f[i][-1] + ((f[ii][0] - f[i][-1]) / (t[ii][0] - t[i][-1])) * (ti - t[i][-1])
        return fi
