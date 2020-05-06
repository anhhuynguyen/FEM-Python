import matplotlib.pyplot as plt
import matplotlib.collections
from matplotlib.animation import FuncAnimation
import numpy as np
from mesh import MeshBox


class Post2D:

    def __init__(self, sobj, mobj):
        """
        Initialize Post2D class
        :param sobj: HeatTransfer object
        :param mobj: Mesh2D/MeshBox object
        """
        self.solver = sobj
        self.mesh = mobj
        self.nodal_flux = np.array([])
        self.element_flux = np.array([])
        self.stress = np.array([])
        self.strain = np.array([])

    def plot_temp(self, cmap='jet', contour=False, meshon=False, levels=60, label=False, **kwargs):
        """
        Plot the temperature distribution
        :param cmap: colormap (default to jet). Details: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        :param contour: True to show contour lines, False to hide them
        :param meshon: True to show mesh, False to turn it off
        :param levels: number of levels
        :param label: show label when contour lines are on
        :param kwargs: only for transient porblem
                    interval: Delay between frames in milliseconds. Defaults to 1.
                    savefig: True to save animation as a gif file.
                    filename: filename of the gif file
                    showfig: True to show animation
                    fit: fit the figure to the surrounding box
        """

        if self.solver.response == 'steady-state':
            self._plot_nodal_values(self.solver.D, 'Temperature distribution', cmap, contour, meshon, levels, label)
        elif self.solver.response == 'transient':
            self._animation(cmap, contour, meshon, levels, label, **kwargs)

    def plot_flux(self, comp=None, cmap='jet', average=True, contour=False, meshon=False, levels=60, label=False,
                  **kwargs):
        """
        Plot heat flux
        :param comp: 'x' -> x direction, 'y' -> y direction, 'mag' -> magnitude
        :param cmap: colormap (default to jet). Details: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        :param average: set True for averaging element output at nodes
        :param contour: True to show contour lines, False to hide them
        :param meshon: True to show mesh, False to turn it off
        :param levels: number of levels
        :param label: show label when contour lines are on
        :param kwargs: only for transient porblem
                    interval: Delay between frames in milliseconds. Defaults to 1.
                    savefig: True to save animation as a gif file.
                    filename: filename of the gif file
                    showfig: True to show animation
                    fit: fit the figure to the surrounding box
        """

        if self.solver.response == 'steady-state':
            if average:
                self._plot_nodal_flux(comp, cmap, contour, meshon, levels, label)
            else:
                self._plot_elemental_flux(comp, meshon, cmap)
        elif self.solver.response == 'transient':
            if average:
                self._animation_nodal_flux(comp, cmap, contour, meshon, **kwargs)
            else:
                self._animation_elemental_flux(comp, meshon, cmap, **kwargs)

    def plot_disp(self, comp='mag', scale_factor=10, undeformed=(True, True), deformed=(True, True),
                  alpha=(0.8, 0.8), cmap='jet', ani=False, **kwargs):
        """
        Plot displacement
        :param comp: 'x' -> x direction, 'y' -> y direction, 'mag' -> magnitude
        :param scale_factor: magnify the displacement
        :param undeformed: (True/False, True/False) -> (draw mesh, draw contour)
        :param deformed: (True/False, True/False) -> (draw mesh, draw contour)
        :param alpha: The alpha blending value, between 0 (transparent) and 1 (opaque).
        :param cmap: colormap (default to jet). Details: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        :param  ani: True
        :param  kwargs: (optional) support when ani is set to True
                savefig: save animation as gif
                filename: filenam to save
                interval: Delay between frames in milliseconds. Defaults to 100.
                frames: number of frames in animation (int).
        """

        if ani:
            self._disp_animation(comp, scale_factor, undeformed, deformed, alpha, cmap, **kwargs)
        else:
            self._plot_disp(comp, scale_factor, undeformed, deformed, alpha, cmap)

    def plot_stress(self, option='mises', undeformed=(False, False), deformed=(True, True), scale_factor=10,
                    alpha=(0.8, 0.8), cmap='jet', ani=False, meshon=True, **kwargs):
        """
        Plot stress
        :param option: 'mises' -> von-mises stress, 'tresca' -> tresca stress, 'pressure' -> pressure/hydrostatic
                       'x' -> x normal stress, 'y' -> y normal stress, 'xy' -> inplane shear stress,
                       'max principal' -> inplane maximum principal stress,
                       'min principal' -> inplane minimum principal stress
        :param deformed: (True/False, True/False) -> (draw mesh, draw contour) (only for static plot)
        :param undeformed: (True/False, True/False) -> (draw mesh, draw contour) (only for static plot)
        :param scale_factor: magnify the displacement
        :param alpha: The alpha blending value, between 0 (transparent) and 1 (opaque).
        :param cmap: colormap (default to jet). Details: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        :param ani: show animation
        :param meshon: plot mesh (only for animation)
        :param kwargs: (optional) support when ani is set to True
               savefig: save animation as gif
               filename: filenam to save
               interval: Delay between frames in milliseconds. Defaults to 100.
               frames: number of frames in animation (int).
               limit_stress: if limit stress is declared, any element with the value of stress higher than the limit
                             stress will be deleted from the model.
        """

        if ani:
            self._stress_animation(option, meshon, scale_factor, alpha, cmap, **kwargs)
        else:
            self._plot_stress(option, deformed, undeformed, scale_factor, alpha, cmap, **kwargs)

    def _plot_nodal_flux(self, comp=None, cmap='jet', contour=False, meshon=False, levels=60, label=False):

        if self.nodal_flux.size == 0:
            self.nodal_flux = self.solver.get_flux(self.solver.D, 'intp', True)

        if comp == 'x':
            q = self.nodal_flux[:, 0]
            title = 'Heat flux in x direction'
        elif comp == 'y':
            q = self.nodal_flux[:, 1]
            title = 'Heat flux in y direction'
        else:
            q = np.linalg.norm(self.nodal_flux, axis=1)
            title = 'Heat flux magnitude'

        self._plot_nodal_values(q, title, cmap, contour, meshon, levels, label)

    def _plot_nodal_values(self, quantity, title, cmap='jet', contour=False, meshon=False, levels=60, label=False):
        # plot results using contourf
        x, y = self.mesh.xseed, self.mesh.yseed
        if contour:
            levels = 12
        else:
            levels = levels
        z = self._arrange_nodal_data(quantity, x, y)
        fig, ax = plt.subplots()
        nx, ny = z.shape
        x, y = self._arrrange_mesh_points(self.mesh.nodes, x, y, nx, ny)
        cs = ax.contourf(x, y, z, corner_mask=False, cmap=cmap, levels=levels)
        if meshon:
            pc = matplotlib.collections.PolyCollection(self.mesh.nodes[self.mesh.elements],
                                                       facecolors='', edgecolors='black', lw=0.5)
            ax.add_collection(pc)
        if contour:
            c = ax.contour(cs)
            if label:
                plt.clabel(c, inline=True, fontsize=8, colors='k')

        ax.axis('equal')
        fig.tight_layout()
        plt.title(title)
        fig.colorbar(cs, format='%.3f', ticks=np.linspace(z.min(), z.max(), 11))
        plt.show()

    def _plot_elemental_flux(self, comp, meshon=True, cmap='jet'):

        if self.element_flux.size == 0:
            self.element_flux = self.solver.get_flux(self.solver.D, 'centroid')

        if comp == 'x':
            q = self.element_flux[:, 0]
            title = 'Heat flux in x direction'
        elif comp == 'y':
            q = self.element_flux[:, 1]
            title = 'Heat flux in y direction'
        else:
            q = np.linalg.norm(self.element_flux, axis=1)
            title = 'Heat flux magnitude'

        z = self._arrange_elemental_data(q)
        nx, ny = z.shape

        fig, ax = plt.subplots()
        x, y = self._arrrange_mesh_points(self.mesh.nodes, self.mesh.xseed, self.mesh.yseed, nx+1, ny+1)
        if meshon:
            c = ax.pcolor(x, y, z, cmap=cmap, edgecolors='k')
        else:
            c = ax.pcolor(x, y, z, cmap=cmap)

        ax.axis('equal')
        fig.tight_layout()
        plt.title(title)
        fig.colorbar(c, ax=ax, format='%.3e', ticks=np.linspace(z.min(), z.max(), 11))
        plt.show()

    def _animation_elemental_flux(self, comp, meshon, cmap, **kwargs):

        x, y = self.mesh.xseed, self.mesh.yseed
        fig, ax = plt.subplots()

        with open(r'C:\Users\ANHHUY\Documents\Python\FEM2D\database\temp.txt', 'r') as file:
            solutions = np.loadtxt(file)

        #########################################################
        flux = []
        if self.element_flux.size == 0:
            for sol in solutions[:, 1:]:
                flux.append(self.solver.get_flux(sol, 'centroid'))
            self.element_flux = np.array(flux)
        #########################################################
        # set up colorbar
        if comp == 'x':
            zo = self.element_flux[-1, :, 0]
        elif comp == 'y':
            zo = self.element_flux[-1, :, 1]
        else:
            zo = np.linalg.norm(self.element_flux[-1], axis=1)
        # Setup colorbar
        zo = self._arrange_elemental_data(zo)
        zmax, zmin = zo.max(), zo.min()
        nx, ny = zo.shape
        x, y = self._arrrange_mesh_points(self.mesh.nodes, x, y, nx+1, ny+1)
        co = ax.pcolor(x, y, zo, cmap=cmap)
        fig.colorbar(co, ticks=np.round(np.linspace(zmin, zmax, 12), 2), format='%.3e')

        def update(i):
            q = self.element_flux[i]
            t = solutions[i][0]

            if comp == 'x':
                q = q[:, 0]
                title = 'Heat flux in x direction'
            elif comp == 'y':
                q = q[:, 1]
                title = 'Heat flux in y direction'
            else:
                q = np.linalg.norm(q, axis=1)
                title = 'Heat flux magnitude'

            z = self._arrange_elemental_data(q)
            ax.collections = []

            if meshon:
                c = ax.pcolor(x, y, z, cmap=cmap, edgecolors='k', vmin=zmin, vmax=zmax)
            else:
                c = ax.pcolor(x, y, z, cmap=cmap, vmin=zmin, vmax=zmax)

            ax.set_title(f'{title}\nTime = {np.round(t, 6)}\nMin:{z.min():.3f}, Max:{z.max():.3f}')
            ax.axis('equal')

            return c

        animation = FuncAnimation(fig, update, repeat=True, interval=kwargs.get('interval', 100),
                                  frames=len(solutions))

        if kwargs.get('savefig', False):
            filename = kwargs.get('filename', 'ani') + '.gif'
            animation.save(filename, dpi=50)
        if kwargs.get('showfig', True):
            plt.show()

    def _animation_nodal_flux(self, comp, cmap, contour, meshon, levels, label, **kwargs):
        x, y = self.mesh.xseed, self.mesh.yseed
        fig, ax = plt.subplots()

        with open(r'C:\Users\ANHHUY\Documents\Python\FEM2D\database\temp.txt', 'r') as file:
            solutions = np.loadtxt(file)

        ################################################
        flux = []
        if self.nodal_flux.size == 0:
            for sol in solutions[:, 1:]:
                flux.append(self.solver.get_flux(sol, 'intp', True))
            self.nodal_flux = np.array(flux)
        ################################################
        # Set up colorbar
        if comp == 'x':
            zo = self.nodal_flux[-1, :, 0]
        elif comp == 'y':
            zo = self.nodal_flux[-1, :, 1]
        else:
            zo = np.linalg.norm(self.nodal_flux[-1], axis=1)

        zo = self._arrange_nodal_data(zo, x, y)
        zmax, zmin = zo.max(), zo.min()
        nx, ny = zo.shape
        xx, yy = self._arrrange_mesh_points(self.mesh.nodes, x, y, nx, ny)
        co = ax.contourf(xx, yy, zo, corner_mask=False, cmap=cmap, levels=levels)
        fig.colorbar(co, ticks=np.round(np.linspace(zmin, zmax, 12), 2), format='%.3e')

        def update(i):
            q = self.nodal_flux[i]
            time = solutions[i][0]

            if comp == 'x':
                q = q[:, 0]
                title = 'Heat flux in x direction'
            elif comp == 'y':
                q = q[:, 1]
                title = 'Heat flux in y direction'
            else:
                q = np.linalg.norm(q, axis=1)
                title = 'Heat flux magnitude'

            ax.collections = []

            z = self._arrange_nodal_data(q, x, y)
            cs = ax.contourf(xx, yy, z, corner_mask=False, cmap=cmap, levels=levels, vmax=zmax, vmin=zmin)
            if meshon:
                pc = matplotlib.collections.PolyCollection(self.mesh.nodes[self.mesh.elements],
                                                           facecolors='', edgecolors='black', lw=0.5)
                ax.add_collection(pc)
            if contour:
                c = ax.contour(cs)
                if label:
                    plt.clabel(c, inline=True, fontsize=8, colors='k')

            ax.set_title(f'{title}\nTime = {np.round(time, 6)}\nMin:{z.min():.3f}, Max:{z.max():.3f}')
            if kwargs.get('fit', True):
                ax.axis('equal')
            return cs

        animation = FuncAnimation(fig, update, repeat=True, interval=kwargs.get('interval', 100),
                                  frames=len(solutions))
        if kwargs.get('savefig', False):
            filename = kwargs.get('filename', 'ani') + '.gif'
            animation.save(filename, dpi=50)
        if kwargs.get('showfig', True):
            plt.show()

    def _animation(self, cmap, contour, meshon, levels, label, **kwargs):
        # plot results using contourf
        x, y = self.mesh.xseed, self.mesh.yseed
        fig, ax = plt.subplots()

        if contour:
            levels = 12
        else:
            levels = levels

        with open(r'C:\Users\ANHHUY\Documents\Python\FEM2D\database\temp.txt', 'r') as file:
            solutions = np.loadtxt(file)
        # Set up colorbar
        data = solutions[:, 1:]
        zmax, zmin = data.max(), data.min()
        zo = self._arrange_nodal_data(data[-1 if zmax in data[-1, :] else 1, :], x, y)
        co = ax.contourf(x, y, zo, corner_mask=False, cmap=cmap, levels=levels)
        fig.colorbar(co, ticks=np.round(np.linspace(zmin, zmax, 12), 3), format=kwargs.get('format', '%.3f'))

        def update(i):
            solution = solutions[i]
            t, quantity = solution[0], solution[1:]
            z = self._arrange_nodal_data(quantity, x, y)
            ax.collections = []
            nx, ny = z.shape
            xx, yy = self._arrrange_mesh_points(self.mesh.nodes, x, y, nx, ny)
            cs = ax.contourf(xx, yy, z, corner_mask=False, cmap=cmap, levels=levels, vmax=zmax, vmin=zmin)

            if meshon:
                pc = matplotlib.collections.PolyCollection(self.mesh.nodes[self.mesh.elements],
                                                           facecolors='', edgecolors='black', lw=0.5)
                ax.add_collection(pc)

            if contour:
                c = ax.contour(cs)
                if label:
                    plt.clabel(c, inline=True, fontsize=8, colors='k')

            if kwargs.get('fit', True):
                ax.axis('equal')

            ax.set_title(f'Time = {np.round(t, 6)}\nTemperature: Min:{z.min():.3f}, Max:{z.max():.3f}')
            return cs

        animation = FuncAnimation(fig, update, repeat=True, interval=kwargs.get('interval', 100),
                                  frames=len(solutions))

        if kwargs.get('savefig', False):
            filename = kwargs.get('filename', 'ani') + '.gif'
            animation.save(filename, dpi=50)

        if kwargs.get('showfig', True):
            plt.show()

    def _plot_disp(self, comp, scale_factor, undeformed, deformed, alpha, cmap):

        fig = plt.figure('Displacements')
        ax = fig.add_subplot(111)
        cs = []

        du = self.solver.D.reshape((self.mesh.nn, 2))
        u = self.mesh.nodes + du * scale_factor
        x, y = self.mesh.xseed, self.mesh.yseed
        if comp == 'x':
            z = self._arrange_nodal_data(du[:, 0], x, y)
        elif comp == 'y':
            z = self._arrange_nodal_data(du[:, 1], x, y)
        else:
            z = self._arrange_nodal_data(np.linalg.norm(du, axis=1), x, y)

        nx, ny = z.shape

        if undeformed[0]:
            pc = matplotlib.collections.PolyCollection(self.mesh.nodes[self.mesh.elements], facecolors='white',
                                                       alpha=0.5, edgecolors='black')
            ax.add_collection(pc)

        if undeformed[1]:
            xx, yy = self._arrrange_mesh_points(u, x, y, nx, ny)
            cs = ax.contourf(xx, yy, z, corner_mask=False, cmap=cmap, levels=50, alpha=alpha[0])

        if deformed[0]:
            pc = matplotlib.collections.PolyCollection(u[self.mesh.elements], facecolors='white',
                                                       alpha=0.5, edgecolors='black')
            ax.add_collection(pc)

        if deformed[1]:
            ux, uy = self._arrrange_mesh_points(u, x, y, nx, ny)
            cs = ax.contourf(ux, uy, z, corner_mask=False, cmap=cmap, levels=50, alpha=alpha[1])

        if deformed[1] or undeformed[1]:
            fig.colorbar(cs, format='%.3e', ticks=np.linspace(z.min(), z.max(), 11))

        margin = scale_factor * np.abs(du).max()
        ax.set_ylim([-1 - margin, self.mesh.width + 1 + margin])
        ax.set_xlim([-1 - margin, self.mesh.length + 1 + margin])
        fig.tight_layout()
        ax.axis('equal')
        ax.set_title('Displacement')
        plt.show()

    def _plot_stress(self, option, deformed, undeformed, scale_factor, alpha, cmap, **kwargs):

        if self.stress.size == 0:
            self.stress, self.strain = self.solver.get_stress()

        fig = plt.figure('Stress')
        ax = fig.add_subplot(111)
        cs = []

        du = self.solver.D.reshape((self.mesh.nn, 2))
        u = self.mesh.nodes + du * scale_factor
        x, y = self.mesh.xseed, self.mesh.yseed

        sxx = self.stress[:, 0, 0]  # x normal stress
        syy = self.stress[:, 1, 0]  # y normal stress
        sxy = self.stress[:, 2, 0]  # xy shear stress

        z, title = self._compute_stress(option, sxx, syy, sxy, kwargs.get('limit_stress', None))
        zmax, zmin = z.max(), z.min()
        nx, ny = z.shape

        if undeformed[0]:
            pc = matplotlib.collections.PolyCollection(self.mesh.nodes[self.mesh.elements], facecolors='white',
                                                       alpha=0.5, edgecolors='black')
            ax.add_collection(pc)

        if undeformed[1]:
            xx, yy = self._arrrange_mesh_points(self.mesh.nodes, x, y, nx+1, ny+1)
            cs = ax.pcolor(xx, yy, z, cmap=cmap, alpha=alpha[0])

        if deformed[0]:
            pc = matplotlib.collections.PolyCollection(u[self.mesh.elements], facecolors='white',
                                                       alpha=0.5, edgecolors='black')
            ax.add_collection(pc)

        if deformed[1]:
            ux, uy = self._arrrange_mesh_points(u, x, y, nx+1, ny+1)
            cs = ax.pcolor(ux, uy, z, cmap=cmap, alpha=alpha[1])

        if deformed[1] or undeformed[1]:
            fig.colorbar(cs, format='%.3e', ticks=np.linspace(zmin, zmax, 11))

        ax.set_title(title + f'\nMax: {np.round(zmax, 4):.3e}, Min: {np.round(zmin, 4):3e}')
        margin = scale_factor * np.abs(du).max()
        ax.set_ylim([-1 - margin, self.mesh.width + 1 + margin])
        ax.set_xlim([-1 - margin, self.mesh.length + 1 + margin])
        ax.axis('equal')
        plt.tight_layout()
        plt.show()

    def _stress_animation(self, option, meshon, scale_factor, alpha, cmap, **kwargs):

        if self.stress.size == 0:
            self.stress, self.strain = self.solver.get_stress()

        fig = plt.figure('Stress')
        ax = fig.add_subplot(111)

        duo = self.solver.D.reshape((self.mesh.nn, 2))
        uo = self.mesh.nodes + duo * scale_factor
        x, y = self.mesh.xseed, self.mesh.yseed

        sxx = self.stress[:, 0, 0]  # x normal stress
        syy = self.stress[:, 1, 0]  # y normal stress
        sxy = self.stress[:, 2, 0]  # xy shear stress
        limit_stress = kwargs.get('limit_stress', None)

        z, title = self._compute_stress(option, sxx, syy, sxy, None)
        nx, ny = z.shape
        zmax, zmin = z.max(), z.min()
        uxo, uyo = self._arrrange_mesh_points(uo, x, y, nx+1, ny+1)
        cs = ax.pcolor(uxo, uyo, z, cmap=cmap, alpha=alpha[1])
        cb = fig.colorbar(cs, format='%.3e', ticks=np.linspace(zmin, zmax, 11))

        margin = scale_factor * abs(duo).max()
        ax.set_ylim([-1 - margin, self.mesh.width + 1 + margin])
        ax.set_xlim([-1 - margin, self.mesh.length + 1 + margin])
        ax.axis('equal')

        def update(i):
            ax.collections, c = [], []
            du = duo * i
            u = self.mesh.nodes + du * scale_factor
            zi = z * i
            if limit_stress is not None:
                zi = np.ma.masked_where(zi >= limit_stress, zi)
                if zi.mask.all():
                    ax.set_title('All failed !!!')
                    return None
                zimax, zimin = zi[zi < limit_stress].max(), zi[zi < limit_stress].min()
            else:
                zimax, zimin = zi.max(), zi.min()

            if i % 2 == 0:
                if meshon:
                    pc = matplotlib.collections.PolyCollection(self.mesh.nodes[self.mesh.elements], facecolors='white',
                                                               alpha=0.5, edgecolors='black')
                    ax.add_collection(pc)
                xx, yy = self._arrrange_mesh_points(self.mesh.nodes, x, y, nx+1, ny+1)
                c = ax.pcolor(xx, yy, zi*0, cmap=cmap, alpha=alpha[1], vmin=zmin, vmax=zmax)
            else:
                if meshon:
                    pc = matplotlib.collections.PolyCollection(u[self.mesh.elements], facecolors='white',
                                                               alpha=0.5, edgecolors='black')
                    ax.add_collection(pc)

                ux, uy = self._arrrange_mesh_points(u, x, y, nx+1, ny+1)
                c = ax.pcolor(ux, uy, zi, cmap=cmap, alpha=alpha[1], vmin=None if limit_stress else zmin,
                              vmax=None if limit_stress else zmax)

            if limit_stress is not None:
                cb.update_normal(c)

            ax.set_title(title + f'\nFrame: {np.round(i, 3)}\n' +
                         f'Max: {np.round(zimax, 4): .3e}, Min: {np.round(zimin, 4): .3e}')
            plt.autoscale(False)
            plt.tight_layout()

        animation = FuncAnimation(fig, update, repeat=True, interval=kwargs.get('interval', 100),
                                  frames=np.linspace(0, 1, kwargs.get('frames', 2)))

        if kwargs.get('savefig', False):
            filename = kwargs.get('filename', 'ani') + '.gif'
            animation.save(filename, dpi=50)

        plt.show()

    def _disp_animation(self, comp, scale_factor, undeformed, deformed, alpha, cmap, **kwargs):

        fig = plt.figure('Displacements')
        ax = fig.add_subplot(111)

        duo = self.solver.D.reshape((self.mesh.nn, 2))
        uo = self.mesh.nodes + duo * scale_factor
        x, y = self.mesh.xseed, self.mesh.yseed
        title = ''

        if comp == 'x':
            zo = self._arrange_nodal_data(duo[:, 0], x, y)
            title = 'Displacement (x)'
        elif comp == 'y':
            zo = self._arrange_nodal_data(duo[:, 1], x, y)
            title = 'Displacement (y)'
        else:
            zo = self._arrange_nodal_data(np.linalg.norm(duo, axis=1), x, y)
            title = 'Displacement (Magnitude)'

        zmax, zmin = zo.max(), zo.min()
        nx, ny = zo.shape

        if deformed[1]:
            uxo, uyo = self._arrrange_mesh_points(uo, x, y, nx, ny)
            cs = ax.contourf(uxo, uyo, zo, corner_mask=False, cmap=cmap, levels=50, alpha=alpha[1])
            fig.colorbar(cs, format='%.3e', ticks=np.linspace(zmin, zmax, 11))
        else:
            p = matplotlib.collections.PolyCollection(uo[self.mesh.elements], facecolors='white',
                                                      alpha=0.5, edgecolors='black')
            ax.add_collection(p)

        margin = scale_factor * np.abs(duo).max()
        ax.set_ylim([-1 - margin, self.mesh.width + 1 + margin])
        ax.set_xlim([-1 - margin, self.mesh.length + 1 + margin])
        ax.axis('equal')

        def update(i):
            ax.collections = []
            du = duo * i
            u = self.mesh.nodes + du * scale_factor

            if comp == 'x':
                z = self._arrange_nodal_data(du[:, 0], x, y)
            elif comp == 'y':
                z = self._arrange_nodal_data(du[:, 1], x, y)
            else:
                z = self._arrange_nodal_data(np.linalg.norm(du, axis=1), x, y)

            if i % 2 == 0:
                if undeformed[0]:
                    pc = matplotlib.collections.PolyCollection(self.mesh.nodes[self.mesh.elements], facecolors='white',
                                                               alpha=0.5, edgecolors='black')
                    ax.add_collection(pc)
                if undeformed[1]:
                    xx, yy = self._arrrange_mesh_points(self.mesh.nodes, x, y, nx, ny)
                    ax.contourf(xx, yy, z * 0, corner_mask=False, cmap=cmap, levels=50, alpha=alpha[0])
            else:
                if deformed[0]:
                    pc = matplotlib.collections.PolyCollection(u[self.mesh.elements], facecolors='white',
                                                               alpha=0.5, edgecolors='black')
                    ax.add_collection(pc)

                if deformed[1]:
                    ux, uy = self._arrrange_mesh_points(u, x, y, nx, ny)
                    ax.contourf(ux, uy, z, corner_mask=False, cmap=cmap, levels=50,
                                alpha=alpha[1], vmin=zmin, vmax=zmax)

            plt.autoscale(False)
            ax.set_title(title + f"\nFrame: {np.round(i, 4)} " +
                         f'Max: {np.round(du.max(), 4): .3e}, Min: {np.round(du.min(), 4): .3e}')
            plt.tight_layout()

        animation = FuncAnimation(fig, update, repeat=True, interval=kwargs.get('interval', 100),
                                  frames=np.linspace(0, 1, kwargs.get('frames', 2)))

        if kwargs.get('savefig', False):
            filename = kwargs.get('filename', 'ani') + '.gif'
            animation.save(filename, dpi=50)

        plt.show()

    def _compute_stress(self, option, sxx, syy, sxy, limit_stress):

        if option == 'x':
            z = self._arrange_elemental_data(sxx)
            title = 'Normal stress x'
        elif option == 'y':
            z = self._arrange_elemental_data(syy)
            title = 'Normal stress y'
        elif option == 'xy':
            z = self._arrange_elemental_data(sxy)
            title = 'In-plane shear stress'
        elif option == 'max principal':
            sp1, sp2 = Post2D.principal_stress(sxx, syy, sxy)
            z = self._arrange_elemental_data(sp1)
            title = 'Max. principal stress'
        elif option == 'min principal':
            sp1, sp2 = Post2D.principal_stress(sxx, syy, sxy)
            z = self._arrange_elemental_data(sp2)
            title = 'Min. principal stress'
        elif option == 'tresca':
            tresca = Post2D.tresca_stress(sxx, syy, sxy)
            z = self._arrange_elemental_data(tresca)
            title = 'Tresca stress'
            if limit_stress is not None:
                z = np.ma.masked_where(z >= limit_stress, z)
        elif option == 'pressure':
            p = Post2D.pressure(sxx, syy, sxy)
            z = self._arrange_elemental_data(p)
            title = 'Pressure'
        else:
            vm = Post2D.von_mises_stress(sxx, syy, sxy)
            z = self._arrange_elemental_data(vm)
            title = 'Von Mises stress'
            if limit_stress is not None:
                z = np.ma.masked_where(z >= limit_stress, z)

        return z, title

    @staticmethod
    def von_mises_stress(sxx, syy, sxy):
        sp1, sp2 = Post2D.principal_stress(sxx, syy, sxy)
        return np.sqrt(sp1 ** 2 - sp1 * sp2 + sp2 ** 2)  # von mises stress

    @staticmethod
    def principal_stress(sxx, syy, sxy):
        sp1 = (sxx + syy) / 2 + np.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)  # pricipal stress 1
        sp2 = (sxx + syy) / 2 - np.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)  # principal stress 2
        return sp1, sp2

    @staticmethod
    def tresca_stress(sxx, syy, sxy):
        sp1 = (sxx + syy) / 2 + np.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)  # pricipal stress 1
        sp2 = (sxx + syy) / 2 - np.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)  # principal stress 2
        return np.amax(np.c_[abs(sp1 - sp2), sp1, sp2], axis=1)  # tresca stress

    @staticmethod
    def pressure(sxx, syy, sxy):
        sp1, sp2 = Post2D.principal_stress(sxx, syy, sxy)
        return (sp1 + sp2 + 0) / 3

    def _arrrange_mesh_points(self, data, x, y, nx, ny):
        if isinstance(self.mesh, MeshBox):
            if len(y) > len(x):
                xx = data[:, 0].reshape((nx, ny), order='C')
                yy = data[:, 1].reshape((nx, ny), order='C')
            else:
                xx = data[:, 0].reshape((nx, ny), order='F')
                yy = data[:, 1].reshape((nx, ny), order='F')
        else:
            xx = self._arrange_nodal_data(data[:, 0], x, y)
            yy = self._arrange_nodal_data(data[:, 1], x, y)
        return xx, yy

    def _arrange_nodal_data(self, quantity, x, y):
        if isinstance(self.mesh, MeshBox):
            if len(y) > len(x):
                z = quantity.reshape(len(y), len(x))
            else:
                z = quantity.reshape(len(x), len(y)).T
        else:
            zval = np.zeros(len(self.mesh.node_number))
            zval[(self.mesh.node_number == -1)] = -1
            mask = np.zeros_like(zval, dtype=bool)
            mask[(self.mesh.node_number == -1)] = True
            zval[zval == 0] = quantity.ravel()

            if len(y) >= len(x):
                zval = np.array(zval).reshape(len(y), len(x))
            else:
                zval = np.array(zval).reshape(len(x), len(y)).T
                mask = mask.reshape(len(x), len(y)).ravel(order='F')

            z = np.ma.array(zval, mask=mask)
        return z

    def _arrange_elemental_data(self, quantity):

        if isinstance(self.mesh, MeshBox):
            z = quantity.reshape(self.mesh.ndy, self.mesh.ndx)
        else:
            zval = np.zeros(len(self.mesh.raw_elements))
            zval[(self.mesh.raw_elements == -1).any(axis=1)] = -1
            zval[zval == 0] = quantity.ravel()
            zval = np.array(zval).reshape(len(self.mesh.yseed) - 1, len(self.mesh.xseed) - 1)
            z = np.ma.masked_where(zval == -1, zval)
        return z

    def disp_animation_2(self, comp='mag', scale_factor=100, undeformed=(0, 0), deformed=(1, 1), alpha=(0.5, 0.9),
                         cmap='jet', **kwargs):

        fig = plt.figure('Displacements')
        ax = fig.add_subplot(111)

        with open(r'C:\Users\ANHHUY\Documents\Python\FEM2D\database\stress_dynamic.txt', 'r') as file:
            sol = np.loadtxt(file)

        d = sol[:, 1:]
        duo = d[-1, :].reshape((self.mesh.nn, 2))
        uo = self.mesh.nodes + duo * scale_factor
        x, y = self.mesh.xseed, self.mesh.yseed
        title = ''

        if comp == 'x':
            zo = self._arrange_nodal_data(duo[:, 0], x, y)
            title = 'Displacement (x)'
        elif comp == 'y':
            zo = self._arrange_nodal_data(duo[:, 1], x, y)
            title = 'Displacement (y)'
        else:
            zo = self._arrange_nodal_data(np.linalg.norm(duo, axis=1), x, y)
            title = 'Displacement (Magnitude)'

        zmax, zmin = zo.max(), zo.min()
        nx, ny = zo.shape

        if deformed[1]:
            uxo, uyo = self._arrrange_mesh_points(uo, x, y, nx, ny)
            cs = ax.contourf(uxo, uyo, zo, corner_mask=False, cmap=cmap, levels=50, alpha=alpha[1])
            fig.colorbar(cs, format='%.3e', ticks=np.linspace(zmin, zmax, 11))
        else:
            p = matplotlib.collections.PolyCollection(uo[self.mesh.elements], facecolors='white',
                                                      alpha=0.5, edgecolors='black')
            ax.add_collection(p)

        ax.axis('equal')

        def update(i):
            ax.collections = []
            t = sol[i, 0]
            du = d[i, :].reshape((self.mesh.nn, 2))
            u = self.mesh.nodes + du * scale_factor

            if comp == 'x':
                z = self._arrange_nodal_data(du[:, 0], x, y)
            elif comp == 'y':
                z = self._arrange_nodal_data(du[:, 1], x, y)
            else:
                z = self._arrange_nodal_data(np.linalg.norm(du, axis=1), x, y)

            if undeformed[0]:
                pc = matplotlib.collections.PolyCollection(self.mesh.nodes[self.mesh.elements], facecolors='white',
                                                           alpha=0.5, edgecolors='black')
                ax.add_collection(pc)

            if deformed[0]:
                pc = matplotlib.collections.PolyCollection(u[self.mesh.elements], facecolors='white',
                                                           alpha=0.5, edgecolors='black')
                ax.add_collection(pc)

            if deformed[1]:
                ux, uy = self._arrrange_mesh_points(u, x, y, nx, ny)
                ax.contourf(ux, uy, z, corner_mask=False, cmap=cmap, levels=50,
                            alpha=alpha[1])

            plt.autoscale(False)
            ax.set_title(title + f"\nFrame: {np.round(t, 4)} " +
                         f'Max: {z.max(): .4e}, Min: {z.min(): .4e}')
            plt.tight_layout()

        animation = FuncAnimation(fig, update, repeat=True, interval=kwargs.get('interval', 100), frames=len(sol))

        if kwargs.get('savefig', False):
            filename = kwargs.get('filename', 'ani') + '.gif'
            animation.save(filename, dpi=100)

        plt.show()

    def normal_modes(self, comp='mag', scale_factor=100, undeformed=(0, 0), deformed=(1, 1), alpha=(0.5, 0.9),
                     cmap='jet', **kwargs):
        with open(r'C:\Users\ANHHUY\Documents\Python\FEM2D\database\normal_modes.txt', 'r') as file:
            sol = np.loadtxt(file)

        w = sol[:, 0]
        d = sol[:, 1:]
        x, y = self.mesh.xseed, self.mesh.yseed

        fig = plt.figure()
        ax = fig.add_subplot(111)

        def update(i):
            ax.collections = []
            du = d[i, :].reshape((self.mesh.nn, 2))
            u = self.mesh.nodes + du * scale_factor

            if comp == 'x':
                z = self._arrange_nodal_data(du[:, 0], x, y)
            elif comp == 'y':
                z = self._arrange_nodal_data(du[:, 1], x, y)
            else:
                z = self._arrange_nodal_data(np.linalg.norm(du, axis=1), x, y)

            nx, ny = z.shape

            if undeformed[0]:
                pc = matplotlib.collections.PolyCollection(self.mesh.nodes[self.mesh.elements], facecolors='white',
                                                           alpha=0.5, edgecolors='black')
                ax.add_collection(pc)

            if deformed[0]:
                pc = matplotlib.collections.PolyCollection(u[self.mesh.elements], facecolors='white',
                                                           alpha=0.5, edgecolors='black')
                ax.add_collection(pc)

            if deformed[1]:
                ux, uy = self._arrrange_mesh_points(u, x, y, nx, ny)
                ax.contourf(ux, uy, z, corner_mask=False, cmap=cmap, levels=50,
                            alpha=alpha[1])

            ax.set_title(f"Normal Modes\nMode: {i+1} Freq: {np.sqrt(w[i]) / (2*np.pi) if w[i]> 0 else 0: .4e}")
            ax.axis('equal')
            plt.tight_layout()

        animation = FuncAnimation(fig, update, repeat=True, interval=kwargs.get('interval', 1000), frames=len(sol))

        if kwargs.get('savefig', False):
            filename = kwargs.get('filename', 'ani') + '.gif'
            animation.save(filename, dpi=100)

        plt.show()
