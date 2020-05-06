import matplotlib.pyplot as plt
import matplotlib.collections
from matplotlib.widgets import RectangleSelector
import numpy as np
import stat


class Mesh2D:
    def __init__(self, length=1, width=1, ndx=(5, 5, 5), ndy=(5, 5, 5), constraints=(0, 0, 0, 0)):
        """
        Initialize rectangle's size
        :param length: length of the rectangle (x)
        :param width: width of the rectangle (y)
        :param ndx: number of elements along x direction (optional)
        :param ndy: number of elements along y direction (optional)

        Example:
        For a 2x2 rectangle plate
        m = Mesh(2, 2), [50, 50)]
        """

        self.length = length
        self.width = width
        self.ndx = ndx
        self.ndy = ndy
        self.constraints = constraints
        self.eltype = 'None'
        self.nod = None  # number of nodes per elements
        self.nelts = None  # number of elements in the mesh
        self.nn = None  # number of nodes in the mesh

        self._xseed, self._yseed = np.array([]), np.array([])
        self._elements, self._nodes = np.array([]), np.array([])
        self._node_number, self._z = np.array([]), np.array([])
        self._selected_nodes, self._selected_elements, self._selected_lines = set(), set(), set()

        self.__elements = np.array([])

    def __repr__(self):
        """ Return repr(self) """

        return f'Length: {self.length}, Width: {self.width}\n' + \
            f'Nodes: {self.nn}\n' + \
            f'Elements: {self.nelts}\n' + \
            f'Element type: {self.eltype}'

    @property
    def ndx(self):
        return self._ndx

    @ndx.setter
    def ndx(self, ndx):
        if type(ndx) is int:
            self._ndx = [ndx, ndx, ndx]
        elif type(ndx) in [tuple, list]:
            self._ndx = list(ndx)

    @property
    def ndy(self):
        return self._ndy

    @ndy.setter
    def ndy(self, ndy):
        if type(ndy) is int:
            self._ndy = [ndy, ndy, ndy]
        elif type(ndy) in [tuple, list]:
            self._ndy = list(ndy)

    @property
    def nodes(self):
        return self._nodes

    @property
    def elements(self):
        return self._elements

    @property
    def selected_nodes(self):
        return np.array(list(self._selected_nodes))

    @property
    def selected_elements(self):
        return np.array(list(self._selected_elements))

    @property
    def selected_lines(self):
        selected_lines = np.array(list(self._selected_lines))
        # convert line number to element number
        el = np.fromiter(map(self.identify_element, selected_lines), dtype=int, count=len(selected_lines))
        # convert line number to local edge number
        f = 4 if self.eltype == 'quad4' else 3
        edge = np.fromiter(map(lambda n: n % f, selected_lines), dtype=int, count=len(selected_lines))
        return np.c_[el, edge]

    @property
    def xseed(self):
        return self._xseed

    @property
    def yseed(self):
        return self._yseed

    @property
    def node_number(self):
        return self._node_number

    @property
    def raw_elements(self):
        return self.__elements

    @property
    def length(self):
        return self.__length

    @length.setter
    def length(self, length):
        if length <= 0:
            self.__length = 1
        else:
            self.__length = length

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, width):
        if width <= 0:
            self.__width = 1
        else:
            self.__width = width

    def seed(self, *args):
        """
        Create seeds along each edge
        :param args: number divisions along x and y directions
        :return: seeds in x and y directions
        Example:
        ndx = [5, 5, 5]
        ndy = [5, 5, 5]
        seed(*ndx, *ndy)
        Each edge is split into 3 segments, so the first number in ndx/ndy denotes the number of divisions in the first
        segments, and it is similar for the other numbers.
        Note: biasing is not supported in this method.
        """
        if not args or len(args) == 3:
            args = list(self.ndx + self.ndy)
        else:
            args = list(args)
        # x direction
        if self.constraints[0] == 0:
            args[0] = 0
        if self.constraints[1] == self.length:
            args[2] = 0
        self._xseed = np.r_[
            np.linspace(0, self.constraints[0], args[0]+1),
            np.linspace(self.constraints[0], self.constraints[1], args[1]+1),
            np.linspace(self.constraints[1], self.length, args[2]+1)
        ]

        # y direction
        if self.constraints[2] == 0:
            args[3] = 0
        if self.constraints[3] == self.width:
            args[5] = 0
        self._yseed = np.r_[
            np.linspace(0, self.constraints[2], args[3]+1),
            np.linspace(self.constraints[2], self.constraints[3], args[4]+1),
            np.linspace(self.constraints[3], self.width, args[5]+1)
        ]

        self._xseed = np.unique(self._xseed)
        self._yseed = np.unique(self._yseed)

    def mesh(self, eltype='quad', hide=True):
        """
        Mshing a rectanglar plate with prescribed seeds
        :param hide: do not plot mesh
        :param eltype: element type (quad/tri)
        :return: None
        """

        assert self._xseed.any() or self._yseed.any(), "Seed the model before meshing, use method seed()."

        self._create_nodes()

        if eltype == 'quad':
            self.eltype = 'quad4'
            self._quad()
            self.nod = 4
        else:
            self.eltype = 'tri3'
            self._tri()
            self.nod = 3

        self.nelts = len(self.elements)
        self.nn = len(self.nodes)

        if not hide:
            self.showmesh()

    def showmesh(self, figout=False, label=False):
        """
        Show mesh
        :param label: show node numbering
        :param figout: True to return the Figure and Axes objects
        :return the Figure and Axes objects
        """

        assert self._nodes.any() or self._elements.any(), \
            "Nodes and elements are not initialized, use method mesh() to create them."

        fig = plt.figure('Mesh', figsize=(8, 7))
        ax = fig.add_subplot(111)
        pc = matplotlib.collections.PolyCollection(self._nodes[self._elements], facecolors='', edgecolors='black')
        ax.add_collection(pc)
        ax.set_title(f'#nodes: {len(self._nodes)}, #elements: {len(self._elements)}')
        ax.axis('equal')
        if label:
            for i, x, y in zip(range(len(self._nodes)), self._nodes[:, 0], self._nodes[:, 1]):
                ax.annotate(str(int(i)), (x, y), size=7)
        if figout:
            return fig, ax
        else:
            plt.show()

    def showmeshstat(self, mesh_metric='AR'):
        """
        Plot mesh statistics
        :param: mesh_metric: 'AR' (aspect ratio), 'EQ" (element quality)
        Element quality : A value of 1 indicates a perfect cube or square while a value of 0 indicates that the element
        has a zero or negative volume.
        Aspect ratio: For triangles, equilateral triangles have an aspect ratio of 1 and the aspect ratio increases the
        more the triangle is stretched from an equilateral shape. For quadrilaterals, squares have an aspect ratio of 1
        and the aspect ratio increases the more the square is stretched to an oblong shape.
        :return: None
        """

        assert self._nodes.any() or self._elements.any(), \
            "Nodes and elements are not initialized, use method mesh()."

        title = ''
        if mesh_metric == 'AR':
            title = 'Aspect ratio\n'
        elif mesh_metric == 'EQ':
            title = 'Element quality\n'

        fig = plt.figure('Mesh metrics', figsize=(8, 7))
        ax = fig.add_subplot(111)

        if self.eltype == 'quad4':

            z = self._meshstat(mesh_metric)
            if isinstance(self, MeshBox):
                zm = z.reshape(self.ndy, self.ndx)
                nx, ny = zm.shape
                if ny > nx:
                    x = self.nodes[:, 0].reshape((nx + 1, ny + 1), order='C')
                    y = self.nodes[:, 1].reshape((nx + 1, ny + 1), order='C')
                else:
                    x = self.nodes[:, 0].reshape((nx + 1, ny + 1), order='F')
                    y = self.nodes[:, 1].reshape((nx + 1, ny + 1), order='F')
            else:
                zval = np.zeros(len(self.__elements))
                zval[(self.__elements == -1).any(axis=1)] = -1
                zval[zval == 0] = z
                zval = np.array(zval).reshape(len(self._yseed) - 1, len(self._xseed) - 1)
                zm = np.ma.masked_where(zval == -1, zval)
                x, y = self.xseed, self.yseed

            zmin, zmax, zavg = z.min(), z.max(), z.mean()
            plt.title(title + f'Min: {zmin:.2f}, Max: {zmax:.2f}, Avg: {zavg:.2f}')
            c = ax.pcolor(x, y, zm,
                          vmin=zmin - 0.1, vmax=zmax + 0.1, cmap='rainbow', edgecolors='k')
            fig.colorbar(c, ax=ax, format='%.2f')
        elif self.eltype == 'tri3':
            z = self._meshstat(mesh_metric)
            zmin, zmax, zavg = z.min(), z.max(), z.mean()
            plt.title(title + f'Min: {zmin:.2f}, Max: {zmax:.2f}, Avg: {zavg:.2f}')
            c = ax.tripcolor(self._nodes[:, 0], self._nodes[:, 1], self._elements,
                             facecolors=z, edgecolors='k', cmap='rainbow')
            fig.colorbar(c, ax=ax, format='%.2f')

        ax.axis('equal')
        plt.show()

    def showseed(self):
        """ Show seeds """

        assert self._xseed.any() or self._yseed.any(), "Seed the model first using method seed()."

        fig = plt.figure('Seeds', figsize=(8, 7))
        ax = fig.add_subplot(111)
        plt.plot(self._xseed, np.zeros(len(self._xseed)), 'k.', markersize=5)
        plt.plot(np.zeros(len(self._yseed)), self._yseed, 'k.', markersize=5)
        ax.set_title(f'x seeds: {len(self._xseed)} y seeds: {len(self._yseed)}')
        ax.axis('equal')
        ax.grid()
        plt.show()

    def get_lines(self):
        """
        Get lines from the mesh to impose pressure or flux.
        """
        # TODO: improve the performance of this method, the midnodes matrix holds a lot of np.nan -> instant solution
        nan = -2e10
        # extract the coordinates of the nodes
        n = self.nodes[self.elements]
        line_numbers = [0, 1, 2, 3] if self.eltype == 'quad4' else [0, 1, 2]
        # compute the midpoint of each edge of an element
        mx = (n[:, :, 0] + np.roll(n[:, :, 0], -1, axis=1)) * 0.5
        my = (n[:, :, 1] + np.roll(n[:, :, 1], -1, axis=1)) * 0.5
        midnodes = np.c_[mx.ravel(), my.ravel(), np.tile(line_numbers, self.nelts)]
        # remove duplicated midside nodes and assign those duplicates as np.nan
        arr, unique_ind = np.unique(midnodes[:, 0:2], return_index=True, axis=0)
        repeated_ind = np.setdiff1d(np.arange(len(midnodes)), unique_ind)
        midnodes[repeated_ind, 0:2] = np.array([nan, nan])

        self._box_selector(self._selected_lines, midnodes, 'lines')

    def get_nodes(self):
        """
        Get node numberings using box selector
        To obtain the selected nodes, use the attribute selected_nodes, e.g, m.selected_nodes
        """
        self._box_selector(self._selected_nodes, self._nodes, 'nodes')

    def get_elements(self):
        """
        Get element numberings using box selector
        To obtain the selected elements, use the attribute selected_elements, e.g, m.selected_elements
        """
        centroids = []
        if self.eltype == 'tri3':
            centroids = self.nodes[self.elements].sum(axis=1) * 1 / 3
        elif self.eltype == 'quad4':
            centroids = self.nodes[self.elements].sum(axis=1) * 1 / 4
        self._box_selector(self._selected_elements, centroids, 'elements')

    def _box_selector(self, selected_items, items, item_type):
        """
        Create a box selector in matplotlib
        :param selected_items: object to store picked items
        :param items: an array of point coordinates dependent on item types
        :param item_type: item type
        """

        fig, ax = self.showmesh(figout=True)

        selected_items.clear()
        ecs_not_pressed = [True]

        def onselect(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata

            # filter the unwanted points
            p = np.flatnonzero((items[:, 0] > x1) & (items[:, 0] < x2) &
                               (items[:, 1] > y1) & (items[:, 1] < y2))

            if ecs_not_pressed[0]:
                if eclick.button == 3:  # 3 means right mouse button to remove items
                    # remove already selected items
                    flag = False
                    p = list(selected_items.intersection(set(p)))
                    old_len = len(selected_items)
                    selected_items.difference_update(set([node for node in p if node in selected_items]))
                    if len(selected_items) != old_len:
                        flag = True
                    pp = np.array(list(selected_items))
                    # update plot
                    if flag:
                        if item_type == 'nodes':
                            ax.clear()
                            pc = matplotlib.collections.PolyCollection(self._nodes[self._elements],
                                                                       facecolors='', edgecolors='black')
                            ax.add_collection(pc)
                            if len(pp) != 0:
                                ax.plot(items[pp, 0], items[pp, 1], 'ro', markersize=3)
                        elif item_type == 'elements':
                            ax.clear()
                            pc1 = matplotlib.collections.PolyCollection(self._nodes[self._elements],
                                                                        facecolors='', edgecolors='black')
                            ax.add_collection(pc1)
                            if len(pp) != 0:
                                pc2 = matplotlib.collections.PolyCollection(self._nodes[self._elements[pp]],
                                                                            facecolors='r', edgecolors='black',
                                                                            alpha=0.5)
                                ax.add_collection(pc2)
                        elif item_type == 'lines':
                            ax.clear()
                            pc1 = matplotlib.collections.PolyCollection(self._nodes[self._elements],
                                                                        facecolors='', edgecolors='black')
                            ax.add_collection(pc1)
                            if len(pp) != 0:
                                els = np.array([self.identify_element(n) for n in pp], dtype=int)
                                ie = items[pp, 2].astype(int)
                                for i, e in zip(ie, self.elements[els]):
                                    line = np.transpose(np.c_[self.nodes[e][i], np.roll(self.nodes[e], -1, axis=0)[i]])
                                    ax.plot(line[:, 0], line[:, 1], c='r')

                elif eclick.button == 1:  # 1 means left mouse button to add items
                    selected_items.update(p)
                    pp = np.array(list(selected_items))
                    pc1 = matplotlib.collections.PolyCollection(self._nodes[self._elements],
                                                                facecolors='', edgecolors='black')
                    ax.add_collection(pc1)
                    if item_type == 'nodes':
                        ax.clear()
                        pp = np.array(list(selected_items))
                        pc = matplotlib.collections.PolyCollection(self._nodes[self._elements],
                                                                   facecolors='', edgecolors='black')
                        ax.add_collection(pc)
                        if len(pp) != 0:
                            ax.plot(items[pp, 0], items[pp, 1], 'ro', markersize=3)
                    elif item_type == 'elements':
                        ax.clear()
                        pc1 = matplotlib.collections.PolyCollection(self._nodes[self._elements],
                                                                    facecolors='', edgecolors='black')
                        ax.add_collection(pc1)
                        if len(pp) != 0:
                            pc2 = matplotlib.collections.PolyCollection(self._nodes[self._elements[pp]],
                                                                        facecolors='r', edgecolors='black', alpha=0.5)
                            ax.add_collection(pc2)
                    elif item_type == 'lines':
                        ax.clear()
                        pc1 = matplotlib.collections.PolyCollection(self._nodes[self._elements],
                                                                    facecolors='', edgecolors='black')
                        ax.add_collection(pc1)
                        if len(pp) != 0:
                            els = np.array([self.identify_element(n) for n in pp], dtype=int)
                            ie = items[pp, 2].astype(int)  # extract local edge numbers e.g, 1, 2, 3, 4
                            for i, e in zip(ie, self.elements[els]):
                                # select nodes based on edge numbers
                                line = np.transpose(np.c_[self.nodes[e][i], np.roll(self.nodes[e], -1, axis=0)[i]])
                                ax.plot(line[:, 0], line[:, 1], c='r')

                plt.title('Right mouse button: Deselect, Left mouse button: Select')
                ax.axis('equal')
                plt.draw()
            ecs_not_pressed[0] = True

        def toggle_selector(event):
            if event.key == 'escape':
                ecs_not_pressed[0] = False

        rectprops = dict(facecolor='white', edgecolor='black', alpha=0.4, fill=True)
        toggle_selector.RS = RectangleSelector(ax, onselect, drawtype='box', useblit=True, rectprops=rectprops)
        fig.canvas.mpl_connect('key_press_event', toggle_selector)
        plt.title('Right mouse button: Deselect, Left mouse button: Select')
        ax.axis('equal')
        plt.show()

    def _meshstat(self, mesh_metric='AR'):
        """
        Compute mesh metrics (aspect ratio, element quality)
        :param: mesh_metric: 'AR' (aspect ratio), 'EQ" (element quality)
        :return: a numpy array of elemental metrics
        """

        if mesh_metric == 'AR':
            return np.fromiter(map(stat.aspect_ratio, self._nodes[self._elements]), float, count=self.nelts)
        elif mesh_metric == 'EQ':
            return np.fromiter(map(stat.element_quality, self._nodes[self._elements]), float, count=self.nelts)

    def _create_nodes(self):

        self._nodes = []
        self._node_number = []
        a, b, c, d = self.constraints
        swap, ii = False, 0

        nnx = len(self._xseed)
        nny = len(self._yseed)
        x, y = np.meshgrid(self._xseed, self._yseed)

        if nnx > nny:
            nnx, nny = nny, nnx
            swap = True

        for row in range(nny):
            for col in range(nnx):
                if swap:
                    i, j = col, row
                else:
                    i, j = row, col
                flag = True
                if not ((a < x[i, j] < b) and (c < y[i, j] < d)):
                    self._nodes.append([x[i, j], y[i, j]])

                    if y[i, j] == d and (a < x[i, j] < b) and d == self.width:  # U
                        flag = False
                    if x[i, j] == b and (c < y[i, j] < d) and b == self.length:  # C
                        flag = False
                    if x[i, j] == a and (c < y[i, j] < d) and a == 0:  # flipped C
                        flag = False
                    if y[i, j] == c and (a < x[i, j] < b) and c == 0:  # upside-down U
                        flag = False

                    if x[i, j] == 0 and y[i, j] == 0 and c == 0 and a == 0:  # flipped upside down L
                        if any(self.constraints):  # in case of a rectangle, keep the point at the origin
                            flag = False
                    elif x[i, j] == self.length and y[i, j] == self.width and d == self.width and b == self.length:
                        flag = False  # L
                    elif x[i, j] == self.length and y[i, j] == self.width and b == self.width and c == 0:
                        flag = False  # upside-down L
                    elif x[i, j] == 0 and y[i, j] == self.width and a == 0 and d == self.width:  # flipped L
                        flag = False

                    if flag:
                        self._node_number.append(ii)
                        ii += 1
                    else:
                        # remove boundary nodes
                        self._nodes.pop()
                        self._node_number.append(-1)
                else:
                    # remove interior nodes
                    self._node_number.append(-1)

        self._nodes = np.array(self._nodes)
        self._node_number = np.array(self._node_number)

    def _quad(self):

        nnx, nny = len(self._xseed), len(self._yseed)
        if nnx > nny:
            node_number = self._node_number.reshape(nnx, nny).T
        else:
            node_number = self._node_number.reshape(nny, nnx)

        self._elements = np.array([
            [
                node_number[row, col],
                node_number[row, col + 1],
                node_number[row + 1, col + 1],
                node_number[row + 1, col]
            ]
            for row in range(nny-1) for col in range(nnx-1)
        ])

        self._elements = np.array(self._elements)
        self.__elements = self.elements  # contain -1 nodes
        self._elements = self._elements[(self._elements != -1).all(axis=1)]

    def _tri(self):

        self._elements = []
        nnx, nny = len(self._xseed), len(self._yseed)
        if nnx > nny:
            node_number = self._node_number.reshape(nnx, nny).T
        else:
            node_number = self._node_number.reshape(nny, nnx)

        for row in range(nny-1):
            for col in range(nnx-1):
                xm, ym = (self._nodes[node_number[row, col]] + self._nodes[node_number[row + 1, col + 1]]) * 0.5
                if not ((self.constraints[0] < xm < self.constraints[1]) and
                        (self.constraints[2] < ym < self.constraints[3])):
                    self._elements.append(
                        [
                            node_number[row, col],
                            node_number[row + 1, col + 1],
                            node_number[row + 1, col]
                        ]
                    )
                    self._elements.append(
                        [
                            node_number[row, col],
                            node_number[row, col + 1],
                            node_number[row + 1, col + 1]
                        ]
                    )

        self._elements = np.array(self._elements)
        self._elements = self._elements[(self._elements != -1).all(axis=1)]

    def identify_element(self, n):
        if self.eltype == 'quad4':
            return np.ceil(n / 4) - 1 if n % 4 != 0 else n / 4
        elif self.eltype == 'tri3':
            return np.ceil(n / 3) - 1 if n % 3 != 0 else n / 3


class MeshBox(Mesh2D):

    def __init__(self, length=1, width=1, ndx=10, ndy=10, verts=None):
        Mesh2D.__init__(self, length, width, ndx, ndy)
        self.verts = verts
        if self.verts.any():
            self.length = self.verts[:, 0].max(axis=0) - self.verts[:, 0].min(axis=0)
            self.width = self.verts[:, 1].max(axis=0) - self.verts[:, 1].min(axis=0)

    @property
    def verts(self):
        return self._verts

    @verts.setter
    def verts(self, verts):
        self._verts = np.array(verts)

    @property
    def ndx(self):
        return self._ndx

    @ndx.setter
    def ndx(self, ndx):
        if ndx <= 0 or type(ndx) is float:
            self._ndx = 10
        else:
            self._ndx = ndx

    @property
    def ndy(self):
        return self._ndy

    @ndy.setter
    def ndy(self, ndy):
        if ndy <= 0 or type(ndy) is float:
            self._ndy = 10
        else:
            self._ndy = ndy

    def seed(self, biasx=('+', 'single', 1), biasy=('+', 'single', 1), **kwargs):
        """
        Create seeding along an edge by prescribing the number of elements ndx and ndy
        :param biasy: a list of biased seeding parameters in x dirrection
        :param biasx: a list of biased seeding parameters in y direction
         ndx: number of elements along x direction (optional)
         ndy: number of elements along y direction (optional)
        :return: None

        Example:
        For a 2x2 rectangle plate modelled as a 50x50 meshgrid with biased seedings
        m = Mesh(2, 2)
        m.seed(('+'/'-', 'single'/'double', bf), ('+'/'-', 'single'/'double', bf))[, ndx=50, ndy=50)]
        where
        '+': inward bias
        '-': outward bias
        'single': a single bias that changes the mesh density from one end of the edge to the other.
        'double': a double bias that changes the mesh density from the center of the edge to each end of the edge.
        'bf': bias factor (>=1)
        """

        def checkbias(bias):
            if bias[0] == '+':
                bias[2] = 1 / bias[2]
            elif bias[0] == '-':
                bias[2] = bias[2]
            return bias[2]  # return bias factor

        if 'ndx' in kwargs:
            self.ndx = kwargs.get('ndx', 10)
        if 'ndx' in kwargs:
            self.ndy = kwargs.get('ndy', 10)

        if self.verts.any():
            self._xseed = np.linspace(0, 1, self.ndx + 1)
            self._yseed = np.linspace(0, 1, self.ndy + 1)
        else:
            # assign a bias factor for each type
            biasx = list(biasx)
            biasy = list(biasy)
            biasx[2] = checkbias(biasx)
            biasy[2] = checkbias(biasy)

            # create seeds
            self._xseed = MeshBox.__create_seed(self.length, self.ndx, biasx[1:])
            self._yseed = MeshBox.__create_seed(self.width, self.ndy, biasy[1:])

    def _quad(self):
        """
        Create quad mesh
        :return: an array of tuples, each of which contains node numbers for a quad element
        """

        nnx, nny = len(self._xseed), len(self._yseed)
        if nnx >= nny:
            node_number = np.arange(0, nnx * nny).reshape(nnx, nny).T
        else:
            node_number = np.arange(0, nnx * nny).reshape(nny, nnx)

        self._elements = np.array([
            [
                node_number[row, col],
                node_number[row, col + 1],
                node_number[row + 1, col + 1],
                node_number[row + 1, col]
            ]
            for row in range(self.ndy) for col in range(self.ndx)
        ])

    def _tri(self):
        """
        Create tri mesh
        :return: an array of nodal coordinates
        """

        self._elements = []
        nnx, nny = len(self._xseed), len(self._yseed)
        if nnx >= nny:
            node_number = np.arange(0, nnx * nny).reshape(nnx, nny).T
        else:
            node_number = np.arange(0, nnx * nny).reshape(nny, nnx)

        for row in range(self.ndy):
            for col in range(self.ndx):
                self._elements.append(
                    [
                        node_number[row, col],
                        node_number[row + 1, col + 1],
                        node_number[row + 1, col]
                    ]
                )
                self._elements.append(
                    [
                        node_number[row, col],
                        node_number[row, col + 1],
                        node_number[row + 1, col + 1]
                    ]
                )

        self._elements = np.array(self._elements)

    def _create_nodes(self):
        """
        Create an array of nodal coordinates
        """

        if self.verts.any():
            xn, yn = self.verts[:, 0], self.verts[:, 1]
            t1, t2 = self._xseed, self._yseed
            # boundary nodes on the bottom edge
            x = (xn[1] - xn[0]) * t1 + xn[0]
            y = (yn[1] - yn[0]) * t1 + yn[0]
            # boundary nodes on the right edge
            x_2 = ((xn[2] - xn[1]) * t2 + xn[1])[1:-1]
            y_2 = ((yn[2] - yn[1]) * t2 + yn[1])[1:-1]
            # boundary nodes on the left edge
            x_4 = ((xn[3] - xn[0]) * t2 + xn[0])[1:-1]
            y_4 = ((yn[3] - yn[0]) * t2 + yn[0])[1:-1]
            # interior nodes
            for i in range(self.ndy-1):
                x_i = (x_2[i] - x_4[i]) * t1 + x_4[i]
                y_i = (y_2[i] - y_4[i]) * t1 + y_4[i]
                # add nodes
                x = np.r_[x, x_i]
                y = np.r_[y, y_i]
            # boundary nodes on the top edge
            x_3 = (xn[2] - xn[3]) * t1 + xn[3]
            y_3 = (yn[2] - yn[3]) * t1 + yn[3]
            # add top nodes
            x = np.r_[x, x_3]
            y = np.r_[y, y_3]
            # sort nodes according to node number
            if self.ndx > self.ndy:
                x = x.reshape(self.ndy + 1, self.ndx + 1).T.ravel()
                y = y.reshape(self.ndy + 1, self.ndx + 1).T.ravel()
            self._nodes = np.c_[x, y]
        else:
            x, y = np.meshgrid(self._xseed, self._yseed)
            nnx, nny = len(self._xseed), len(self._yseed)
            if nnx >= nny:
                node_number = np.arange(0, nnx * nny, dtype=int).reshape(nnx, nny).ravel(order='F')
            else:
                node_number = np.arange(0, nnx * nny, dtype=int).reshape(nny, nnx).ravel()

            self._nodes = np.c_[x.ravel(), y.ravel()]
            self._nodes[node_number] = self._nodes[:, [0, 1]]  # sort nodal coordinates according to theirs numbering

    @staticmethod
    def __create_seed(edgelen, ndivs, bias):
        """
        Create seeds from prescribed parameters in seed()
        :return: an array of seed coordinates.
        """

        iseven, flag = None, None

        if bias[1] != 1 and ndivs >= 2:
            if bias[0] == 'double':
                edgelen /= 2
                iseven = True if ndivs % 2 == 0 else False
                ndivs = np.ceil(ndivs / 2).astype(int)

            r = bias[1] ** (1 / (ndivs - 1))
            eltlen = [edgelen * (r - 1) / (r ** ndivs - 1)]
            for i in range(1, ndivs):
                eltlen.append(eltlen[i - 1] * r)

            # increase each element's length for the midpoint of
            # a coincident with the midpoint of the last element in eltlen
            if bias[0] == 'double' and not iseven:
                eltlen = np.array(eltlen) + eltlen[-1] / (2 * ndivs)

            seeds = np.append([0], [sum(eltlen[:i + 1]) for i in range(ndivs)])

            if bias[0] == 'double':
                if iseven:
                    seeds = np.concatenate((seeds[:-1], 2 * edgelen - seeds[::-1]))
                else:
                    # discard the last element
                    seeds = seeds[:-1]
                    seeds = np.concatenate((seeds, 2 * edgelen - seeds[::-1]))
        else:
            seeds = np.linspace(0, edgelen, ndivs + 1)

        return seeds


if __name__ == '__main__':
    m = MeshBox()
    m.seed(ndx=2, ndy=2)
    m.mesh('tri')
    m.get_lines()
    print(m.selected_lines)
