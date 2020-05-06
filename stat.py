"""
This file includes programs to compute some common mesh metrics such as aspect ratio and element quality, and
other utilites including triangle area and quad area.
"""
import numpy as np


def tri_aspect_ratio(coords):
    """
    Compute the aspect ratio of a triangle
    Reference: https://www.sharcnet.ca/Software/Ansys/17.0/en-us/help/ans_thry/thy_et7.html#b16r108jjw
    :param coords: coordinates of three corners of a triangle
    :return: aspect ratio
    """
    coords = np.array(coords)
    am = []

    for i in range(3):
        ai = coords[i-2] - coords[i]
        bi = coords[i-1] - coords[i]
        costh = abs(np.dot(ai, bi) / (np.linalg.norm(ai) * np.linalg.norm(bi)))

        if 0.5 <= costh < 1:
            am.append(0.5 / (1 - costh))
        elif -1 <= costh < 0.5:
            am.append(1.5 / (1 + costh))

    return (np.sqrt(sum(am)/3) - 1)*2 + 1


def quad_aspect_ratio(coords):
    """
        Compute the aspect ratio of a quadrilateral
        Reference: https://www.sharcnet.ca/Software/Ansys/17.0/en-us/help/ans_thry/thy_et7.html#b16r108jjw
        :param coords: coordinates of three corners of a quad
        :return: aspect ratio
        """

    coords = np.array(coords)
    # compute midpoints of each edge of the quad
    midpts = 0.5 * (coords + np.roll(coords, -1, axis=0))

    # distance from a point to a line
    def dist(a, b, x, xo):
        return np.abs(a*(x[0] - xo[0]) + b*(x[1] - xo[1])) / np.sqrt(a**2 + b**2)

    # construct a rectangle with edges passing through the quad midpoints, and two of them are parallel
    # with a line bisecting the opposing pairs of the quad edges.
    asprat = []
    for i in range(2):
        t = midpts[i] - midpts[i+2]
        n = np.array([t[1], -t[0]])
        len1 = dist(*n, midpts[i], midpts[i+1]) + dist(*n, midpts[i], midpts[i-1])
        len2 = np.linalg.norm(midpts[i] - midpts[i+2])
        asprat.append(len1 / len2 if len1 >= len2 else len2 / len1)

    return max(asprat)


def aspect_ratio(coords):
    """
    Compute aspect ratio of quad/tri element
    :param coords: elemental coordiantes
    :return: aspect ratio
    """

    if len(coords) == 3:
        return tri_aspect_ratio(coords)
    elif len(coords) == 4:
        return quad_aspect_ratio(coords)


def tri_area(a, b, c):
    return 0.5 * abs(a[0]*b[1] + b[0]*c[1] + c[0]*a[1] - b[0]*a[1] - c[0]*b[1] - a[0]*c[1])


def quad_area(a, b, c, d):
    return 0.5 * abs(a[0]*b[1] + b[0]*c[1] + c[0]*d[1] + d[0]*a[1] - b[0]*a[1] - c[0]*b[1] - d[0]*c[1] - a[0]*d[1])


def element_quality(coords):
    """
    Compute the quality of each element
    Reference: https://www.sharcnet.ca/Software/Ansys/17.0/en-us/help/ans_thry/thy_et7.html#b16r108jjw
    :param coords: coordinates of coners
    :return: element quality
    """

    if len(coords) == 3:
        return 6.92820323 * (tri_area(*coords) / sum(np.linalg.norm(coords - np.roll(coords, -1, axis=0), axis=1)**2))
    elif len(coords) == 4:
        return 4.0 * (quad_area(*coords) / sum(np.linalg.norm(coords - np.roll(coords, -1, axis=0), axis=1)**2))
