import os
import math
import numpy as np
import pandas as pd


def concentration(db, col1, col2):
    """Computes the composition of binary material

    Parameters
    ----------
    db : DataFrame
        The DataFrame object containing materials of interest
    col1, col2 : str
        Names of columns for elements 1 and 2

    Returns
    -------
    tuple
        a compositions of elements 1 and 2
    """

    x1 = db[col1] / (db[col1] + db[col2])
    x2 = db[col2] / (db[col1] + db[col2])
    return (x1, x2)


def arithmetic_mean(db, col1, col2, weighted=False):
    """Computes arithmetic average of elemental features

    Parameters
    ----------
    db : DataFrame
        The DataFrame object containing materials of interest
    col1, col2 : str
        Names of columns for elemental features of elements 1 and 2
    weighted : bool, optional
        A flag used to compute weighted statistics (default is False)

    Returns
    -------
    tuple
        a compositions of elements 1 and 2
    """

    x1, x2 = concentration(db, col1, col2)
    if weighted:
        return x1 * db[col1] + x2 * db[col2]
    else:
        return 0.5 * db[col1] + 0.5 * db[col2]


def standard_deviation_mean(db, col1, col2, weighted=False):
    """Computes scaled standard deviation of elemental features"""

    x1, x2 = concentration(db, col1, col2)
    if weighted:
        return abs(x1 * db[col1] - x2 * db[col2])
    else:
        return abs(0.5 * db[col1] - 0.5 * db[col2])


def harmonic_mean(db, col1, col2, weighted=False):
    """Computes harmonic average of elemental features"""
    x1, x2 = concentration(db, col1, col2)
    if weighted:
        return 1 / (x1 / db[col1] + x2 / db[col2])
    else:
        return 1 / (0.5 / db[col1] + 0.5 / db[col2])


def geometric_mean(db, col1, col2, weighted=False):
    """Computes geometric average of elemental features"""
    x1, x2 = concentration(db, col1, col2)
    if weighted:
        return db[col1] ** x1 * db[col2] ** x2
    else:
        return db[col1] ** 0.5 * db[col2] ** 0.5


def quadratic_mean(db, col1, col2, weighted=False):
    """Computes quadratic average of elemental features"""
    x1, x2 = concentration(db, col1, col2)
    if weighted:
        return (x1 * db[col1] ** 2 + x2 * db[col2] ** 2) ** 0.5
    else:
        return (0.5 * db[col1] ** 2 + 0.5 * db[col2] ** 2) ** 0.5
