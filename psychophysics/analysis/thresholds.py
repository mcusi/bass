import numpy as np
from scipy.special import expit as sigmoid


def average_crossing_points_log(x, log_odds):
    y = sigmoid(-log_odds)
    f = np.cumsum(1 - y)
    b = y[::-1].cumsum()[::-1]
    boundary = 0.5*(x[(f >= b).argmax()] + x[(f <= b).argmin()])
    return boundary


def average_crossing_points_log_axis(x, log_odds, axis):
    y = sigmoid(-log_odds)
    f = np.cumsum(1 - y, axis=axis)
    b = y[::-1].cumsum(axis)[::-1]
    boundary = 0.5*(x[(f >= b).argmax(axis)] + x[(f <= b).argmin(axis)])
    return boundary


def average_crossing_points(x, y):
    f = np.cumsum(1 - y)
    b = y[::-1].cumsum()[::-1]
    boundary = 0.5*(x[(f >= b).argmax()] + x[(f <= b).argmin()])
    return boundary
