import numba
import numpy as np
import matplotlib.pyplot as plt

def read_simdat(path_to_simdat):
    with open(path_to_simdat, "r") as f:
        header = f.readline()
        lines = f.readlines()
        x = []
        y = []
        p_true = []
        for line in lines:
            xn, yn, pn = line.split(',')
            x.append(float(xn))
            y.append(float(yn))
            p_true.append(float(pn))
        return {'x': x, 'y': y, 'p_true': p_true}

if __name__ == '__main__':
    # Path to simulated data
    path_to_simdat = '../../dat/dat.txt'

    # Read simulated data
    data = read_simdat(path_to_simdat)
    

