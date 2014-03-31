#!/usr/bin/env python

from astropy.io import fits
import numpy as np
import pylab as plt

def plot_u_gr(u, g, r):
    """Plot u vs. g-r for sample SDSS data

    Input: u, g, r
    Output: Displays plot of u vs. g-r to screen
""" 
    plt.plot(g-r, u, 'o')
    plt.xlabel('g-r [mag]')
    plt.ylabel('u [mag]')
    plt.show()

if __name__ == "__main__":
    file='result.fits'

    data=fits.getdata(file,1)

    plot_u_gr(data['u'], data['g'], data['r'])

