#!/usr/bin/env python

from astropy.io import fits
import numpy as np
import pylab as plt

file='result.fits'

data=fits.getdata(file,1)

def plot_u_gr(u, g, r):
    """Plot u vs. g-r for sample SDSS data

    Input: u, g, r
    Output: Displays plot of u vs. g-r to screen
""" 
    py.plot(g-r, u, 'o')
    py.xlabel('g-r [mag]')
    py.ylabel('u [mag]')

