#!/usr/bin/env python

from astropy.io import fits
file='result.fits'

data=fits.getdata(file,1)

py.plot(data['g']-data['r'], data['u'], 'o')
py.xlabel('g-r [mag]')
py.ylabel('u [mag]')

