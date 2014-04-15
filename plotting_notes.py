### Organizing data to make it easy to analyze and plot. ###

import numpy as np
from matplotlib import pyplot as plt

# First let's create the data.  Default step size of 1.
xdata = np.arange(1,100)/(2*pi)
ydata = sin(xdata)
ysigma = 0.05
# Resample data points from assume error distribution
ydata += np.random.normal(0, ysigma, size=(len(xdata)))
ydataerr = ysigma * np.ones_like(ydata)
ydataerr = ysigma * np.random.lognormal(mean=1, sigma=0.5, size=(len(ydata)))
##########

input='foo.txt'
np.savetxt(input, np.column_stack((xdata, ydata, ydataerr)), header="x    y    yerr")
# If you just do
## np.savetxt(input, (xdata, ydata, ydataerr))
## You get the tuples saved, so you get a row of xdata, then a row of ydata, and a row of ydataerr
## It works, but it's almost never what you actually wanted.
## So we use 'column_stack' to align our tuple in the right way.
# If you know the format of the numbers that you want, you can add that.
# For example I know these are all floats, so let's pick a default '%f'
np.savetxt(input, np.column_stack((xdata, ydata, ydataerr)), header="x    y   yerr", fmt="%f")

# One quick trick is that if you know the number of decimal places you want to save to, 
#  you can just specify that, without having to also specify the total length of the number.  E.g.,
np.savetxt(input, np.column_stack((xdata, ydata, ydataerr)), header="x    y      yerr", fmt="%.4f")

# For more details on formatting than you really want to know, see
#  http://docs.python.org/library/string.html#format-specification-mini-language

# Notice that the default is to overwrite the file.

##########
(x, y, yerr) = np.loadtxt(input, unpack=True)
# Magic shorthand for color and symbol characters together as third argument
plt.plot(x, y, 'ko')
plt.xlabel('Time [units]')
plt.ylabel('Signal [units]')
# To get full list of symbol, line, and color options
help(plt.plot)

plt.clf()
# But we had error bars, so let's use those
plt.errorbar(x, y, yerr, fmt='o', color='k')
plt.xlabel('Time [units]')
plt.ylabel('Signal [units]')

## There are legends, learn about them
help(plt.legend)

plt.clf()
# Or, astropy to create a Table object
# Table objects are really convenient.  They maintin coherence of the data
# allow you to slice row or column, and you can create subsets most easily
from astropy.io import ascii
# Which is almost from astropy.table import Table
data = ascii.read(input)

plt.plot(data['x'], data['y'], 'ko')
plt.xlabel('Time [units]')
plt.ylabel('Signal [units]')

plt.clf()
plt.errorbar(data['x'], data['y'], data['yerr'], fmt='o', color='k')
plt.xlabel('Time [units]')
plt.ylabel('Signal [units]')

# What do our errors loook like.  
#  Perhaps we wish to select only data points with errors less than some characteristic value.
#  Let's make a histogram to investigate the data.
#  When we make a histogram, we are making a choice about how to display the data
#  By default 'matplotlib.pyplot.hist' uses 10 bins that span the range of the input data
plt.clf()
plt.hist(data['yerr'], histtype='stepfilled')
plt.xlabel('Signal [units]')
plt.ylabel('# / bin')
### That's right, histograms have units of (# / binsize)

# We can explicitly calculate the bin size
# by using the object returned by 'hist'
# (This is an exercise in using the information returned by hist, and using off-by-one slicing, 
#  disguised as a calculation of binsize.)
plt.clf()
h1 = plt.hist(data['yerr'], histtype='stepfilled')
binsize=np.mean(h1[1][1:] - h1[1][:-1])
plt.xlabel('Signal [units]')
plt.ylabel('# / (%f [units])' % binsize)

# We can add a second axis 
#  we do this to add a histogram with a different bin size to clarify the units point.
ax=plt.gca()   # Get current axis   
ax2=ax.twinx()
h2 = plt.hist(data['yerr'], bins=20, histtype='stepfilled', color='green')
ax2.set_ylim(ax.get_ylim())
binsize2=np.mean(h2[1][1:] - h2[1][:-1])
plt.ylabel('# / (%f [units])' % binsize2)

### How do we choose the binning for histograms in general?
# For some more disussion, see
#  http://www.astroml.org/user_guide/density_estimation.html#bayesian-blocks-histograms-the-right-way

### What if we show just good data based on some cut in yerr, say 0.2.
# First, let's overplot that cut line on our previous plot
yerr_cut = 0.20
ax.axvline(yerr_cut, linestyle='--', linewidth=2, color='red')
### We actually wanted ax2.axvline.  
### I used the above so you can think through why the vertical line appears as it does.

# And then save our plot with:
plt.savefig("yerr_hist.pdf")

### You can easily control the file format of your output plot just based on suffix 
###
### PDF is generically better than PNG (which is unfortunately the implict default in matplotlib)
### PDF is a vector-based representation and so can be scaled to different sizes and still look good
### PNG is a pixel-based representation and so looks bad scaled (up, in particular) 
###   and you have to pick a DPI and size, etc. -- which are not things you actually wanted to worry about in making the plot.
### If you have actual image data from a telescope, say, then PNG is very likely what you want.
### PDF and PNG are equally accepted by 'pdflatex'
### PNG are probably easier to post to a wiki, and you can always easily 'convert foo.pdf foo.png' 

# Good data.
w, = np.where(data['yerr'] < yerr_cut)
gooddata = data[w]

plt.clf()
plt.errorbar(data['x'], data['y'], data['yerr'], fmt='o', color='k', label='raw data')
plt.xlabel('Time [units]')
plt.ylabel('Signal [units]')
plt.errorbar(gooddata['x'], gooddata['y'], gooddata['yerr'], fmt='o', color='g', ecolor='g', label='good data')
plt.legend(numpoints=1)
plt.legend(loc='lower left', numpoints=1)

### FITTING ###

# Basic fitting.
#  How do we fit this data?
# http://wiki.scipy.org/Cookbook/FittingData
#### (But if you _ever_ use Tx and tX to represent different things, you deserve what you will get.)

from scipy import optimize

### Basic function
sinfunc = lambda p, x: p[0]*np.sin(x * p[1] - p[2])
### Delta between data and model
errfunc = lambda p, x, y: sinfunc(p, x) - y

### We're going to use least-square optimization here, but there's a whole family of optimizing functions under 'optimize'
help(optimize)
p0 = [0.5, 1, 0]
fit_p, success = optimize.leastsq(errfunc, p0, args=(data['x'], data['y']))

plot(data['x'], sinfunc(fit_p, data['x']), color='blue', label='least-sq good fit: %f %f %f' % tuple(fit_p))

# But note that if we start with a very bad guess, we don't find the right solution
#  even though 'optimize.leastsq' says it was "success"ful
p1 = [0.5, 0.1, 0]
fit_p1, success = optimize.leastsq(errfunc, p1, args=(data['x'], data['y']))
print fit_p1, success

plot(data['x'], sinfunc(fit_p1, data['x']), color='red', linestyle='--', label='least-sq bad fit: %f %f %f' % tuple(fit_p1))
# Now let's update the legend
plt.legend(loc='lower left', numpoints=1)

### What happened?  It's tricky to fit functions with phases using simple least squares.
### Let's be a little more general and use 'curve_fit' and the uncertainty information we have.

### Curve fit is actually generally what you want as it will take the function you probably would have written
help(optimize.curve_fit)

modelfunc = lambda p, a, f, t: a *np.sin(x * f - t)
pc, covar = optimize.curve_fit(modelfunc, xdata=data['x'], ydata=data['y'], sigma=data['yerr'], p0=p0)

plot(data['x'], modelfunc(data['x'], pc[0], pc[1], pc[2]), color='magenta', linestyle='--', label='curve fit: %f %f %f' % tuple(pc))
# Now let's update the legend
plt.legend(loc='lower left', numpoints=1)

