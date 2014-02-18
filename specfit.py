# make sure to set PYTHONPATH to ""  not a syntax error
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import emcee
import astropy.io.fits as pyfits
import triangle
import time
import math
import acor

# things to do to improve the code:
# - Which parameters affect the fit: Macroturbulent velocity, Alpha enhancement, NEW/OLD models
#   (plot to see the differences between these - or add them to the code to test their impact?)
# - Implement variable sky spectrum (different fibers or mixture of fibers)
# - Work out best Teff interpolation method (linear, logarithmic, polynomial, Geert's gaussians?)

# For the future:
# - Create template and then use cross-correlation to determine best fitting RV
#   (Should we vary the best fitting template according to uncertainties?)
#   (Should we add in a noise component to see how that affects the RV?)

# List of things to do:
# - E - Add RV into the MCMC simulation to see what results we get
#       - Looks like we get a poorly fitted RV (10 km/s uncertainty), and little Teff improvement
#       - Testing with 1500 runs to compare uncertainties
# - E - Test Macroturbulent velocity, Alpha enhancement, & NEW/OLD models in the MCMC simulation
# - E - Add extra polynomial to the fit and see what happens
#       - Looks like the uncertainties don't really change
#       - Testing with 1500 runs to compare uncertainties
# - H - Find fiber list & positions, and write code to randomly pick fibers as a free parameter
# - E - Try logarithmic interpolation for Teff grid to see if the results vary from linear interp
# - M - Test running the code with IPython.parallel
#       Transfer all the data to the cluster
#       Test running the code on the cluster and time it to check the improvement
# - E - How are the fits and uncertainties affected when using/not using certain parameters

# define the likelihood function
def lnlike(theta, star, wave):
	teff, logg, vel, met, sky = theta
	teff_below = '{0:05.0f}'.format(int(teff/250.)*250.)
	teff_above = '{0:05.0f}'.format(int(teff/250.)*250. + 250.)
	logg_unit = '{0:02.0f}'.format(round(logg/0.5)*5.)
	met_unit = '{0:02.0f}'.format(math.fabs(round(met/0.5)*5.))
	if met >= 0. or met_unit == "00":
		met_sign = "P"
	else:
		met_sign = "M"

	if vel <= 5.:
		vel_unit = "000"
	elif vel <= 15.:
		vel_unit = "010"
	elif vel <= 25.:
		vel_unit = "020"
	elif vel <= 35.:
		vel_unit = "030"
	elif vel <= 45.:
		vel_unit = "040"
	elif vel <= 62.5:
		vel_unit = "050"
	elif vel <= 87.5:
		vel_unit = "075"
	else:
		vel_unit = "100"

	filename1 = "templates/Munari/CAII/T" + teff_below + "G" + logg_unit + met_sign + met_unit + "V" + vel_unit + "K2SNWNVD01F.FITS"
	filename2 = "templates/Munari/CAII/T" + teff_above + "G" + logg_unit + met_sign + met_unit + "V" + vel_unit + "K2SNWNVD01F.FITS"
	model1 = pyfits.getdata(filename1).__array__().astype('float64')
	model2 = pyfits.getdata(filename2).__array__().astype('float64')
	fraction = (teff - float(teff_below))/250.
	model = model1*(1.-fraction) + model2*fraction
	model = np.interp(wave,modelwave,model)
	sigma2 = np.array([0.03]*len(wave))
	icov=np.linalg.inv(np.diag(sigma2))
	diff = star-(model+skyspec*sky)
	return -0.5*np.dot(diff,np.dot(icov,diff))

# define the log probability function
def lnprob(theta, star, wave):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lnlike(theta, star, wave)

# define the priors:
def lnprior(theta):
	teff, logg, vel, met, sky = theta
	if 3500. <= teff <= 8000. and 1.0 <= logg <= 5.0 and 0.0 <= vel <= 100. and -2.5 <= met <= 0.5 and 0.0 <= sky <= 2.0:
		return 0.0
	return -np.inf

# a subroutine to write the full numerical results of the MCMCM code to a file
def writeout(sampler):
	# This first file lists all nwalkers and nruns for all ndim
	flatchains = mcmc_results(sampler,param_keys).flatchain
	outdata = Table(flatchains.values(), names=flatchains.keys())
	filename=('tables/ndim_' + str(ndim) + '_walkers_' + str(nwalkers) + '.fits')
	Table.write(outdata,filename,overwrite=True)

# a subroutine to plot a triangle plot for the existing data
def mcmctriangle(ID):
	filename=('tables/ndim_' + str(ndim) + '_walkers_' + str(nwalkers) + '.fits')
	data = Table.read(filename)
	data_t = np.array([data[key] for key in param_keys]).transpose()
	truths = [np.median(data[key]) for key in param_keys]
	triangle.corner(data_t,labels=param_labels,quantiles=[0.1587,0.5000,0.8413])
	plt.savefig('plots/triangle_ndim_' + str(ndim) + '_walkers_' + str(nwalkers) + '.pdf')
	plt.close()

# a subroutine to calculate the averages and uncertainties from the data and write them out to another file
def bestfits(ID):
	means = []
	medians = []
	q_84s = []
	q_16s = []
	filename=('tables/ndim_' + str(ndim) + '_walkers_' + str(nwalkers) + '_nruns_'+ str(nruns) + '.fits')
	data = Table.read(filename)
	mean = [np.mean(data[key]) for key in param_keys]
	means.append(mean)
	median = [np.median(data[key]) for key in param_keys]
	medians.append(median)
	q_84 = [ np.percentile(data[key],84) for key in param_keys ]
	q_84s.append(q_84)
	q_16 = [ np.percentile(data[key],16) for key in param_keys ]
	q_16s.append(q_16)

	q_84s = np.array(q_84s)
	q_16s = np.array(q_16s)
	means = np.array(means)
	medians = np.array(medians)

	outdata = [ means[:,0], means[:,1], means[:,2], means[:,3], means[:,4], medians[:,0], medians[:,1], medians[:,2], medians[:,3], medians[:,4], q_16s[:,0], q_16s[:,1], q_16s[:,2], q_16s[:,3], q_16s[:,4], q_84s[:,0], q_84s[:,1],q_84s[:,2], q_84s[:,3], q_84s[:,4] ]
	colnames = ['Teff_mean','logg_mean','Vrot_mean','[M/H]_mean','Sky_mean','Teff_median','logg_median','Vrot_median','[M/H]_median','Sky_median','Teff_q_16','logg_q_16','Vrot_q_16','[M/H]_q_16','Sky_q_16','Teff_q_84','logg_q_84','Vrot_q_84','[M/H]_q_84','Sky_q_84']
	table = Table(outdata, names=colnames)
	filename = 'tables/averages.fits'
	Table.write(table, filename, overwrite=True)

# a class of programs to manipulate the MCMC results
class mcmc_results:
	def __init__(self, sampler, param_keys):
		self.param_keys = param_keys
		self.sampler = sampler
		self.flatchain = self.get_flatchains()

	def get_flatchains(self):
		results = [self.sampler.flatchain[:,i] for i in range(len(param_keys))]
		return dict(zip(self.param_keys, results))

	def chain(self,walker):
		chains = [self.sampler.chain[walker][:,i] for i in range(len(param_keys))]
		return dict(zip(self.param_keys, chains))

# a subroutine to run some more iterations of the MCMC code
def runmore(pos,nruns):
	start=time.time()
	sampler.run_mcmc(pos, nruns)
	x = sampler.flatchain[:,0]
	tau, mean, sigma = acor.acor(x)
	runsdone = len(x)/nwalkers
	indsamples = runsdone/tau
	end=time.time()
	total = str("%.4f" % ((end-start)/(nruns*nwalkers)))
	print "Time per run and per walker is " + total + " seconds (" + str(runsdone) + " runs / " + str(nwalkers) + " walkers)"
	print "Autocorrelation time = " + str(tau) + " and independent samples = " + str(indsamples)



##### MAIN CODE STARTS HERE #####

# read in the template wavelength file first
modelwave = pyfits.getdata("templates/Munari/wavelengths.fits").__array__().astype('float64')

# now read in the star file
starfile = "spectra/2011b/extract/016_t.fits"
star, header = pyfits.getdata(starfile, 0, header=True)
wave = (np.linspace(1., header['NAXIS1'], header['NAXIS1'])-header['CRPIX1'])*header['CD1_1'] + header['CRVAL1']
fit = np.polyfit(wave,star,2)
newfit = fit[2] + fit[1]*wave + fit[0]*wave**2
star = star / newfit

# Get sky file
skyfile = "spectra/2011b/extract/016_sky.fits"
skyspec = pyfits.getdata(skyfile, 0, header=False)/newfit

# set parameters for emcee and set starting position guess
# Have lots of walkers (like hundreds): 
ndim, nwalkers, nruns = 5, 100, 100
param_keys = ['logteff','log g','Vrot','[M/H]','Sky']
param_labels=[r'Teff',r'log g',r'Vrot',r'[M/H]',r'Sky']
# walkers currently randomly distributed across parameter space - should change this to tight balls around likely results
teff0 = np.arange(4000., 8000., 250.)
logg0 = np.arange(1.0, 5.0, 0.1)
vel0 = np.arange(0., 100., 10.)
met0 = np.arange(-2.5, 0.5, 0.1)
sky0 = np.arange(0.0, 2.0, 0.1)
pos = [np.array([np.random.choice(teff0), np.random.choice(logg0), np.random.choice(vel0), np.random.choice(met0), np.random.choice(sky0)]) for i in range(nwalkers)]

# set up the emcee sampler, then run the MCMC for 100 steps from the stating position 'pos0', then reset the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(star, wave))
start=time.time()
pos, prob, state = sampler.run_mcmc(pos, nruns)
sampler.reset()
middle=time.time()
total = str("%.4f" % (middle-start))
print "Burn-in time is " + total + " seconds (for nwalkers=" + str(nwalkers) + ", nruns=" + str(nruns) + ")." 
print "Starting main MCMC simulation now."

runsdone = 0

while (runsdone < 100 or indsamples < 10):
	sampler.run_mcmc(pos, nruns)
	x = sampler.flatchain[:,0]
	tau, mean, sigma = acor.acor(x)
	runsdone = len(x)/nwalkers
	indsamples = runsdone/tau
	print "Done " + str(runsdone) + " runs and so far have " + str(indsamples) + " independent samples."

end=time.time()
total = str("%.4f" % (end-start))
timeper = str("%.4f" % ((end-start)/(runsdone*nwalkers)))
print "Converged after " + str(runsdone) + " runs, which took " + total + " seconds"
print "Time per run and per walker is " + timeper + " seconds"
print "Autocorrelation time = " + str(tau) + " and independent samples = " + str(indsamples)
