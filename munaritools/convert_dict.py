from __future__ import print_function, division
from astropy.io import ascii
import numpy as np
import os


def fetch_model(teff, fn):
    fn = "T_{0:05}/{1}".format(teff, fn)
    model_spectrum = np.loadtxt(basedir+fn)
    model_spectrum = model_spectrum[wavemask]
    # If masking enabled, set mask to regions defined in lims.
    # Otherwise mask should encompass whole spectrum.
    if do_masking:
        mask = np.where((modelwave > lims[0][0]) & (modelwave < lims[0][1]))
        for x in lims[1:]:
            masktemp = np.where((modelwave > x[0]) & (modelwave < x[1]))
            mask = np.append(mask, masktemp)
    else:
        mask = np.where((modelwave > 0))
        # Generate fit to masked spectrum, normalize
    fit = np.polyfit(modelwave[mask], model_spectrum[mask], 2)
    newfit = fit[2] + fit[1]*modelwave + fit[0]*modelwave**2
    return model_spectrum / newfit

if (os.uname()[1]).startswith('uhppc'):
    uname = os.getlogin()
    munari_dir = "/local/home/{0}/Munari/".format(uname)
    basedir = "/local/home/{0}/spectra/fluxed_spectra/".format(uname)
elif (os.uname()[1]).startswith('node'):
    munari_dir = "/car-data/hfarnhill/Munari/"
    basedir = "/car-data/hfarnhill/spectra/fluxed_spectra/"

do_masking = False
print("Masking = {0}".format(do_masking))

# Get model wavelength array, cut down to desired range
modelwave = np.loadtxt(basedir+"LAMBDA_01A.ASC")
wavemask = np.where((modelwave > 8300) & (modelwave < 8935))
modelwave = modelwave[wavemask]

# Read file containing all library spectrum info
munari_info = ascii.read('{0}file_details.txt'.format(basedir))

# Set up dictionary for spectrum
specdict = {}

# Regions that seem "safe" from Paschen/Ca lines
lims = [[8300, 8480], [8510, 8530], [8555, 8580], [8720, 8755], [8782, 8788],
        [8796, 8806], [8814, 8822], [8854, 8860], [8872, 8882], [8924, 8930]]

# Append Munari spectra to disctionary after loading, trimming and normalizing
for spectrum in munari_info:
    teff = spectrum['Temp']
    print(teff)
    fn = spectrum['Filename']
    model_spectrum = fetch_model(teff, fn)
    keyn = fn.split('.')[0]
    specdict[keyn] = model_spectrum

if do_masking:
    np.save("{0}Munari_masked.dict".format(basedir), specdict)
else:
    np.save("{0}Munari.dict".format(basedir), specdict)
