from __future__ import print_function, division
from astropy.table import Table
from astropy.io import fits
import numpy as np
import glob
import os


def trimspec(data, header, trimto):
    """Trims the supplied spectrum to specified wavelength range.

    Also modifies the relevant keywords in the supplied header.

    """
    wave = ((np.linspace(1., header['NAXIS1'], header['NAXIS1']) -
             header['CRPIX1']) * header['CD1_1'] + header['CRVAL1'])
    mask = np.where((wave > trimto[0]) & (wave < trimto[1]))
    wavenew = wave[mask]
    data = data[mask]
    header['NAXIS1'] = len(wavenew)
    header['CRVAL1'] = wavenew[0]
    return data, header


def sort(flatten, extension, trim, trimto, clobber, pattern):
    """Processes fits files matching {pattern}.fits in the working directory.

    Extracts single extension (if multi-extension files are supplied and
    specified) and trims to requested wavelength range.

    """
    flist = glob.glob("./{0}.fits".format(pattern))
    if len(flist) == 0:
        print("Could not find any files matching the supplied pattern!\n"
              "Make sure you have provided a sensible pattern, and have"
              "specified the right directory!")
        return

    for spectrum in flist:
        fn = spectrum.split('/')[1]
        f = fits.open(spectrum)
        header = f[0].header

        # If skysub/raw/sky Hectospec files are supplied, this will
        # strip all but the spectrum contained in the specified extension
        if flatten:
            header['NAXIS3'] = 1
            data = f[0].data[extension][0]
        else:
            data = f[0].data

        # Trim the spectrum to the requested wavelength range
        if trim:
            data, header = trimspec(data, header, trimto)

        # Check that obj and sky subfolders exist. If not, create them.
        if not os.path.isdir('./obj'):
            os.mkdir('obj')
        if not os.path.isdir('./sky'):
            os.mkdir('sky')
        # Save the new spectra into the subfolders.
        if header['TYPE'] == 1:
            fits.writeto("obj/{0}".format(fn), data=data, header=header,
                         clobber=clobber)
        elif header['TYPE'] == 0:
            fits.writeto("sky/{0}".format(fn), data=data, header=header,
                         clobber=clobber)


def mklst(fdr, tablename, clobber):
    """Generate fits file listing all spectra in the {fdr} subdirectory.

    Table contains spectrum filenames and the RA/DEC position of the spectrum.

    """
    flist = glob.glob("./{0}/*.fits".format(fdr))
    fns, RAs, DECs = [], [], []
    for spectrum in flist:
        fn = spectrum.split('/')[2]
        fns.append(fn)
        header = fits.getheader(spectrum)
        RAarr = (header['RA']).split(':')
        DECarr = (header['DEC']).split(':')
        RA = ((15. * float(RAarr[0])) +
              (0.25 * float(RAarr[1])) +
              ((1. / 240.) * float(RAarr[2])))
        DEC = (float(DECarr[0]) +
               ((1. / 60.) * float(DECarr[1])) +
               ((1. / 3600.) * float(DECarr[2])))
        RAs.append(RA)
        DECs.append(DEC)
    t1 = Table()
    t1['filename'] = fns
    t1['RA'] = RAs
    t1['DEC'] = DECs
    t1.write("{0}.fits".format(tablename), overwrite=clobber)


def mkarr(fdr, arrname):
    """Generate numpy array containing all spectra in {fdr} subdirectory.

    Resulting file can be read in with np.load, and contains records with
    'name','wavelengths','spectrum' fields.

    """
    flist = glob.glob("./{0}/*.fits".format(fdr))
    dtype = [('name', '|S10'), ('wavelengths', object), ('spectrum', object)]
    datalist = []
    for spectrum in flist:
        fn = spectrum.split('/')[2]
        f = fits.open(spectrum)
        header = f[0].header
        wave = ((np.linspace(1., header['NAXIS1'], header['NAXIS1']) -
                 header['CRPIX1']) * header['CD1_1'] + header['CRVAL1'])
        data = f[0].data
        datalist.append(tuple([fn, wave, data]))
    arr = np.rec.fromrecords(datalist, dtype=dtype)
    np.save(arrname, arr)


def run(cwd, flatten=True, trim=True, genfits=True, genarr=True,
        clobber=False, pattern="*", extension=1, trimto=[8300, 8925]):
    # Allow user to specify folder to process, but return them to cwd
    orig_dir = os.getcwd()
    os.chdir(cwd)

    # Process folder by
    # 1) extracting single extension (default: second extension)
    # 2) trimming to desired wl range (default 8300 - 8925 Å)
    # Operates on fits files in folder that match {pattern}.fits
    sort(flatten, extension, trim, trimto, clobber, pattern)

    # Generate files objs.fits and skies.fits, which list all object and sky
    # spectra respectively, along with their RA and DEC in degrees.
    if genfits:
        mklst("obj", "objs", clobber)
        mklst("sky", "skies", clobber)

    # Generate numpy recarrays containing all object and sky spectra in
    # numpy arrays as objs.npy and skies.npy
    if genarr:
        mkarr("obj", "objs")
        mkarr("sky", "skies")

    # Returns user to the folder where they first ran the script
    os.chdir(orig_dir)
