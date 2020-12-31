from datetime import datetime
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

def lc(events, time, cps, cps_err=None, bins=300, gti=None, author=None, SIM_id=0, name=None, overwrite=0):
    """
    Write X-ray lightcurve to .fits similar to CIAO formatted files. 

    Parameters
    ----------
    events: [s] array_like,
        Photon arrival times / time-tagged events (TTEs);
    
    time: [s] array_like,
        time values associated to cps;
    
    cps: [count/s] array_like,
        count rate values;

    (cps_err=None): [count/s] array_like,
        error on count rate values;
    
    (bins=300): [s] scalar,
        bin value to apply on provided data;

    (gti=None): [s] array_like,
        Good time intervals in the format [[gti1_start, gti1_end], [gti2_start, gti2_end], ...];

    (author=None):  string,
        Author of simulation;
    
    (SIM_id=0): integer,
        Simulation number;

    (name=None): string,
        Name of file to be created;

    (overwrite=0): T/F,
        Overwrite existing file;

    Returns
    -------
    Nothing (lc.fits file)

    """
    #----------------------------------------------------------------
    # Formatting data for binning purposes
    #----------------------------------------------------------------
    if (time[-1]-time[0]) % bins != 0:
        time = np.insert(time, len(time), time[-1] + bins - (time[-1]-time[0]) % bins)
        cps = np.insert(cps, len(cps), cps[-1])
        if cps_err is None:
            cps_err = np.zeros((len(cps)))
        else:
            cps_err = np.insert(cps_err, len(cps_err), cps_err[-1])
    bins_array = np.arange(0, (time[-1]-time[0])/bins, 1, dtype=int) + 1 # numbered bins
    bins_left = np.linspace(time[0], time[-1], int((time[-1]-time[0])/bins)+1)[:-1]
    bins_mid = bins_left + bins/2
    bins_right = bins_left + bins 
    # binning arrays
    counts = np.hstack(map(len, np.split(events, np.searchsorted(events, bins_right))))[:-1] # binning counts
    cps_binned = np.hstack(map(np.mean, np.split(cps, np.searchsorted(time, bins_right))))[:-1]    
    cps_err_binned = np.hstack(map(np.mean, np.split(cps_err, np.searchsorted(time, bins_right))))[:-1]
    totcts = len(events)
    #----------------------------------------------------------------
    # primary HDU
    #----------------------------------------------------------------
    lc_hdr = fits.Header()
    lc_hdr['NAXIS'] = (0, 'number of data axes')
    lc_hdr['EXTEND'] = ('T', 'FITS dataset may contain extensions')
    lc_hdr['HDUNAME'] = ('PRIMARY')
    lc_hdr['ORIGIN'] = ('writeto_FITS.py', 'Source of FITS file')
    lc_hdr['DATE'] = (datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 'Date and time of file creation')
    lc_hdr['TIMESYS'] = ('TT', 'Time system')
    lc_hdr['TIMEZERO'] = (0, '[s] Clock correction')
    lc_hdr['TIMEUNIT'] = ('s', 'Time unit')
    lc_hdr['MJDREF'] = (time[0]/86400, '[d] MJD zero point for times')
    lc_hdr['TSTART'] = (time[0], '[s] Simulation start time')
    lc_hdr['TSTOP'] = (time[-1], '[s] Simulation end time')
    lc_hdr['OBS_ID'] = (SIM_id, 'Simulation id')
    lc_hdr['DATE-OBS'] = (datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 'File creation date')
    lc_hdr['INSTRUME'] = ('PYTHON', 'Instrument')
    lc_hdr['TELESCOP'] = ('NONE', 'Telescope')
    lc_empty_primary = fits.PrimaryHDU(header=lc_hdr)
    #----------------------------------------------------------------
    # Binary Table HDU
    #----------------------------------------------------------------
    lc_col0 = fits.Column(name='TIME_BIN', array=bins_array, unit='s', format='1J')
    lc_col1 = fits.Column(name='TIME_MIN', array=bins_left, unit='s', format='1D')
    lc_col2 = fits.Column(name='TIME', array=bins_mid, unit='s', format='1D')
    lc_col3 = fits.Column(name='TIME_MAX', array=bins_right, unit='s', format='1D')
    lc_col4 = fits.Column(name='COUNTS', array=counts, unit='count', format='1J')
    lc_col5 = fits.Column(name='COUNT_RATE', array=cps_binned, unit='count/s', format='1D')
    lc_col6 = fits.Column(name='COUNT_RATE_ERR', array=cps_err_binned, unit='count/s', format='1D')
    lc_col7 = fits.Column(name='NET_COUNTS', array=counts, unit='count', format='1D')
    lc_col8 = fits.Column(name='NET_RATE', array=cps_binned, unit='count/s', format='1D')
    lc_col9 = fits.Column(name='ERR_RATE', array=cps_err_binned, unit='count/s', format='1D')
    lc_tableHDU = fits.BinTableHDU.from_columns([lc_col0, lc_col1, lc_col2, lc_col3, lc_col4, lc_col5, lc_col6, lc_col7, lc_col8, lc_col9])
    lc_tableHDU.header['EXTNAME'] = ('LIGHTCURVE', 'Name of this extension')
    lc_tableHDU.header['DATACLAS'] = ('SIMULATED', 'default')
    lc_tableHDU.header['TIMEDEL'] = (bins, '[s] Delta-T between time records')
    lc_tableHDU.header['ONTIME'] = (time[-1] - time[0], '[s] Sum of GTIs')
    lc_tableHDU.header['LIVETIME'] = (time[-1] - time[0], '[s] Livetime')
    lc_tableHDU.header['EXPOSURE'] = (time[-1] - time[0], '[s] Exposure time')
    lc_tableHDU.header['TOTCTS'] = (totcts, '[count] Total counts')
    lc_tableHDU.header['TITLE'] = (name, 'File title')
    lc_tableHDU.header['CREATOR'] = (author, "Author of simulation")
    lc_tableHDU.header['OBS_ID'] = (SIM_id, 'Simulation number')
    lc_tableHDU.header['CONTENT'] = ('LIGHTCURVE', 'Lightcurve file')
    lc_tableHDU.header['TLMIN1'] = (1)
    lc_tableHDU.header['TLMAX1'] = int(bins_array[-1])
    #----------------------------------------------------------------
    # Good Time Intervals (GTIs)
    #----------------------------------------------------------------
    if gti is None:
        lc_gti_col0 = fits.Column(name='START', array=np.array([time[0]]), unit='s', format='1D')
        lc_gti_col1 = fits.Column(name='STOP', array=np.array([time[-1]]), unit='s', format='1D')
    else:
        gti = np.asarray(gti)
        lc_gti_col0 = fits.Column(name='START', array=gti[:,0], unit='s', format='1D')
        lc_gti_col1 = fits.Column(name='STOP', array=gti[:,1], unit='s', format='1D')
    lc_gti = fits.BinTableHDU.from_columns([lc_gti_col0, lc_gti_col1])
    lc_gti.header['EXTNAME'] = ('GTI', 'Name of this binary table extension')
    lc_gti.header['MJDREF'] = (time[0]/86400, '[d] MJD zero point for times')
    lc_gti.header['TSTART'] = (time[0], '[s] Simulation start time')
    lc_gti.header['TSTOP'] = (time[-1], '[s] Simulation end time')
    lc_gti.header['TIMESYS'] = ('TT', 'Time system')
    lc_gti.header['TIMEUNIT'] = ('s', 'Time unit')
    lc_gti.header['CCD_ID'] = (0, 'CCD for this table, set to 0 for simulations')
    #----------------------------------------------------------------
    # Write to .fits
    #----------------------------------------------------------------
    lc_file = fits.HDUList([lc_empty_primary, lc_tableHDU, lc_gti])
    if name is None:
        lc_file.writeto("Sim" + str(SIM_id) + "_lc" + str(bins) + ".fits", overwrite=overwrite)
    else:
        lc_file.writeto(str(name) + "_lc" + str(bins) + ".fits", overwrite=overwrite)


#____________________________________________________________________________________________________

def evt(events, photon_E, gti=None, author=None, SIM_id=0, name=None, overwrite=0):
    """
    Write X-ray events to .fits similar to CIAO formatted files. 

    Parameters
    ----------
    events: [s] array_like,
        Photon arrival times / time-tagged events (TTEs);
    
    photon_E: [eV] float or list,
        Energy of each photon. If scalar then each photon has the same energy;

    (gti=None): [s] array_like,
        Good time intervals in the format [[gti1_start, gti1_end], [gti2_start, gti2_end], ...];

    (author=None):  string,
        Author of simulation;
    
    (SIM_id=0): integer,
        Simulation number;

    (name=None): string,
        Name of file to be created;

    (overwrite=0): T/F,
        Overwrite file with same name;

    Returns
    -------
    Nothing (evt.fits file)

    """
    photon_E = np.asarray(photon_E).tolist()
    #----------------------------------------------------------------
    # primary HDU
    #----------------------------------------------------------------
    evt_hdr = fits.Header()
    evt_hdr['NAXIS'] = (0, 'number of data axes')
    evt_hdr['EXTEND'] = ('T', "FITS dataset may contain extensions")
    evt_hdr['HDUNAME'] = ('PRIMARY')
    evt_hdr['ORIGIN'] = ('create_CIAOfits', 'Source of FITS file')
    evt_hdr['DATE'] = (datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 'Date and time of file creation')
    evt_hdr['TIMESYS'] = ('TT', 'Time system')
    evt_hdr['MJDREF'] = (events[0]/86400, '[d] MJD zero point for times')
    evt_hdr['TIMEZERO'] = (0, '[s] Clock correction')
    evt_hdr['TIMEUNIT'] = ('s', 'Time unit')
    evt_hdr['CLOCKAPP'] = ('T', 'default')
    evt_hdr['TSTART'] = (events[0], '[s] Simulation start time')
    evt_hdr['TSTOP'] = (events[-1], '[s] Simulation end time')
    evt_hdr['OBS_ID'] = (SIM_id, 'Simulation id')
    evt_hdr['DATE-OBS'] = (datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 'File creation date')
    evt_hdr['MJD-OBS'] = (events[0]/86400, 'Modified Julian date of simulation')
    evt_hdr['INSTRUME'] = ('PYTHON', 'Intrument') 
    evt_hdr['TELESCOP'] = ('NONE', 'Telescope')
    evt_empty_primary = fits.PrimaryHDU(header=evt_hdr)
    #----------------------------------------------------------------
    # Binary Table HDU
    #----------------------------------------------------------------
    evt_col0 = fits.Column(name='time', array=events, unit='s', format='1D')
    evt_col1 = fits.Column(name='ccd_id', array=np.zeros((len(events))), format='1J')
    if isinstance(photon_E, list):
        evt_col2 = fits.Column(name='energy', array=photon_E, unit='eV', format='1D')
    else:
        evt_col2 = fits.Column(name='energy', array=photon_E*np.ones((len(events))), unit='eV', format='1D')
    evt_tableHDU = fits.BinTableHDU.from_columns([evt_col0, evt_col1, evt_col2])
    evt_tableHDU.header['EXTNAME'] = ('EVENTS', 'Name of this binary table extension')
    evt_tableHDU.header['DSVAL1'] = ('TABLE', '[s]')
    evt_tableHDU.header['DSREF1'] = (':GTI7') 
    evt_tableHDU.header['DSVAL3'] = ('2000:8000', '[eV]')
    #----------------------------------------------------------------
    # Good Time Intervals (GTIs)
    #----------------------------------------------------------------
    if gti is None:
        evt_gti_col0 = fits.Column(name='START', array=np.array([events[0]]), unit='s', format='1D')
        evt_gti_col1 = fits.Column(name='STOP', array=np.array([events[-1]]), unit='s', format='1D')
    else:
        gti = np.asarray(gti)
        evt_gti_col0 = fits.Column(name='START', array=gti[:,0], unit='s', format='1D')
        evt_gti_col1 = fits.Column(name='STOP', array=gti[:,1], unit='s', format='1D')
    evt_gti = fits.BinTableHDU.from_columns([evt_gti_col0, evt_gti_col1])
    evt_gti.header['EXTNAME'] = ('GTI', 'name of this binary table extension')
    evt_gti.header['MJDREF'] = (events[0]/86400, '[d] MJD zero point for times')
    evt_gti.header['TSTART'] = (events[0], '[s] Simulation start time')
    evt_gti.header['TSTOP'] = (events[-1], '[s] Simulation end time')
    evt_gti.header['TIMESYS'] = ('TT', 'Time system')
    evt_gti.header['TIMEUNIT'] = ('s', 'Time unit')
    evt_gti.header['CCD_ID'] = (0, 'CCD for this table, simulation is 0')
    evt = fits.HDUList([evt_empty_primary, evt_tableHDU, evt_gti])
    #----------------------------------------------------------------
    # Writing to .fits 
    #----------------------------------------------------------------
    if name is None:
        evt.writeto("Sim" + str(SIM_id) + "_evt.fits", overwrite=overwrite)
    else:
        evt.writeto(str(name) + "_evt.fits", overwrite=overwrite)
        