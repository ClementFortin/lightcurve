import numpy as np
import sys

def flux_to_cps(time, flux, conversion_factor, time_unit='s', distance=8178, distance_err=35):
    """
    Convert from flux units to cps (count per second). It is assumed that the flux is coming from the SMBH Sgr A* at the center of the Milky Way, as per http://dx.doi.org/10.1051/0004-6361/201935656. 
    This can be changed by specifying the distance variable.

    Parameters
    ----------
    time: array_like,
        time values, does not have to possess any structure;

    flux: [erg/cm^2/sec] array_like,
        array of flux values associated to time array;

    conversion_factor: [count/10**34 erg] float_like,
        factor corresponding to each photon's average energy;
    
    (time_unit = 's'): string,
        units of time array, default is seconds. The possible entries are
        's' = seconds; 'min' = minutes; 'h' = hours; 'd' = days; 'yrs' = years; 

    (distance = 8178): [pc] float_like,
        distance to the flux emitting object;

    (distance_err = 35): [pc] float_like,
        error on the distance;

    Returns
    -------
    time_sec: [s] ndarray,
        time array converted to seconds (if not already the case);

    cps: [count/s] ndarray,
        count rate values associated with time array;

    cps_err: [count/s] ndarray,
        count rate error coming from uncertainty in the distance;
    

    """
    # changing units, from erg/cm^2/s to cps
    distance_cm = distance * 3.086*10**18 #distance in cm
    cps = flux*4*np.pi*distance_cm**2*conversion_factor*10**(-34) #counts/s
    # changing time units to seconds if necessary
    if time_unit != 's':
        if time_unit == 'min':
            time_sec = time * 60
        elif time_unit == 'h': 
            time_sec = time * 3600
        elif time_unit == 'd':
            time_sec = time * 24 * 3600
        elif time_unit == 'yrs':
            time_sec = time * 365 * 24 * 3600
        else:
            sys.exit("Time unit not valid, please enter a valid unit as a string (e.g. 's').")
    else:
        time_sec = time
    # uncertainties
    dsq_err = (distance_err*(13/distance_cm)*distance_cm**2)
    cps_err = flux*4*np.pi*dsq_err*conversion_factor*10**(-34)

    return time_sec, cps, cps_err


#____________________________________________________________________________________________________

def create_events(time, cps, cps_err=0):
    """
    Simulate photon arrival times as events from a lightcurve using Poisson statistics for each interval. 
    All operations are vectorized, replacing loop of np.linspace() using builtin_function map().

    Parameters
    ----------
    time: [s] array_like,
        time values, does not have to possess any internal structure or regularities;
    
    cps: [count/s] array_like,
        count rate values associated to the time values;
    
    (cps_err=0): array_like,
        error on count rate values;

    Returns
    -------
    events: [s] ndarray,
        photon arrival times;
    """
    
    # Initiate random sampling of cps values (range determined by error on each value)
    cps_low = cps - cps_err # lower bounds
    cps_low[cps_low < 0] = 0 # negative cps -> 0 ct/s
    cps_high = cps + cps_err # upper bounds
    # Sample required values
    cps_rand = np.random.uniform(cps_low, cps_high)
    # Reshaping time array to create intervals (adding element t[-1] + dt[-1] at the end to not lose last cps value)
    tstart = time
    tstop = np.concatenate([ time[time != time[0]], [2*time[-1] - time[-2]] ])
    intervals = np.column_stack((tstart, tstop))
    Ncounts = np.random.poisson(cps_rand * (tstop - tstart)) 
    print("Total number of events generated:", sum(Ncounts))
    # Consider only nonempty intervals
    Ncounts_nonzero = Ncounts[Ncounts != 0] 
    intervals_nonempty = intervals[Ncounts != 0]
    steps = (intervals_nonempty[:,1] - intervals_nonempty[:,0]) / Ncounts_nonzero
    divs = Ncounts_nonzero.astype(int)  
    # Photons are stacked at "start" value of nonempty intervals
    events_stack = np.repeat(intervals_nonempty[:,0], divs)
    # Find offset for each photon
    repeat_1d = np.repeat(steps, divs)
    split = np.split(repeat_1d, np.cumsum(divs)[:-1])
    offset = np.hstack(map(np.cumsum, split)) - repeat_1d/2 # last term for centering

    events = events_stack + offset
    return events

#____________________________________________________________________________________________________
