from pynbody import units
import numpy as np



# Define a list of IllustrisTNG simulation runs available for analysis
global illustrisTNGruns
illustrisTNGruns=['TNG50-1','TNG50-2','TNG50-3','TNG50-4',
                  'TNG100-1','TNG100-2','TNG100-3',
                  'TNG300-1','TNG300-2','TNG300-3',]

# Define common units used in TNG simulations
UnitLength=units.kpc/units.h
UnitMass=1e10*units.Msol/units.h
UnitVel=units.km/units.s
UnitMassdTime=UnitMass/(0.978*units.Gyr/units.h)
UnitComvingLength=units.a*UnitLength
UnitPressure=(UnitMass/UnitLength)*(units.km/units.s/units.kpc)**2
UnitNo=units.no_unit

# Define parameters that will not be converted in physical_units()
global NotneedtransGCPa
NotneedtransGCPa=['SubhaloSFR','SubhaloSFRinHalfRad','SubhaloSFRinMaxRad','SubhaloSFRinRad','SubhaloStellarPhotometrics',
                  'GroupSFR']


def HaloPaName(field :str ,) -> str:
    """
    This function modifies the name of a halo parameter to a custom-defined name.

    Parameters:
    -----------
    field : str
        The standard name of the halo parameter as it is typically used in the data.

    Returns:
    --------
    str
        The custom name corresponding to the input parameter. If the input parameter
        does not match any predefined names, the function returns the input parameter 
        name unchanged.

    Notes:
    ------
    The function uses a dictionary `Matchfield` to map standard halo parameter names to 
    their custom names. Currently, `Matchfield` is empty. If the input `field` is found 
    in `Matchfield`, the corresponding custom name is returned. Otherwise, the function 
    returns the original `field` name.
    """
    Matchfield={

    }
    if field in Matchfield.keys():
        return Matchfield[field]
    else:
        return field

def SubhaloPaName(field :str,) -> str:
    """
    This function modifies the name of a subhalo parameter to a custom-defined name.

    Parameters:
    -----------
    field : str
        The standard name of the halo parameter as it is typically used in the data.

    Returns:
    --------
    str
        The custom name corresponding to the input parameter. If the input parameter
        does not match any predefined names, the function returns the input parameter 
        name unchanged.

    Notes:
    ------
    The function uses a dictionary `Matchfield` to map standard subhalo parameter names to 
    their custom names. Currently, `Matchfield` is empty. If the input `field` is found 
    in `Matchfield`, the corresponding custom name is returned. Otherwise, the function 
    returns the original `field` name.
    """
    Matchfield={

    }
    if field in Matchfield.keys():
        return Matchfield[field]
    else:
        return field
        


def SnapshotPaName(field : str,) -> str:
    """
    This function modifies the name of a particle parameter to a custom-defined name.

    Parameters:
    -----------
    field : str
        The standard name of the parameter as it is typically used in the data.

    Returns:
    --------
    str
        The custom name corresponding to the input parameter. If the input parameter
        does not match any predefined names, the function returns the input parameter 
        name unchanged.

    Notes:
    ------
    The function uses a dictionary `Matchfield` to map standard parameter names to 
    their custom names. If the input `field` is found in `Matchfield`, the corresponding 
    custom name is returned. Otherwise, the function returns the original `field` name.
    """
    Matchfield={
        'Coordinates': 'pos',
        'Density': 'rho',
        'ParticleIDs': 'iord',
        'Potential': 'pot',
        'Masses': 'mass',
        'Velocities': 'vel',
        'GFM_StellarFormationTime': 'aform',
        'GFM_Metallicity': 'metals',
        'InternalEnergy': 'u',
        
        'StarFormationRate': 'sfr',
    }
    if field in Matchfield.keys():
        return Matchfield[field]
    else:
        return field

def SnapshotsUnits(field : str,) -> units.Unit:
    """
    This function provides the unit corresponding to a given particle parameter.

    Parameters:
    -----------
    field : str
        The name of the particle parameter for which the unit is requested.

    Returns:
    --------
    unit
        The unit associated with the input parameter. If the parameter is not 
        defined in `Matchfieldunits`, the function will raise a KeyError.

    Notes:
    ------
    The function uses a dictionary `Matchfieldunits` to map parameter names to their 
    respective units. The units are specified based on the TNG project's data 
    specifications as detailed in their documentation:
    https://www.tng-project.org/data/docs/specifications/#sec1b
    
    Examples:
    ---------
    >>> SnapshotsUnits('Density')
    (UnitMass)/(UnitComvingLength)**3
    
    >>> SnapshotsUnits('Velocity')
    KeyError: 'Velocity'
    """
    Matchfieldunits={
        'CenterOfMass': UnitComvingLength,
        'Coordinates': UnitComvingLength,
        'Density':(UnitMass)/(UnitComvingLength)**3,
        'ElectronAbundance':UnitNo,
        'EnergyDissipation':(1/units.a*1e10*units.Msol)/(units.a*units.kpc)*(UnitVel)**3,
        'GFM_AGNRadiation':units.erg/units.s/units.cm**2*(4*np.pi),
        'GFM_CoolingRate':units.erg*units.cm**3/units.s,
        'GFM_Metallicity': UnitNo,
        'GFM_Metals':UnitNo,
        'GFM_MetalsTagged':UnitNo,
        'GFM_WindDMVelDisp':UnitVel,
        'GFM_WindHostHaloMass':UnitMass,
        'InternalEnergy':(UnitVel)**2,
        'InternalEnergyOld':(UnitVel)**2,
        'Machnumber':UnitNo,
        'units.magneticField':(units.h/units.a**2)*UnitPressure**(1,2),
        'units.magneticFieldDivergence':(units.h**3/units.a*2)*(1e10*units.Msol)**(1,2)*(UnitVel)*(units.a*units.kpc)**(-5,2),
        'Masses':(UnitMass),
        'NeutralHydrogenAbundance':UnitNo,
        'ParticleIDs': UnitNo,
        'Potential': (UnitVel)**2/units.a,
        'StarFormationRate':  units.Msol/units.yr,
        'SubfindDMDensity': (UnitMass)/(UnitComvingLength)**3,
        'SubfindDensity': (UnitMass)/(UnitComvingLength)**3,
        'SubfindHsml': UnitComvingLength,
        'SubfindVelDisp': UnitVel,
        'Velocities': units.km*units.a**(1,2)/units.s,
        'BirthPos': UnitComvingLength,
        'BirthVel': units.km*units.a**(1,2)/units.s,
        'GFM_InitialMass': UnitMass,
        'GFM_StellarFormationTime': UnitNo,
        'GFM_StellarPhotometrics': UnitNo,
        'StellarHsml': UnitComvingLength,
        'BH_BPressure': (units.h/units.a)**4*1e10*units.Msol*(UnitVel)**2/(units.a*units.kpc)**3,
        'BH_CumEgyInjection_QM': UnitMass*(UnitComvingLength)**2/(0.978*units.Gyr/units.h)**2,
        'BH_CumEgyInjection_RM': UnitMass*(UnitComvingLength)**2/(0.978*units.Gyr/units.h)**2,
        'BH_CumMassGrowth_QM': UnitMass,
        'BH_CumMassGrowth_RM': UnitMass,
        'BH_Density': UnitMass/(UnitComvingLength)**3,
        'BH_HostHaloMass': UnitMass,
        'BH_Hsml': UnitComvingLength,
        'BH_Mass':  UnitMass,
        'BH_Mdot': UnitMassdTime,
        'BH_MdotBondi': UnitMassdTime,
        'BH_MdotEddington':UnitMassdTime,
        'BH_Pressure':UnitMass/(UnitComvingLength)/(0.978*units.Gyr/units.h)**2,
        'BH_Progs': UnitNo,
        'BH_U': (UnitVel)**2,
    }
    if field in Matchfieldunits:
        return Matchfieldunits[field]
    else:
        raise KeyError(f"Parameter '{field}' not found in Matchfieldunits.")

def GroupcatUnits(field : str,) -> units.Unit:
    """
    This function provides the unit corresponding to a given halo or subhalo parameter.

    Parameters:
    -----------
    field : str
        The name of the halo or subhalo parameter for which the unit is requested.

    Returns:
    --------
    unit
        The unit associated with the input parameter. If the parameter is not 
        defined in `Matchfieldunits`, the function will raise a KeyError.

    Notes:
    ------
    The function uses a dictionary `Matchfieldunits` to map parameter names to 
    their respective units. The units are specified based on the TNG project's data 
    specifications as detailed in their documentation:
    https://www.tng-project.org/data/docs/specifications/#sec2
    """
    Matchfieldunits={
        ### halo properties
        'GroupBHMass': UnitMass,
        'GroupBHMdot': UnitMassdTime,
        'GroupCM': UnitComvingLength,
        'GroupFirstSub': UnitNo,
        'GroupGasMetalFractions': UnitNo,
        'GroupGasMetallicity': UnitNo,
        'GroupLen': UnitNo,
        'GroupLenType': UnitNo,
        'GroupMass': UnitMass,
        'GroupMassType': UnitMass,
        'GroupNsubs': UnitNo,
        'GroupPos': UnitComvingLength,
        'GroupSFR': units.Msol/units.yr,
        'GroupStarMetalFractions': UnitNo,
        'GroupStarMetallicity': UnitNo,
        'GroupVel': units.km/units.s/units.a,
        'GroupWindMass': UnitMass,
        'Group_M_Crit200': UnitMass,
        'Group_M_Crit500': UnitMass,
        'Group_M_Mean200': UnitMass,
        'Group_M_TopHat200': UnitMass,
        'Group_R_Crit200': UnitComvingLength,
        'Group_R_Crit500': UnitComvingLength,
        'Group_R_Mean200': UnitComvingLength,
        'Group_R_TopHat200': UnitComvingLength,

        ### subhalo properties
        'SubhaloFlag': UnitNo,
        'SubhaloBHMass': UnitMass,
        'SubhaloBHMdot': UnitMassdTime,
        'SubhaloBfldDisk': (units.h/units.a**2)*UnitPressure**(1,2),
        'SubhaloBfldHalo': (units.h/units.a**2)*UnitPressure**(1,2),
        'SubhaloCM': UnitComvingLength,
        'SubhaloGasMetalFractions': UnitNo,
        'SubhaloGasMetalFractionsHalfRad': UnitNo,
        'SubhaloGasMetalFractionsMaxRad': UnitNo,
        'SubhaloGasMetalFractionsSfr': UnitNo,
        'SubhaloGasMetalFractionsSfrWeighted': UnitNo,
        'SubhaloGasMetallicity': UnitNo,
        'SubhaloGasMetallicityHalfRad': UnitNo,
        'SubhaloGasMetallicityMaxRad': UnitNo,
        'SubhaloGasMetallicitySfr': UnitNo,
        'SubhaloGasMetallicitySfrWeighted': UnitNo,
        'SubhaloGrNr': UnitNo,
        'SubhaloHalfmassRad': UnitComvingLength,
        'SubhaloHalfmassRadType': UnitComvingLength,
        'SubhaloIDMostbound': UnitNo,
        'SubhaloLen': UnitNo,
        'SubhaloLenType': UnitNo,
        'SubhaloMass': UnitMass,
        'SubhaloMassInHalfRad': UnitMass,
        'SubhaloMassInHalfRadType': UnitMass,
        'SubhaloMassInMaxRad': UnitMass,
        'SubhaloMassInMaxRadType': UnitMass,
        'SubhaloMassInRad':UnitMass,
        'SubhaloMassInRadType': UnitMass,
        'SubhaloMassType': UnitMass,
        'SubhaloParent': UnitNo,
        'SubhaloPos': UnitComvingLength,
        'SubhaloSFR': units.Msol/units.yr,
        'SubhaloSFRinHalfRad': units.Msol/units.yr,
        'SubhaloSFRinMaxRad': units.Msol/units.yr,
        'SubhaloSFRinRad': units.Msol/units.yr,
        'SubhaloSpin': UnitLength*UnitVel,
        'SubhaloStarMetalFractions': UnitNo,
        'SubhaloStarMetalFractionsHalfRad': UnitNo,
        'SubhaloStarMetalFractionsMaxRad': UnitNo,
        'SubhaloStarMetallicity': UnitNo,
        'SubhaloStarMetallicityHalfRad': UnitNo,
        'SubhaloStarMetallicityMaxRad': UnitNo,
        'SubhaloStellarPhotometrics': UnitNo,
        'SubhaloStellarPhotometricsMassInRad': UnitMass,
        'SubhaloStellarPhotometricsRad': UnitComvingLength,
        'SubhaloVel': UnitVel,
        'SubhaloVelDisp': UnitVel,
        'SubhaloVmax': UnitVel,
        'SubhaloVmaxRad': UnitComvingLength,
        'SubhaloWindMass': UnitMass
    }
    if field in Matchfieldunits:
        return Matchfieldunits[field]
    else:
        raise KeyError(f"Parameter '{field}' not found in Matchfieldunits.")

