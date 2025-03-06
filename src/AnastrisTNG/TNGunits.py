'''
units and descriptions of parameters
'''

from pynbody import units
import numpy as np

# Define a list of IllustrisTNG simulation runs available for analysis
# global illustrisTNGruns
illustrisTNGruns = [
    'TNG50-1',
    'TNG50-2',
    'TNG50-3',
    'TNG50-4',
    'TNG100-1',
    'TNG100-2',
    'TNG100-3',
    'TNG300-1',
    'TNG300-2',
    'TNG300-3',
    'TNG-Cluster'
]

# Define common units used in TNG simulations
UnitLength = units.kpc / units.h
UnitMass = 1e10 * units.Msol / units.h
UnitVel = units.km / units.s
UnitMassdTime = UnitMass / (0.978 * units.Gyr / units.h)
UnitComvingLength = units.a * UnitLength
UnitPressure = (UnitMass / UnitLength) * (units.km / units.s / units.kpc) ** 2
UnitNo = units.no_unit

# Define parameters that will not be converted in physical_units()
# global NotneedtransGCPa
NotneedtransGCPa = [
    'SubhaloSFR',
    'SubhaloSFRinHalfRad',
    'SubhaloSFRinMaxRad',
    'SubhaloSFRinRad',
    'SubhaloStellarPhotometrics',
    'GroupSFR',
]


def halo_pa_name(
    field: str,
) -> str:
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
    Matchfield = {}
    if field in Matchfield:
        return Matchfield[field]
    else:
        return field


def subhalo_pa_name(
    field: str,
) -> str:
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
    Matchfield = {}
    if field in Matchfield:
        return Matchfield[field]
    else:
        return field


def snapshot_pa_name(
    field: str,
) -> str:
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
    Matchfield = {
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
    if field in Matchfield:
        return Matchfield[field]
    else:
        return field


def snapshot_units(
    field: str,
) -> units.Unit:
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
    Matchfieldunits = {
        'CenterOfMass': UnitComvingLength,
        'Coordinates': UnitComvingLength,
        'Density': (UnitMass) / (UnitComvingLength) ** 3,
        'ElectronAbundance': UnitNo,
        'EnergyDissipation': (1 / units.a * 1e10 * units.Msol)
        / (units.a * units.kpc)
        * (UnitVel) ** 3,
        'GFM_AGNRadiation': units.erg / units.s / units.cm**2 * (4 * np.pi),
        'GFM_CoolingRate': units.erg * units.cm**3 / units.s,
        'GFM_Metallicity': UnitNo,
        'GFM_Metals': UnitNo,
        'GFM_MetalsTagged': UnitNo,
        'GFM_WindDMVelDisp': UnitVel,
        'GFM_WindHostHaloMass': UnitMass,
        'InternalEnergy': (UnitVel) ** 2,
        'InternalEnergyOld': (UnitVel) ** 2,
        'Machnumber': UnitNo,
        'MagneticField': (units.h / units.a**2) * UnitPressure ** (1, 2),
        'MagneticFieldDivergence': (units.h**3 / units.a * 2)
        * (1e10 * units.Msol) ** (1, 2)
        * (UnitVel)
        * (units.a * units.kpc) ** (-5, 2),
        'Masses': (UnitMass),
        'NeutralHydrogenAbundance': UnitNo,
        'ParticleIDs': UnitNo,
        'Potential': (UnitVel) ** 2 / units.a,
        'StarFormationRate': units.Msol / units.yr,
        'SubfindDMDensity': (UnitMass) / (UnitComvingLength) ** 3,
        'SubfindDensity': (UnitMass) / (UnitComvingLength) ** 3,
        'SubfindHsml': UnitComvingLength,
        'SubfindVelDisp': UnitVel,
        'Velocities': units.km * units.a ** (1, 2) / units.s,
        'BirthPos': UnitComvingLength,
        'BirthVel': units.km * units.a ** (1, 2) / units.s,
        'GFM_InitialMass': UnitMass,
        'GFM_StellarFormationTime': UnitNo,
        'GFM_StellarPhotometrics': UnitNo,
        'StellarHsml': UnitComvingLength,
        'BH_BPressure': (units.h / units.a) ** 4
        * 1e10
        * units.Msol
        * (UnitVel) ** 2
        / (units.a * units.kpc) ** 3,
        'BH_CumEgyInjection_QM': UnitMass
        * (UnitComvingLength) ** 2
        / (0.978 * units.Gyr / units.h) ** 2,
        'BH_CumEgyInjection_RM': UnitMass
        * (UnitComvingLength) ** 2
        / (0.978 * units.Gyr / units.h) ** 2,
        'BH_CumMassGrowth_QM': UnitMass,
        'BH_CumMassGrowth_RM': UnitMass,
        'BH_Density': UnitMass / (UnitComvingLength) ** 3,
        'BH_HostHaloMass': UnitMass,
        'BH_Hsml': UnitComvingLength,
        'BH_Mass': UnitMass,
        'BH_Mdot': UnitMassdTime,
        'BH_MdotBondi': UnitMassdTime,
        'BH_MdotEddington': UnitMassdTime,
        'BH_Pressure': UnitMass
        / (UnitComvingLength)
        / (0.978 * units.Gyr / units.h) ** 2,
        'BH_Progs': UnitNo,
        'BH_U': (UnitVel) ** 2,
        'AllowRefinement': UnitNo,
        'BH_WindCount': UnitNo,
        'BH_WindTimes': UnitNo,
        'BH_MPB_CumEgyLow': UnitMass
        * (UnitComvingLength) ** 2
        / (0.978 * units.Gyr / units.h) ** 2,
        'BH_MPB_CumEgyHigh': UnitMass
        * (UnitComvingLength) ** 2
        / (0.978 * units.Gyr / units.h) ** 2,
    }
    if field in Matchfieldunits:
        return Matchfieldunits[field]
    else:
        raise KeyError(f"Parameter '{field}' not found in Matchfieldunits.")


def groupcat_units(
    field: str,
) -> units.Unit:
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
    Matchfieldunits = {
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
        'GroupSFR': units.Msol / units.yr,
        'GroupStarMetalFractions': UnitNo,
        'GroupStarMetallicity': UnitNo,
        'GroupVel': units.km / units.s / units.a,
        'GroupWindMass': UnitMass,
        'Group_M_Crit200': UnitMass,
        'Group_M_Crit500': UnitMass,
        'Group_M_Mean200': UnitMass,
        'Group_M_TopHat200': UnitMass,
        'Group_R_Crit200': UnitComvingLength,
        'Group_R_Crit500': UnitComvingLength,
        'Group_R_Mean200': UnitComvingLength,
        'Group_R_TopHat200': UnitComvingLength,
        
        #TNG-Cluster 
        'GroupContaminationFracByMass': UnitNo,
        'GroupContaminationFracByNumPart': UnitNo,
        'GroupOrigHaloID': UnitNo,
        'GroupPrimaryZoomTarget': UnitNo,
        'GroupOffsetType': UnitNo,
        
        ### subhalo properties
        'SubhaloFlag': UnitNo,
        'SubhaloBHMass': UnitMass,
        'SubhaloBHMdot': UnitMassdTime,
        'SubhaloBfldDisk': (units.h / units.a**2) * UnitPressure ** (1, 2),
        'SubhaloBfldHalo': (units.h / units.a**2) * UnitPressure ** (1, 2),
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
        'SubhaloMassInRad': UnitMass,
        'SubhaloMassInRadType': UnitMass,
        'SubhaloMassType': UnitMass,
        'SubhaloParent': UnitNo,
        'SubhaloPos': UnitComvingLength,
        'SubhaloSFR': units.Msol / units.yr,
        'SubhaloSFRinHalfRad': units.Msol / units.yr,
        'SubhaloSFRinMaxRad': units.Msol / units.yr,
        'SubhaloSFRinRad': units.Msol / units.yr,
        'SubhaloSpin': UnitLength * UnitVel,
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
        'SubhaloWindMass': UnitMass,
        
        #TNG-Cluster
        'SubhaloOrigHaloID': UnitNo,
        'SubhaloOffsetType': UnitNo,
        
        #TNG-Cluster Snap 99 only
        'TracerLengthType': UnitNo,
        'TracerOffsetType': UnitNo,
        'SubhaloLengthType': UnitNo,
        'SubhaloOffsetType': UnitNo,
    }
    if field in Matchfieldunits:
        return Matchfieldunits[field]
    else:
        raise KeyError(f"Parameter '{field}' not found in Matchfieldunits.")


def parameter_all_Description(table: str, contents: str, parameters: str) -> str:
    '''
    Input:
        table: str,
            'snapshots' or 'groupcatalogs'.
        contents: str,
            'halo' or 'subhalo' for table = 'groupcatalogs'.
            'Gas', 'DM', 'Stars', 'BH' for table = 'snapshots'.
        parameters: str,
            specified parameter.

    Output: str
        description of the parameter.
    '''

    Gas_parameter = {
        'CenterOfMass': '''Spatial position of the center of mass, which in general differs from the geometrical center of the Voronoi cell (the offset should be small). Comoving coordinate.''',
        'Coordinates': '''Spatial position within the periodic simulation domain of BoxSize. Comoving coordinate.''',
        'Density': '''Comoving mass density of cell (calculated as mass/volume).''',
        'ElectronAbundance': '''Fractional electron number density with respect to the total hydrogen number density, so ne=ElectronAbundance*nH where nH=XH*rho/mp. Use with caution for star-forming gas (see comment below for NeutralHydrogenAbundance).''',
        'EnergyDissipation': '''Shock finder output: the dissipated energy rate (amount of kinetic energy irreversibly transformed into thermal energy). Note units correspond to (Energy/Time). ''',
        'GFM_AGNRadiation': '''Bolometric intensity (physical units) at the position of this cell arising from the radiation fields of nearby AGN. One should divide by 4π to obtain the flux at this location, in the sense of F=L/(4πR2).''',
        'GFM_CoolingRate': '''The instantaneous net cooling rate experienced by this gas cell, in cgs units (e.g. Λnet/n2H).''',
        'GFM_Metallicity': '''The ratio MZ/Mtotal where MZ is the total mass all metal elements (above He). Is NOT in solar units. To convert to solar metallicity, divide by 0.0127 (the primordial solar metallicity).''',
        'GFM_Metals': '''Individual abundances of nine species: H, He, C, N, O, Ne, Mg, Si, Fe (in this order). Each is the dimensionless ratio of mass in that species to the total gas cell mass. The tenth entry contains the 'total' of all other (i.e. untracked) metals.''',
        'GFM_MetalsTagged': '''Six additional metal-origin tracking fields in this order: SNIa (0), SNII (1), AGB (2), NSNS (3), FeSNIa (4), FeSNII (5). Each keeps track of heavy elements arising from particular processes. Full description below.''',
        'GFM_WindDMVelDisp': '''Equal to SubfindVelDisp (redundant).''',
        'GFM_WindHostHaloMass': '''Mass of the parent FoF halo of this gas cell (redundant).''',
        'InternalEnergy': '''Internal (thermal) energy per unit mass for this gas cell. See FAQ for conversion to gas temperature. Use with caution for star-forming gas, as this corresponds to the 'effective' temperature of the equation of state, which is not a physical temperature. Note: this field has "corrected" values, and is generally recommended for all uses, see the data release background for details.''',
        'InternalEnergyOld': '''Old internal (thermal) energy per unit mass for this gas cell. See FAQ for conversion to gas temperature. This field holds the original values, and is not recommended for use, see the data release background for details. (!) Note that subboxes do not have corrected values, so the InternalEnergy field for subboxes contains the uncorrected values, and no InternalEnergyOld exists.''',
        'Machnumber': '''Shock finder output: The Mach number (ratio of fluid velocity to sound speed) of the gas cell, zero if no shock is present.''',
        'MagneticField': '''The (comoving) magnetic field 3-vector (x,y,z) of this gas cell.''',
        'MagneticFieldDivergence': '''The divergence of the magnetic field in this cell. ''',
        'Masses': '''Gas mass in this cell. Refinement/derefinement attempts to keep this value within a factor of two of the targetGasMass for every cell.''',
        'NeutralHydrogenAbundance': '''Fraction of the hydrogen cell mass (or density) in neutral hydrogen, so nH0=NeutralHydrogenAbundance*nH. (So note that nH+=nH-nH0). Use with caution for star-forming gas, as the calculation is based on the 'effective' temperature of the equation of state, which is not a physical temperature.''',
        'ParticleIDs': '''The unique ID (uint64) of this gas cell. Constant for the duration of the simulation. May cease to exist (as gas) in a future snapshot due to conversion into a star/wind particle, accretion into a BH, or a derefinement event.''',
        'Potential': '''Gravitational potential energy.''',
        'StarFormationRate': '''Instantaneous star formation rate of this gas cell.''',
        'SubfindDMDensity': '''The local total comoving mass density, estimated using the standard cubic-spline SPH kernel over all DM particles within a radius of SubfindHsml.''',
        'SubfindDensity': '''The local total comoving mass density, estimated using the standard cubic-spline SPH kernel over all particles/cells within a radius of SubfindHsml.''',
        'SubfindHsml': '''The comoving radius of the sphere centered on this cell enclosing the 64±1 nearest dark matter particles.''',
        'SubfindVelDisp': '''The 3D velocity dispersion of all dark matter particles within a radius of SubfindHsml of this cell.''',
        'Velocities': '''Spatial velocity''',
        'AllowRefinement': '''Flag which takes a value of either 0 or a positive integer. If positive, then this gas cell was part of the high resolution region of the targeted zoom halo. If zero, it was in the low resolution background, and could be considered 'contamination' if found within/near a halo of interest. (Not present in mini snapshots).'''
    }

    DM_parameter = {
        'Coordinates': '''Spatial position within the periodic simulation domain of BoxSize. Comoving coordinate.''',
        'ParticleIDs': '''The unique ID (uint64) of this DM particle. Constant for the duration of the simulation.''',
        'Potential': '''Gravitational potential energy.''',
        'SubfindDMDensity': '''The local total comoving mass density, estimated using the standard cubic-spline SPH kernel over all DM particles within a radius of SubfindHsml.''',
        'SubfindDensity': '''The local total comoving mass density, estimated using the standard cubic-spline SPH kernel over all particles/cells within a radius of SubfindHsml.''',
        'SubfindHsml': '''The comoving radius of the sphere centered on this particle enclosing the 64±1 nearest dark matter particles.''',
        'SubfindVelDisp': '''The 3D velocity dispersion of all dark matter particles within a radius of SubfindHsml of this particle.''',
        'Velocities': '''Spatial velocity.''',
    }

    Star_parameter = {
        'BirthPos': '''Spatial position within the periodic box where this star particle initially formed. Comoving coordinate.''',
        'BirthVel': '''Spatial velocity of the parent star-forming gas cell at the time of formation. ''',
        'Coordinates': '''Spatial position within the periodic simulation domain of BoxSize. Comoving coordinate.''',
        'GFM_InitialMass': '''Mass of this star particle when it was formed (will subsequently decrease due to stellar evolution).''',
        'GFM_Metallicity': '''See entry under gas. Inherited from the gas cell spawning/converted into this star, at the time of birth.''',
        'GFM_Metals': '''See entry under gas. Inherited from the gas cell spawning/converted into this star, at the time of birth.''',
        'GFM_MetalsTagged': '''See entry under gas. This field is identical for star particles, and note that it is simply inherited at the time of formation from the gas cell from which the star was born. It does not then evolve or change in any way (i.e. no self-enrichment), so these values describe the 'inherited' wind/SN/NSNS material from the gas.''',
        'GFM_StellarFormationTime': '''The exact time (given as the scalefactor) when this star was formed. Note: The only differentiation between a real star (>0) and a wind phase gas cell (<=0) is the sign of this quantity.''',
        'GFM_StellarPhotometrics': '''Stellar magnitudes in eight bands: U, B, V, K, g, r, i, z. In detail, these are: Buser's X filter, where X=U,B3,V (Vega magnitudes), then IR K filter + Palomar 200 IR detectors + atmosphere.57 (Vega), then SDSS Camera X Response Function, airmass = 1.3 (June 2001), where X=g,r,i,z (AB magnitudes). They can be found in the filters.log file in the BC03 package. The details on the four SDSS filters can be found in Stoughton et al. 2002, section 3.2.1.''',
        'Masses': '''Mass of this star or wind phase cell.''',
        'ParticleIDs': '''The unique ID (uint64) of this star/wind cell. Constant for the duration of the simulation.''',
        'Potential': '''Gravitational potential energy.''',
        'StellarHsml': '''The comoving radius of the sphere centered on this particle enclosing the 32±1 nearest particles of this same type. Useful for visualization.''',
        'SubfindDMDensity': '''The local total comoving mass density, estimated using the standard cubic-spline SPH kernel over all DM particles within a radius of SubfindHsml.''',
        'SubfindDensity': '''The local total comoving mass density, estimated using the standard cubic-spline SPH kernel over all particles/cells within a radius of SubfindHsml.''',
        'SubfindHsml': '''The comoving radius of the sphere centered on this particle enclosing the 64±1 nearest dark matter particles.''',
        'SubfindVelDisp': '''The 3D velocity dispersion of all dark matter particles within a radius of SubfindHsml of this particle.''',
        'Velocities': '''Spatial velocity.''',
    }

    BH_parameter = {
        'BH_BPressure': '''The mean magnetic pressure of gas cells within a radius of BH_Hsml, kernel and volume weighted (kernel weight clipped at a maximum of wt=2.5). Units are those of MagneticField2. Note: is still in Heavyside-Lorentz, not Gauss, so multiply by 4π to be unit consistent with MagneticField.''',
        'BH_CumEgyInjection_QM': '''Cumulative amount of thermal AGN feedback energy injected into surrounding gas in the high accretion-state (quasar) mode, total over the entire lifetime of this blackhole. Field summed during BH-BH merger.''',
        'BH_CumEgyInjection_RM': '''Cumulative amount of kinetic AGN feedback energy injected into surrounding gas in the low accretion-state (wind) mode, total over the entire lifetime of this blackhole. Field summed during BH-BH merger.''',
        'BH_CumMassGrowth_QM': '''Cumulative mass accreted onto the BH in the high accretion-state (quasar) mode, total over the entire lifetime of this blackhole. Field summed during BH-BH merger.''',
        'BH_CumMassGrowth_RM': '''Cumulative mass accreted onto the BH in the low accretion-state (kinetic wind) mode, total over the entire lifetime of this blackhole. Field summed during BH-BH merger.''',
        'BH_Density': '''Local comoving gas density averaged over the nearest neighbors of the BH.''',
        'BH_HostHaloMass': '''Mass of the parent FoF halo of this blackhole.''',
        'BH_Hsml': '''The comoving radius of the sphere enclosing the 64,128,256 (for TNG100-3, -2, and -1 resolutions) ±4 nearest-neighbor gas cells around the BH.''',
        'BH_Mass': '''Actual mass of the BH; does not include gas reservoir. Monotonically increases with time according to the accretion prescription, starting from the seed mass.''',
        'BH_Mdot': '''The mass accretion rate onto the black hole, instantaneous.''',
        'BH_MdotBondi': '''Current estimate of the Bondi accretion rate for this BH.''',
        'BH_MdotEddington': '''Current estimate of the Eddington accretion rate for this BH.''',
        'BH_Pressure': '''Physical gas pressure (in comoving units) near the BH, defined as (gama-1)*rho*u, where rho is the local comoving gas density (BH_Density, as above) u is BH_U (defined below)''',
        'BH_Progs': '''Total number of BHs that have merged into this BH.''',
        'BH_U': '''Thermal energy per unit mass in quasar-heated bubbles near the BH. Used to define the BH_Pressure. Not to be confused with the "radio mode" bubbles injected via the unified feedback model.''',
        'Coordinates': '''Spatial position within the periodic simulation domain of BoxSize. Comoving coordinate.''',
        'Masses': '''Total mass of the black hole particle. Includes the gas reservoir from which accretion is tracked onto the actual BH mass (see BH_Mass).''',
        'ParticleIDs': '''The unique ID (uint64) of this star/wind cell. Constant for the duration of the simulation.''',
        'Potential': '''Gravitational potential energy.''',
        'StellarHsml': '''The comoving radius of the sphere centered on this particle enclosing the 32±1 nearest particles of this same type. Useful for visualization.''',
        'SubfindDMDensity': '''The local total comoving mass density, estimated using the standard cubic-spline SPH kernel over all DM particles within a radius of SubfindHsml.''',
        'SubfindDensity': '''The local total comoving mass density, estimated using the standard cubic-spline SPH kernel over all particles/cells within a radius of SubfindHsml.''',
        'SubfindHsml': '''The comoving radius of the sphere centered on this particle enclosing the 64±1 nearest dark matter particles.''',
        'SubfindVelDisp': '''The 3D velocity dispersion of all dark matter particles within a radius of SubfindHsml of this particle.''',
        'Velocities': '''Spatial velocity.''',
        'BH_WindCount': '''Number of kinetic feedback events (winds) of this black hole. These correspond to discrete times of energy output while the SMBH is in the low/kinetic feedback mode of the TNG model. This quantity was meant to be accumulated along the main progenitor branch (MPB) of a black hole, but see caveat above.''',
        'BH_WindTimes': '''Times (i.e. scalefactors) of the last five kinetic feedback events of this black hole. These correspond to discrete times of energy output while the SMBH is in the low/kinetic feedback mode of the TNG model. The events are ordered in time, but the array is rolled/periodic, i.e. the smallest time is not necessarily the first entry. See caveat above about MPB.''',
        'BH_MPB_CumEgyLow': '''Cumulative amount of kinetic AGN feedback energy injected into surrounding gas in the low accretion-state (wind) mode, total over the entire lifetime of this blackhole. See caveat above about MPB.''',
        'BH_MPB_CumEgyHigh': '''Cumulative amount of thermal AGN feedback energy injected into surrounding gas in the high accretion-state (quasar) mode, total over the entire lifetime of this blackhole. See caveat above about MPB.'''
    }

    FoF_halos = {
        'GroupBHMass': '''Sum of the BH_Mass field of all blackholes (type 5) in this group.''',
        'GroupBHMdot': '''Sum of the BH_Mdot field of all blackholes (type 5) in this group.''',
        'GroupCM': '''Center of mass of the group, computed as the sum of the mass weighted relative coordinates of all particles/cells in the group, of all types. Comoving coordinate. (Available only for the Illustris-3 run)''',
        'GroupFirstSub': '''Index into the Subhalo table of the first (i.e. central/primary) Subfind subhalo within this FoF group. The subhalos of a group are ordered in descending total number of bound particles/cells. The first/central subhalo is usually, but not always, the most massive. Note: This value is signed (or should be interpreted as signed)! In this case, a value of -1 indicates that this FoF group has no subhalos.''',
        'GroupGasMetalFractions': '''Individual abundances: H, He, C, N, O, Ne, Mg, Si, Fe, total (in this order). Each is the dimensionless ratio of the total mass in that species divided by the total gas mass, for all gas cells in the group. The tenth entry contains the 'total' of all other (i.e. untracked) metals.''',
        'GroupGasMetallicity': '''Mass-weighted average metallicity (Mz/Mtot, where Z = any element above He) of all gas cells in this FOF group.''',
        'GroupLen': '''Integer counter of the total number of particles/cells of all types in this group.''',
        'GroupLenType': '''Integer counter of the total number of particles/cells, split into the six different types, in this group. Note: Wind phase cells are counted as stars (type 4) for GroupLenType.''',
        'GroupMass': '''Sum of the individual masses of every particle/cell, of all types, in this group.''',
        'GroupMassType': '''Sum of the individual masses of every particle/cell, split into the six different types, in this group. Note: Wind phase cells are counted as gas (type 0) for GroupMassType.''',
        'GroupNsubs': '''Count of the total number of Subfind groups within this FoF group.''',
        'GroupPos': '''Spatial position within the periodic box (of the particle with the minimum gravitational potential energy). Comoving coordinate.''',
        'GroupSFR': '''Sum of the individual star formation rates of all gas cells in this group.''',
        'GroupStarMetalFractions': '''Individual abundances: H, He, C, N, O, Ne, Mg, Si, Fe, total (in this order). Each is the dimensionless ratio of the total mass in that species divided by the total stellar mass, for all stars in the group. The tenth entry contains the 'total' of all other (i.e. untracked) metals.''',
        'GroupStarMetallicity': '''Mass-weighted average metallicity (Mz/Mtot, where Z = any element above He) of all star particles in this FOF group.''',
        'GroupVel': '''Velocity of the group, computed as the sum of the mass weighted velocities of all particles/cells in this group, of all types.''',
        'GroupWindMass': '''Sum of the individual masses of all wind phase gas cells (type 4, BirthTime <= 0) in this group.''',
        'Group_M_Crit200': '''Total Mass of this group enclosed in a sphere whose mean density is 200 times the critical density of the Universe, at the time the halo is considered.''',
        'Group_M_Crit500': '''Total Mass of this group enclosed in a sphere whose mean density is 500 times the critical density of the Universe, at the time the halo is considered.''',
        'Group_M_Mean200': '''Total Mass of this group enclosed in a sphere whose mean density is 200 times the mean density of the Universe, at the time the halo is considered.''',
        'Group_M_TopHat200': '''Total Mass of this group enclosed in a sphere whose mean density is Δc times the critical density' of the Universe, at the time the halo is considered. Δc derives from the solution of the collapse of a spherical top-hat perturbation (fitting formula from Bryan+ 1998). The subscript 200 can be ignored.''',
        'Group_R_Crit200': '''Comoving Radius of a sphere centered at the GroupPos of this Group whose mean density is 200 times the critical density of the Universe, at the time the halo is considered.''',
        'Group_R_Crit500': '''Comoving Radius of a sphere centered at the GroupPos of this Group whose mean density is 500 times the critical density of the Universe, at the time the halo is considered.''',
        'Group_R_Mean200': '''Comoving Radius of a sphere centered at the GroupPos of this Group whose mean density is 200 times the mean density of the Universe, at the time the halo is considered.''',
        'Group_R_TopHat200': '''Comoving Radius of a sphere centered at the GroupPos of this Group whose mean density is Δc times the critical density of the Universe, at the time the halo is considered.''',
        'GroupContaminationFracByMass': '''Fraction of 'low resolution contamination', from zero (good) to one (bad), equal to the number of PartType2 particles (low-res DM) divided by the number of PartType1 particles (high-res DM) in this FoF halo. In general, halos (and their subhalos) should only be analyzed if they have zero, or near zero, contamination.''',
        'GroupContaminationFracByNumPart': '''Fraction of 'low resolution contamination', from zero (good) to one (bad), equal to the mass of PartType2 particles (low-res DM) divided by the mass of PartType1 particles (high-res DM) in this FoF halo. In general, halos (and their subhalos) should only be analyzed if they have zero, or near zero, contamination.''',
        'GroupOrigHaloID': '''Integer which gives the original FoF halo ID from the parent DMO box from which the TNG-Cluster halos were selected. Ranges from 0 (the most massive halo of the parent box) to 5711 (the least massive halo simulated). Note that each zoom run (i.e. "original FoF halo ID") contains many FoF halos: the primary zoom target, as well as all other halos in the high resolution region / containing high resolution particles. The set of all halos coming from a single zoom run can be chosen as those which have the same GroupOrigHaloID.''',
        'GroupPrimaryZoomTarget': '''Flag with a value of either zero or one. If one, this was the first (i.e. most massive) FoF halo of an original zoom simulation. At z=0, this is exactly the set of halos that were the original zoom targets. Most analyses of TNG-Cluster will use only the 352 halos with GroupPrimaryZoomTarget > 0. The total number of unity values in this dataset equals the number of unique values of GroupOrigHaloID by definition. At higher redshift, z>0, a complication exists: the first FoF halo of each original zoom does not necessarily correspond to the SubLink main progenitor of the z=0 halo. We therefore flag (the parent groups of) all SubLink main progenitors at z>0 with GroupPrimaryZoomTarget == 2, if not already flagged by unity. At z>0 one can decide, e.g. by looking at contamination, which (or both) of these two sets of halos are appropriate for a given analysis.''',
        'GroupOffsetType': '''	These are the same "offsets" as in the offsets files, which identify the starting index of member particles of this FoF halo in the snapshot, for a given type. Copied into the group catalogs for convenience (new convention).''',
    }

    Subfind_subhalos = {
        'SubhaloFlag': '''Flag field indicating suitability of this subhalo for certain types of analysis. If zero, this subhalo should generally be excluded, and is not thought to be of cosmological origin. That is, it may have formed within an existing halo, or is possibly a baryonic fragment of a disk or other galactic structure identified by Subfind. If one, this subhalo should be considered a 'galaxy' or 'satellite' of cosmological origin. (Note: always true for centrals). This field is only present for baryonic runs, and is absent for dark matter only runs. See the data release background for details.''',
        'SubhaloBHMass': '''Sum of the masses of all blackholes in this subhalo.''',
        'SubhaloBHMdot': '''Sum of the instantaneous accretion rates M˙ of all blackholes in this subhalo.''',
        'SubhaloBfldDisk': '''The square root of the volume weighted value of B2for all gas cells within the canonical two times the stellar half mass radius. This value gives a magnetic field strength which would have the same amount of mean magnetic energy as the galaxy cells. (*) Only available for full snapshots.''',
        'SubhaloBfldHalo': '''The square root of the volume weighted value of B2 for all gas cells in the subhalo. This value gives a magnetic field strength which would have the same amount of mean magnetic energy as the subhalo cells. (*) Only available for full snapshots.''',
        'SubhaloCM': '''Comoving center of mass of the Subhalo, computed as the sum of the mass weighted relative coordinates of all particles/cells in the Subhalo, of all types.''',
        'SubhaloGasMetalFractions': '''Individual abundances: H, He, C, N, O, Ne, Mg, Si, Fe, total (in this order). Each is the dimensionless ratio of the total mass in that species divided by the total gas mass, both restricted to gas cells within twice the stellar half mass radius. The tenth entry contains the 'total' of all other (i.e. untracked) metals.''',
        'SubhaloGasMetalFractionsHalfRad': '''Same as SubhaloGasMetalFractions, but restricted to cells within the stellar half mass radius.''',
        'SubhaloGasMetalFractionsMaxRad': '''Same as SubhaloGasMetalFractions, but restricted to cells within the radius of Vmax.''',
        'SubhaloGasMetalFractionsSfr': '''Same as SubhaloGasMetalFractions, but restricted to cells which are star-forming.''',
        'SubhaloGasMetalFractionsSfrWeighted': '''Same as SubhaloGasMetalFractionsSfr, but weighted by the cell star-formation rate rather than the cell mass.''',
        'SubhaloGasMetallicity': '''Mass-weighted average metallicity (Mz/Mtot, where Z = any element above He) of the gas cells bound to this Subhalo, but restricted to cells within twice the stellar half mass radius.''',
        'SubhaloGasMetallicityHalfRad': '''Same as SubhaloGasMetallicity, but restricted to cells within the stellar half mass radius.''',
        'SubhaloGasMetallicityMaxRad': '''Same as SubhaloGasMetallicity, but restricted to cells within the radius of Vmax.''',
        'SubhaloGasMetallicitySfr': '''Mass-weighted average metallicity (Mz/Mtot, where Z = any element above He) of the gas cells bound to this Subhalo, but restricted to cells which are star forming.''',
        'SubhaloGasMetallicitySfrWeighted': '''Same as SubhaloGasMetallicitySfr, but weighted by the cell star-formation rate rather than the cell mass.''',
        'SubhaloGrNr': '''Index into the Group table of the FOF host/parent of this Subhalo.''',
        'SubhaloHalfmassRad': '''Comoving radius containing half of the total mass (SubhaloMass) of this Subhalo.''',
        'SubhaloHalfmassRadType': '''Comoving radius containing half of the mass of this Subhalo split by Type (SubhaloMassType).''',
        'SubhaloIDMostbound': '''The ID of the particle with the smallest binding energy (could be any type).''',
        'SubhaloLen': '''Total number of member particle/cells in this Subhalo, of all types.''',
        'SubhaloLenType': '''Total number of member particle/cells in this Subhalo, separated by type.''',
        'SubhaloMass': '''Total mass of all member particle/cells which are bound to this Subhalo, of all types. Particle/cells bound to subhaloes of this Subhalo are NOT accounted for.''',
        'SubhaloMassInHalfRad': '''Sum of masses of all particles/cells within the stellar half mass radius.''',
        'SubhaloMassInHalfRadType': '''Sum of masses of all particles/cells (split by type) within the stellar half mass radius.''',
        'SubhaloMassInMaxRad': '''Sum of masses of all particles/cells within the radius of Vmax.''',
        'SubhaloMassInMaxRadType': '''Sum of masses of all particles/cells (split by type) within the radius of Vmax.''',
        'SubhaloMassInRad': '''Sum of masses of all particles/cells within twice the stellar half mass radius.''',
        'SubhaloMassInRadType': '''Sum of masses of all particles/cells (split by type) within twice the stellar half mass radius.''',
        'SubhaloMassType': '''Total mass of all member particle/cells which are bound to this Subhalo, separated by type. Particle/cells bound to subhaloes of this Subhalo are NOT accounted for. Note: Wind phase cells are counted as gas (type 0) for SubhaloMassType.''',
        'SubhaloParent': '''Index back into this same Subhalo table of the unique Subfind host/parent of this Subhalo. This index is local to the group (i.e. 2 indicates the third most massive subhalo of the parent halo of this subhalo, not the third most massive of the whole snapshot). The values are often zero for all subhalos of a group, indicating that there is no resolved hierarchical structure in that group, beyond the primary subhalo having as direct children all of the secondary subhalos.''',
        'SubhaloPos': '''Spatial position within the periodic box (of the particle with the minium gravitational potential energy). Comoving coordinate.''',
        'SubhaloSFR': '''Sum of the individual star formation rates of all gas cells in this subhalo.''',
        'SubhaloSFRinHalfRad': '''Same as SubhaloSFR, but restricted to cells within the stellar half mass radius.''',
        'SubhaloSFRinMaxRad': '''Same as SubhaloSFR, but restricted to cells within the radius of Vmax.''',
        'SubhaloSFRinRad': '''Same as SubhaloSFR, but restricted to cells within twice the stellar half mass radius.''',
        'SubhaloSpin': '''Total spin per axis, computed for each as the mass weighted sum of the relative coordinate times relative velocity of all member particles/cells.''',
        'SubhaloStarMetalFractions': '''Individual abundances: H, He, C, N, O, Ne, Mg, Si, Fe, total (in this order). Each is the dimensionless ratio of the total mass in that species divided by the total stellar mass, both restricted to stars within twice the stellar half mass radius. The tenth entry contains the 'total' of all other (i.e. untracked) metals.''',
        'SubhaloStarMetalFractionsHalfRad': '''Same as SubhaloStarMetalFractions, but restricted to stars within the stellar half mass radius.''',
        'SubhaloStarMetalFractionsMaxRad': '''Same as SubhaloStarMetalFractions, but restricted to stars within the radius of Vmax.''',
        'SubhaloStarMetallicity': '''Mass-weighted average metallicity (Mz/Mtot, where Z = any element above He) of the star particles bound to this Subhalo, but restricted to stars within twice the stellar half mass radius.''',
        'SubhaloStarMetallicityHalfRad': '''Same as SubhaloStarMetallicity, but restricted to stars within the stellar half mass radius.''',
        'SubhaloStarMetallicityMaxRad': '''Same as SubhaloStarMetallicity, but restricted to stars within the radius of Vmax.''',
        'SubhaloStellarPhotometrics': '''Eight bands: U, B, V, K, g, r, i, z. Magnitudes based on the summed-up luminosities of all the stellar particles of the group. For details on the bands, see snapshot table for stars.''',
        'SubhaloStellarPhotometricsMassInRad': '''Sum of the mass of the member stellar particles, but restricted to stars within the radius SubhaloStellarPhotometricsRad.''',
        'SubhaloStellarPhotometricsRad': '''Radius at which the surface brightness profile (computed from all member stellar particles) drops below the limit of 20.7 mag arcsec^-2 in the K band (in comoving units).''',
        'SubhaloVel': '''Peculiar velocity of the group, computed as the sum of the mass weighted velocities of all particles/cells in this group, of all types.''',
        'SubhaloVelDisp': '''One-dimensional velocity dispersion of all the member particles/cells (the 3D dispersion divided by sqrt(3)).''',
        'SubhaloVmax': '''Maximum value of the spherically-averaged rotation curve. All available particle types (e.g. gas, stars, DM, and SMBHs) are included in this calculation.''',
        'SubhaloVmaxRad': '''Comoving radius of rotation curve maximum (where Vmax is achieved). As above, all available particle types are used in this calculation.''',
        'SubhaloWindMass': '''Sum of masses of all wind-phase cells in this subhalo (with Type==4 and BirthTime<=0).''',
        'SubhaloOrigHaloID': '''Integer giving the original FoF halo ID from the parent DMO box from which the TNG-Cluster halos were selected. Exactly the same as GroupOrigHaloID, i.e. all subhalos have the same value of this field as their parent halos.''',
        'SubhaloOffsetType': '''These are the same "offsets" as in the offsets files, which identify the starting index of member particles of this subhalo in the snapshot, for a given type. Copied into the group catalogs for convenience (new convention).''',
        'SubhaloOrigHaloID': '''Integer giving the original FoF halo ID from the parent DMO box from which the TNG-Cluster halos were selected. Exactly the same as GroupOrigHaloID, i.e. all subhalos have the same value of this field as their parent halos.''',
        'SubhaloOffsetType': '''These are the same "offsets" as in the offsets files, which identify the starting index of member particles of this subhalo in the snapshot, for a given type. Copied into the group catalogs for convenience (new convention).''',
    }
    Description = {
        'groupcatalogs': {'halo': FoF_halos, 'subhalo': Subfind_subhalos},
        'snapshots': {
            'gas': Gas_parameter,
            'dm': DM_parameter,
            'star': Star_parameter,
            'bh': BH_parameter,
        },
    }
    return ((Description.get(table.lower(), {})).get(contents.lower(), {})).get(
        parameters
    )
