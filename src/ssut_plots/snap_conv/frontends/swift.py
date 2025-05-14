import h5py
import numpy as np
import unyt as u

from snap_conv.util import git_version

from .hdf5 import Hdf5Frontend
from .header import Header

_units = [u.Ampere, u.cm, u.g, u.K, u.s]


class SwiftFrontend(Hdf5Frontend):
    def _make_aliases(self):
        self.gas.alias("Density", "Densities")
        self.gas.alias("SmoothingLength", "SmoothingLengths")
        self.gas.alias("StarFormationRate", SwiftFrontend.sanitize_sfr)
        self.gas.alias("InternalEnergy", "InternalEnergies")
        self.stars.alias("SmoothingLength", "SmoothingLengths")
        self.stars.alias("InitialMass", "InitialMasses")
        self.stars.alias("StellarFormationTime", "BirthScaleFactors")
        self.black_holes.alias("Masses", "SubgridMasses")
        self.black_holes.alias("SmoothingLength", "SmoothingLengths")
        self.black_holes.alias("Mdot", "AccretionRates")

    def sanitize_sfr(self):
        if (data := self.gas.check_cache("StarFormationRate")) is not None:
            return data
        data = self.gas.StarFormationRates.copy()
        data[data < 0] = 0
        self.gas.add_cache(data, "StarFormationRate")
        return data

    def _get_unit(self, group, key):
        with h5py.File(self.fname) as f:
            attrs = f[group][key].attrs
            factor = attrs[
                "Conversion factor to CGS (not including cosmological corrections)"
            ][0]
            exponents = [attrs[f"U_{c} exponent"][0] for c in "ILMTt"]
            unit = 1.0
            for part, exp in zip(_units, exponents):
                unit = unit * part**exp
            if unit == 1 and factor == 1:
                return None
            return factor * unit

    @classmethod
    def _get_output_unit(cls, group, key):
        units = dict(
            mass=1e10 * u.Msun,
            length=u.Mpc,
            velocity=u.km / u.s,
        )
        units["time"] = units["length"] / units["velocity"]

        gas_units = dict(
            Coordinates=units["length"],
            StarFormationRate=u.Msun / u.yr,
            Masses=units["mass"],
            InternalEnergy=units["velocity"] ** 2,
            Density=units["mass"] / units["length"] ** 3,
            Velocities=units["velocity"],
            SmoothingLength=units["length"],
        )
        dm_units = dict(
            Coordinates=units["length"],
            Masses=units["mass"],
            Velocities=units["velocity"],
        )
        star_units = dict(
            Coordinates=units["length"],
            Masses=units["mass"],
            Velocities=units["velocity"],
            SmoothingLength=units["length"],
            InitialMass=units["mass"],
        )
        bh_units = dict(
            Coordinates=units["length"],
            Masses=units["mass"],
            Velocities=units["velocity"],
            SmoothingLength=units["length"],
            Mdot=units["mass"] / units["time"],
        )
        field_units = dict(
            PartType0=gas_units,
            PartType1=dm_units,
            PartType4=star_units,
            PartType5=bh_units,
        )

        if group in field_units:
            if key in field_units[group]:
                return field_units[group][key]
        return None

    def load_header(self):
        with h5py.File(self.fname) as f:
            header = f["Header"].attrs
            cosmo = f["Cosmology"].attrs

            redshift = header["Redshift"][0]
            scale = header["Scale-factor"][0]
            h = cosmo["H0 [internal units]"][0] / 100
            H = cosmo["H0 [internal units]"][0] * u.km / u.s / u.Mpc
            box_size = header["BoxSize"] * u.Mpc
            num_part = header["NumPart_Total"]
            Omega_cdm = cosmo["Omega_cdm"]
            Omega_b = cosmo["Omega_b"]
            Omega_m = cosmo["Omega_m"]
            Omega_Lambda = cosmo["Omega_lambda"]

            return Header(
                redshift=redshift,
                scale=scale,
                h=h,
                H=H,
                box_size=box_size,
                num_part=num_part,
                Omega_cdm=Omega_cdm,
                Omega_b=Omega_b,
                Omega_m=Omega_m,
                Omega_Lambda=Omega_Lambda,
            )

    def __str__(self) -> str:
        return "SWIFT " + super().__str__()
