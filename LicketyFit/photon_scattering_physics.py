"""Physics-only optical-scattering kernels for LicketyFit.

Rayleigh is derived from the Einstein--Smoluchowski--Cabannes fluctuation
formula using the IAPWS refractive index of liquid water. Water Raman
scattering is included as a competing first interaction with an OH-stretch
Stokes redistribution. No WCSim-derived scattering scale is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple
import math
import numpy as np

BOLTZMANN_J_PER_K = 1.380649e-23
C_MM_PER_NS = 299.792458

@dataclass(frozen=True)
class WaterState:
    temperature_K: float = 293.15
    density_kg_m3: float = 998.2071
    isothermal_compressibility_Pa_inv: float = 4.58e-10
    def validate(self) -> None:
        if not (250.0 <= self.temperature_K <= 650.0):
            raise ValueError("temperature_K outside model domain")
        if not (500.0 <= self.density_kg_m3 <= 1200.0):
            raise ValueError("density_kg_m3 not plausible")
        if not (0.0 < self.isothermal_compressibility_Pa_inv < 1e-8):
            raise ValueError("compressibility not plausible")

# IAPWS R9-97 coefficients.
_IAPWS_A0 = 0.244257733
_IAPWS_A1 = 9.74634476e-3
_IAPWS_A2 = -3.73234996e-3
_IAPWS_A3 = 2.68678472e-4
_IAPWS_A4 = 1.58920570e-3
_IAPWS_A5 = 2.45934259e-3
_IAPWS_A6 = 0.900704920
_IAPWS_A7 = -1.66626219e-2
_IAPWS_LAMBDA_UV_BAR = 0.2292020
_IAPWS_LAMBDA_IR_BAR = 5.432937
_IAPWS_T_REF_K = 273.15
_IAPWS_RHO_REF_KG_M3 = 1000.0
_IAPWS_LAMBDA_REF_UM = 0.589

R14374_QE_WAVELENGTH_NM = np.asarray(
    [300,320,340,360,380,400,420,440,460,480,500,520,540,560,580,600,620,640,660,680,700],
    dtype=np.float64,
)
R14374_QE_RELATIVE = np.asarray(
    [0.0787,0.1838,0.2401,0.2521,0.2695,0.2676,0.2593,0.2472,0.2276,0.1970,0.1777,0.1547,0.1033,0.0727,0.0587,0.0470,0.0372,0.0285,0.0220,0.0130,0.0084],
    dtype=np.float64,
)

class SpectralQuadrature(NamedTuple):
    wavelength_nm: np.ndarray
    normalized_weight: np.ndarray
    phase_index: np.ndarray
    group_index: np.ndarray
    scattering_coefficient_m_inv: np.ndarray
    depolarization_ratio: np.ndarray

class RamanShiftQuadrature(NamedTuple):
    shift_cm_inv: np.ndarray
    normalized_weight: np.ndarray
    depolarization_ratio: np.ndarray


def _as_f64(x):
    return np.asarray(x, dtype=np.float64)


def water_refractive_index_iapws(wavelength_nm, *, temperature_K=293.15, density_kg_m3=998.2071):
    wl = _as_f64(wavelength_nm)
    if np.any(~np.isfinite(wl)) or np.any(wl <= 0.0):
        raise ValueError("wavelength_nm must be finite and positive")
    rb = float(density_kg_m3) / _IAPWS_RHO_REF_KG_M3
    tb = float(temperature_K) / _IAPWS_T_REF_K
    lb = (wl * 1e-3) / _IAPWS_LAMBDA_REF_UM
    l2 = lb * lb
    F = (_IAPWS_A0 + _IAPWS_A1*rb + _IAPWS_A2*tb + _IAPWS_A3*l2*tb
         + _IAPWS_A4/l2 + _IAPWS_A5/(l2-_IAPWS_LAMBDA_UV_BAR**2)
         + _IAPWS_A6/(l2-_IAPWS_LAMBDA_IR_BAR**2) + _IAPWS_A7*rb*rb)
    q = rb * F
    if np.any(q >= 1.0):
        raise ValueError("IAPWS Lorentz--Lorenz denominator non-positive")
    return np.sqrt((1.0 + 2.0*q) / (1.0 - q))


def water_group_index_iapws(wavelength_nm, *, temperature_K=293.15, density_kg_m3=998.2071):
    wl = _as_f64(wavelength_nm)
    rb = float(density_kg_m3) / _IAPWS_RHO_REF_KG_M3
    tb = float(temperature_K) / _IAPWS_T_REF_K
    lb = (wl * 1e-3) / _IAPWS_LAMBDA_REF_UM
    l2 = lb * lb
    F = (_IAPWS_A0 + _IAPWS_A1*rb + _IAPWS_A2*tb + _IAPWS_A3*l2*tb
         + _IAPWS_A4/l2 + _IAPWS_A5/(l2-_IAPWS_LAMBDA_UV_BAR**2)
         + _IAPWS_A6/(l2-_IAPWS_LAMBDA_IR_BAR**2) + _IAPWS_A7*rb*rb)
    q = rb * F
    n = np.sqrt((1.0 + 2.0*q)/(1.0-q))
    dF_dlb = (2.0*_IAPWS_A3*lb*tb - 2.0*_IAPWS_A4/(lb**3)
              - 2.0*_IAPWS_A5*lb/(l2-_IAPWS_LAMBDA_UV_BAR**2)**2
              - 2.0*_IAPWS_A6*lb/(l2-_IAPWS_LAMBDA_IR_BAR**2)**2)
    dq_dlb = rb * dF_dlb
    dn2_dq = 3.0/(1.0-q)**2
    dn_dlb = 0.5*dn2_dq*dq_dlb/n
    return n - lb*dn_dlb


def rho_d_n2_d_rho_iapws(wavelength_nm, *, temperature_K=293.15, density_kg_m3=998.2071):
    wl = _as_f64(wavelength_nm)
    rb = float(density_kg_m3) / _IAPWS_RHO_REF_KG_M3
    tb = float(temperature_K) / _IAPWS_T_REF_K
    lb = (wl*1e-3)/_IAPWS_LAMBDA_REF_UM
    l2 = lb*lb
    F = (_IAPWS_A0 + _IAPWS_A1*rb + _IAPWS_A2*tb + _IAPWS_A3*l2*tb
         + _IAPWS_A4/l2 + _IAPWS_A5/(l2-_IAPWS_LAMBDA_UV_BAR**2)
         + _IAPWS_A6/(l2-_IAPWS_LAMBDA_IR_BAR**2) + _IAPWS_A7*rb*rb)
    dF = _IAPWS_A1 + 2.0*_IAPWS_A7*rb
    q = rb*F
    dq = F + rb*dF
    return rb * 3.0/(1.0-q)**2 * dq


def water_depolarization_ratio(wavelength_nm, *, model: Literal["spectral_2026","constant_0p04","zero"]="spectral_2026"):
    wl = _as_f64(wavelength_nm)
    if model == "zero": return np.zeros_like(wl)
    if model == "constant_0p04": return np.full_like(wl, 0.04)
    if model != "spectral_2026": raise ValueError(f"unknown depolarization model {model!r}")
    return np.interp(wl, [491.0,532.0,660.0], [0.0417,0.0393,0.0363], left=0.0417, right=0.0363)


def water_rayleigh_scattering_coefficient_m_inv(wavelength_nm, *, water_state=WaterState(), depolarization_model="spectral_2026"):
    water_state.validate()
    wl_nm = _as_f64(wavelength_nm)
    wl_m = wl_nm*1e-9
    optical = rho_d_n2_d_rho_iapws(wl_nm, temperature_K=water_state.temperature_K, density_kg_m3=water_state.density_kg_m3)
    delta = water_depolarization_ratio(wl_nm, model=depolarization_model)
    cabannes = (6.0+6.0*delta)/(6.0-7.0*delta)
    beta90 = ((math.pi**2/(2.0*wl_m**4))*optical**2*BOLTZMANN_J_PER_K
              *water_state.temperature_K*water_state.isothermal_compressibility_Pa_inv*cabannes)
    return (8.0*math.pi/3.0)*beta90*(2.0+delta)/(1.0+delta)


def water_rayleigh_scattering_length_m(*args, **kwargs):
    b=water_rayleigh_scattering_coefficient_m_inv(*args, **kwargs)
    return np.divide(1.0,b,out=np.full_like(b,np.inf),where=b>0.0)


def relative_r14374_qe(wavelength_nm):
    return np.interp(_as_f64(wavelength_nm), R14374_QE_WAVELENGTH_NM, R14374_QE_RELATIVE, left=0.0, right=0.0)


def detected_cherenkov_spectral_quadrature(beta: float, *, n_nodes=8, wavelength_min_nm=300.0, wavelength_max_nm=660.0, water_state=WaterState(), detector_response="r14374_relative_qe", depolarization_model="spectral_2026"):
    if not (0.0 < beta <= 1.0): raise ValueError("beta must lie in (0,1]")
    x,w=np.polynomial.legendre.leggauss(int(n_nodes))
    half=0.5*(wavelength_max_nm-wavelength_min_nm); mid=0.5*(wavelength_max_nm+wavelength_min_nm)
    wl=mid+half*x; qw=half*w
    npidx=water_refractive_index_iapws(wl,temperature_K=water_state.temperature_K,density_kg_m3=water_state.density_kg_m3)
    ng=water_group_index_iapws(wl,temperature_K=water_state.temperature_K,density_kg_m3=water_state.density_kg_m3)
    ft=np.maximum(1.0-1.0/(beta*beta*npidx*npidx),0.0)
    resp=relative_r14374_qe(wl) if detector_response=="r14374_relative_qe" else np.ones_like(wl)
    spec=qw*resp*ft/(wl*wl); total=float(np.sum(spec))
    if total<=0.0: raise ValueError("zero detected spectrum")
    spec/=total
    br=water_rayleigh_scattering_coefficient_m_inv(wl,water_state=water_state,depolarization_model=depolarization_model)
    dep=water_depolarization_ratio(wl,model=depolarization_model)
    return SpectralQuadrature(*(np.ascontiguousarray(a,dtype=np.float64) for a in (wl,spec,npidx,ng,br,dep)))

_WATER_RAMAN_CENTERS_CM_INV=np.asarray([3020.0,3219.0,3422.0,3545.0,3626.0])
_WATER_RAMAN_FWHM_CM_INV=np.asarray([165.0,209.0,226.0,127.0,106.0])
_WATER_RAMAN_RELATIVE_AREAS=np.asarray([0.034,0.367,0.516,0.039,0.047])


def water_raman_scattering_coefficient_m_inv(wavelength_nm, *, reference_wavelength_nm=532.0, reference_coefficient_m_inv=1.84e-4, photon_number_exponent=5.3):
    wl=_as_f64(wavelength_nm)
    return float(reference_coefficient_m_inv)*(float(reference_wavelength_nm)/wl)**float(photon_number_exponent)


def water_raman_depolarization_ratio(shift_cm_inv, *, model="constant_0p20"):
    x=_as_f64(shift_cm_inv)
    if model=="constant_0p20": return np.full_like(x,0.20)
    if model=="constant_0p17": return np.full_like(x,0.17)
    if model=="isotropic": return np.full_like(x,0.75)
    raise ValueError(f"unknown Raman depolarization model {model!r}")


def water_raman_shift_pdf_cm(shift_cm_inv, *, centers_cm_inv=_WATER_RAMAN_CENTERS_CM_INV, fwhm_cm_inv=_WATER_RAMAN_FWHM_CM_INV, relative_areas=_WATER_RAMAN_RELATIVE_AREAS):
    x=_as_f64(shift_cm_inv); centers=_as_f64(centers_cm_inv); fwhm=_as_f64(fwhm_cm_inv); areas=_as_f64(relative_areas)
    sigma=fwhm/(2.0*math.sqrt(2.0*math.log(2.0))); an=areas/np.sum(areas)
    z=(x[...,None]-centers[None,...])/sigma[None,...]
    return np.sum(an[None,...]*np.exp(-0.5*z*z)/(math.sqrt(2.0*math.pi)*sigma[None,...]),axis=-1)


def water_raman_shift_quadrature(*, n_nodes=8, shift_min_cm_inv=2700.0, shift_max_cm_inv=3900.0, depolarization_model="constant_0p20"):
    x,w=np.polynomial.legendre.leggauss(int(n_nodes)); half=0.5*(shift_max_cm_inv-shift_min_cm_inv); mid=0.5*(shift_max_cm_inv+shift_min_cm_inv)
    shift=mid+half*x; weight=half*w*water_raman_shift_pdf_cm(shift); weight/=np.sum(weight)
    rho=water_raman_depolarization_ratio(shift,model=depolarization_model)
    return RamanShiftQuadrature(*(np.ascontiguousarray(a,dtype=np.float64) for a in (shift,weight,rho)))


def raman_scattered_wavelength_nm(wavelength_in_nm, shift_cm_inv):
    wl,shift=np.broadcast_arrays(_as_f64(wavelength_in_nm),_as_f64(shift_cm_inv)); nu=1e7/wl-shift
    return np.divide(1e7,nu,out=np.full_like(nu,np.inf),where=nu>0.0)


def polarized_rayleigh_phase_sr_inv(polarization_dot_outgoing, depolarization_ratio=0.0):
    x=np.clip(_as_f64(polarization_dot_outgoing),-1.0,1.0)
    d=np.broadcast_to(_as_f64(depolarization_ratio),np.broadcast(x,depolarization_ratio).shape); x=np.broadcast_to(x,d.shape)
    f=3.0*d/(2.0+d)
    return (1.0-f)*3.0*(1.0-x*x)/(8.0*math.pi)+f/(4.0*math.pi)


def depolarized_dipole_phase_sr_inv(polarization_dot_outgoing, depolarization_ratio):
    return polarized_rayleigh_phase_sr_inv(polarization_dot_outgoing,depolarization_ratio)


def total_water_scattering_coefficient_m_inv(wavelength_nm, *, water_state=WaterState(), rayleigh_depolarization_model="spectral_2026", include_rayleigh=True, include_raman=True):
    wl=_as_f64(wavelength_nm); out=np.zeros_like(wl)
    if include_rayleigh: out += water_rayleigh_scattering_coefficient_m_inv(wl,water_state=water_state,depolarization_model=rayleigh_depolarization_model)
    if include_raman: out += water_raman_scattering_coefficient_m_inv(wl)
    return out


def validate_raman_quadrature(n_nodes=24):
    q=water_raman_shift_quadrature(n_nodes=n_nodes); mean=float(np.sum(q.normalized_weight*q.shift_cm_inv))
    return {"weight_sum":float(np.sum(q.normalized_weight)),"mean_shift_cm_inv":mean,"rms_shift_cm_inv":float(np.sqrt(np.sum(q.normalized_weight*(q.shift_cm_inv-mean)**2)))}
