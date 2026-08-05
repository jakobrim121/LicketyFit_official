"""Physics-grounded first-interaction photon transport for LicketyFit.

This is the production-candidate Rayleigh + water-Raman first-interaction
transport. It uses wavelength-dependent water physics, direct zero-interaction
survival, distributed interaction positions, Cherenkov polarization, the exact
16-sided WCTE water prism/endcaps, incident mPMT-dome interception, finite PMT
apertures, and source-resolved arrival-time nodes.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import math
import os
import hashlib
from pathlib import Path
from typing import Callable, NamedTuple
import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads

try:
    from .photon_scattering_native import (
        accumulate_fused_selected_native, native_receiver_available,
        ensure_native_receiver_built,
    )
except ImportError:
    try:
        from photon_scattering_native import (
            accumulate_fused_selected_native, native_receiver_available,
            ensure_native_receiver_built,
        )
    except ImportError:
        accumulate_fused_selected_native = None
        native_receiver_available = lambda: False
        ensure_native_receiver_built = lambda required=False: None

try:
    from .photon_scattering_physics import (
        C_MM_PER_NS, WaterState, relative_r14374_qe,
        water_refractive_index_iapws, water_group_index_iapws,
        water_rayleigh_scattering_coefficient_m_inv, water_depolarization_ratio,
        water_raman_scattering_coefficient_m_inv, water_raman_shift_quadrature,
        raman_scattered_wavelength_nm, total_water_scattering_coefficient_m_inv,
    )
except ImportError:
    from photon_scattering_physics import (
        C_MM_PER_NS, WaterState, relative_r14374_qe,
        water_refractive_index_iapws, water_group_index_iapws,
        water_rayleigh_scattering_coefficient_m_inv, water_depolarization_ratio,
        water_raman_scattering_coefficient_m_inv, water_raman_shift_quadrature,
        raman_scattered_wavelength_nm, total_water_scattering_coefficient_m_inv,
    )

CHANNEL_RAYLEIGH=np.int8(0)
CHANNEL_RAMAN=np.int8(1)

# -----------------------------------------------------------------------------
# Deterministic spectral-moment acceleration tables
# -----------------------------------------------------------------------------
# These tables are generated from the same analytic water/QE/Rayleigh/Raman
# functions used above. They are not WCSim-derived. The lookup is enabled only
# for the exact configuration encoded in the table metadata; all other
# configurations fall back to the direct spectral calculation.
_SPECTRAL_MOMENT_LUT_CACHE = {}
_DEFAULT_SPECTRAL_MOMENT_LUT_PATH = Path(__file__).resolve().parents[1] / "tables" / "photon_scatter_spectral_moment_lut_v1.npz"
_SPECTRAL_PARAM_NAMES = (
    "hazard_mm_inv", "incident_b_mm_inv", "outgoing_b_mm_inv",
    "n_phase", "n_group_in", "n_group_out", "depolarization",
    "wavelength_in_nm", "wavelength_out_nm",
)

def _spectral_moment_lut_path():
    raw = os.environ.get("EMITTER_PHOTON_SCATTER_SPECTRAL_LUT", "").strip()
    return Path(raw).expanduser() if raw else _DEFAULT_SPECTRAL_MOMENT_LUT_PATH

def _spectral_lut_matches_config(payload, config):
    try:
        return (
            str(np.asarray(payload["spectral_mode"]).item()).lower() == "moment"
            and int(np.asarray(payload["n_wavelength_nodes"]).item()) == int(config.n_wavelength_nodes)
            and int(np.asarray(payload["n_raman_shift_nodes"]).item()) == int(config.n_raman_shift_nodes)
            and abs(float(np.asarray(payload["wavelength_min_nm"]).item()) - float(config.wavelength_min_nm)) < 1e-12
            and abs(float(np.asarray(payload["wavelength_max_nm"]).item()) - float(config.wavelength_max_nm)) < 1e-12
            and str(config.detector_response) == "r14374_relative_qe"
            and str(config.rayleigh_depolarization_model) == "spectral_2026"
            and str(config.raman_depolarization_model) == "constant_0p20"
            and bool(config.enable_rayleigh)
            and bool(config.enable_raman)
            and abs(float(config.water_state.temperature_K) - 293.15) < 1e-9
            and abs(float(config.water_state.density_kg_m3) - 998.2071) < 1e-6
        )
    except Exception:
        return False

def _load_spectral_moment_lut(config):
    path = _spectral_moment_lut_path()
    key = (str(path), int(config.n_wavelength_nodes), int(config.n_raman_shift_nodes))
    if key in _SPECTRAL_MOMENT_LUT_CACHE:
        return _SPECTRAL_MOMENT_LUT_CACHE[key]
    if not path.is_file():
        _SPECTRAL_MOMENT_LUT_CACHE[key] = None
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if not _spectral_lut_matches_config(z, config):
                out = None
            else:
                names = tuple(str(x) for x in np.asarray(z["param_names"]).tolist())
                if names != _SPECTRAL_PARAM_NAMES:
                    out = None
                else:
                    out = {
                        "beta": np.ascontiguousarray(z["beta_grid"], dtype=np.float64),
                        "valid": np.ascontiguousarray(z["channel_valid"], dtype=np.uint8),
                        "params": np.ascontiguousarray(z["channel_params"], dtype=np.float64),
                        "direct_beta": np.ascontiguousarray(z["direct_beta_grid"], dtype=np.float64),
                        "direct_path": np.ascontiguousarray(z["direct_path_grid_mm"], dtype=np.float64),
                        "direct_survival": np.ascontiguousarray(z["direct_survival"], dtype=np.float64),
                        "direct_group": np.ascontiguousarray(z["direct_group_index"], dtype=np.float64),
                    }
    except Exception:
        out = None
    _SPECTRAL_MOMENT_LUT_CACHE[key] = out
    return out

def _interpolate_channel_arrays_from_lut(betas, config):
    lut = _load_spectral_moment_lut(config)
    if lut is None:
        return None
    b = np.asarray(betas, dtype=np.float64)
    grid = lut["beta"]
    idx = np.searchsorted(grid, b, side="right") - 1
    idx = np.clip(idx, 0, grid.size - 2)
    frac = np.divide(
        b - grid[idx], grid[idx + 1] - grid[idx],
        out=np.zeros_like(b), where=(grid[idx + 1] > grid[idx]),
    )
    frac = np.clip(frac, 0.0, 1.0)
    p0 = lut["params"][idx]
    p1 = lut["params"][idx + 1]
    params = p0 + frac[:, None, None] * (p1 - p0)
    # Channel validity is equivalent to positive interpolated hazard. This keeps
    # the threshold transition continuous and avoids a boolean staircase.
    valid = params[:, :, 0] > 0.0
    return valid, params

@njit(cache=True, fastmath=True)
def _bilinear_direct_lut_numba(beta, path, bg, rg, survival_table, group_table):
    n = beta.size
    out_s = np.empty(n, dtype=np.float64)
    out_g = np.empty(n, dtype=np.float64)
    nb = bg.size
    nr = rg.size
    r0 = rg[0]
    dr = (rg[nr - 1] - r0) / (nr - 1)
    for k in range(n):
        b = beta[k]
        r = path[k]
        lo = 0
        hi = nb
        while lo < hi:
            mid = (lo + hi) // 2
            if bg[mid] < b:
                lo = mid + 1
            else:
                hi = mid
        ib = lo - 1
        if ib < 0:
            ib = 0
        elif ib > nb - 2:
            ib = nb - 2
        den = bg[ib + 1] - bg[ib]
        tb = (b - bg[ib]) / den if den > 0.0 else 0.0
        fr = (r - r0) / dr
        ir = int(math.floor(fr))
        if ir < 0:
            ir = 0
        elif ir > nr - 2:
            ir = nr - 2
        tr = fr - ir
        if tr < 0.0:
            tr = 0.0
        elif tr > 1.0:
            tr = 1.0
        s00 = survival_table[ib, ir]
        s10 = survival_table[ib + 1, ir]
        s01 = survival_table[ib, ir + 1]
        s11 = survival_table[ib + 1, ir + 1]
        g00 = group_table[ib, ir]
        g10 = group_table[ib + 1, ir]
        g01 = group_table[ib, ir + 1]
        g11 = group_table[ib + 1, ir + 1]
        out_s[k] = (1.0 - tb) * ((1.0 - tr) * s00 + tr * s01) + tb * ((1.0 - tr) * s10 + tr * s11)
        out_g[k] = (1.0 - tb) * ((1.0 - tr) * g00 + tr * g01) + tb * ((1.0 - tr) * g10 + tr * g11)
    return out_s, out_g

def _bilinear_direct_lut(beta, path_length_mm, config):
    lut = _load_spectral_moment_lut(config)
    if lut is None:
        return None
    b, r = np.broadcast_arrays(np.asarray(beta, dtype=np.float64), np.asarray(path_length_mm, dtype=np.float64))
    flat_b = b.ravel(); flat_r = r.ravel()
    bg = lut["direct_beta"]; rg = lut["direct_path"]
    inside = (flat_b >= bg[0]) & (flat_b <= bg[-1]) & (flat_r >= rg[0]) & (flat_r <= rg[-1])
    if not np.all(inside):
        return None
    out_s, out_g = _bilinear_direct_lut_numba(
        np.ascontiguousarray(flat_b), np.ascontiguousarray(flat_r),
        bg, rg, lut["direct_survival"], lut["direct_group"],
    )
    return out_s.reshape(b.shape), out_g.reshape(b.shape)

@dataclass(frozen=True)
class WCTEPrism:
    n_sides:int=16
    apothem_mm:float=3075.926/2.0
    height_mm:float=2714.235
    y_center_mm:float=424.763
    @property
    def y_min_mm(self): return self.y_center_mm-0.5*self.height_mm
    @property
    def y_max_mm(self): return self.y_center_mm+0.5*self.height_mm

@dataclass(frozen=True)
class WCTEScatteringGeometry:
    dome_centres_mm:np.ndarray
    dome_axes:np.ndarray
    dome_slot_ids:np.ndarray
    prism:WCTEPrism=WCTEPrism()
    dome_outer_radius_mm:float=347.0
    dome_cap_cut_mm:float=235.0
    boundary_plane_points_mm:np.ndarray|None=None
    boundary_inward_normals:np.ndarray|None=None
    is_wcte_like:bool=True
    @classmethod
    def from_wcd(cls,wcd,*,prism=WCTEPrism()):
        centres=[];axes=[];slots=[];locations=[];raw_axes=[]; cyl=2.0*77.785
        for slot,mpmt in enumerate(getattr(wcd,"mpmts",[])):
            if mpmt is None: continue
            pl=mpmt.get_placement("design",wcd)
            a=np.asarray(pl["direction_z"],dtype=np.float64);a/=max(float(np.linalg.norm(a)),1e-300)
            loc=np.asarray(pl["location"],dtype=np.float64)
            locations.append(loc);raw_axes.append(a);slots.append(int(slot))
        if locations:
            loc_array=np.asarray(locations,dtype=np.float64)
            axis_array=np.asarray(raw_axes,dtype=np.float64)
            centre=np.median(loc_array,axis=0)
            flip=np.einsum("ij,ij->i",centre[None,:]-loc_array,axis_array)<0.0
            axis_array[flip]*=-1.0
            for loc,a in zip(loc_array,axis_array):
                centres.append(loc+(cyl-cls.dome_cap_cut_mm)*a);axes.append(a)
        else:
            loc_array=np.empty((0,3),dtype=np.float64)
            axis_array=np.empty((0,3),dtype=np.float64)
        label=(str(getattr(wcd,"name",""))+" "+str(getattr(wcd,"kind",""))).lower()
        labelled_wcte="wcte" in label
        extent_wcte=False
        if loc_array.size:
            lo=np.min(loc_array,axis=0);hi=np.max(loc_array,axis=0)
            span=hi-lo;bounding_center=0.5*(lo+hi)
            extent_wcte=bool(
                95<=loc_array.shape[0]<=110
                and 3250.0<=span[0]<=3550.0
                and 2850.0<=span[1]<=3250.0
                and 3250.0<=span[2]<=3550.0
                and abs(float(span[0]-span[2]))<=120.0
                and 300.0<=float(bounding_center[1])<=550.0
            )
        # The hard-coded WCTE prism must never be authorized by a label alone.
        is_wcte=bool(
            extent_wcte
            and (labelled_wcte or len(getattr(wcd,"mpmts",[]))==106)
        )
        return cls(
            np.ascontiguousarray(centres,dtype=np.float64),
            np.ascontiguousarray(axes,dtype=np.float64),
            np.ascontiguousarray(slots,dtype=np.int32),
            prism=prism,
            boundary_plane_points_mm=np.ascontiguousarray(loc_array,dtype=np.float64),
            boundary_inward_normals=np.ascontiguousarray(axis_array,dtype=np.float64),
            is_wcte_like=is_wcte,
        )
    def receiver_dome_arrays(self,pmt_slots):
        lookup={int(s):i for i,s in enumerate(self.dome_slot_ids)}; slots=np.asarray(pmt_slots,dtype=np.int32)
        c=np.empty((slots.size,3));a=np.empty_like(c)
        for i,s in enumerate(slots):
            j=lookup.get(int(s));
            if j is None: c[i]=np.nan;a[i]=np.nan
            else: c[i]=self.dome_centres_mm[j];a[i]=self.dome_axes[j]
        return np.ascontiguousarray(c),np.ascontiguousarray(a)
    def distance_to_boundary_mm(self,source_mm,direction,*,include_domes=True,boundary_model="auto"):
        source=np.asarray(source_mm,dtype=np.float64);k=np.asarray(direction,dtype=np.float64);k/=max(float(np.linalg.norm(k)),1e-300)
        mode=str(boundary_model).strip().lower().replace("-","_")
        if mode=="auto":mode="wcte_prism" if self.is_wcte_like else "convex_mpmt_planes"
        if mode=="convex_mpmt_planes" and self.boundary_plane_points_mm is not None:
            points=np.asarray(self.boundary_plane_points_mm,dtype=np.float64)
            normals=np.asarray(self.boundary_inward_normals,dtype=np.float64)
            signed=np.einsum("ij,ij->i",source[None,:]-points,normals)
            # A source outside any defining half-space is not a physical photon
            # source for this detector. Return zero immediately rather than
            # intersecting whichever other planes happen to face the ray.
            if np.any(signed < -1e-6):
                return 0.0
            den=normals@k
            valid=den<-1e-14
            t=np.divide(-signed,den,out=np.full_like(signed,np.inf),where=valid)
            t=t[(t>1e-10)&np.isfinite(t)]
            best=float(np.min(t)) if t.size else 0.0
        else:
            best=distance_to_regular_prism_boundary_mm(source,k,self.prism)
        if (not include_domes) or self.dome_centres_mm.size==0:return float(best)
        q=source[None,:]-self.dome_centres_mm;b=q@k;c=np.einsum("ij,ij->i",q,q)-self.dome_outer_radius_mm**2
        disc=b*b-c;good=disc>=0.0
        if not np.any(good):return float(best)
        root=np.sqrt(np.maximum(disc,0.0))
        for t in (-b-root,-b+root):
            valid=good&(t>1e-7)
            if not np.any(valid):continue
            hit=source[None,:]+t[:,None]*k[None,:]
            valid &= np.einsum("ij,ij->i",hit-self.dome_centres_mm,self.dome_axes)>=self.dome_cap_cut_mm-1e-6
            if np.any(valid):
                cand=float(np.min(t[valid]));
                if best<=0.0 or cand<best:best=cand
        return float(best)

@dataclass(frozen=True)
class PhotonScatteringTransportConfig:
    n_track_nodes:int=6
    n_azimuth_nodes:int=12
    n_scatter_nodes:int=6
    n_wavelength_nodes:int=6
    n_raman_shift_nodes:int=8
    n_timing_bins:int=16
    spectral_mode:str="full"
    pmt_aperture_radius_mm:float=45.0
    pmt_facing_soft_width:float=0.02
    pmt_response_model:str="legacy_power"
    reflector_model:str="spherical"
    reflector_reflectivity:float=1.0
    reflector_active_radius_mm:float=36.0
    reflector_sphere_radius_mm:float=53.0
    reflector_side_angle_deg:float=31.7
    reflector_lut_size:int=401
    wavelength_min_nm:float=300.0
    wavelength_max_nm:float=660.0
    detector_response:str="r14374_relative_qe"
    rayleigh_depolarization_model:str="spectral_2026"
    raman_depolarization_model:str="constant_0p20"
    enable_rayleigh:bool=True
    enable_raman:bool=True
    parallel_pmt_loop:bool=False
    native_receiver:bool=False
    native_receiver_threads:int=1
    native_receiver_required:bool=False
    include_mpmt_domes:bool=True
    enforce_receiver_dome_visibility:bool=False
    receiver_mode:str="sparse_moment"
    receiver_moment_table_path:str=""
    receiver_moment_table_required:bool=False
    receiver_dome_outer_radius_mm:float=347.0
    receiver_dome_cap_cut_mm:float=235.0
    boundary_model:str="auto"
    water_state:WaterState=WaterState()
    prism:WCTEPrism=WCTEPrism()
    def validate(self):
        for name in ("n_track_nodes","n_azimuth_nodes","n_scatter_nodes","n_wavelength_nodes","n_raman_shift_nodes","n_timing_bins"):
            if int(getattr(self,name))<1:raise ValueError(f"{name} must be positive")
        if self.n_wavelength_nodes<2 or self.n_raman_shift_nodes<2:raise ValueError("spectral quadratures need >=2 nodes")
        if str(self.spectral_mode).lower() not in {"full","moment","moments","effective"}:raise ValueError("invalid spectral_mode")
        if self.pmt_aperture_radius_mm<=0.0:raise ValueError("invalid aperture")
        if not (self.enable_rayleigh or self.enable_raman):raise ValueError("no scattering channel enabled")
        if str(self.receiver_mode).strip().lower() not in {"exact","exact_all","all_pmts","sparse_moment","sparse","moment"}:raise ValueError("invalid receiver_mode")
        if str(self.boundary_model).strip().lower().replace("-","_") not in {"auto","wcte_prism","convex_mpmt_planes"}:raise ValueError("invalid boundary_model")
        self.water_state.validate()

class PhotonScatterNodes(NamedTuple):
    position_mm:np.ndarray
    incident_direction:np.ndarray
    polarization:np.ndarray
    charge_weight:np.ndarray
    channel:np.ndarray
    outgoing_scattering_coefficient_mm_inv:np.ndarray
    depolarization_ratio:np.ndarray
    outgoing_group_index:np.ndarray
    base_time_ns:np.ndarray
    wavelength_in_nm:np.ndarray
    wavelength_out_nm:np.ndarray
    source_coordinate_mm:np.ndarray
    incident_path_mm:np.ndarray

class PhotonScatterPrediction(NamedTuple):
    charge:np.ndarray
    rayleigh_charge:np.ndarray
    raman_charge:np.ndarray
    timing_node_charge:np.ndarray|None
    timing_node_time_ns:np.ndarray|None
    timing_active_indices:np.ndarray|None
    timing_bin_edges_ns:np.ndarray|None

@lru_cache(maxsize=32)
def _leggauss_cached(n):
    x,w=np.polynomial.legendre.leggauss(int(n));return np.ascontiguousarray(x),np.ascontiguousarray(w)

def _stable_transverse_basis(direction):
    d=np.asarray(direction,dtype=np.float64);d/=max(float(np.linalg.norm(d)),1e-300)
    ref=np.array([1.,0.,0.]) if abs(d[0])<0.8 else np.array([0.,1.,0.])
    e1=np.cross(d,ref);e1/=max(float(np.linalg.norm(e1)),1e-300);e2=np.cross(d,e1);e2/=max(float(np.linalg.norm(e2)),1e-300)
    return e1,e2

def distance_to_regular_prism_boundary_mm(source_mm,direction,prism=WCTEPrism()):
    x=np.asarray(source_mm,dtype=np.float64);k=np.asarray(direction,dtype=np.float64);k/=max(float(np.linalg.norm(k)),1e-300);cand=[]
    for j in range(prism.n_sides):
        ang=2.0*math.pi*j/prism.n_sides;nx,nz=math.cos(ang),math.sin(ang);den=nx*k[0]+nz*k[2]
        if den>1e-14:
            t=(prism.apothem_mm-nx*x[0]-nz*x[2])/den
            if t>1e-10:
                y=x[1]+t*k[1]
                if prism.y_min_mm-1e-7<=y<=prism.y_max_mm+1e-7:cand.append(float(t))
    for ycap,sgn in ((prism.y_max_mm,1.0),(prism.y_min_mm,-1.0)):
        if sgn*k[1]<=1e-14:continue
        t=(ycap-x[1])/k[1]
        if t<=1e-10:continue
        p=x+t*k;inside=True
        for j in range(prism.n_sides):
            ang=2.0*math.pi*j/prism.n_sides
            if math.cos(ang)*p[0]+math.sin(ang)*p[2]>prism.apothem_mm+1e-7:inside=False;break
        if inside:cand.append(float(t))
    return min(cand) if cand else 0.0

def _frank_tamm_integrated(beta,n_reference=1.344):
    if beta<=0.0 or n_reference*beta<=1.0:return 0.0
    return max((1.0-1.0/(n_reference*n_reference*beta*beta))/(1.0-1.0/(n_reference*n_reference)),0.0)

def _empty_nodes():
    e3=np.empty((0,3));e1=np.empty(0)
    return PhotonScatterNodes(e3,e3.copy(),e3.copy(),e1,np.empty(0,dtype=np.int8),e1.copy(),e1.copy(),e1.copy(),e1.copy(),e1.copy(),e1.copy(),e1.copy(),e1.copy())

@lru_cache(maxsize=64)
def _static_spectral_arrays(n_nodes,wmin,wmax,detector_response,dep_model,T,rho,kappa,enable_rayleigh,enable_raman):
    x,w=_leggauss_cached(int(n_nodes));half=0.5*(wmax-wmin);mid=0.5*(wmax+wmin);wl=mid+half*x;qw=half*w
    npidx=water_refractive_index_iapws(wl,temperature_K=T,density_kg_m3=rho);ng=water_group_index_iapws(wl,temperature_K=T,density_kg_m3=rho)
    resp=relative_r14374_qe(wl) if detector_response=="r14374_relative_qe" else np.ones_like(wl)
    st=WaterState(T,rho,kappa)
    br=water_rayleigh_scattering_coefficient_m_inv(wl,water_state=st,depolarization_model=dep_model) if enable_rayleigh else np.zeros_like(wl)
    bm=water_raman_scattering_coefficient_m_inv(wl) if enable_raman else np.zeros_like(wl)
    dep=water_depolarization_ratio(wl,model=dep_model)
    return tuple(np.ascontiguousarray(a,dtype=np.float64) for a in (wl,qw,resp,npidx,ng,br,bm,dep))

def _detected_spectrum_matrix(beta,config):
    b=np.asarray(beta,dtype=np.float64);flat=b.ravel();st=config.water_state
    wl,qw,resp,npidx,ng,br,bm,dep=_static_spectral_arrays(config.n_wavelength_nodes,config.wavelength_min_nm,config.wavelength_max_nm,config.detector_response,config.rayleigh_depolarization_model,st.temperature_K,st.density_kg_m3,st.isothermal_compressibility_Pa_inv,config.enable_rayleigh,config.enable_raman)
    ft=np.maximum(1.0-1.0/np.maximum(flat[:,None]**2*npidx[None,:]**2,1e-300),0.0)
    spec=qw[None,:]*resp[None,:]*ft/(wl[None,:]**2);total=np.sum(spec,axis=1)
    weights=np.divide(spec,total[:,None],out=np.zeros_like(spec),where=total[:,None]>0.0)
    return weights,(wl,npidx,ng,br,bm,dep),b.shape


@lru_cache(maxsize=32)
def _raman_shift_quadrature_cached(n_nodes, depolarization_model):
    """Cache the detector-independent Raman shift quadrature.

    The previous hot path rebuilt the same Legendre rule for every longitudinal
    source point and every FCN.  The returned arrays are immutable in use and
    depend only on the configured node count and depolarization model.
    """
    q = water_raman_shift_quadrature(
        n_nodes=int(n_nodes),
        depolarization_model=str(depolarization_model),
    )
    return tuple(
        np.ascontiguousarray(a, dtype=np.float64)
        for a in (q.shift_cm_inv, q.normalized_weight, q.depolarization_ratio)
    )

def _moment_channel_parameters(beta,config):
    weights,arrays,_=_detected_spectrum_matrix(np.asarray([float(beta)]),config);w=weights[0]
    wl,npidx,ng,brm,bmm,dep=arrays;br=brm/1000.0;bm=bmm/1000.0;btot=br+bm;out={}
    h=w*br;hs=float(np.sum(h))
    if config.enable_rayleigh and hs>0.0:
        hn=h/hs;out[int(CHANNEL_RAYLEIGH)]={
            "hazard_mm_inv":hs,"incident_b_mm_inv":float(np.sum(hn*btot)),"outgoing_b_mm_inv":float(np.sum(hn*btot)),
            "n_phase":float(np.sum(hn*npidx)),"n_group_in":float(np.sum(hn*ng)),"n_group_out":float(np.sum(hn*ng)),
            "depolarization":float(np.sum(hn*dep)),"wavelength_in_nm":float(np.sum(hn*wl)),"wavelength_out_nm":float(np.sum(hn*wl)),}
    if config.enable_raman:
        _qshift, _qweight, _qdep = _raman_shift_quadrature_cached(
            config.n_raman_shift_nodes, config.raman_depolarization_model
        )
        wlo=raman_scattered_wavelength_nm(wl[:,None],_qshift[None,:]);qei=relative_r14374_qe(wl)[:,None];qeo=relative_r14374_qe(wlo)
        ratio=np.divide(qeo,qei,out=np.zeros_like(qeo),where=qei>0.0)
        branch=w[:,None]*bm[:,None]*_qweight[None,:]*ratio;hs=float(np.sum(branch))
        if hs>0.0:
            bn=branch/hs;ngo=water_group_index_iapws(wlo,temperature_K=config.water_state.temperature_K,density_kg_m3=config.water_state.density_kg_m3)
            bo=total_water_scattering_coefficient_m_inv(wlo,water_state=config.water_state,rayleigh_depolarization_model=config.rayleigh_depolarization_model,include_rayleigh=config.enable_rayleigh,include_raman=config.enable_raman)/1000.0
            out[int(CHANNEL_RAMAN)]={
                "hazard_mm_inv":hs,"incident_b_mm_inv":float(np.sum(bn*btot[:,None])),"outgoing_b_mm_inv":float(np.sum(bn*bo)),
                "n_phase":float(np.sum(bn*npidx[:,None])),"n_group_in":float(np.sum(bn*ng[:,None])),"n_group_out":float(np.sum(bn*ngo)),
                "depolarization":float(np.sum(bn*_qdep[None,:])),"wavelength_in_nm":float(np.sum(bn*wl[:,None])),"wavelength_out_nm":float(np.sum(bn*wlo)),}
    return out


@njit(cache=True)
def _distance_to_boundary_fast_numba(
    sx, sy, sz, kx, ky, kz,
    face_nx, face_nz,
    n_sides, apothem_mm, y_min_mm, y_max_mm,
    boundary_points, boundary_normals, use_convex_planes,
    dome_centres, dome_axes, dome_radius_mm, dome_cap_cut_mm,
    include_domes,
):
    """Distance to either the exact WCTE prism or a convex plane envelope."""
    best = 0.0
    if use_convex_planes:
        for j in range(boundary_points.shape[0]):
            nx = boundary_normals[j, 0]
            ny = boundary_normals[j, 1]
            nz = boundary_normals[j, 2]
            signed = (
                (sx - boundary_points[j, 0]) * nx
                + (sy - boundary_points[j, 1]) * ny
                + (sz - boundary_points[j, 2]) * nz
            )
            # A negative signed distance means the source is already outside
            # the conservative convex envelope; do not create scatter nodes.
            if signed < -1.0e-6:
                return 0.0
            den = nx * kx + ny * ky + nz * kz
            if den < -1.0e-14:
                t = -signed / den
                if t > 1.0e-10 and (best <= 0.0 or t < best):
                    best = t
    else:
        for j in range(n_sides):
            nx = face_nx[j]
            nz = face_nz[j]
            den = nx * kx + nz * kz
            if den > 1.0e-14:
                t = (apothem_mm - nx * sx - nz * sz) / den
                if t > 1.0e-10:
                    y = sy + t * ky
                    if y_min_mm - 1.0e-7 <= y <= y_max_mm + 1.0e-7:
                        if best <= 0.0 or t < best:
                            best = t

        if ky > 1.0e-14:
            t = (y_max_mm - sy) / ky
            if t > 1.0e-10:
                px = sx + t * kx
                pz = sz + t * kz
                inside = True
                for j in range(n_sides):
                    if face_nx[j] * px + face_nz[j] * pz > apothem_mm + 1.0e-7:
                        inside = False
                        break
                if inside and (best <= 0.0 or t < best):
                    best = t
        elif ky < -1.0e-14:
            t = (y_min_mm - sy) / ky
            if t > 1.0e-10:
                px = sx + t * kx
                pz = sz + t * kz
                inside = True
                for j in range(n_sides):
                    if face_nx[j] * px + face_nz[j] * pz > apothem_mm + 1.0e-7:
                        inside = False
                        break
                if inside and (best <= 0.0 or t < best):
                    best = t

    if include_domes:
        nd = dome_centres.shape[0]
        rr = dome_radius_mm * dome_radius_mm
        for m in range(nd):
            qx = sx - dome_centres[m, 0]
            qy = sy - dome_centres[m, 1]
            qz = sz - dome_centres[m, 2]
            b = qx * kx + qy * ky + qz * kz
            c = qx * qx + qy * qy + qz * qz - rr
            disc = b * b - c
            if disc < 0.0:
                continue
            root = math.sqrt(disc)
            t1 = -b - root
            t2 = -b + root
            for ir in range(2):
                t = t1 if ir == 0 else t2
                if t <= 1.0e-7:
                    continue
                hx = sx + t * kx - dome_centres[m, 0]
                hy = sy + t * ky - dome_centres[m, 1]
                hz = sz + t * kz - dome_centres[m, 2]
                cap = (
                    hx * dome_axes[m, 0]
                    + hy * dome_axes[m, 1]
                    + hz * dome_axes[m, 2]
                )
                if cap >= dome_cap_cut_mm - 1.0e-6:
                    if best <= 0.0 or t < best:
                        best = t
    return best


@njit(cache=True)
def _build_nodes_moment_numba(
    start, direction, e1, e2,
    sn, sw, betas, particle_time, ft_weight,
    phis, scatter_x, scatter_w,
    valid_channel, hazard, incident_b, outgoing_b,
    n_phase, n_group_in, n_group_out, depolarization,
    wavelength_in, wavelength_out,
    intensity, primary_ngeo_normalization, start_time_ns,
    face_nx, face_nz, n_sides, apothem_mm, y_min_mm, y_max_mm,
    boundary_points, boundary_normals, use_convex_planes,
    dome_centres, dome_axes, dome_radius_mm, dome_cap_cut_mm,
    include_domes,
):
    nt = sn.size
    nphi = phis.size
    nu = scatter_x.size
    max_nodes = nt * 2 * nphi * nu
    position = np.empty((max_nodes, 3), dtype=np.float64)
    incident = np.empty((max_nodes, 3), dtype=np.float64)
    polarization = np.empty((max_nodes, 3), dtype=np.float64)
    weight = np.empty(max_nodes, dtype=np.float64)
    channel_out = np.empty(max_nodes, dtype=np.int8)
    outgoing_coeff = np.empty(max_nodes, dtype=np.float64)
    dep_out = np.empty(max_nodes, dtype=np.float64)
    ng_out = np.empty(max_nodes, dtype=np.float64)
    base_time = np.empty(max_nodes, dtype=np.float64)
    wl_in_out = np.empty(max_nodes, dtype=np.float64)
    wl_out_out = np.empty(max_nodes, dtype=np.float64)
    source_s = np.empty(max_nodes, dtype=np.float64)
    incident_path = np.empty(max_nodes, dtype=np.float64)
    count = 0
    dphi = 2.0 * math.pi / nphi

    dx0 = direction[0]
    dy0 = direction[1]
    dz0 = direction[2]

    for it in range(nt):
        beta = betas[it]
        ft = ft_weight[it]
        if not (0.0 < beta <= 1.0) or ft <= 0.0:
            continue
        s = sn[it]
        sx = start[0] + s * dx0
        sy = start[1] + s * dy0
        sz = start[2] + s * dz0
        pref = intensity * primary_ngeo_normalization * ft * sw[it] * dphi

        for ch in range(2):
            if valid_channel[it, ch] == 0:
                continue
            ctc = 1.0 / (beta * n_phase[it, ch])
            if ctc >= 1.0:
                continue
            if ctc < -1.0:
                ctc = -1.0
            stc = math.sqrt(max(1.0 - ctc * ctc, 0.0))
            for iphi in range(nphi):
                phi = phis[iphi]
                cp = math.cos(phi)
                sp = math.sin(phi)
                kx = ctc * dx0 + stc * cp * e1[0] + stc * sp * e2[0]
                ky = ctc * dy0 + stc * cp * e1[1] + stc * sp * e2[1]
                kz = ctc * dz0 + stc * cp * e1[2] + stc * sp * e2[2]
                kn = math.sqrt(kx * kx + ky * ky + kz * kz)
                if kn <= 0.0:
                    continue
                kx /= kn
                ky /= kn
                kz /= kn

                dk = dx0 * kx + dy0 * ky + dz0 * kz
                ex = dx0 - dk * kx
                ey = dy0 - dk * ky
                ez = dz0 - dk * kz
                en = math.sqrt(ex * ex + ey * ey + ez * ez)
                if en <= 0.0:
                    continue
                ex /= en
                ey /= en
                ez /= en

                T = _distance_to_boundary_fast_numba(
                    sx, sy, sz, kx, ky, kz,
                    face_nx, face_nz,
                    n_sides, apothem_mm, y_min_mm, y_max_mm,
                    boundary_points, boundary_normals, use_convex_planes,
                    dome_centres, dome_axes, dome_radius_mm, dome_cap_cut_mm,
                    include_domes,
                )
                if T <= 0.0:
                    continue
                halfT = 0.5 * T
                for iu in range(nu):
                    u = halfT * (scatter_x[iu] + 1.0)
                    du = halfT * scatter_w[iu]
                    amp = (
                        pref * hazard[it, ch] * du
                        * math.exp(-incident_b[it, ch] * u)
                    )
                    if amp <= 0.0:
                        continue
                    position[count, 0] = sx + u * kx
                    position[count, 1] = sy + u * ky
                    position[count, 2] = sz + u * kz
                    incident[count, 0] = kx
                    incident[count, 1] = ky
                    incident[count, 2] = kz
                    polarization[count, 0] = ex
                    polarization[count, 1] = ey
                    polarization[count, 2] = ez
                    weight[count] = amp
                    channel_out[count] = np.int8(ch)
                    outgoing_coeff[count] = outgoing_b[it, ch]
                    dep_out[count] = depolarization[it, ch]
                    ng_out[count] = n_group_out[it, ch]
                    base_time[count] = (
                        start_time_ns + particle_time[it]
                        + n_group_in[it, ch] * u / C_MM_PER_NS
                    )
                    wl_in_out[count] = wavelength_in[it, ch]
                    wl_out_out[count] = wavelength_out[it, ch]
                    source_s[count] = s
                    incident_path[count] = u
                    count += 1
    return (
        position, incident, polarization, weight, channel_out,
        outgoing_coeff, dep_out, ng_out, base_time,
        wl_in_out, wl_out_out, source_s, incident_path, count,
    )


def _build_photon_scatter_nodes_moment(*,start_position_mm,track_direction,visible_length_mm,beta_at_s,particle_time_at_s_ns,intensity,primary_ngeo_normalization,start_time_ns,config,geometry):
    L=float(visible_length_mm)
    if L<=0.0:return _empty_nodes()
    x0=np.asarray(start_position_mm,dtype=np.float64)
    d=np.asarray(track_direction,dtype=np.float64);d/=max(float(np.linalg.norm(d)),1e-300)
    e1,e2=_stable_transverse_basis(d)
    xs,ws=_leggauss_cached(config.n_track_nodes)
    sn=.5*L*(xs+1.0);sw=.5*L*ws
    betas=np.asarray(beta_at_s(sn),dtype=np.float64)
    tp=np.asarray(particle_time_at_s_ns(sn),dtype=np.float64)
    xu,wu=_leggauss_cached(config.n_scatter_nodes)
    phis=(np.arange(config.n_azimuth_nodes,dtype=np.float64)+.5)*2.0*math.pi/config.n_azimuth_nodes

    nt=sn.size
    valid=np.zeros((nt,2),dtype=np.uint8)
    hazard=np.zeros((nt,2),dtype=np.float64)
    incident_b=np.zeros((nt,2),dtype=np.float64)
    outgoing_b=np.zeros((nt,2),dtype=np.float64)
    n_phase=np.ones((nt,2),dtype=np.float64)
    n_group_in=np.ones((nt,2),dtype=np.float64)
    n_group_out=np.ones((nt,2),dtype=np.float64)
    dep=np.zeros((nt,2),dtype=np.float64)
    wlin=np.zeros((nt,2),dtype=np.float64)
    wlout=np.zeros((nt,2),dtype=np.float64)
    ft=np.zeros(nt,dtype=np.float64)
    _lut_channels = _interpolate_channel_arrays_from_lut(betas, config)
    if _lut_channels is not None:
        _valid_lut, _par_lut = _lut_channels
        valid[:] = _valid_lut.astype(np.uint8)
        hazard[:] = _par_lut[:, :, 0]
        incident_b[:] = _par_lut[:, :, 1]
        outgoing_b[:] = _par_lut[:, :, 2]
        n_phase[:] = _par_lut[:, :, 3]
        n_group_in[:] = _par_lut[:, :, 4]
        n_group_out[:] = _par_lut[:, :, 5]
        dep[:] = _par_lut[:, :, 6]
        wlin[:] = _par_lut[:, :, 7]
        wlout[:] = _par_lut[:, :, 8]
        for i, beta in enumerate(betas):
            if 0.0 < float(beta) <= 1.0:
                ft[i] = _frank_tamm_integrated(float(beta))
    else:
        for i,beta in enumerate(betas):
            if not (0.0<float(beta)<=1.0):
                continue
            ft[i]=_frank_tamm_integrated(float(beta))
            params=_moment_channel_parameters(float(beta),config)
            for ch,par in params.items():
                c=int(ch)
                valid[i,c]=1
                hazard[i,c]=par["hazard_mm_inv"]
                incident_b[i,c]=par["incident_b_mm_inv"]
                outgoing_b[i,c]=par["outgoing_b_mm_inv"]
                n_phase[i,c]=par["n_phase"]
                n_group_in[i,c]=par["n_group_in"]
                n_group_out[i,c]=par["n_group_out"]
                dep[i,c]=par["depolarization"]
                wlin[i,c]=par["wavelength_in_nm"]
                wlout[i,c]=par["wavelength_out_nm"]

    prism=config.prism
    face_ang=2.0*math.pi*np.arange(prism.n_sides,dtype=np.float64)/prism.n_sides
    face_nx=np.ascontiguousarray(np.cos(face_ang),dtype=np.float64)
    face_nz=np.ascontiguousarray(np.sin(face_ang),dtype=np.float64)
    boundary_mode=str(getattr(config,"boundary_model","auto")).strip().lower().replace("-","_")
    if geometry is None:
        dome_centres=np.empty((0,3),dtype=np.float64)
        dome_axes=np.empty((0,3),dtype=np.float64)
        boundary_points=np.empty((0,3),dtype=np.float64)
        boundary_normals=np.empty((0,3),dtype=np.float64)
        use_convex_planes=False
        dome_radius=347.0
        dome_cut=235.0
    else:
        dome_centres=np.ascontiguousarray(geometry.dome_centres_mm,dtype=np.float64)
        dome_axes=np.ascontiguousarray(geometry.dome_axes,dtype=np.float64)
        boundary_points=np.ascontiguousarray(
            np.empty((0,3),dtype=np.float64)
            if geometry.boundary_plane_points_mm is None
            else geometry.boundary_plane_points_mm,
            dtype=np.float64,
        )
        boundary_normals=np.ascontiguousarray(
            np.empty((0,3),dtype=np.float64)
            if geometry.boundary_inward_normals is None
            else geometry.boundary_inward_normals,
            dtype=np.float64,
        )
        if boundary_mode=="auto":
            use_convex_planes=not bool(getattr(geometry,"is_wcte_like",True))
        else:
            use_convex_planes=boundary_mode=="convex_mpmt_planes"
        if use_convex_planes and boundary_points.shape[0]==0:
            raise ValueError("convex_mpmt_planes requested but detector has no boundary planes")
        dome_radius=float(geometry.dome_outer_radius_mm)
        dome_cut=float(geometry.dome_cap_cut_mm)

    out=_build_nodes_moment_numba(
        np.ascontiguousarray(x0,dtype=np.float64),
        np.ascontiguousarray(d,dtype=np.float64),
        np.ascontiguousarray(e1,dtype=np.float64),
        np.ascontiguousarray(e2,dtype=np.float64),
        np.ascontiguousarray(sn,dtype=np.float64),
        np.ascontiguousarray(sw,dtype=np.float64),
        np.ascontiguousarray(betas,dtype=np.float64),
        np.ascontiguousarray(tp,dtype=np.float64),
        np.ascontiguousarray(ft,dtype=np.float64),
        np.ascontiguousarray(phis,dtype=np.float64),
        np.ascontiguousarray(xu,dtype=np.float64),
        np.ascontiguousarray(wu,dtype=np.float64),
        np.ascontiguousarray(valid,dtype=np.uint8),
        np.ascontiguousarray(hazard,dtype=np.float64),
        np.ascontiguousarray(incident_b,dtype=np.float64),
        np.ascontiguousarray(outgoing_b,dtype=np.float64),
        np.ascontiguousarray(n_phase,dtype=np.float64),
        np.ascontiguousarray(n_group_in,dtype=np.float64),
        np.ascontiguousarray(n_group_out,dtype=np.float64),
        np.ascontiguousarray(dep,dtype=np.float64),
        np.ascontiguousarray(wlin,dtype=np.float64),
        np.ascontiguousarray(wlout,dtype=np.float64),
        float(intensity),float(primary_ngeo_normalization),float(start_time_ns),
        face_nx,face_nz,int(prism.n_sides),float(prism.apothem_mm),
        float(prism.y_min_mm),float(prism.y_max_mm),
        boundary_points,boundary_normals,bool(use_convex_planes),
        dome_centres,dome_axes,dome_radius,dome_cut,
        bool(config.include_mpmt_domes),
    )
    count=int(out[-1])
    if count<=0:return _empty_nodes()
    arrays=out[:-1]
    return PhotonScatterNodes(
        *(
            np.ascontiguousarray(a[:count])
            for a in arrays
        )
    )

def build_photon_scatter_nodes(**kwargs):
    config=kwargs.get("config",PhotonScatteringTransportConfig());config.validate()
    # Production uses the validated spectral-moment representation.
    return _build_photon_scatter_nodes_moment(**kwargs)

@njit(cache=True,inline='always')
def _legacy_response(c):
    if c<0.0:c=0.0
    elif c>1.0:c=1.0
    cn=c**3.0777000000000001
    return (0.1209+(1.6396999999999999-0.1209)*(cn/(cn+0.79428866592713121)))/1.002379253316015

@njit(cache=True)
def _facing_visibility(c,width):
    if width<=0.0:return 1.0 if c>0.0 else 0.0
    if c<=-width:return 0.0
    if c>=width:return 1.0
    u=(c+width)/(2.0*width);return 3.0*u*u-2.0*u*u*u



# High-resolution deterministic lookup tables for the two expensive scalar
# functions in the node-PMT hot loop.  The optical model is unchanged.  At
# 65,536 samples, linear interpolation changes the complete scattered field by
# <3e-10 relative in the validation suite, far below floating-point/model
# precision, while roughly halving the serial transport cost.
_SCATTER_LUT_SIZE = 65536
_SCATTER_ATTENUATION_X_MAX = 0.125
_scatter_cost_grid = np.linspace(0.0, 1.0, _SCATTER_LUT_SIZE)
_scatter_cost_power = _scatter_cost_grid ** 3.0777000000000001
_SCATTER_RESPONSE_LUT = np.ascontiguousarray(
    (0.1209 + (1.6396999999999999 - 0.1209)
     * (_scatter_cost_power / (_scatter_cost_power + 0.79428866592713121)))
    / 1.002379253316015,
    dtype=np.float64,
)
_SCATTER_ATTENUATION_LUT = np.ascontiguousarray(
    np.exp(-np.linspace(0.0, _SCATTER_ATTENUATION_X_MAX, _SCATTER_LUT_SIZE)),
    dtype=np.float64,
)


@njit(cache=True, inline='always')
def _scatter_response_lut(c, lut):
    if c <= 0.0:
        return lut[0]
    if c >= 1.0:
        return lut[lut.size - 1]
    f = c * (lut.size - 1)
    i = int(f)
    t = f - i
    return lut[i] + t * (lut[i + 1] - lut[i])


@njit(cache=True, inline='always')
def _scatter_attenuation_lut(x, lut, xmax):
    if x <= 0.0:
        return 1.0
    if x >= xmax:
        return math.exp(-x)
    f = x * (lut.size - 1) / xmax
    i = int(f)
    if i >= lut.size - 1:
        return lut[lut.size - 1]
    t = f - i
    return lut[i] + t * (lut[i + 1] - lut[i])


@njit(cache=True, inline='always')
def _node_to_pmt_amplitude_lut(
    px, py, pz, nx, ny, nz,
    j, node_pos, node_pol, node_w, node_b, node_dep,
    a2, width, inv4pi, dip, response_lut, attenuation_lut, attenuation_xmax,
):
    dx = px - node_pos[j, 0]
    dy = py - node_pos[j, 1]
    dz = pz - node_pos[j, 2]
    r2 = dx * dx + dy * dy + dz * dz
    if r2 <= 1.0e-12:
        return 0.0, 0.0
    r = math.sqrt(r2)
    kx = dx / r
    ky = dy / r
    kz = dz / r
    facing = -(nx * kx + ny * ky + nz * kz)
    vis = _facing_visibility(facing, width)
    if vis <= 0.0:
        return 0.0, r
    pd = node_pol[j, 0] * kx + node_pol[j, 1] * ky + node_pol[j, 2] * kz
    if pd < -1.0:
        pd = -1.0
    elif pd > 1.0:
        pd = 1.0
    dlt = node_dep[j]
    fiso = 3.0 * dlt / (2.0 + dlt)
    phase = (1.0 - fiso) * dip * (1.0 - pd * pd) + fiso * inv4pi
    omega_per_area = 2.0 * (1.0 - r / math.sqrt(r2 + a2)) / a2
    response = _scatter_response_lut(facing, response_lut)
    attenuation = _scatter_attenuation_lut(
        node_b[j] * r, attenuation_lut, attenuation_xmax
    )
    amp = node_w[j] * phase * omega_per_area * response * vis * attenuation
    if amp <= 0.0 or not math.isfinite(amp):
        return 0.0, r
    return amp, r



@njit(cache=True, fastmath=True)
def _accumulate_charge_serial_lut(
    p, n, node_pos, node_pol, node_phase_a, node_phase_b, node_ch, node_b,
    a, width, response_lut, attenuation_lut, attenuation_xmax,
):
    npm = p.shape[0]
    nn = node_pos.shape[0]
    charge = np.zeros(npm)
    ray = np.zeros(npm)
    ram = np.zeros(npm)
    a2 = a * a
    inv4pi = 1.0 / (4.0 * math.pi)
    dip = 3.0 / (8.0 * math.pi)
    nl = response_lut.size
    ne = attenuation_lut.size
    for i in range(npm):
        px = p[i, 0]; py = p[i, 1]; pz = p[i, 2]
        nx = n[i, 0]; ny = n[i, 1]; nz = n[i, 2]
        acc = 0.0; ar = 0.0; am = 0.0
        for j in range(nn):
            dx = px - node_pos[j, 0]
            dy = py - node_pos[j, 1]
            dz = pz - node_pos[j, 2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 1.0e-12:
                continue
            r = math.sqrt(r2)
            kx = dx / r; ky = dy / r; kz = dz / r
            facing = -(nx * kx + ny * ky + nz * kz)
            vis = _facing_visibility(facing, width)
            if vis <= 0.0:
                continue
            pd = node_pol[j, 0] * kx + node_pol[j, 1] * ky + node_pol[j, 2] * kz
            if pd < -1.0: pd = -1.0
            elif pd > 1.0: pd = 1.0
            phase_weight = (
                node_phase_a[j] * (1.0 - pd * pd) + node_phase_b[j]
            )
            omega = 2.0 * (1.0 - r / math.sqrt(r2 + a2)) / a2
            c = facing
            if c < 0.0: c = 0.0
            elif c > 1.0: c = 1.0
            f = c * (nl - 1)
            ir = int(f)
            if ir >= nl - 1:
                response = response_lut[nl - 1]
            else:
                tr = f - ir
                response = response_lut[ir] + tr * (response_lut[ir + 1] - response_lut[ir])
            xatt = node_b[j] * r
            if xatt >= attenuation_xmax:
                attenuation = math.exp(-xatt)
            elif xatt <= 0.0:
                attenuation = 1.0
            else:
                fe = xatt * (ne - 1) / attenuation_xmax
                ie = int(fe)
                if ie >= ne - 1:
                    attenuation = attenuation_lut[ne - 1]
                else:
                    te = fe - ie
                    attenuation = attenuation_lut[ie] + te * (attenuation_lut[ie + 1] - attenuation_lut[ie])
            amp = phase_weight * omega * response * vis * attenuation
            if amp <= 0.0 or not math.isfinite(amp):
                continue
            acc += amp
            if node_ch[j] == 0: ar += amp
            else: am += amp
        charge[i] = acc; ray[i] = ar; ram[i] = am
    return charge, ray, ram


@njit(cache=True, fastmath=True)
def _accumulate_fused_serial_lut(
    p, n, node_pos, node_pol, node_phase_a, node_phase_b, node_ch, node_b, node_ng, node_bt,
    a, width, active_col, nactive, nbin, tmin, tmax,
    response_lut, attenuation_lut, attenuation_xmax,
):
    npm = p.shape[0]
    nn = node_pos.shape[0]
    charge = np.zeros(npm)
    ray = np.zeros(npm)
    ram = np.zeros(npm)
    node_mu = np.zeros((nbin, nactive))
    node_mt = np.zeros((nbin, nactive))
    dt = (tmax - tmin) / nbin if nbin > 0 else 1.0
    a2 = a * a
    inv4pi = 1.0 / (4.0 * math.pi)
    dip = 3.0 / (8.0 * math.pi)
    nl = response_lut.size
    ne = attenuation_lut.size
    for i in range(npm):
        px = p[i, 0]; py = p[i, 1]; pz = p[i, 2]
        nx = n[i, 0]; ny = n[i, 1]; nz = n[i, 2]
        ia = active_col[i]
        acc = 0.0; ar = 0.0; am = 0.0
        for j in range(nn):
            dx = px - node_pos[j, 0]
            dy = py - node_pos[j, 1]
            dz = pz - node_pos[j, 2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 1.0e-12: continue
            r = math.sqrt(r2)
            kx = dx / r; ky = dy / r; kz = dz / r
            facing = -(nx * kx + ny * ky + nz * kz)
            vis = _facing_visibility(facing, width)
            if vis <= 0.0: continue
            pd = node_pol[j, 0] * kx + node_pol[j, 1] * ky + node_pol[j, 2] * kz
            if pd < -1.0: pd = -1.0
            elif pd > 1.0: pd = 1.0
            phase_weight = (
                node_phase_a[j] * (1.0 - pd * pd) + node_phase_b[j]
            )
            omega = 2.0 * (1.0 - r / math.sqrt(r2 + a2)) / a2
            c = facing
            if c < 0.0: c = 0.0
            elif c > 1.0: c = 1.0
            f = c * (nl - 1); ir = int(f)
            if ir >= nl - 1: response = response_lut[nl - 1]
            else:
                tr = f - ir
                response = response_lut[ir] + tr * (response_lut[ir + 1] - response_lut[ir])
            xatt = node_b[j] * r
            if xatt >= attenuation_xmax: attenuation = math.exp(-xatt)
            elif xatt <= 0.0: attenuation = 1.0
            else:
                fe = xatt * (ne - 1) / attenuation_xmax; ie = int(fe)
                if ie >= ne - 1: attenuation = attenuation_lut[ne - 1]
                else:
                    te = fe - ie
                    attenuation = attenuation_lut[ie] + te * (attenuation_lut[ie + 1] - attenuation_lut[ie])
            amp = phase_weight * omega * response * vis * attenuation
            if amp <= 0.0 or not math.isfinite(amp): continue
            acc += amp
            if node_ch[j] == 0: ar += amp
            else: am += amp
            if ia >= 0 and nbin > 0:
                tt = node_bt[j] + node_ng[j] * r / C_MM_PER_NS
                ib = int((tt - tmin) / dt)
                if ib < 0: ib = 0
                elif ib >= nbin: ib = nbin - 1
                node_mu[ib, ia] += amp
                node_mt[ib, ia] += amp * tt
        charge[i] = acc; ray[i] = ar; ram[i] = am
    return charge, ray, ram, node_mu, node_mt




@njit(cache=True, parallel=True, fastmath=True)
def _accumulate_charge_parallel_lut(
    p, n, node_pos, node_pol, node_phase_a, node_phase_b, node_ch, node_b,
    a, width, response_lut, attenuation_lut, attenuation_xmax,
):
    """PMT-parallel charge-only kernel; avoids zero-column timing arrays."""
    npm = p.shape[0]
    nn = node_pos.shape[0]
    charge = np.zeros(npm)
    ray = np.zeros(npm)
    ram = np.zeros(npm)
    a2 = a * a
    nl = response_lut.size
    ne = attenuation_lut.size
    for i in prange(npm):
        px = p[i, 0]; py = p[i, 1]; pz = p[i, 2]
        nx = n[i, 0]; ny = n[i, 1]; nz = n[i, 2]
        acc = 0.0; ar = 0.0; am = 0.0
        for j in range(nn):
            dx = px - node_pos[j, 0]
            dy = py - node_pos[j, 1]
            dz = pz - node_pos[j, 2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 1.0e-12: continue
            r = math.sqrt(r2)
            invr = 1.0 / r
            kx = dx * invr; ky = dy * invr; kz = dz * invr
            facing = -(nx * kx + ny * ky + nz * kz)
            vis = _facing_visibility(facing, width)
            if vis <= 0.0: continue
            pd = node_pol[j, 0] * kx + node_pol[j, 1] * ky + node_pol[j, 2] * kz
            if pd < -1.0: pd = -1.0
            elif pd > 1.0: pd = 1.0
            phase_weight = node_phase_a[j] * (1.0 - pd * pd) + node_phase_b[j]
            omega = 2.0 * (1.0 - r / math.sqrt(r2 + a2)) / a2
            c = facing
            if c < 0.0: c = 0.0
            elif c > 1.0: c = 1.0
            f = c * (nl - 1); ir = int(f)
            if ir >= nl - 1: response = response_lut[nl - 1]
            else:
                tr = f - ir
                response = response_lut[ir] + tr * (response_lut[ir + 1] - response_lut[ir])
            xatt = node_b[j] * r
            if xatt >= attenuation_xmax: attenuation = math.exp(-xatt)
            elif xatt <= 0.0: attenuation = 1.0
            else:
                fe = xatt * (ne - 1) / attenuation_xmax; ie = int(fe)
                if ie >= ne - 1: attenuation = attenuation_lut[ne - 1]
                else:
                    te = fe - ie
                    attenuation = attenuation_lut[ie] + te * (attenuation_lut[ie + 1] - attenuation_lut[ie])
            amp = phase_weight * omega * response * vis * attenuation
            if amp <= 0.0 or not math.isfinite(amp): continue
            acc += amp
            if node_ch[j] == 0: ar += amp
            else: am += amp
        charge[i] = acc; ray[i] = ar; ram[i] = am
    return charge, ray, ram


@njit(cache=True, parallel=True, fastmath=True)
def _accumulate_fused_parallel_lut(
    p, n, node_pos, node_pol, node_phase_a, node_phase_b, node_ch, node_b, node_ng, node_bt,
    a, width, active_col, nactive, nbin, tmin, tmax,
    response_lut, attenuation_lut, attenuation_xmax,
):
    npm = p.shape[0]
    nn = node_pos.shape[0]
    charge = np.zeros(npm)
    ray = np.zeros(npm)
    ram = np.zeros(npm)
    node_mu = np.zeros((nbin, nactive))
    node_mt = np.zeros((nbin, nactive))
    dt = (tmax - tmin) / nbin if nbin > 0 else 1.0
    a2 = a * a
    inv4pi = 1.0 / (4.0 * math.pi)
    dip = 3.0 / (8.0 * math.pi)
    nl = response_lut.size
    ne = attenuation_lut.size
    for i in prange(npm):
        px = p[i, 0]; py = p[i, 1]; pz = p[i, 2]
        nx = n[i, 0]; ny = n[i, 1]; nz = n[i, 2]
        ia = active_col[i]
        acc = 0.0; ar = 0.0; am = 0.0
        for j in range(nn):
            dx = px - node_pos[j, 0]
            dy = py - node_pos[j, 1]
            dz = pz - node_pos[j, 2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 1.0e-12: continue
            r = math.sqrt(r2)
            kx = dx / r; ky = dy / r; kz = dz / r
            facing = -(nx * kx + ny * ky + nz * kz)
            vis = _facing_visibility(facing, width)
            if vis <= 0.0: continue
            pd = node_pol[j, 0] * kx + node_pol[j, 1] * ky + node_pol[j, 2] * kz
            if pd < -1.0: pd = -1.0
            elif pd > 1.0: pd = 1.0
            phase_weight = (
                node_phase_a[j] * (1.0 - pd * pd) + node_phase_b[j]
            )
            omega = 2.0 * (1.0 - r / math.sqrt(r2 + a2)) / a2
            c = facing
            if c < 0.0: c = 0.0
            elif c > 1.0: c = 1.0
            f = c * (nl - 1); ir = int(f)
            if ir >= nl - 1: response = response_lut[nl - 1]
            else:
                tr = f - ir
                response = response_lut[ir] + tr * (response_lut[ir + 1] - response_lut[ir])
            xatt = node_b[j] * r
            if xatt >= attenuation_xmax: attenuation = math.exp(-xatt)
            elif xatt <= 0.0: attenuation = 1.0
            else:
                fe = xatt * (ne - 1) / attenuation_xmax; ie = int(fe)
                if ie >= ne - 1: attenuation = attenuation_lut[ne - 1]
                else:
                    te = fe - ie
                    attenuation = attenuation_lut[ie] + te * (attenuation_lut[ie + 1] - attenuation_lut[ie])
            amp = phase_weight * omega * response * vis * attenuation
            if amp <= 0.0 or not math.isfinite(amp): continue
            acc += amp
            if node_ch[j] == 0: ar += amp
            else: am += amp
            if ia >= 0 and nbin > 0:
                tt = node_bt[j] + node_ng[j] * r / C_MM_PER_NS
                ib = int((tt - tmin) / dt)
                if ib < 0: ib = 0
                elif ib >= nbin: ib = nbin - 1
                node_mu[ib, ia] += amp
                node_mt[ib, ia] += amp * tt
        charge[i] = acc; ray[i] = ar; ram[i] = am
    return charge, ray, ram, node_mu, node_mt





@njit(cache=True, fastmath=True)
def _accumulate_charge_node_major_lut(
    p, n, node_pos, node_pol, node_phase_a, node_phase_b, node_ch, node_b,
    a, width, response_lut, attenuation_lut, attenuation_xmax,
):
    """Charge-only node-major equivalent of the validated serial kernel."""
    npm = p.shape[0]
    nn = node_pos.shape[0]
    charge = np.zeros(npm)
    ray = np.zeros(npm)
    ram = np.zeros(npm)
    a2 = a * a
    nl = response_lut.size
    ne = attenuation_lut.size
    for j in range(nn):
        sx = node_pos[j, 0]; sy = node_pos[j, 1]; sz = node_pos[j, 2]
        polx = node_pol[j, 0]; poly = node_pol[j, 1]; polz = node_pol[j, 2]
        pha = node_phase_a[j]; phb = node_phase_b[j]
        bout = node_b[j]
        is_ray = node_ch[j] == 0
        for i in range(npm):
            dx = p[i, 0] - sx
            dy = p[i, 1] - sy
            dz = p[i, 2] - sz
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 1.0e-12:
                continue
            r = math.sqrt(r2)
            invr = 1.0 / r
            kx = dx * invr; ky = dy * invr; kz = dz * invr
            facing = -(n[i, 0] * kx + n[i, 1] * ky + n[i, 2] * kz)
            vis = _facing_visibility(facing, width)
            if vis <= 0.0:
                continue
            pd = polx * kx + poly * ky + polz * kz
            if pd < -1.0: pd = -1.0
            elif pd > 1.0: pd = 1.0
            phase_weight = pha * (1.0 - pd * pd) + phb
            omega = 2.0 * (1.0 - r / math.sqrt(r2 + a2)) / a2
            c = facing
            if c < 0.0: c = 0.0
            elif c > 1.0: c = 1.0
            f = c * (nl - 1); ir = int(f)
            if ir >= nl - 1: response = response_lut[nl - 1]
            else:
                tr = f - ir
                response = response_lut[ir] + tr * (response_lut[ir + 1] - response_lut[ir])
            xatt = bout * r
            if xatt >= attenuation_xmax: attenuation = math.exp(-xatt)
            elif xatt <= 0.0: attenuation = 1.0
            else:
                fe = xatt * (ne - 1) / attenuation_xmax; ie = int(fe)
                if ie >= ne - 1: attenuation = attenuation_lut[ne - 1]
                else:
                    te = fe - ie
                    attenuation = attenuation_lut[ie] + te * (attenuation_lut[ie + 1] - attenuation_lut[ie])
            amp = phase_weight * omega * response * vis * attenuation
            if amp <= 0.0 or not math.isfinite(amp):
                continue
            charge[i] += amp
            if is_ray: ray[i] += amp
            else: ram[i] += amp
    return charge, ray, ram


@njit(cache=True, fastmath=True)
def _accumulate_fused_node_major_lut(
    p, n, node_pos, node_pol, node_phase_a, node_phase_b, node_ch, node_b, node_ng, node_bt,
    a, width, active_col, nactive, nbin, tmin, tmax,
    response_lut, attenuation_lut, attenuation_xmax,
):
    """Serial node-major equivalent of the validated PMT-major kernel.

    The summation order for every PMT remains the original node order, but all
    node-only quantities are loaded once outside the PMT loop.  This is an exact
    loop interchange, not a quadrature or physics approximation.
    """
    npm = p.shape[0]
    nn = node_pos.shape[0]
    charge = np.zeros(npm)
    ray = np.zeros(npm)
    ram = np.zeros(npm)
    node_mu = np.zeros((nbin, nactive))
    node_mt = np.zeros((nbin, nactive))
    dt = (tmax - tmin) / nbin if nbin > 0 else 1.0
    a2 = a * a
    nl = response_lut.size
    ne = attenuation_lut.size
    for j in range(nn):
        sx = node_pos[j, 0]; sy = node_pos[j, 1]; sz = node_pos[j, 2]
        polx = node_pol[j, 0]; poly = node_pol[j, 1]; polz = node_pol[j, 2]
        pha = node_phase_a[j]; phb = node_phase_b[j]
        bout = node_b[j]; ngout = node_ng[j]; bt = node_bt[j]
        is_ray = node_ch[j] == 0
        for i in range(npm):
            dx = p[i, 0] - sx
            dy = p[i, 1] - sy
            dz = p[i, 2] - sz
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 1.0e-12:
                continue
            r = math.sqrt(r2)
            invr = 1.0 / r
            kx = dx * invr; ky = dy * invr; kz = dz * invr
            facing = -(n[i, 0] * kx + n[i, 1] * ky + n[i, 2] * kz)
            vis = _facing_visibility(facing, width)
            if vis <= 0.0:
                continue
            pd = polx * kx + poly * ky + polz * kz
            if pd < -1.0: pd = -1.0
            elif pd > 1.0: pd = 1.0
            phase_weight = pha * (1.0 - pd * pd) + phb
            omega = 2.0 * (1.0 - r / math.sqrt(r2 + a2)) / a2
            c = facing
            if c < 0.0: c = 0.0
            elif c > 1.0: c = 1.0
            f = c * (nl - 1); ir = int(f)
            if ir >= nl - 1: response = response_lut[nl - 1]
            else:
                tr = f - ir
                response = response_lut[ir] + tr * (response_lut[ir + 1] - response_lut[ir])
            xatt = bout * r
            if xatt >= attenuation_xmax: attenuation = math.exp(-xatt)
            elif xatt <= 0.0: attenuation = 1.0
            else:
                fe = xatt * (ne - 1) / attenuation_xmax; ie = int(fe)
                if ie >= ne - 1: attenuation = attenuation_lut[ne - 1]
                else:
                    te = fe - ie
                    attenuation = attenuation_lut[ie] + te * (attenuation_lut[ie + 1] - attenuation_lut[ie])
            amp = phase_weight * omega * response * vis * attenuation
            if amp <= 0.0 or not math.isfinite(amp):
                continue
            charge[i] += amp
            if is_ray: ray[i] += amp
            else: ram[i] += amp
            ia = active_col[i]
            if ia >= 0 and nbin > 0:
                tt = bt + ngout * r / C_MM_PER_NS
                ib = int((tt - tmin) / dt)
                if ib < 0: ib = 0
                elif ib >= nbin: ib = nbin - 1
                node_mu[ib, ia] += amp
                node_mt[ib, ia] += amp * tt
    return charge, ray, ram, node_mu, node_mt



# -----------------------------------------------------------------------------
# Sparse receiver acceleration
# -----------------------------------------------------------------------------
# The event charge prediction is normalized to the observed total charge.  For
# PMTs with q=0, the Poisson term depends only on the *sum* of their expectation,
# not on how that expectation is distributed among those PMTs.  We therefore
# evaluate molecular scattering exactly for every hit/timed PMT and preserve the
# all-PMT normalization with a deterministic detector-response moment table.
# The table is built only from the analytic WCTE PMT geometry and response; it is
# not a WCSim light template.
_RECEIVER_MOMENT_CACHE = {}
_RECEIVER_MOMENT_DIAGNOSTIC_CACHE = {}
_RECEIVER_GEOMETRY_HASH_CACHE = {}
_RECEIVER_GEOMETRY_SIGNATURE_CACHE = {}
_RECEIVER_MOMENT_DEFAULT_NAME = "photon_scatter_receiver_moments_wcte_50mm_v1.npz"
_SELECTED_GEOMETRY_CACHE = {}

# The receiver moment table represents an all-PMT SUM.  PMT ordering is therefore
# not physically relevant to the table, even though the historical schema-1
# compatibility check hashed the raw float64 arrays byte-for-byte.  That old
# check was over-strict in two ways:
#   * harmless PMT reordering changed the hash;
#   * sub-micron floating-point differences from geometry/Python versions
#     changed the hash.
#
# Schema 2 uses an order-independent signature of paired (position, normal)
# rows after conservative quantization.  The tolerances below are much smaller
# than either the 50 mm receiver-moment grid spacing or the 45 mm PMT aperture,
# so they remove numerical false negatives without accepting a meaningfully
# different detector geometry.
_RECEIVER_GEOMETRY_SIGNATURE_VERSION = 2
_RECEIVER_POSITION_QUANTUM_MM = 0.1      # 0.1 mm
_RECEIVER_NORMAL_QUANTUM = 1.0e-6       # dimensionless direction cosine
_RECEIVER_POSITION_ATOL_MM = 0.1        # physically negligible vs 50 mm grid
_RECEIVER_NORMAL_ATOL = 1.0e-5          # about 5.7e-4 degrees

# Backward compatibility for the supplied schema-1 WCTE table.  The key is the
# legacy raw-byte hash pair stored in that trusted table; the value is the
# order-independent, quantized signature of the same geometry.  A schema-1
# table is accepted through this path only if the CURRENT runtime geometry has
# the same robust signature.  Thus this is not a filename-based bypass.
_TRUSTED_LEGACY_RECEIVER_GEOMETRIES = {
    (
        "23a69c2bf379d94840bc2f4155bd0860e846d5646460f154fa896d0e7bab3134",
        "c45ad5ea798dbc18dc953242b4acfdd5c1ba92c606371c61de13059a4e53fd2a",
    ): {
        "geometry_signature_version": 2,
        "geometry_signature_sha256": (
            "12616e905f6bc61da21e683930e710f93b920d8e73891e95376dc92ed68f969f"
        ),
        "position_quantum_mm": _RECEIVER_POSITION_QUANTUM_MM,
        "normal_quantum": _RECEIVER_NORMAL_QUANTUM,
        "description": (
            "WCTE 106-slot design geometry with inactive slots "
            "27,32,45,74,77,79,85,91,99"
        ),
    },
}


def _compact_selected_geometry(p, n, selected):
    """Return compact PMT positions/normals for the event's fixed hit support."""
    selected = np.ascontiguousarray(selected, dtype=np.int32)
    key = (id(p), id(n), p.shape, n.shape, selected.tobytes())
    cached = _SELECTED_GEOMETRY_CACHE.get(key)
    if cached is not None:
        return cached
    out = (
        np.ascontiguousarray(p[selected], dtype=np.float64),
        np.ascontiguousarray(n[selected], dtype=np.float64),
    )
    if len(_SELECTED_GEOMETRY_CACHE) >= 64:
        _SELECTED_GEOMETRY_CACHE.clear()
    _SELECTED_GEOMETRY_CACHE[key] = out
    return out


def _receiver_geometry_hash(p, n):
    """Return the historical exact raw-byte hashes for diagnostics/schema 1."""
    key = (id(p), id(n), p.shape, n.shape)
    cached = _RECEIVER_GEOMETRY_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    hp = hashlib.sha256(
        np.ascontiguousarray(p, dtype=np.float64).tobytes()
    ).hexdigest()
    hn = hashlib.sha256(
        np.ascontiguousarray(n, dtype=np.float64).tobytes()
    ).hexdigest()
    cached = (hp, hn)
    if len(_RECEIVER_GEOMETRY_HASH_CACHE) >= 8:
        _RECEIVER_GEOMETRY_HASH_CACHE.clear()
    _RECEIVER_GEOMETRY_HASH_CACHE[key] = cached
    return cached


def _receiver_geometry_signature(
    p,
    n,
    *,
    position_quantum_mm=_RECEIVER_POSITION_QUANTUM_MM,
    normal_quantum=_RECEIVER_NORMAL_QUANTUM,
):
    """Return an order-independent, tolerance-aware PMT geometry signature.

    Each PMT remains a paired position/normal row.  Rows are quantized to signed
    int64 values and then lexicographically sorted before hashing.  Therefore a
    PMT permutation does not alter the signature, while pairing a normal with a
    different PMT still does.
    """
    p = np.asarray(p, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    if p.ndim != 2 or n.shape != p.shape or p.shape[1] != 3:
        raise ValueError(
            "PMT positions and normals must have matching shape (n_pmts, 3)"
        )
    pq = float(position_quantum_mm)
    nq = float(normal_quantum)
    if not (np.isfinite(pq) and pq > 0.0):
        raise ValueError("position_quantum_mm must be finite and positive")
    if not (np.isfinite(nq) and nq > 0.0):
        raise ValueError("normal_quantum must be finite and positive")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(n))):
        raise ValueError("PMT positions/normals contain non-finite values")

    cache_key = (id(p), id(n), p.shape, n.shape, pq, nq)
    cached = _RECEIVER_GEOMETRY_SIGNATURE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    p_integer = np.rint(p / pq).astype("<i8")
    n_integer = np.rint(n / nq).astype("<i8")
    rows = np.concatenate((p_integer, n_integer), axis=1)
    # np.lexsort uses the last key as primary.  Reverse the column list so
    # column 0, then 1, ... defines the conventional row ordering.
    order = np.lexsort(
        tuple(rows[:, j] for j in range(rows.shape[1] - 1, -1, -1))
    )
    rows = np.ascontiguousarray(rows[order], dtype="<i8")
    signature = hashlib.sha256(rows.tobytes()).hexdigest()
    cached = {
        "version": int(_RECEIVER_GEOMETRY_SIGNATURE_VERSION),
        "sha256": signature,
        "position_quantum_mm": pq,
        "normal_quantum": nq,
    }
    if len(_RECEIVER_GEOMETRY_SIGNATURE_CACHE) >= 16:
        _RECEIVER_GEOMETRY_SIGNATURE_CACHE.clear()
    _RECEIVER_GEOMETRY_SIGNATURE_CACHE[cache_key] = cached
    return cached



def _match_receiver_geometry_arrays(
    runtime_positions,
    runtime_normals,
    table_positions,
    table_normals,
    *,
    position_atol_mm=_RECEIVER_POSITION_ATOL_MM,
    normal_atol=_RECEIVER_NORMAL_ATOL,
):
    """Match two PMT geometry sets without assuming identical PMT ordering.

    A small spatial-cell lookup is used instead of a raw-byte hash.  PMTs are
    paired by position and their normals are checked as part of the same match.
    The tolerances are orders of magnitude below the receiver table's 50 mm
    spatial grid and therefore cannot hide a physically meaningful geometry
    change.
    """
    rp = np.asarray(runtime_positions, dtype=np.float64)
    rn = np.asarray(runtime_normals, dtype=np.float64)
    tp = np.asarray(table_positions, dtype=np.float64)
    tn = np.asarray(table_normals, dtype=np.float64)
    if (
        rp.ndim != 2 or rp.shape[1] != 3 or rn.shape != rp.shape
        or tp.ndim != 2 or tp.shape[1] != 3 or tn.shape != tp.shape
        or tp.shape[0] < rp.shape[0]
    ):
        return {
            "matches": False,
            "reason": (
                "stored/runtime PMT geometry arrays must both have shape "
                "(n_pmts, 3), and the stored table must contain every runtime PMT"
            ),
            "max_position_difference_mm": np.inf,
            "max_normal_difference": np.inf,
        }
    if not (
        np.all(np.isfinite(rp)) and np.all(np.isfinite(rn))
        and np.all(np.isfinite(tp)) and np.all(np.isfinite(tn))
    ):
        return {
            "matches": False,
            "reason": "stored/runtime PMT geometry contains non-finite values",
            "max_position_difference_mm": np.inf,
            "max_normal_difference": np.inf,
        }

    pos_tol = float(position_atol_mm)
    norm_tol = float(normal_atol)
    if pos_tol <= 0.0 or norm_tol <= 0.0:
        raise ValueError("receiver geometry tolerances must be positive")

    # A cell much wider than the tolerance makes boundary crossings harmless.
    # Neighbouring cells are always searched, so no PMT is rejected merely for
    # lying close to a cell edge.
    cell_size = max(1.0, 8.0 * pos_tol)
    table_cells = np.floor(tp / cell_size).astype(np.int64)
    lookup = {}
    for index, cell in enumerate(table_cells):
        lookup.setdefault(tuple(int(x) for x in cell), []).append(index)

    used = np.zeros(tp.shape[0], dtype=bool)
    max_position_difference = 0.0
    max_normal_difference = 0.0
    offsets = (
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    )
    neighbour_offsets = tuple(offsets)

    for i in range(rp.shape[0]):
        base = np.floor(rp[i] / cell_size).astype(np.int64)
        candidates = []
        for dx, dy, dz in neighbour_offsets:
            key = (
                int(base[0] + dx),
                int(base[1] + dy),
                int(base[2] + dz),
            )
            candidates.extend(lookup.get(key, ()))

        best_index = -1
        best_metric = np.inf
        best_position_difference = np.inf
        best_normal_difference = np.inf
        for j in candidates:
            if used[j]:
                continue
            position_difference = float(np.max(np.abs(rp[i] - tp[j])))
            if position_difference > pos_tol:
                continue
            normal_difference = float(np.max(np.abs(rn[i] - tn[j])))
            if normal_difference > norm_tol:
                continue
            metric = position_difference / pos_tol + normal_difference / norm_tol
            if metric < best_metric:
                best_metric = metric
                best_index = int(j)
                best_position_difference = position_difference
                best_normal_difference = normal_difference

        if best_index < 0:
            return {
                "matches": False,
                "reason": (
                    f"runtime PMT {i} has no stored position/normal match within "
                    f"{pos_tol:g} mm and {norm_tol:g}"
                ),
                "unmatched_runtime_index": int(i),
                "max_position_difference_mm": max_position_difference,
                "max_normal_difference": max_normal_difference,
            }

        used[best_index] = True
        max_position_difference = max(
            max_position_difference, best_position_difference
        )
        max_normal_difference = max(
            max_normal_difference, best_normal_difference
        )

    matched_table_indices = np.flatnonzero(used).astype(np.int64)
    missing_table_indices = np.flatnonzero(~used).astype(np.int64)
    is_exact_set = bool(missing_table_indices.size == 0)
    return {
        "matches": True,
        "runtime_is_subset": bool(not is_exact_set),
        "reason": (
            "all PMT position/normal pairs match within tolerance"
            if is_exact_set else
            "every runtime PMT matches a stored table PMT; unmatched stored "
            "receivers will be subtracted from the all-PMT moment field"
        ),
        "max_position_difference_mm": max_position_difference,
        "max_normal_difference": max_normal_difference,
        "position_atol_mm": pos_tol,
        "normal_atol": norm_tol,
        "matched_table_indices": matched_table_indices.tolist(),
        "missing_table_indices": missing_table_indices.tolist(),
        "stored_pmt_count": int(tp.shape[0]),
        "runtime_pmt_count": int(rp.shape[0]),
    }


@njit(cache=True, parallel=True, fastmath=True)
def _receiver_moment_contributions_grid(
    xs, ys, zs, pmt_positions, pmt_normals,
    aperture_radius_mm, facing_soft_width, b_ref,
):
    """Build all-PMT receiver moments on a Cartesian grid.

    This is the same analytic finite-aperture/legacy-angular receiver used by
    the exact molecular-scattering kernel.  The three moment planes are the
    zeroth, first and second path-length coefficients of the attenuation
    expansion around ``b_ref``.  It is used both to generate geometry tables and
    to subtract run-masked PMTs from a stored superset table exactly.
    """
    nx = xs.size
    ny = ys.size
    nz = zs.size
    ngrid = nx * ny * nz
    out = np.zeros((ngrid, 3, 7), dtype=np.float32)
    a2 = aperture_radius_mm * aperture_radius_mm
    yz = ny * nz
    for flat in prange(ngrid):
        ix = flat // yz
        rem = flat - ix * yz
        iy = rem // nz
        iz = rem - iy * nz
        sx = xs[ix]
        sy = ys[iy]
        sz = zs[iz]
        m = np.zeros((3, 7), dtype=np.float64)
        for i in range(pmt_positions.shape[0]):
            dx = pmt_positions[i, 0] - sx
            dy = pmt_positions[i, 1] - sy
            dz = pmt_positions[i, 2] - sz
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 1.0e-12:
                continue
            r = math.sqrt(r2)
            invr = 1.0 / r
            kx = dx * invr
            ky = dy * invr
            kz = dz * invr
            facing = -(
                pmt_normals[i, 0] * kx
                + pmt_normals[i, 1] * ky
                + pmt_normals[i, 2] * kz
            )
            vis = _facing_visibility(facing, facing_soft_width)
            if vis <= 0.0:
                continue
            omega = 2.0 * (1.0 - r / math.sqrt(r2 + a2)) / a2
            base = omega * _legacy_response(facing) * vis * math.exp(-b_ref * r)
            if base <= 0.0 or not math.isfinite(base):
                continue
            q0 = 1.0
            q1 = kx * kx
            q2 = ky * ky
            q3 = kz * kz
            q4 = kx * ky
            q5 = kx * kz
            q6 = ky * kz
            for order in range(3):
                weight = base
                if order == 1:
                    weight *= r
                elif order == 2:
                    weight *= r2
                m[order, 0] += weight * q0
                m[order, 1] += weight * q1
                m[order, 2] += weight * q2
                m[order, 3] += weight * q3
                m[order, 4] += weight * q4
                m[order, 5] += weight * q5
                m[order, 6] += weight * q6
        for order in range(3):
            for q in range(7):
                out[flat, order, q] = m[order, q]
    return out.reshape((nx, ny, nz, 3, 7))


def build_receiver_moment_contributions(
    xs, ys, zs, pmt_positions_mm, pmt_normals, *,
    pmt_aperture_radius_mm=45.0,
    pmt_facing_soft_width=0.02,
    b_ref=1.0e-5,
):
    """Public deterministic builder for a geometry-derived receiver table."""
    return _receiver_moment_contributions_grid(
        np.ascontiguousarray(xs, dtype=np.float64),
        np.ascontiguousarray(ys, dtype=np.float64),
        np.ascontiguousarray(zs, dtype=np.float64),
        np.ascontiguousarray(pmt_positions_mm, dtype=np.float64),
        np.ascontiguousarray(pmt_normals, dtype=np.float64),
        float(pmt_aperture_radius_mm),
        float(pmt_facing_soft_width),
        float(b_ref),
    )


def _subtract_missing_receiver_moments(
    moments, xs, ys, zs, table_positions, table_normals,
    missing_table_indices, *, aperture_radius_mm, facing_soft_width, b_ref,
):
    missing = np.asarray(missing_table_indices, dtype=np.int64)
    if missing.size == 0:
        return np.ascontiguousarray(moments, dtype=np.float32)
    missing_positions = np.ascontiguousarray(
        np.asarray(table_positions, dtype=np.float64)[missing], dtype=np.float64
    )
    missing_normals = np.ascontiguousarray(
        np.asarray(table_normals, dtype=np.float64)[missing], dtype=np.float64
    )
    subtraction = build_receiver_moment_contributions(
        xs, ys, zs, missing_positions, missing_normals,
        pmt_aperture_radius_mm=aperture_radius_mm,
        pmt_facing_soft_width=facing_soft_width,
        b_ref=b_ref,
    )
    adjusted = np.asarray(moments, dtype=np.float64) - np.asarray(
        subtraction, dtype=np.float64
    )
    # Roundoff can make a physically non-negative scalar moment infinitesimally
    # negative.  Preserve tensor cross terms, which legitimately have either sign.
    adjusted[..., :, 0] = np.maximum(adjusted[..., :, 0], 0.0)
    return np.ascontiguousarray(adjusted, dtype=np.float32)


def _receiver_moment_candidate_paths(config):
    explicit = str(getattr(config, "receiver_moment_table_path", "") or "").strip()
    env = os.environ.get("EMITTER_PHOTON_SCATTER_RECEIVER_TABLE", "").strip()
    if explicit:
        return [Path(explicit).expanduser()]
    if env:
        return [Path(env).expanduser()]
    module_dir = Path(__file__).resolve().parent
    candidates = []
    for name in ("LF_TABLE_DIR", "LF_OFFICIAL_TABLE_DIR"):
        raw = os.environ.get(name, "").strip()
        if raw:
            candidates.append(Path(raw).expanduser() / _RECEIVER_MOMENT_DEFAULT_NAME)
    candidates.extend([
        module_dir.parent / "tables" / _RECEIVER_MOMENT_DEFAULT_NAME,
        module_dir / "tables" / _RECEIVER_MOMENT_DEFAULT_NAME,
        Path.cwd() / "tables" / _RECEIVER_MOMENT_DEFAULT_NAME,
    ])
    out = []
    seen = set()
    for q in candidates:
        q = q.resolve()
        if str(q) not in seen:
            seen.add(str(q))
            out.append(q)
    return out


def _payload_scalar(payload, name, default=None):
    if name not in payload.files:
        return default
    return np.asarray(payload[name]).item()


def _inspect_receiver_moment_table(path, config, p, n, hp, hn):
    """Inspect one candidate without loading its large moment array."""
    path = Path(path)
    result = {
        "path": str(path),
        "exists": bool(path.is_file()),
        "compatible": False,
        "compatibility_method": None,
        "reason": None,
    }
    if not path.is_file():
        result["reason"] = "file does not exist"
        return result

    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "table_kind", "schema_version", "n_pmts",
                "pmt_aperture_radius_mm", "pmt_facing_soft_width",
                "xs", "ys", "zs", "moments", "bref",
            }
            missing = sorted(required.difference(payload.files))
            if missing:
                result["reason"] = "missing fields: " + ", ".join(missing)
                return result

            table_kind = str(_payload_scalar(payload, "table_kind"))
            schema_version = int(_payload_scalar(payload, "schema_version"))
            n_pmts = int(_payload_scalar(payload, "n_pmts"))
            aperture = float(_payload_scalar(payload, "pmt_aperture_radius_mm"))
            facing = float(_payload_scalar(payload, "pmt_facing_soft_width"))
            stored_hp = str(_payload_scalar(payload, "pmt_positions_sha256", ""))
            stored_hn = str(_payload_scalar(payload, "pmt_normals_sha256", ""))
            result.update({
                "table_kind": table_kind,
                "schema_version": schema_version,
                "table_pmt_count": n_pmts,
                "table_pmt_positions_sha256": stored_hp,
                "table_pmt_normals_sha256": stored_hn,
                "table_pmt_aperture_radius_mm": aperture,
                "table_pmt_facing_soft_width": facing,
            })

            if table_kind != "licketyfit_photon_scatter_receiver_moments":
                result["reason"] = f"unexpected table_kind={table_kind!r}"
                return result
            if schema_version not in (1, 2):
                result["reason"] = f"unsupported schema_version={schema_version}"
                return result
            if n_pmts < int(p.shape[0]):
                result["reason"] = (
                    "PMT count mismatch: the receiver table cannot be a subset "
                    f"of the runtime detector (table={n_pmts}, runtime={p.shape[0]})"
                )
                return result
            if not np.isclose(
                aperture, float(config.pmt_aperture_radius_mm),
                rtol=0.0, atol=1.0e-12,
            ):
                result["reason"] = (
                    "PMT aperture mismatch: "
                    f"table={aperture}, runtime={config.pmt_aperture_radius_mm}"
                )
                return result
            if not np.isclose(
                facing, float(config.pmt_facing_soft_width),
                rtol=0.0, atol=1.0e-12,
            ):
                result["reason"] = (
                    "PMT facing-width mismatch: "
                    f"table={facing}, runtime={config.pmt_facing_soft_width}"
                )
                return result

            # Exact schema-1 behavior remains the first and strongest check.
            if stored_hp == hp and stored_hn == hn:
                result["compatible"] = True
                result["compatibility_method"] = "exact_raw_float64_hash"
                result["reason"] = "exact PMT position and normal hashes match"
                return result

            # Preferred schema-2 path: compare the stored PMT geometry
            # directly, without assuming the same array ordering or exact
            # floating-point bytes.
            if (
                "pmt_positions_mm" in payload.files
                and "pmt_normals" in payload.files
            ):
                pos_atol = float(_payload_scalar(
                    payload,
                    "geometry_position_atol_mm",
                    _RECEIVER_POSITION_ATOL_MM,
                ))
                normal_atol = float(_payload_scalar(
                    payload,
                    "geometry_normal_atol",
                    _RECEIVER_NORMAL_ATOL,
                ))
                comparison = _match_receiver_geometry_arrays(
                    p,
                    n,
                    np.asarray(payload["pmt_positions_mm"], dtype=np.float64),
                    np.asarray(payload["pmt_normals"], dtype=np.float64),
                    position_atol_mm=pos_atol,
                    normal_atol=normal_atol,
                )
                result["geometry_position_atol_mm"] = pos_atol
                result["geometry_normal_atol"] = normal_atol
                result["max_position_difference_mm"] = float(
                    comparison.get("max_position_difference_mm", np.inf)
                )
                result["max_normal_difference"] = float(
                    comparison.get("max_normal_difference", np.inf)
                )
                if comparison.get("matches", False):
                    result["compatible"] = True
                    runtime_subset = bool(comparison.get("runtime_is_subset", False))
                    result["compatibility_method"] = (
                        "runtime_geometry_subset_of_table_with_exact_moment_subtraction"
                        if runtime_subset else
                        "order_independent_position_normal_tolerance"
                    )
                    result["runtime_geometry_is_table_subset"] = runtime_subset
                    result["matched_table_indices"] = list(
                        comparison.get("matched_table_indices", [])
                    )
                    result["missing_table_indices"] = list(
                        comparison.get("missing_table_indices", [])
                    )
                    result["missing_table_pmt_count"] = int(
                        len(result["missing_table_indices"])
                    )
                    result["reason"] = (
                        comparison.get("reason")
                        or "stored PMT positions/normals match the runtime geometry"
                    )
                    return result
                result["reason"] = (
                    "stored PMT geometry arrays are incompatible: "
                    + str(comparison.get("reason", "unknown mismatch"))
                )
                return result

            # Native schema-2 robust signature.
            stored_signature = _payload_scalar(
                payload, "geometry_signature_sha256", None
            )
            if stored_signature is not None:
                signature_version = int(_payload_scalar(
                    payload,
                    "geometry_signature_version",
                    _RECEIVER_GEOMETRY_SIGNATURE_VERSION,
                ))
                pq = float(_payload_scalar(
                    payload,
                    "geometry_position_quantum_mm",
                    _RECEIVER_POSITION_QUANTUM_MM,
                ))
                nq = float(_payload_scalar(
                    payload,
                    "geometry_normal_quantum",
                    _RECEIVER_NORMAL_QUANTUM,
                ))
                if signature_version != _RECEIVER_GEOMETRY_SIGNATURE_VERSION:
                    result["reason"] = (
                        "unsupported geometry signature version: "
                        f"{signature_version}"
                    )
                    return result
                runtime_signature = _receiver_geometry_signature(
                    p, n, position_quantum_mm=pq, normal_quantum=nq
                )
                result["runtime_geometry_signature_sha256"] = (
                    runtime_signature["sha256"]
                )
                result["table_geometry_signature_sha256"] = str(stored_signature)
                result["geometry_position_quantum_mm"] = pq
                result["geometry_normal_quantum"] = nq
                if runtime_signature["sha256"] == str(stored_signature):
                    result["compatible"] = True
                    result["compatibility_method"] = (
                        "order_independent_quantized_geometry_signature"
                    )
                    result["reason"] = (
                        "robust PMT geometry signature matches; raw-byte "
                        "difference is only ordering/numerical precision"
                    )
                    return result
                result["reason"] = (
                    "robust PMT geometry signature mismatch: "
                    f"table={stored_signature}, "
                    f"runtime={runtime_signature['sha256']}"
                )
                return result

            # Trusted migration path for the supplied legacy schema-1 table.
            trusted = _TRUSTED_LEGACY_RECEIVER_GEOMETRIES.get(
                (stored_hp, stored_hn)
            )
            if trusted is not None:
                runtime_signature = _receiver_geometry_signature(
                    p,
                    n,
                    position_quantum_mm=trusted["position_quantum_mm"],
                    normal_quantum=trusted["normal_quantum"],
                )
                result["runtime_geometry_signature_sha256"] = (
                    runtime_signature["sha256"]
                )
                result["table_geometry_signature_sha256"] = trusted[
                    "geometry_signature_sha256"
                ]
                result["geometry_position_quantum_mm"] = trusted[
                    "position_quantum_mm"
                ]
                result["geometry_normal_quantum"] = trusted["normal_quantum"]
                result["trusted_legacy_geometry"] = trusted["description"]
                if (
                    runtime_signature["sha256"]
                    == trusted["geometry_signature_sha256"]
                ):
                    result["compatible"] = True
                    result["compatibility_method"] = (
                        "trusted_schema1_order_independent_quantized_signature"
                    )
                    result["reason"] = (
                        "trusted legacy WCTE table matches the runtime geometry "
                        "after order-independent, physically negligible "
                        "quantization"
                    )
                    return result
                result["reason"] = (
                    "legacy raw hashes differ and the robust geometry signature "
                    "also differs; this is a real geometry mismatch, not merely "
                    "floating-point precision or PMT ordering"
                )
                return result

            result["reason"] = (
                "raw PMT hashes differ and this schema-1 table has no trusted "
                "robust geometry signature"
            )
            return result
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result


def _find_receiver_moment_table(config, p, n):
    mode = str(getattr(config, "receiver_mode", "exact")).strip().lower()
    if mode not in {"sparse_moment", "sparse", "moment"}:
        return None, {
            "compatible": False,
            "compatibility_method": None,
            "reason": "exact receiver mode requested",
            "candidate_diagnostics": [],
        }

    p = np.ascontiguousarray(p, dtype=np.float64)
    n = np.ascontiguousarray(n, dtype=np.float64)
    hp, hn = _receiver_geometry_hash(p, n)
    cache_key = (
        hp,
        hn,
        float(config.pmt_aperture_radius_mm),
        float(config.pmt_facing_soft_width),
        str(getattr(config, "receiver_moment_table_path", "")),
        os.environ.get("EMITTER_PHOTON_SCATTER_RECEIVER_TABLE", ""),
    )
    if cache_key in _RECEIVER_MOMENT_CACHE:
        return (
            _RECEIVER_MOMENT_CACHE[cache_key],
            _RECEIVER_MOMENT_DIAGNOSTIC_CACHE.get(cache_key, {}),
        )

    diagnostics = []
    for path in _receiver_moment_candidate_paths(config):
        diagnostic = _inspect_receiver_moment_table(
            path, config, p, n, hp, hn
        )
        diagnostics.append(diagnostic)
        if not diagnostic.get("compatible", False):
            continue
        try:
            with np.load(path, allow_pickle=False) as payload:
                xs = np.ascontiguousarray(payload["xs"], dtype=np.float64)
                ys = np.ascontiguousarray(payload["ys"], dtype=np.float64)
                zs = np.ascontiguousarray(payload["zs"], dtype=np.float64)
                moments = np.ascontiguousarray(
                    payload["moments"], dtype=np.float32
                )
                b_ref = float(np.asarray(payload["bref"]).item())
                missing = diagnostic.get("missing_table_indices", [])
                if missing:
                    moments = _subtract_missing_receiver_moments(
                        moments, xs, ys, zs,
                        np.asarray(payload["pmt_positions_mm"], dtype=np.float64),
                        np.asarray(payload["pmt_normals"], dtype=np.float64),
                        missing,
                        aperture_radius_mm=float(
                            np.asarray(payload["pmt_aperture_radius_mm"]).item()
                        ),
                        facing_soft_width=float(
                            np.asarray(payload["pmt_facing_soft_width"]).item()
                        ),
                        b_ref=b_ref,
                    )
                    diagnostic = dict(diagnostic)
                    diagnostic["moment_subset_adjustment_applied"] = True
                    diagnostic["moment_subset_subtracted_pmt_count"] = int(
                        len(missing)
                    )
                table = (xs, ys, zs, moments, b_ref, str(path))
        except Exception as exc:
            diagnostic = dict(diagnostic)
            diagnostic["compatible"] = False
            diagnostic["reason"] = (
                "metadata passed but table arrays could not be loaded: "
                f"{type(exc).__name__}: {exc}"
            )
            diagnostics[-1] = diagnostic
            continue

        if len(_RECEIVER_MOMENT_CACHE) >= 4:
            _RECEIVER_MOMENT_CACHE.clear()
            _RECEIVER_MOMENT_DIAGNOSTIC_CACHE.clear()
        final_diagnostic = dict(diagnostic)
        final_diagnostic["candidate_diagnostics"] = diagnostics
        _RECEIVER_MOMENT_CACHE[cache_key] = table
        _RECEIVER_MOMENT_DIAGNOSTIC_CACHE[cache_key] = final_diagnostic
        return table, final_diagnostic

    final_diagnostic = {
        "compatible": False,
        "compatibility_method": None,
        "reason": "no compatible receiver moment table was found",
        "candidate_diagnostics": diagnostics,
    }
    _RECEIVER_MOMENT_CACHE[cache_key] = None
    _RECEIVER_MOMENT_DIAGNOSTIC_CACHE[cache_key] = final_diagnostic
    if bool(getattr(config, "receiver_moment_table_required", False)):
        detail = "; ".join(
            f"{d.get('path')}: {d.get('reason')}" for d in diagnostics
        )
        raise FileNotFoundError(
            "No compatible photon-scatter receiver moment table was found. "
            + detail
        )
    return None, final_diagnostic


def _load_receiver_moment_table(config, p, n):
    table, _ = _find_receiver_moment_table(config, p, n)
    return table


def receiver_moment_table_status(
    config,
    pmt_positions_mm,
    pmt_normals,
):
    """Describe the effective molecular-scattering receiver implementation."""
    p = np.ascontiguousarray(pmt_positions_mm, dtype=np.float64)
    n = np.ascontiguousarray(pmt_normals, dtype=np.float64)
    mode = str(getattr(config, "receiver_mode", "exact")).strip().lower()
    sparse_requested = mode in {"sparse_moment", "sparse", "moment"}
    hp, hn = _receiver_geometry_hash(p, n)
    robust = _receiver_geometry_signature(p, n)
    candidates = [str(path) for path in _receiver_moment_candidate_paths(config)]

    table = None
    diagnostic = {
        "compatible": False,
        "compatibility_method": None,
        "reason": "exact receiver mode requested",
        "candidate_diagnostics": [],
    }
    error = None
    if sparse_requested:
        try:
            table, diagnostic = _find_receiver_moment_table(config, p, n)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    compatible_path = None if table is None else str(table[-1])
    if not sparse_requested:
        effective_mode = "exact_all_pmts"
        reason = "exact receiver mode requested"
    elif compatible_path is not None:
        effective_mode = "sparse_moment"
        reason = diagnostic.get("reason", "compatible receiver table loaded")
    else:
        effective_mode = "exact_all_pmts_fallback"
        reason = error or diagnostic.get(
            "reason", "no compatible receiver moment table was found"
        )

    return {
        "requested_mode": mode,
        "effective_mode": effective_mode,
        "sparse_requested": bool(sparse_requested),
        "compatible_table": compatible_path is not None,
        "compatible_table_path": compatible_path,
        "compatibility_method": diagnostic.get("compatibility_method"),
        "candidate_paths": candidates,
        "candidate_diagnostics": diagnostic.get("candidate_diagnostics", []),
        "pmt_count": int(p.shape[0]),
        "pmt_positions_sha256": hp,
        "pmt_normals_sha256": hn,
        "geometry_signature_version": int(robust["version"]),
        "geometry_signature_sha256": robust["sha256"],
        "geometry_position_quantum_mm": float(robust["position_quantum_mm"]),
        "geometry_normal_quantum": float(robust["normal_quantum"]),
        "pmt_aperture_radius_mm": float(config.pmt_aperture_radius_mm),
        "pmt_facing_soft_width": float(config.pmt_facing_soft_width),
        "native_receiver_requested": bool(getattr(config, "native_receiver", False)),
        "native_receiver_available": bool(
            native_receiver_available()
            if bool(getattr(config, "native_receiver", False))
            else False
        ),
        "native_receiver_threads": int(
            max(1, int(getattr(config, "native_receiver_threads", 1)))
        ),
        "reason": reason,
    }


@njit(cache=True, inline="always")
def _receiver_axis_index(x, grid):
    if x <= grid[0]:
        return 0, 0.0
    if x >= grid[grid.size - 1]:
        return grid.size - 2, 1.0
    lo = 0; hi = grid.size
    while lo < hi:
        mid = (lo + hi) // 2
        if grid[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    i = lo - 1
    return i, (x - grid[i]) / (grid[i + 1] - grid[i])


@njit(cache=True, fastmath=True)
def _receiver_total_from_moments(
    position, polarization, phase_a, phase_b, outgoing_b, channel,
    xs, ys, zs, moments, b_ref,
):
    total = 0.0; ray = 0.0; ram = 0.0
    for j in range(position.shape[0]):
        ix, tx = _receiver_axis_index(position[j, 0], xs)
        iy, ty = _receiver_axis_index(position[j, 1], ys)
        iz, tz = _receiver_axis_index(position[j, 2], zs)
        m = np.zeros(7, dtype=np.float64)
        db = outgoing_b[j] - b_ref
        for ax in range(2):
            wx = (1.0 - tx) if ax == 0 else tx
            for ay in range(2):
                wy = (1.0 - ty) if ay == 0 else ty
                for az in range(2):
                    wz = (1.0 - tz) if az == 0 else tz
                    wxyz = wx * wy * wz
                    for q in range(7):
                        m0 = float(moments[ix + ax, iy + ay, iz + az, 0, q])
                        m1 = float(moments[ix + ax, iy + ay, iz + az, 1, q])
                        m2 = float(moments[ix + ax, iy + ay, iz + az, 2, q])
                        m[q] += wxyz * (m0 - db * m1 + 0.5 * db * db * m2)
        ex = polarization[j, 0]; ey = polarization[j, 1]; ez = polarization[j, 2]
        quad = (
            m[1] * ex * ex + m[2] * ey * ey + m[3] * ez * ez
            + 2.0 * m[4] * ex * ey + 2.0 * m[5] * ex * ez
            + 2.0 * m[6] * ey * ez
        )
        amp = phase_a[j] * (m[0] - quad) + phase_b[j] * m[0]
        if amp < 0.0 and amp > -1.0e-10:
            amp = 0.0
        total += amp
        if channel[j] == 0:
            ray += amp
        else:
            ram += amp
    return total, ray, ram


@njit(cache=True, fastmath=True)
def _accumulate_charge_selected_lut(
    p, n, selected, node_pos, node_pol, node_phase_a, node_phase_b,
    node_ch, node_b, a, width, response_lut, attenuation_lut,
    attenuation_xmax,
):
    nsel = selected.size; nn = node_pos.shape[0]
    charge = np.zeros(nsel); ray = np.zeros(nsel); ram = np.zeros(nsel)
    a2 = a * a; nl = response_lut.size; ne = attenuation_lut.size
    for jj in range(nsel):
        i = selected[jj]
        px = p[i, 0]; py = p[i, 1]; pz = p[i, 2]
        nx = n[i, 0]; ny = n[i, 1]; nz = n[i, 2]
        acc = 0.0; ar = 0.0; am = 0.0
        for j in range(nn):
            dx = px - node_pos[j, 0]; dy = py - node_pos[j, 1]; dz = pz - node_pos[j, 2]
            r2 = dx*dx + dy*dy + dz*dz
            if r2 <= 1.0e-12: continue
            r = math.sqrt(r2); invr = 1.0/r
            kx = dx*invr; ky = dy*invr; kz = dz*invr
            facing = -(nx*kx + ny*ky + nz*kz)
            vis = _facing_visibility(facing, width)
            if vis <= 0.0: continue
            pd = node_pol[j,0]*kx + node_pol[j,1]*ky + node_pol[j,2]*kz
            if pd < -1.0: pd = -1.0
            elif pd > 1.0: pd = 1.0
            phase = node_phase_a[j]*(1.0-pd*pd) + node_phase_b[j]
            omega = 2.0*(1.0-r/math.sqrt(r2+a2))/a2
            c = facing
            if c < 0.0: c = 0.0
            elif c > 1.0: c = 1.0
            f = c*(nl-1); ir = int(f)
            if ir >= nl-1: response = response_lut[nl-1]
            else:
                tr = f-ir; response = response_lut[ir] + tr*(response_lut[ir+1]-response_lut[ir])
            xatt = node_b[j]*r
            if xatt >= attenuation_xmax: attenuation = math.exp(-xatt)
            elif xatt <= 0.0: attenuation = 1.0
            else:
                fe = xatt*(ne-1)/attenuation_xmax; ie = int(fe)
                if ie >= ne-1: attenuation = attenuation_lut[ne-1]
                else:
                    te = fe-ie; attenuation = attenuation_lut[ie] + te*(attenuation_lut[ie+1]-attenuation_lut[ie])
            amp = phase*omega*response*vis*attenuation
            if amp <= 0.0 or not math.isfinite(amp): continue
            acc += amp
            if node_ch[j] == 0: ar += amp
            else: am += amp
        charge[jj] = acc; ray[jj] = ar; ram[jj] = am
    return charge, ray, ram


@njit(cache=True, parallel=True, fastmath=True)
def _accumulate_charge_selected_parallel_lut(
    p, n, selected, node_pos, node_pol, node_phase_a, node_phase_b,
    node_ch, node_b, a, width, response_lut, attenuation_lut,
    attenuation_xmax,
):
    nsel = selected.size; nn = node_pos.shape[0]
    charge = np.zeros(nsel); ray = np.zeros(nsel); ram = np.zeros(nsel)
    a2 = a * a; nl = response_lut.size; ne = attenuation_lut.size
    for jj in prange(nsel):
        i = selected[jj]
        px = p[i, 0]; py = p[i, 1]; pz = p[i, 2]
        nx = n[i, 0]; ny = n[i, 1]; nz = n[i, 2]
        acc = 0.0; ar = 0.0; am = 0.0
        for j in range(nn):
            dx = px - node_pos[j, 0]; dy = py - node_pos[j, 1]; dz = pz - node_pos[j, 2]
            r2 = dx*dx + dy*dy + dz*dz
            if r2 <= 1.0e-12: continue
            r = math.sqrt(r2); invr = 1.0/r
            kx = dx*invr; ky = dy*invr; kz = dz*invr
            facing = -(nx*kx + ny*ky + nz*kz)
            vis = _facing_visibility(facing, width)
            if vis <= 0.0: continue
            pd = node_pol[j,0]*kx + node_pol[j,1]*ky + node_pol[j,2]*kz
            if pd < -1.0: pd = -1.0
            elif pd > 1.0: pd = 1.0
            phase = node_phase_a[j]*(1.0-pd*pd) + node_phase_b[j]
            omega = 2.0*(1.0-r/math.sqrt(r2+a2))/a2
            c = facing
            if c < 0.0: c = 0.0
            elif c > 1.0: c = 1.0
            f = c*(nl-1); ir = int(f)
            if ir >= nl-1: response = response_lut[nl-1]
            else:
                tr = f-ir; response = response_lut[ir] + tr*(response_lut[ir+1]-response_lut[ir])
            xatt = node_b[j]*r
            if xatt >= attenuation_xmax: attenuation = math.exp(-xatt)
            elif xatt <= 0.0: attenuation = 1.0
            else:
                fe = xatt*(ne-1)/attenuation_xmax; ie = int(fe)
                if ie >= ne-1: attenuation = attenuation_lut[ne-1]
                else:
                    te = fe-ie; attenuation = attenuation_lut[ie] + te*(attenuation_lut[ie+1]-attenuation_lut[ie])
            amp = phase*omega*response*vis*attenuation
            if amp <= 0.0 or not math.isfinite(amp): continue
            acc += amp
            if node_ch[j] == 0: ar += amp
            else: am += amp
        charge[jj] = acc; ray[jj] = ar; ram[jj] = am
    return charge, ray, ram


@njit(cache=True, fastmath=True)
def _accumulate_fused_selected_lut(
    p, n, selected, node_pos, node_pol, node_phase_a, node_phase_b,
    node_ch, node_b, node_ng, node_bt, a, width, nbin, tmin, tmax,
    response_lut, attenuation_lut, attenuation_xmax,
):
    nsel = selected.size; nn = node_pos.shape[0]
    charge = np.zeros(nsel); ray = np.zeros(nsel); ram = np.zeros(nsel)
    node_mu = np.zeros((nbin,nsel)); node_mt = np.zeros((nbin,nsel))
    dt = (tmax-tmin)/nbin if nbin>0 else 1.0
    a2 = a*a; nl=response_lut.size; ne=attenuation_lut.size
    for jj in range(nsel):
        i=selected[jj]; px=p[i,0];py=p[i,1];pz=p[i,2];nx=n[i,0];ny=n[i,1];nz=n[i,2]
        acc=0.0;ar=0.0;am=0.0
        for j in range(nn):
            dx=px-node_pos[j,0];dy=py-node_pos[j,1];dz=pz-node_pos[j,2];r2=dx*dx+dy*dy+dz*dz
            if r2<=1e-12:continue
            r=math.sqrt(r2);invr=1.0/r;kx=dx*invr;ky=dy*invr;kz=dz*invr
            facing=-(nx*kx+ny*ky+nz*kz);vis=_facing_visibility(facing,width)
            if vis<=0.0:continue
            pd=node_pol[j,0]*kx+node_pol[j,1]*ky+node_pol[j,2]*kz
            if pd<-1.0:pd=-1.0
            elif pd>1.0:pd=1.0
            phase=node_phase_a[j]*(1.0-pd*pd)+node_phase_b[j]
            omega=2.0*(1.0-r/math.sqrt(r2+a2))/a2
            c=facing
            if c<0.0:c=0.0
            elif c>1.0:c=1.0
            f=c*(nl-1);ir=int(f)
            if ir>=nl-1:response=response_lut[nl-1]
            else:
                tr=f-ir;response=response_lut[ir]+tr*(response_lut[ir+1]-response_lut[ir])
            xatt=node_b[j]*r
            if xatt>=attenuation_xmax:attenuation=math.exp(-xatt)
            elif xatt<=0.0:attenuation=1.0
            else:
                fe=xatt*(ne-1)/attenuation_xmax;ie=int(fe)
                if ie>=ne-1:attenuation=attenuation_lut[ne-1]
                else:
                    te=fe-ie;attenuation=attenuation_lut[ie]+te*(attenuation_lut[ie+1]-attenuation_lut[ie])
            amp=phase*omega*response*vis*attenuation
            if amp<=0.0 or not math.isfinite(amp):continue
            acc+=amp
            if node_ch[j]==0:ar+=amp
            else:am+=amp
            if nbin>0:
                tt=node_bt[j]+node_ng[j]*r/C_MM_PER_NS;ib=int((tt-tmin)/dt)
                if ib<0:ib=0
                elif ib>=nbin:ib=nbin-1
                node_mu[ib,jj]+=amp;node_mt[ib,jj]+=amp*tt
        charge[jj]=acc;ray[jj]=ar;ram[jj]=am
    return charge,ray,ram,node_mu,node_mt


@njit(cache=True, parallel=True, fastmath=True)
def _accumulate_fused_selected_parallel_lut(
    p, n, selected, node_pos, node_pol, node_phase_a, node_phase_b,
    node_ch, node_b, node_ng, node_bt, a, width, nbin, tmin, tmax,
    response_lut, attenuation_lut, attenuation_xmax,
):
    nsel = selected.size; nn = node_pos.shape[0]
    charge = np.zeros(nsel); ray = np.zeros(nsel); ram = np.zeros(nsel)
    node_mu = np.zeros((nbin,nsel)); node_mt = np.zeros((nbin,nsel))
    dt = (tmax-tmin)/nbin if nbin>0 else 1.0
    a2=a*a;nl=response_lut.size;ne=attenuation_lut.size
    for jj in prange(nsel):
        i=selected[jj];px=p[i,0];py=p[i,1];pz=p[i,2];nx=n[i,0];ny=n[i,1];nz=n[i,2]
        acc=0.0;ar=0.0;am=0.0
        for j in range(nn):
            dx=px-node_pos[j,0];dy=py-node_pos[j,1];dz=pz-node_pos[j,2];r2=dx*dx+dy*dy+dz*dz
            if r2<=1e-12:continue
            r=math.sqrt(r2);invr=1.0/r;kx=dx*invr;ky=dy*invr;kz=dz*invr
            facing=-(nx*kx+ny*ky+nz*kz);vis=_facing_visibility(facing,width)
            if vis<=0.0:continue
            pd=node_pol[j,0]*kx+node_pol[j,1]*ky+node_pol[j,2]*kz
            if pd<-1.0:pd=-1.0
            elif pd>1.0:pd=1.0
            phase=node_phase_a[j]*(1.0-pd*pd)+node_phase_b[j]
            omega=2.0*(1.0-r/math.sqrt(r2+a2))/a2
            c=facing
            if c<0.0:c=0.0
            elif c>1.0:c=1.0
            f=c*(nl-1);ir=int(f)
            if ir>=nl-1:response=response_lut[nl-1]
            else:
                tr=f-ir;response=response_lut[ir]+tr*(response_lut[ir+1]-response_lut[ir])
            xatt=node_b[j]*r
            if xatt>=attenuation_xmax:attenuation=math.exp(-xatt)
            elif xatt<=0.0:attenuation=1.0
            else:
                fe=xatt*(ne-1)/attenuation_xmax;ie=int(fe)
                if ie>=ne-1:attenuation=attenuation_lut[ne-1]
                else:
                    te=fe-ie;attenuation=attenuation_lut[ie]+te*(attenuation_lut[ie+1]-attenuation_lut[ie])
            amp=phase*omega*response*vis*attenuation
            if amp<=0.0 or not math.isfinite(amp):continue
            acc+=amp
            if node_ch[j]==0:ar+=amp
            else:am+=amp
            if nbin>0:
                tt=node_bt[j]+node_ng[j]*r/C_MM_PER_NS;ib=int((tt-tmin)/dt)
                if ib<0:ib=0
                elif ib>=nbin:ib=nbin-1
                node_mu[ib,jj]+=amp;node_mt[ib,jj]+=amp*tt
        charge[jj]=acc;ray[jj]=ar;ram[jj]=am
    return charge,ray,ram,node_mu,node_mt


def _scatter_sparse_fill(npm, selected, sel_ray, sel_ram, total_ray, total_ram):
    ray = np.zeros(npm, dtype=np.float64); ram = np.zeros(npm, dtype=np.float64)
    ray[selected] = sel_ray; ram[selected] = sel_ram
    inactive = np.ones(npm, dtype=bool); inactive[selected] = False
    ni = int(np.count_nonzero(inactive))
    miss_ray = float(total_ray) - float(np.sum(sel_ray))
    miss_ram = float(total_ram) - float(np.sum(sel_ram))
    # The total moment interpolation should exceed the exact hit-PMT sum.  If it
    # does not, the table is incompatible with this geometry/hypothesis and the
    # caller falls back to the exact all-PMT calculation.
    if ni <= 0 or miss_ray < -1.0e-8 or miss_ram < -1.0e-8:
        return None
    if miss_ray > 0.0: ray[inactive] = miss_ray / ni
    if miss_ram > 0.0: ram[inactive] = miss_ram / ni
    return ray + ram, ray, ram

def _configure_numba_scatter_threads(config):
    if not bool(config.parallel_pmt_loop):
        return 1
    raw = os.environ.get("EMITTER_PHOTON_SCATTER_THREADS", "").strip()
    if raw:
        requested = max(1, int(float(raw)))
        try:
            set_num_threads(min(requested, get_num_threads()))
        except Exception:
            pass
    try:
        return int(get_num_threads())
    except Exception:
        return 1


def accumulate_photon_scatter_prediction(
    nodes, pmt_positions_mm, pmt_normals, *, timing_active_indices=None,
    charge_active_indices=None, config=PhotonScatteringTransportConfig(),
    receiver_dome_centres_mm=None, receiver_dome_axes=None,
):
    p = np.ascontiguousarray(pmt_positions_mm, dtype=np.float64)
    n = np.ascontiguousarray(pmt_normals, dtype=np.float64)
    config.validate()
    if nodes.position_mm.shape[0] == 0:
        z = np.zeros(p.shape[0])
        return PhotonScatterPrediction(z, z.copy(), z.copy(), None, None, None, None)

    if timing_active_indices is None:
        timing_active = np.empty(0, dtype=np.int32)
        nbin = 0; tmin = tmax = 0.0
    else:
        timing_active = np.ascontiguousarray(timing_active_indices, dtype=np.int32)
        nbin = int(config.n_timing_bins)
        pmin = np.min(p, axis=0); pmax = np.max(p, axis=0)
        corners = np.asarray([[x,y,z] for x in (pmin[0],pmax[0]) for y in (pmin[1],pmax[1]) for z in (pmin[2],pmax[2])])
        maxr = max(float(np.max(np.linalg.norm(nodes.position_mm-c[None,:],axis=1))) for c in corners)
        tmin = float(np.min(nodes.base_time_ns))
        tmax = float(np.max(nodes.base_time_ns + nodes.outgoing_group_index*maxr/C_MM_PER_NS))
        tmax = max(tmax, tmin+1.0)

    if charge_active_indices is None:
        charge_active = timing_active
    else:
        charge_active = np.ascontiguousarray(charge_active_indices, dtype=np.int32)

    dep = np.asarray(nodes.depolarization_ratio, dtype=np.float64)
    fiso = 3.0*dep/(2.0+dep)
    phase_a = np.ascontiguousarray(np.asarray(nodes.charge_weight,dtype=np.float64)*(1.0-fiso)*(3.0/(8.0*math.pi)))
    phase_b = np.ascontiguousarray(np.asarray(nodes.charge_weight,dtype=np.float64)*fiso*(1.0/(4.0*math.pi)))

    nthreads = _configure_numba_scatter_threads(config)
    use_parallel = bool(config.parallel_pmt_loop) and nthreads > 1
    table = _load_receiver_moment_table(config, p, n)
    sparse_ok = (
        table is not None and charge_active.size > 0 and charge_active.size < p.shape[0]
        and (timing_active.size == 0 or np.array_equal(timing_active, charge_active))
    )

    if sparse_ok:
        xs,ys,zs,moments,bref,_table_path = table
        total, total_ray, total_ram = _receiver_total_from_moments(
            nodes.position_mm, nodes.polarization, phase_a, phase_b,
            nodes.outgoing_scattering_coefficient_mm_inv, nodes.channel,
            xs,ys,zs,moments,float(bref),
        )
        native_result = None
        if bool(getattr(config, "native_receiver", False)) and accumulate_fused_selected_native is not None:
            psel, nsel = _compact_selected_geometry(p, n, charge_active)
            native_result = accumulate_fused_selected_native(
                psel, nsel, nodes.position_mm, nodes.polarization, phase_a, phase_b,
                nodes.channel, nodes.outgoing_scattering_coefficient_mm_inv,
                nodes.outgoing_group_index, nodes.base_time_ns,
                float(config.pmt_aperture_radius_mm),
                float(config.pmt_facing_soft_width),
                nbin if timing_active.size else 0, float(tmin), float(tmax),
                _SCATTER_RESPONSE_LUT, _SCATTER_ATTENUATION_LUT,
                float(_SCATTER_ATTENUATION_X_MAX),
                n_threads=max(1, int(getattr(config, "native_receiver_threads", 1))),
                required=bool(getattr(config, "native_receiver_required", False)),
            )
        if native_result is not None:
            sel_charge, sel_ray, sel_ram, nmu, nmt = native_result
        elif timing_active.size:
            kernel = _accumulate_fused_selected_parallel_lut if use_parallel else _accumulate_fused_selected_lut
            sel_charge,sel_ray,sel_ram,nmu,nmt = kernel(
                p,n,charge_active,nodes.position_mm,nodes.polarization,phase_a,phase_b,
                nodes.channel,nodes.outgoing_scattering_coefficient_mm_inv,
                nodes.outgoing_group_index,nodes.base_time_ns,
                float(config.pmt_aperture_radius_mm),float(config.pmt_facing_soft_width),
                nbin,float(tmin),float(tmax),_SCATTER_RESPONSE_LUT,
                _SCATTER_ATTENUATION_LUT,float(_SCATTER_ATTENUATION_X_MAX),
            )
        else:
            kernel = _accumulate_charge_selected_parallel_lut if use_parallel else _accumulate_charge_selected_lut
            sel_charge,sel_ray,sel_ram = kernel(
                p,n,charge_active,nodes.position_mm,nodes.polarization,phase_a,phase_b,
                nodes.channel,nodes.outgoing_scattering_coefficient_mm_inv,
                float(config.pmt_aperture_radius_mm),float(config.pmt_facing_soft_width),
                _SCATTER_RESPONSE_LUT,_SCATTER_ATTENUATION_LUT,
                float(_SCATTER_ATTENUATION_X_MAX),
            )
            nmu=np.empty((0,0));nmt=np.empty((0,0))
        filled = _scatter_sparse_fill(p.shape[0], charge_active, sel_ray, sel_ram, total_ray, total_ram)
        if filled is not None:
            charge,ray,ram = filled
            if timing_active.size == 0:
                return PhotonScatterPrediction(charge,ray,ram,None,None,None,None)
            nt=np.divide(nmt,nmu,out=np.full_like(nmt,np.inf),where=nmu>0.0)
            return PhotonScatterPrediction(
                charge,ray,ram,np.ascontiguousarray(nmu,dtype=np.float32),
                np.ascontiguousarray(nt,dtype=np.float32),timing_active,
                np.linspace(tmin,tmax,nbin+1),
            )

    # Exact all-PMT reference/fallback path.
    active = timing_active
    if use_parallel:
        if active.size:
            active_col=np.full(p.shape[0],-1,dtype=np.int32);active_col[active]=np.arange(active.size,dtype=np.int32)
            charge,ray,ram,nmu,nmt=_accumulate_fused_parallel_lut(
                p,n,nodes.position_mm,nodes.polarization,phase_a,phase_b,nodes.channel,
                nodes.outgoing_scattering_coefficient_mm_inv,nodes.outgoing_group_index,nodes.base_time_ns,
                float(config.pmt_aperture_radius_mm),float(config.pmt_facing_soft_width),
                np.ascontiguousarray(active_col),int(active.size),nbin,float(tmin),float(tmax),
                _SCATTER_RESPONSE_LUT,_SCATTER_ATTENUATION_LUT,float(_SCATTER_ATTENUATION_X_MAX))
        else:
            charge,ray,ram=_accumulate_charge_parallel_lut(
                p,n,nodes.position_mm,nodes.polarization,phase_a,phase_b,nodes.channel,
                nodes.outgoing_scattering_coefficient_mm_inv,float(config.pmt_aperture_radius_mm),
                float(config.pmt_facing_soft_width),_SCATTER_RESPONSE_LUT,
                _SCATTER_ATTENUATION_LUT,float(_SCATTER_ATTENUATION_X_MAX))
            nmu=np.empty((0,0));nmt=np.empty((0,0))
    else:
        if active.size:
            active_col=np.full(p.shape[0],-1,dtype=np.int32);active_col[active]=np.arange(active.size,dtype=np.int32)
            charge,ray,ram,nmu,nmt=_accumulate_fused_node_major_lut(
                p,n,nodes.position_mm,nodes.polarization,phase_a,phase_b,nodes.channel,
                nodes.outgoing_scattering_coefficient_mm_inv,nodes.outgoing_group_index,nodes.base_time_ns,
                float(config.pmt_aperture_radius_mm),float(config.pmt_facing_soft_width),
                np.ascontiguousarray(active_col),int(active.size),nbin,float(tmin),float(tmax),
                _SCATTER_RESPONSE_LUT,_SCATTER_ATTENUATION_LUT,float(_SCATTER_ATTENUATION_X_MAX))
        else:
            charge,ray,ram=_accumulate_charge_node_major_lut(
                p,n,nodes.position_mm,nodes.polarization,phase_a,phase_b,nodes.channel,
                nodes.outgoing_scattering_coefficient_mm_inv,float(config.pmt_aperture_radius_mm),
                float(config.pmt_facing_soft_width),_SCATTER_RESPONSE_LUT,
                _SCATTER_ATTENUATION_LUT,float(_SCATTER_ATTENUATION_X_MAX))
            nmu=np.empty((0,0));nmt=np.empty((0,0))
    if active.size==0:
        return PhotonScatterPrediction(charge,ray,ram,None,None,None,None)
    nt=np.divide(nmt,nmu,out=np.full_like(nmt,np.inf),where=nmu>0.0)
    return PhotonScatterPrediction(charge,ray,ram,np.ascontiguousarray(nmu,dtype=np.float32),np.ascontiguousarray(nt,dtype=np.float32),active,np.linspace(tmin,tmax,nbin+1))

def direct_survival_and_group_index(beta,path_length_mm,*,config=PhotonScatteringTransportConfig()):
    """Return direct zero-interaction survival and surviving group index in one pass."""
    accelerated = _bilinear_direct_lut(beta, path_length_mm, config)
    if accelerated is not None:
        return accelerated
    bta,rr=np.broadcast_arrays(np.asarray(beta,dtype=np.float64),np.asarray(path_length_mm,dtype=np.float64))
    weights,arrays,shape=_detected_spectrum_matrix(bta,config);_,_,ng,br,bm,_=arrays
    physical_path=np.maximum(rr.ravel(),0.0)
    survived=weights*np.exp(-physical_path[:,None]*(br+bm)[None,:]/1000.0)
    sw=np.sum(survived,axis=1)
    survival=sw.copy()
    survival[(np.sum(weights,axis=1)<=0.0)|(rr.ravel()<=0.0)]=1.0
    group=np.divide(np.sum(survived*ng[None,:],axis=1),sw,out=np.full(sw.shape,1.384730463081079),where=sw>0.0)
    return survival.reshape(shape),group.reshape(shape)


def direct_zero_interaction_survival(beta,path_length_mm,*,config=PhotonScatteringTransportConfig()):
    return direct_survival_and_group_index(beta,path_length_mm,config=config)[0]


def direct_surviving_group_index(beta,path_length_mm,*,config=PhotonScatteringTransportConfig()):
    return direct_survival_and_group_index(beta,path_length_mm,config=config)[1]

def convergence_signature(prediction):
    x=np.asarray(prediction,dtype=float);total=float(np.sum(x));p=x/total if total>0 else np.zeros_like(x)
    return {"sum":total,"l1_normalized":float(np.sum(np.abs(p))),"l2_normalized":float(np.sqrt(np.sum(p*p))),"max_fraction":float(np.max(p)) if p.size else 0.0,"nonzero":int(np.count_nonzero(x>0.0))}
