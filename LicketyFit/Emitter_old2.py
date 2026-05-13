import numpy as np
from typing import List, Tuple

from model_muon_cherenkov_collapse import (
    find_scale_for_pmts,
    get_cerenkov_angle_table,
    get_energy_distance_tables,
    theta_c_func
)

from n_model_wrapper import *


def _get_tables():
    c_ang, energy_for_angle = get_cerenkov_angle_table()
    overall_distances, energy_rows, distance_rows = get_energy_distance_tables()
    #r_rdep, E_rdep, n_rdep = get_rdep_tables()
    return c_ang, energy_for_angle, overall_distances, energy_rows, distance_rows#, r_rdep, E_rdep, n_rdep


class Emitter:
    """
    Optimized Cherenkov emitter model used by the fitter.

    The public fit-facing API is preserved, but the hot methods avoid:
      - pickle-based copying
      - debug prints in the fit loop
      - repeated temporary allocations when not needed
    """

    def __init__(self, starting_time, start_coord, direction, beta, length, intensity):
        if not isinstance(starting_time, (int, float)):
            raise TypeError("starting_time must be a number")
        if not (
            isinstance(start_coord, tuple)
            and len(start_coord) == 3
            and all(isinstance(c, (int, float)) for c in start_coord)
        ):
            raise TypeError("start_coord must be a tuple of three numbers")
        if not (
            isinstance(direction, tuple)
            and len(direction) == 3
            and all(isinstance(c, (int, float)) for c in direction)
        ):
            raise TypeError("direction must be a tuple of three numbers")
        if not isinstance(beta, (int, float)) or not (0 < beta < 1):
            raise ValueError("beta must be a number between 0 and 1")
        if not isinstance(length, (int, float)) or length <= 0:
            raise ValueError("length must be a positive number")
        if not isinstance(intensity, (int, float)) or intensity <= 0:
            raise ValueError("intensity must be a positive number")

        self.starting_time = float(starting_time)
        self.start_coord = tuple(float(c) for c in start_coord)
        self.direction = tuple(float(c) for c in direction)
        self.length = float(length)
        self.intensity = float(intensity)

        self.mu_mass = 105.658  # MeV/c^2
        self.n = 1.344
        self.c = 299.792458  # mm/ns

        self.beta = float(beta)
        self.v = self.beta * self.c
        self.cos_tq = None
        self.cot_tq = None
        self.interp_E_init = None

        # Match the original behavior: initialise beta from the length-dependent
        # lookup table rather than trusting the constructor beta argument.
        self.refresh_kinematics_from_length(self.length)

    def __repr__(self):
        return (
            f"Emitter(starting_time={self.starting_time}, start_coord={self.start_coord}, "
            f"direction={self.direction}, beta={self.beta}, length={self.length}, "
            f"intensity={self.intensity})"
        )

    def copy(self):
        """
        Lightweight copy.

        The original version used pickle for every copy, which is much more
        expensive than needed for this small numeric state.
        """
        new = self.__class__.__new__(self.__class__)
        new.__dict__ = self.__dict__.copy()
        return new

    def calc_constants(self, n):
        self.n = float(n)
        self.cos_tq = 1.0 / (self.beta * self.n)
        self.cos_tq = np.clip(self.cos_tq, -1.0, 1.0)
        sin_tq = np.sqrt(max(1e-15, 1.0 - self.cos_tq**2))
        self.cot_tq = self.cos_tq / sin_tq
        self.c = 299.792458
        self.v = self.beta * self.c

    @staticmethod
    def nearest_main_idx(length_mm):
        idx = np.searchsorted(_get_tables()[2], float(length_mm))
        idx = np.clip(idx, 1, len(_get_tables()[2]) - 1)
        left = _get_tables()[2][idx - 1]
        right = _get_tables()[2][idx]
        if (float(length_mm) - left) <= (right - float(length_mm)):
            idx -= 1
        return int(idx)

    def refresh_kinematics_from_energy(self, initial_KE):
        self.interp_E_init = float(initial_KE)
        self.beta = np.sqrt(
            1.0 - (self.mu_mass / (self.interp_E_init + self.mu_mass)) ** 2
        )
        self.calc_constants(self.n)
        return self.interp_E_init

    def refresh_kinematics_from_length(self, length_mm):
        self.length = float(length_mm)
        main_idx = self.nearest_main_idx(self.length)
        return self.refresh_kinematics_from_energy(_get_tables()[3][main_idx][0])

    def set_nominal_track_parameters(self, starting_time, start_coord, direction, length):
        self.starting_time = float(starting_time)
        self.start_coord = tuple(float(c) for c in start_coord)
        self.direction = tuple(float(c) for c in direction)
        self.length = float(length)

    def set_wall_track_parameters(self, starting_time, y_w, phi_w, d_w, w_y, w_phi, length, r, sign_cz=+1):
        """ Set the track parameters of the emitter using "wall" parameters.

        Args:
            starting_time (float): The time that emitter starts emission in nanoseconds.
            y_w (float): y coordinate of the wall intersection point
            phi_w (float): azimuthal angle of the wall intersection point
            d_w (float): distance from start to wall intersection point
            w_y (float): cosine of angle between line direction and y-axis
            w_phi (float): cosine of angle in x-z plane between line direction and tangent to cylinder at wall point
            length (float): The length of the path for the emitter (mm).
            r (float): radius of the cylinder
            sign_cz (int): sign of c_z to choose branch (+1 or -1)
        """
        (x_0, y_0, z_0, c_x, c_y), _ = self.inverse_transform_and_jacobian(y_w, phi_w, d_w, w_y, w_phi, r, sign_cz)
        self.starting_time = float(starting_time)
        self.start_coord = (x_0, y_0, z_0)
        self.direction = (c_x, c_y)
        self.length = float(length)

    def get_wall_parameters_and_jacobian(self, r, sign_cz=+1):
        """
        Forward: (x_0, y_0, z_0, c_x, c_y) -> (y_w, phi_w, d_w, w_y, w_phi), J_f (5x5)
        Cylinder axis is y; wall: x^2 + z^2 = r^2.  phi_w = atan2(x_w, z_w).
        """
        def _safe_sqrt(x):
            return np.sqrt(np.maximum(0.0, x))

        (x_0, y_0, z_0) = self.start_coord
        (c_x, c_y) = self.direction

        # Direction and checks
        beta_xy = c_x ** 2 + c_y ** 2
        c_z = sign_cz * _safe_sqrt(1.0 - beta_xy)
        beta = c_x ** 2 + c_z ** 2  # = 1 - c_y**2 = ||c_perp||^2
        if beta <= 0:
            raise ValueError("Degenerate direction: c_x=c_z=0 (parallel to axis).")

        # Solve (x0 + t cx)^2 + (z0 + t cz)^2 = r^2 for first t>0
        alpha = x_0 * c_x + z_0 * c_z
        rho0_sq = x_0 ** 2 + z_0 ** 2
        disc = alpha ** 2 + beta * (r ** 2 - rho0_sq)
        if disc < 0:
            raise ValueError("No intersection with cylinder (discriminant < 0).")
        d_w = (-alpha + np.sqrt(disc)) / beta

        # Hit point and cylindrical coords
        x_w = x_0 + d_w * c_x
        z_w = z_0 + d_w * c_z
        phi_w = np.arctan2(x_w, z_w)  # φ=0 on +z
        y_w = y_0 + d_w * c_y

        # Cosines
        w_y = c_y
        S, C = np.sin(phi_w), np.cos(phi_w)
        sqrt_beta = np.sqrt(beta)
        # with t_phi=(C,0,-S):  w_phi = (c_perp·t_phi)/||c_perp|| = (c_x C - c_z S)/sqrt_beta
        w_phi = (c_x * C - c_z * S) / sqrt_beta

        # Jacobian building blocks
        a = c_x * S + c_z * C  # c_perp · n (>=0 for outward hit)

        # ∂d_w/∂(x0,z0,cx,cz) at fixed (cx,cz)
        dd_dx0_ind = -S / a
        dd_dz0_ind = -C / a
        dd_dcx_ind = -d_w * S / a
        dd_dcz_ind = -d_w * C / a

        # c_z depends on (c_x, c_y):  ∂c_z/∂c_x = -c_x/c_z,  ∂c_z/∂c_y = -c_y/c_z
        dcz_dcx = -c_x / (c_z if c_z != 0 else 1e-300)
        dcz_dcy = -c_y / (c_z if c_z != 0 else 1e-300)

        # Chain to (c_x, c_y)
        dd_dx0 = dd_dx0_ind
        dd_dz0 = dd_dz0_ind
        dd_dcx = dd_dcx_ind + dd_dcz_ind * dcz_dcx
        dd_dcy = dd_dcz_ind * dcz_dcy

        # φ partials:  dφ = (-x_w dz_w + z_w dx_w)/r^2  ⇒  at wall: dφ = (C dx - S dz)/r
        dphi_dx0_ind = C / (r * a)
        dphi_dz0_ind = -S / (r * a)
        dphi_dcx_ind = d_w * C / (r * a)
        dphi_dcz_ind = -d_w * S / (r * a)

        dphi_dx0 = dphi_dx0_ind
        dphi_dz0 = dphi_dz0_ind
        dphi_dcx = dphi_dcx_ind + dphi_dcz_ind * dcz_dcx
        dphi_dcy = dphi_dcz_ind * dcz_dcy  # dφ/dc_y via c_z only
        # dφ/dy0 = 0

        # Assemble forward Jacobian J_f
        J = np.zeros((5, 5), dtype=float)

        # (1) y_w = y_0 + d_w c_y
        J[0, 0] = c_y * dd_dx0
        J[0, 1] = 1.0
        J[0, 2] = c_y * dd_dz0
        J[0, 3] = c_y * dd_dcx
        J[0, 4] = d_w + c_y * dd_dcy

        # (2) φ_w
        J[1, 0] = dphi_dx0
        J[1, 1] = 0.0
        J[1, 2] = dphi_dz0
        J[1, 3] = dphi_dcx
        J[1, 4] = dphi_dcy

        # (3) d_w
        J[2, 0] = dd_dx0
        J[2, 1] = 0.0
        J[2, 2] = dd_dz0
        J[2, 3] = dd_dcx
        J[2, 4] = dd_dcy

        # (4) w_y = c_y
        J[3, 0] = 0.0
        J[3, 1] = 0.0
        J[3, 2] = 0.0
        J[3, 3] = 0.0
        J[3, 4] = 1.0

        # (5) w_phi = (c_x C - c_z S)/sqrt_beta
        inv_sqrtb = 1.0 / (sqrt_beta if sqrt_beta != 0 else 1e-300)
        inv_beta = 1.0 / (beta if beta != 0 else 1e-300)

        # φ-coupling factor:  ∂w_phi/∂φ at fixed (cx,cz) equals (-a)/sqrt_beta
        fac = (-a) * inv_sqrtb

        # wrt (x0,y0,z0): only via φ
        J[4, 0] = fac * dphi_dx0
        J[4, 1] = 0.0
        J[4, 2] = fac * dphi_dz0

        # wrt (c_x, c_y) including c_z and φ dependences
        # general: dw = (C dcx - S dcz)/sqrtβ + fac dφ - w_phi/β (c_x dcx + c_z dcz)
        coeff_dcz = (-S) * inv_sqrtb - w_phi * c_z * inv_beta
        coeff_dcx = (C) * inv_sqrtb - w_phi * c_x * inv_beta

        J[4, 3] = coeff_dcx + coeff_dcz * dcz_dcx + fac * dphi_dcx  # ∂/∂c_x
        J[4, 4] = coeff_dcz * dcz_dcy + fac * dphi_dcy  # ∂/∂c_y

        return (y_w, phi_w, d_w, w_y, w_phi), J

    def inverse_transform_and_jacobian(y_w, phi_w, d_w, w_y, w_phi, r, sign_cz=+1):
        """
        Inverse: (y_w, phi_w, d_w, w_y, w_phi) -> (x_0, y_0, z_0, c_x, c_y), J_g (5x5)
        Using t_phi = (cosφ, 0, -sinφ).
        """
        def _safe_sqrt(x):
            return np.sqrt(np.maximum(0.0, x))

        S, C = np.sin(phi_w), np.cos(phi_w)
        s = _safe_sqrt(1.0 - w_phi ** 2)  # = sin(angle to t_phi) in xz-plane
        sb = _safe_sqrt(1.0 - w_y ** 2)  # = ||c_perp||

        # Direction (c_perp = sb*(w_phi t_phi + s n))
        c_y = w_y
        c_x = sb * (w_phi * C + s * S)
        c_z = sb * (-w_phi * S + s * C)

        # Optional: enforce chosen c_z branch sign
        if sign_cz < 0 and c_z > 0: c_z = -c_z
        if sign_cz > 0 and c_z < 0: c_z = -c_z

        # Wall point and start point
        x_w = r * S
        z_w = r * C
        x_0 = x_w - d_w * c_x
        y_0 = y_w - d_w * c_y
        z_0 = z_w - d_w * c_z

        # Inverse Jacobian J_g
        J = np.zeros((5, 5), dtype=float)

        # helpers
        dsb_dwy = -(w_y / (sb if sb != 0 else 1e-300))
        ds_dwp = -(w_phi / (s if s != 0 else 1e-300))

        # Direction partials
        # c_x = sb*( w_phi*C + s*S)
        dcx_dphi = sb * (w_phi * (-S) + s * C)
        dcx_dwy = dsb_dwy * (w_phi * C + s * S)
        dcx_dwp = sb * (C + ds_dwp * S)

        # c_z = sb*(-w_phi*S + s*C)
        dcz_dphi = sb * (-w_phi * C - s * S)
        dcz_dwy = dsb_dwy * (-w_phi * S + s * C)
        dcz_dwp = sb * (-S + ds_dwp * C)

        # Rows for start point: x_0 = r*S - d_w*c_x;  z_0 = r*C - d_w*c_z;  y_0 = y_w - d_w*c_y
        J[0, 0] = 0.0
        J[0, 1] = r * C - d_w * dcx_dphi
        J[0, 2] = -c_x
        J[0, 3] = -d_w * dcx_dwy
        J[0, 4] = -d_w * dcx_dwp

        J[1, 0] = 1.0
        J[1, 1] = 0.0
        J[1, 2] = -c_y
        J[1, 3] = -d_w
        J[1, 4] = 0.0

        J[2, 0] = 0.0
        J[2, 1] = -r * S - d_w * dcz_dphi  # because d(r*C)/dφ = -r*S
        J[2, 2] = -c_z
        J[2, 3] = -d_w * dcz_dwy
        J[2, 4] = -d_w * dcz_dwp

        # Rows for direction (outputs 4,5)
        J[3, 0] = 0.0
        J[3, 1] = dcx_dphi
        J[3, 2] = 0.0
        J[3, 3] = dcx_dwy
        J[3, 4] = dcx_dwp

        J[4, 0] = 0.0
        J[4, 1] = 0.0
        J[4, 2] = 0.0
        J[4, 3] = 1.0
        J[4, 4] = 0.0

        return (x_0, y_0, z_0, c_x, c_y), J

    def get_emission_point(self, pmt_coord, initial_KE):
        """
        Emission point for a single PMT.
        """
        x0, y0, z0 = self.start_coord
        cx, cy, cz = self.direction
        px, py, pz = pmt_coord

        dx = px - x0
        dy = py - y0
        dz = pz - z0

        self.refresh_kinematics_from_energy(initial_KE)

        u = cx * dx + cy * dy + cz * dz
        A = dx**2 + dy**2 + dz**2

        if A <= u**2:
            return u
        return u - self.cot_tq * np.sqrt(A - u**2)

    def get_emission_points(self, p_locations, initial_KE):
        """
        Vectorized Cherenkov emission-point calculation for many PMTs.
        """
        x0, y0, z0 = self.start_coord
        cx, cy, cz = self.direction

        p_locations = np.asarray(p_locations, dtype=np.float64)
        dx = p_locations[:, 0] - x0
        dy = p_locations[:, 1] - y0
        dz = p_locations[:, 2] - z0

        self.refresh_kinematics_from_energy(initial_KE)

        u = cx * dx + cy * dy + cz * dz
        A = dx * dx + dy * dy + dz * dz

        ss = np.empty(p_locations.shape[0], dtype=np.float64)
        valid = A > u * u
        ss[valid] = u[valid] - self.cot_tq * np.sqrt(A[valid] - u[valid] * u[valid])
        ss[~valid] = u[~valid]
        return ss

    def power_law(self, x):
        y0_fit = 0.1209
        yinf = 1.6397
        x50 = 0.9279
        n_fit = 3.0777

        x = np.clip(np.asarray(x, dtype=np.float64), 0.0, None)
        xn = x**n_fit
        x50n = x50**n_fit

        max_ = 0.967354918872639
        return (y0_fit + (yinf - y0_fit) * (xn / (xn + x50n))) / max_

    def wl_corr(self, x):
        x = np.asarray(x, dtype=np.float64)
        x_safe = np.maximum(x, 1e-12)

        ymin_wl = 0.1399
        ymax_wl = 1.0
        x50_wl = 3.7620
        n_wl = 2.1020

        return ymin_wl + (ymax_wl - ymin_wl) / (1.0 + (x50_wl / x_safe) ** n_wl)

    def get_expected_pes_ts(
        self,
        wcd,
        s,
        p_locations,
        direction_zs,
        corr_pos,
        obs_pes,
    ):
        """
        Expected PE and first-hit-time model used by the fit.

        The heavy cone-collapse work is delegated to the optimized solver in
        model_muon_cherenkov_collapse.py.
        """
        pmt_radius = (
            wcd.mpmts[0].pmts[0].get_properties("design")["size"] / 2.0
        )

        p_locations = np.asarray(p_locations, dtype=np.float64)
        direction_zs = np.asarray(direction_zs, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        obs_pes = np.asarray(obs_pes, dtype=np.float64)

        n_pmts = s.size

        start_pos = np.asarray(self.start_coord, dtype=np.float64)
        track_dir = np.asarray(self.direction, dtype=np.float64)
        track_dir = track_dir / np.linalg.norm(track_dir)

        scale = np.zeros(n_pmts, dtype=np.float64)
        s_b = np.zeros(n_pmts, dtype=np.float64)
        E_b = np.zeros(n_pmts, dtype=np.float64)

        # Use the current Cherenkov angle as the default for PMTs that never
        # receive collapse treatment.  Their final PE is zero anyway.
        base_theta = np.arccos(np.clip(self.cos_tq, -1.0, 1.0))
        theta_c_b = np.full(n_pmts, base_theta, dtype=np.float64)

        collapse_mask = s > -200.0
        idx = np.flatnonzero(collapse_mask)

        if idx.size:
            scale_sub, s_b_sub, E_b_sub = find_scale_for_pmts(
                pmt_pos=p_locations[idx],
                start_pos=start_pos,
                track_dir=track_dir,
                s_a_mm=0.001,
                s_max_mm=self.length,
                theta_c_func=theta_c_func,
                n_scan=150,
                near_cross_tol=0.02,
            )

            scale[idx] = scale_sub
            s_b[idx] = s_b_sub
            E_b[idx] = E_b_sub
            theta_c_b[idx] = theta_c_func(_get_tables()[0], _get_tables()[1], E_b_sub)

        use_collapse = scale > 0.0
        s_eff = np.where(use_collapse, s_b, s)

        front_mask = s_eff < pmt_radius
        if np.any(front_mask):
            scale[front_mask] *= (s_eff[front_mask] + pmt_radius) / (2.0 * pmt_radius)

        valid_s = s_eff >= -pmt_radius
        scale *= valid_s
        s_eff = np.where(valid_s, s_eff, 0.0)

        e_pos = start_pos[None, :] + s_eff[:, None] * track_dir[None, :]
        to_pmt = p_locations - e_pos
        r = np.linalg.norm(to_pmt, axis=1) + 0.01
        
        
        cost = -np.einsum("ij,ij->i", to_pmt, direction_zs) / r
        valid_cost = cost > 0.0
        scale *= valid_cost

        pwr_corr = self.power_law(cost)

        max_cher_angle = 0.732
#         corr = (
#             1000.0 * np.sin(theta_c_b)**(1)
#         ) / (r * np.sin(max_cher_angle)**(1)) * pwr_corr


        corr = n_from_E_r(E_b, r) * pwr_corr
        
    
        corr[~valid_cost] = 0.0

#         corr = (
#             1000.0 * np.sin(theta_c_b)**2
#         ) / (r * np.sin(max_cher_angle)**2) * pwr_corr
#         corr[~valid_cost] = 0.0

        # This quantity is currently not applied, but it is left here because it
        # exists in the original model and can be re-enabled easily later.
        _dist_from_stop = self.length - s_b
        _ = self.wl_corr(_dist_from_stop / 10.0)

        mean_pes = self.intensity * corr * scale
        mean_pes[mean_pes < 1e-3] = 0.0

        if corr_pos is not None and "wut" in corr_pos and len(corr_pos["wut"]) > 0:
            p_int = p_locations.astype(int)
            matches = (p_int[:, None, :] == corr_pos["wut"][None, :, :]).all(axis=2)
            mask = matches.any(axis=1)
            mean_pes[mask] *= 0.5

        t_light = r * self.n / self.c
        t_emitter = s_eff / self.v
        t_hits = self.starting_time + t_emitter + t_light

        # Preserve the original global intensity normalization step.
        mean_mean = float(np.mean(mean_pes))
        if mean_mean > 0.0:
            mean_pes *= float(np.mean(obs_pes)) / mean_mean

        return mean_pes, t_hits

    @staticmethod
    def get_pmt_placements(event, wcd, place_info):
        """
        Cache-friendly PMT geometry extraction.

        The fitter usually calls this once per detector configuration, so a
        simple straight loop is enough here.
        """
        p_locations = []
        direction_zs = []

        for i_mpmt in range(event.n_mpmt):
            if not event.mpmt_status[i_mpmt]:
                continue

            mpmt = wcd.mpmts[i_mpmt]
            if mpmt is None:
                continue

            for i_pmt in range(event.npmt_per_mpmt):
                if not event.pmt_status[i_mpmt][i_pmt]:
                    continue

                pmt = mpmt.pmts[i_pmt]
                if pmt is None:
                    continue

                placement = pmt.get_placement(place_info, wcd)
                p_locations.append(np.asarray(placement["location"], dtype=np.float64))
                direction_zs.append(np.asarray(placement["direction_z"], dtype=np.float64))

        return np.asarray(p_locations, dtype=np.float64), np.asarray(direction_zs, dtype=np.float64)

    def get_cone_can_intersection_points(self,
            r: float,  # cylinder radius
            ht: float, hb: float,  # top and bottom endcap y (ht > hb)
            n: int,  # number of azimuth samples
            flen: float = 0.  # fractional position along cone axis for apex (0=start, 1=end)
    ) -> List[Tuple[float, float, float]]:
        """
        Return n+1 intersection points (last repeats first) of a right circular cone
        (apex at (x0,y0,z0), axis with direction cosines (cx,cy,cz), half-angle q)
        with the finite cylinder (axis = y, radius r, endcaps at y = hb and y = ht).

        For each azimuth ray on the cone, there is exactly one intersection with the
        cylindrical can (your assumption). If the side intersection is outside the
        y-interval, the intersection is on the corresponding endcap.

        Returns: list of (xi, yi, zi), length n+1 with points[0] == points[-1].
        """
        (x0, y0, z0) = self.start_coord + flen * self.length * np.array(self.direction)
        (cx, cy, cz) = self.direction
        q = np.arccos(self.cos_tq)  # half-angle in radians
        if not (0.0 < q < 0.5 * np.pi):
            raise ValueError("Cone half-angle q must be in (0, pi/2) radians.")
        if ht <= hb:
            raise ValueError("Cylinder top ht must be greater than bottom hb.")
        if r <= 0.0:
            raise ValueError("Cylinder radius r must be positive.")
        if n < 3:
            raise ValueError("Number of azimuth samples n must be at least 3.")

        eps = 1e-12

        # Normalize axis c
        c = np.array([cx, cy, cz], dtype=float)
        c_norm = np.linalg.norm(c)
        if c_norm == 0:
            raise ValueError("Axis direction (cx,cy,cz) must be nonzero.")
        c = c / c_norm

        # Build orthonormal basis {u, v, c} with u,v ⟂ c
        # Choose a helper not nearly parallel to c
        helper = np.array([1.0, 0.0, 0.0]) if abs(c[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(c, helper)
        u_norm = np.linalg.norm(u)
        if u_norm < eps:
            helper = np.array([0.0, 0.0, 1.0])
            u = np.cross(c, helper)
            u_norm = np.linalg.norm(u)
            if u_norm < eps:
                raise ValueError("Failed to construct basis perpendicular to axis.")
        u = u / u_norm
        v = np.cross(c, u)  # already unit

        # Precompute constants
        cosq = np.cos(q)
        sinq = np.sin(q)
        apex = np.array([x0, y0, z0], dtype=float)

        # Azimuth samples
        theta = np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)
        ct = np.cos(theta)
        st = np.sin(theta)

        # Generator directions (unit) for each azimuth
        dirs = (cosq * c)[None, :] + (sinq * ct)[:, None] * u[None, :] + (sinq * st)[:, None] * v[None, :]
        dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]

        # Quadratic for intersection with infinite cylinder x^2 + z^2 = r^2:
        # a t^2 + b t + c0 = 0 for (x,y,z) = apex + t*dir
        a = dx * dx + dz * dz
        b = 2.0 * (x0 * dx + z0 * dz)
        c0 = x0 * x0 + z0 * z0 - r * r

        # Discriminant and roots (vectorized)
        disc = np.maximum(0.0, b * b - 4.0 * a * c0)  # clamp tiny negatives to 0
        sqrt_disc = np.sqrt(disc)
        denom = 2.0 * a

        # Two roots; pick the smallest positive t
        t1 = (-b - sqrt_disc) / denom
        t2 = (-b + sqrt_disc) / denom

        # Mask out non-forward intersections; choose min positive
        t_candidates = np.stack([np.where(t1 > eps, t1, np.inf),
                                 np.where(t2 > eps, t2, np.inf)], axis=0)
        t_side = np.min(t_candidates, axis=0)

        # Side hit position
        x_side = x0 + t_side * dx
        y_side = y0 + t_side * dy
        z_side = z0 + t_side * dz

        # Decide final intersection:
        # - if hb <= y_side <= ht: keep side hit
        # - if y_side < hb: snap to bottom cap at y=hb
        # - if y_side > ht: snap to top cap at y=ht
        y_plane = np.where(y_side < hb, hb, np.where(y_side > ht, ht, np.nan))

        # For plane hits, recompute t from y = y_plane
        # Guard against |dy| ~ 0 by nudging with eps; your uniqueness guarantee
        # implies this division should be safe, but we keep it numerically stable.
        dy_safe = np.where(np.abs(dy) < eps, np.sign(dy) * eps, dy)
        t_cap = (y_plane - y0) / dy_safe  # valid only where y_plane is not nan

        # Compute cap positions
        x_cap = x0 + t_cap * dx
        z_cap = z0 + t_cap * dz

        # Choose between side and cap per-ray
        use_cap = ~np.isnan(y_plane)
        xi = np.where(use_cap, x_cap, x_side)
        yi = np.where(use_cap, y_plane, y_side)
        zi = np.where(use_cap, z_cap, z_side)

        # Stack and append first point to close the loop
        pts = np.stack([xi, yi, zi], axis=1)
        pts_closed = np.vstack([pts, pts[:1]])

        # Convert to list of tuples
        return [tuple(row) for row in pts_closed]
