import numpy as np
import pickle
from typing import List, Tuple
import sys
sys.path.insert(0, "/eos/user/j/jrimmer/SWAN_projects/beam/LicketFit2/LicketyFit")
from model_muon_cherenkov_collapse import *

sys.path.insert(0, "/eos/user/j/jrimmer/Geometry") # Input path to Geometry repo
#sys.path.insert(0, "../")

sys.path.insert(0, "../tables")
from Geometry.Device import Device


#init_E_vs_dist_travelled = np.load('../tables/init_E_vs_dist_travelled.npy')
# init_E_vs_dist_travelled = np.load('../tables/E_vs_dist.npy')
# dist_travelled = init_E_vs_dist_travelled[:,0]
# init_energy = init_E_vs_dist_travelled[:,1]
E_vs_dist = np.load('/eos/user/j/jrimmer/SWAN_projects/beam/LicketyFit2/tables/E_vs_dist_cm.npy', allow_pickle=True) # need to convert this from cm to mm later
overall_distances = np.load('/eos/user/j/jrimmer/SWAN_projects/beam/LicketyFit2/tables/overall_distances_cm.npy')*10 # convert to mm

init_energy = np.array([a[:, 1] for a in E_vs_dist], dtype=object)
dist_travelled = np.array([a[:, 0] for a in E_vs_dist], dtype=object)*10 #change to mm


c_ang_vs_E = np.load('/eos/user/j/jrimmer/SWAN_projects/beam/LicketyFit2/tables/mu_cAng_vs_E_n1344.npy')
c_ang = c_ang_vs_E[:,0]
energy_for_angle = c_ang_vs_E[:,1]


class Emitter:
    """
    An emitter of Cherenkov radiation in a medium

    Attributes:
        starting_time (float): The time that emitter starts emission in nanoseconds.
        start_coord (tuple): The starting coordinates (x, y, z) of the emitter.
        direction (tuple): The direction cosines vector (cx, cy, cz) of the emitter.
        beta (float): The velocity of the emitter as a fraction of the speed of light.
        length (float): The length of the path for the emitter (mm).
        intensity (float): The intensity of the emitted radiation (scaled
                           to mean PE for a PMT at distance 1m at normal incidence).
    """
    def __init__(self, starting_time, start_coord, direction, beta, length, intensity):
        if not isinstance(starting_time, (int, float)):
            raise TypeError("starting_time must be a number")
        if not (isinstance(start_coord, tuple) and len(start_coord) == 3 and all(isinstance(c, (int, float)) for c in start_coord)):
            raise TypeError("start_coord must be a tuple of three numbers")
        if not (isinstance(direction, tuple) and len(direction) == 3 and all(isinstance(c, (int, float)) for c in direction)):
            raise TypeError("direction must be a tuple of three numbers")
        if not isinstance(beta, (int, float)) or not (0 < beta < 1):
            raise ValueError("beta must be a number between 0 and 1")
        if not isinstance(length, (int, float)):
            raise TypeError("length must be a number")
        if length <= 0:
            raise ValueError("length must be positive")
        if not isinstance(intensity, (int, float)):
            raise TypeError("intensity must be a number")
        if intensity <= 0:
            raise ValueError("intensity must be positive")

        self.starting_time = float(starting_time)
        self.start_coord = tuple(float(c) for c in start_coord)
        self.direction = tuple(float(c) for c in direction)
        #self.beta = float(beta)
        self.length = float(length)
        self.intensity = float(intensity)

        self.cos_tq = None # cosine of the Cherenkov angle in water
        self.cot_tq = None
        self.n = None  # refractive index of water
        self.c = None
        self.v = None
        #self.calc_constants(1.36) # hard coded refractive index for water at 300 nm
        
        
        self.mu_mass = 105.658 # In MeV/c^2
        
        #self.interp_E_init = np.interp(self.length, dist_travelled*10, init_energy)
        main_idx = np.searchsorted(overall_distances, self.length)
        main_idx = np.clip(main_idx, 1, len(overall_distances) - 1)
        left = overall_distances[main_idx - 1]
        right = overall_distances[main_idx]
        main_idx -= (self.length - left) <= (right - self.length)
        
        

        self.interp_E_init = init_energy[main_idx][0]
        
        self.beta = np.sqrt(1-(self.mu_mass/(self.interp_E_init+self.mu_mass))**2)
        
        self.calc_constants(1.344)
        print('INIT ENERGY',self.interp_E_init)
        print('BETA VALUE IS',self.beta)

    def __repr__(self):
        return (f"Emitter(starting_time={self.starting_time}, start_coord={self.start_coord}, "
                f"direction={self.direction}, beta={self.beta}, length={self.length}, "
                f"intensity={self.intensity})")

    def copy(self):
        """Create a deep copy of the Emitter object.

        Returns:
            Emitter: A deep copy of the current Emitter object.
        """
        return pickle.loads(pickle.dumps(self))

    def calc_constants(self, n):
        """Calculate and store constants related to the Cherenkov angle."""
        self.n = n
        self.cos_tq = 1./self.beta/self.n
        print('COSINE',self.cos_tq)
        sin_tq = np.sqrt(1. - self.cos_tq**2)
        self.cot_tq = self.cos_tq / sin_tq
        self.c = 299.792458  # speed of light in mm/ns
        self.v = self.beta * self.c  # velocity of the emitter in mm/ns

    def set_nominal_track_parameters(self, starting_time, start_coord, direction, length):
        """Set the nominal track parameters of the emitter."""

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

    # Change of variable equations for nominal parameters to alternative parameters (thanks to ChatGPT5).
    # This is considered as a way to eliminate the strong correlation between x0 and cx and between y0 and cy.
    # Label this choice of parameters as "wall" parameters since they relate to a cylinder wall intersection.

    # A line starts at Cartesian coordinates (x_0,y_0,z_0) within a cylinder of radius r and axis being the y-axis.
    # The direction of the line is given by the unit vector (c_x, c_y, c_z) with c_z^2 = (1-c_x^2 -c_y^2).
    # The line intersects the cylinder at the point in cylindrical coordinates (r=r, h=y_w, phi=phi_w) and that point is a distance d_w from the start of the line.
    # The cylindrical coordinate phi=0 coincides with the z-axis
    # At the intersection point, w_y is the cosine of the angle between the line direction and the y-axis, and w_phi is the cosine of the angle in
    # the x-y plane between the line direction and the tangent to the cylinder that points in the +phi direction.

    # ============================================================
    #  Cylinder axis = y,  phi_w = atan2(x_w, z_w)  (φ=0 on +z)
    #  x_w = r*sinφ,  z_w = r*cosφ
    #  n = (sinφ, 0, cosφ),  t_phi = (cosφ, 0, -sinφ)
    # ============================================================

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

    @staticmethod
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
        """ Find the emission point of Cherenkov radiation along the emitter's path.
            Given the direction cosines  (cx, cy, cz) and start of the track (x0, y0, z0), and the centre of
            the PMT (px, py, pz), the distance parameter s along the track to the emission point is given by:

            s = u - cot_q sqrt(A - u^2)
            where
            u = cx (px - x0) + cy (py - y0) + cz (pz - z0)
            A = (px - x0)^2 + (py - y0)^2 + (pz - z0)^2

        Args:
            pmt_coord (tuple): The coordinates (px, py, pz) of the PMT

        Returns:
            s: The distance parameter from the start of the track

        Note: if the PMT centre is exactly aligned with the emitter track, then A = u^2 and there is no valid
              emission point. In this case s = u, and the light intensity at the PMT is infinite (unphysical).

        """
        x0, y0, z0 = self.start_coord
        cx, cy, cz = self.direction
        px, py, pz = pmt_coord

        dx = px - x0
        dy = py - y0
        dz = pz - z0
        
        self.beta = np.sqrt(1-(self.mu_mass/(initial_KE+self.mu_mass))**2)
        #print('BETA AFTER',self.beta)
        self.cos_tq = 1./self.beta/self.n
        
        sin_tq = np.sqrt(1. - self.cos_tq**2)
        self.cot_tq = self.cos_tq / sin_tq
        

        u = cx * dx + cy * dy + cz * dz
        A = dx**2 + dy**2 + dz**2

        if A <= u**2: # perfect alignment case
            return u
        s = u - self.cot_tq * np.sqrt(A - u**2)
        return s

    # This is the model as a substitute for cos theta PMT acceptance
    def power_law(x, x0_fit, n_fit):
        max_ = np.power(1 - x0_fit, n_fit)
        norm = np.power(x - x0_fit, n_fit)/max_
        
        return norm
    
    def get_emission_points(self, p_locations, initial_KE):
        """ Perform the same calculation as get_emission_point for a list of PMT coordinates.
        Use NumPy vectorization for speeding up the Likelihood calculation."""
        x0, y0, z0 = self.start_coord
        cx, cy, cz = self.direction
        p_locations = np.array(p_locations)
        px = p_locations[:, 0]
        py = p_locations[:, 1]
        pz = p_locations[:, 2]

        dx = px - x0
        dy = py - y0
        dz = pz - z0
        
        self.beta = np.sqrt(1-(self.mu_mass/(initial_KE+self.mu_mass))**2)
        #print('BETA AFTER',self.beta)
        self.cos_tq = 1./self.beta/self.n
        
        sin_tq = np.sqrt(1. - self.cos_tq**2)
        self.cot_tq = self.cos_tq / sin_tq

        u = cx * dx + cy * dy + cz * dz
        A = dx**2 + dy**2 + dz**2

        valid = A > u**2
        invalid = A <= u**2
        ss = np.full(len(p_locations), np.nan)
        #ss = np.full(len(p_locations), None)
        ss[valid] = u[valid] - self.cot_tq * np.sqrt(A[valid] - u[valid]**2)
        ss[invalid] = u[invalid]
        return ss
    
    def power_law(self,x):
        # x must be >= 0; enforce numerical safety
        y0_fit   = 0.1209
        #yinf = 41.3407 # plateau
        yinf = 1.6397 # make it less steep at large cost
        x50  = 0.9279    #(half-saturation)
        n_fit    = 3.0777    #  (steepness)
        
        x = np.clip(x, 0.0, None)
        xn = x**n_fit
        x50n = x50**n_fit

        max_ = 0.967354918872639
        norm = (y0_fit + (yinf - y0_fit) * (xn / (xn + x50n)))/max_


        return norm
    
    def wl_corr(self,x):
        x = np.asarray(x)
        # avoid divide-by-zero if you ever have x=0
        x_safe = np.maximum(x, 1e-12)
        
        ymin_wl = 0.1399
        #ymin_wl = 0.05
        ymax_wl = 1
        x50_wl  = 3.7620
        n_wl    = 2.1020
        
        return ymin_wl + (ymax_wl - ymin_wl) / (1.0 + (x50_wl / x_safe)**n_wl)


    

                
    
    
    def emit(self, simulated_event, wcd, place_info):
        """Simulate the emission of Cherenkov radiation and record hits in a simulated event for the water cherenkov
        detector, wcd. NOTE: changes to this method need to be reflected in get_expected_pes().

        Args:
            simulated_event (SimulatedEvent): The simulated event to record hits in.
            wcd (Geometry.wcd): The water Cherenkov detector with the PMT geometry.
            place_info (string): specify which geometric info to use, e.g. 'design', 'est' (as-built)
        """

        costs = []
        mean_pes = []
        
        test_scale = []

        #diameter of a PMT active area
        pmt_radius = wcd.mpmts[0].pmts[0].get_properties('design')['size']/2.
        print('PMT RADIUS', pmt_radius)
        
        #interp_E_init = np.interp(self.length, dist_travelled*10, init_energy) # Convert distance to mm
        #interp_E_init = np.interp(self.length, dist_travelled*10, init_energy) # Convert distance to mm
        main_idx = np.searchsorted(overall_distances, self.length)
        main_idx = np.clip(main_idx, 1, len(overall_distances) - 1)
        left = overall_distances[main_idx - 1]
        right = overall_distances[main_idx]
        main_idx -= (self.length - left) <= (right - self.length)
        interp_E_init = init_energy[main_idx][0]
        
        # loop over all active mPMTs in the WCD
        for i_mpmt in range(simulated_event.n_mpmt):
            #if i_mpmt == 53:
           #     print('mPMT 53')
           # print('mPMT', i_mpmt)
        
            if not simulated_event.mpmt_status[i_mpmt]:
                continue

            mpmt = wcd.mpmts[i_mpmt]
            print('mPMT',i_mpmt)
            
            # check emission point for i_pmt = 0 in this mPMT. If far outside emission path, skip this mPMT
            pmt = mpmt.pmts[0]
            p = pmt.get_placement(place_info,wcd)
            p_location = p['location']
            s = self.get_emission_point(tuple(p_location),interp_E_init)
            #if s is None or s < -200 or s > self.length + 200:
            #if s is None or s < -200:
             #   continue

            # loop over all active PMTs in the mPMT
            for i_pmt in range(simulated_event.npmt_per_mpmt):
                
                if not simulated_event.pmt_status[i_mpmt][i_pmt]:
                    continue
                print('PMT',i_pmt)
                pmt = mpmt.pmts[i_pmt]
                p = pmt.get_placement(place_info, wcd)
                p_location = p['location']
                s = self.get_emission_point(tuple(p_location),interp_E_init)
                
               
                
                frac_area = 1.
                
#                 if s is None or s < -pmt_radius:
#                     continue
                    # if close to edge of emitter track, calculate the fraction of PMT area that is illuminated
                
                #if s < pmt_radius:
                if s<-1000:
                    continue
                
                else:
                    
                    #pmt_pos = np.array([500,500,1000])
                    start_pos = np.array([self.start_coord[0],self.start_coord[1],self.start_coord[2]])
                    track_dir = np.array([self.direction[0],self.direction[1],self.direction[2]])
                    s_max_mm = self.length
                    #s_a_mm = self.length - collapse_length
                    s_a_mm = 0.00001
                    
                    #E_at_sa_MeV = 200
                    if i_mpmt == 37:
                        mpmt_bool = False
                        
                    else:
                        mpmt_bool = False
                    
                    E_at_sa_MeV = interp_E_init #(self.length/10.)*2 #Estimate initial energy as 2 x track length
                    #dEdx_MeV_per_cm = 2

                    scale_sub, s_b_sub, E_b_sub = find_scale_for_pmts(
                        pmt_pos   = np.array([p_location]),
                        start_pos = np.array([start_pos]),
                        track_dir = track_dir,
                        s_a_mm    = 0.000001,
                        s_max_mm  = self.length,
                        #E_at_sa_MeV=(self.length / 10.0) * 2.0,  # ~2 MeV/cm
                        #E_at_sa_MeV = interp_E_init,
                        #dEdx_MeV_per_cm=2.0,
                        theta_c_func=theta_c_func ,
                        mpmt_bool = mpmt_bool
                    )
#                     if i_mpmt == 37:
#                         print('scale_sub',scale_sub)
#                         print('s_b_sub',s_b_sub)
                    
                    #frac_area = (self.length - collapse_length)/s
                    frac_area = scale_sub
                    #test_scale.append(test_scale_sub)
                    
                    if frac_area <= 0.:
                        continue
                    #print(frac_area)
                    #if i_mpmt == 53:
                     #   print('scale',frac_area)
                 
                try:
                    s_b = int(s_b_sub)
                    E_b = int(E_b_sub)
                    
                except:
                    continue
                
                
                if s_b is None or s_b < -pmt_radius:
                    continue
                    
                
                
                if s_b < pmt_radius:
                    
                    frac_area = frac_area *((s_b+pmt_radius) / (2.*pmt_radius))
                   
                    
                
                #print(s_b)
                if frac_area is None:
                    continue
                if frac_area <= 0.:
                    continue

                # calculate the emission point
                ex = self.start_coord[0] + self.direction[0] * s_b
                ey = self.start_coord[1] + self.direction[1] * s_b
                ez = self.start_coord[2] + self.direction[2] * s_b
                e_location = [ex, ey, ez]

                to_pmt = np.array(p_location) - np.array(e_location)
                r = np.linalg.norm(to_pmt) + 0.01 # add 1 mm to avoid possible div by zero (see note in get_emission_point)
                #print('r',r)

                direction_z = p['direction_z']
                cost = -1.*np.dot(to_pmt,direction_z)/r
                
                #print('cost',cost)
                # ignore light hitting the back of the PMT
                #if cost <= 0.:
                #    continue

                # corr is the simple geometric correction factor based on the distance and angle
                # r0 is the reference distance (1 m), cost0 is the reference costheta
                # note that Cherenkov light density scales as 1/r, not 1/r^2 (parallel rays from the emitter)
                r0 = 1000.
                cost0 = 1.0
                
                # This is the model as a substitute for cos theta PMT acceptance
                
                
                

#                 if i_mpmt == 37 or i_mpmt == 52:
#                     print('mPMT',i_mpmt)
#                     print('PMT',i_pmt)
#                     print('cost',cost)
#                     print('theta',np.arccos(cost)*180/np.pi)
#                     print('')

                if cost<0:
                    continue
                pwr_corr = self.power_law(cost)
                
                max_cher_angle = 0.732
                
                theta_c_b = theta_c_func(c_ang, energy_for_angle, E_b)
                
                new_corr = (r0*np.sin(theta_c_b))/(r*np.sin(max_cher_angle))*pwr_corr
               
                #corr = r0/r*cost/cost0
                corr = r0/r*pwr_corr/cost0 #Try the power law instead of simple cosine
                print('normal corr',corr)
                print('new corr', new_corr)
                
                # Lastly, apply a correction based on the wavelength of light that is emitted at the end of the track
                
                # Get distance from end of track
                dist_from_stop = self.length - s_b
                
                
                
                wl_fac = self.wl_corr(dist_from_stop/10)
                
                #print(wl_fac)
                
                # expected number of PE from this emitter in this PMT
                mean_pe = self.intensity * corr * frac_area 
                #print(mean_pe)
                
               
                
                if mean_pe <= 0.0001:
                    continue
                    
                # actual number of PE from Poisson distribution
                n_pe = np.random.poisson(mean_pe)
                
                # time for light to reach the PMT
                t_light = r * self.n / self.c  # in ns
                # time for emitter to reach the emission point
                t_emitter = s_b / self.v  # in ns
                #t_emitter = s / self.v 
                # total time of hit
                t_hit = self.starting_time + t_emitter + t_light
                
                # record the truth information about the hit in the simulated event
                simulated_event.expected_hit_times[i_mpmt][i_pmt].append(t_hit)
                simulated_event.expected_hit_pe[i_mpmt][i_pmt].append(mean_pe)
                simulated_event.true_hit_pe[i_mpmt][i_pmt].append(float(n_pe))

                costs.append(cost)
                mean_pes.append(mean_pe)
                

        simulated_event.emitters.append(self)
        

        return costs, mean_pes
   

    #def emit(self, simulated_event, wcd, place_info):
#         """Simulate the emission of Cherenkov radiation and record hits in a simulated event for the water cherenkov
#         detector, wcd. NOTE: changes to this method need to be reflected in get_expected_pes().

#         Args:
#             simulated_event (SimulatedEvent): The simulated event to record hits in.
#             wcd (Geometry.wcd): The water Cherenkov detector with the PMT geometry.
#             place_info (string): specify which geometric info to use, e.g. 'design', 'est' (as-built)
#         """

#         costs = []
#         mean_pes = []

#         #diameter of a PMT active area
#         pmt_radius = wcd.mpmts[0].pmts[0].get_properties('design')['size']/2.

#         # loop over all active mPMTs in the WCD
#         for i_mpmt in range(simulated_event.n_mpmt):
#             if not simulated_event.mpmt_status[i_mpmt]:
#                 continue

#             mpmt = wcd.mpmts[i_mpmt]
#             # check emission point for i_pmt = 0 in this mPMT. If far outside emission path, skip this mPMT
#             pmt = mpmt.pmts[0]
#             p = pmt.get_placement(place_info,wcd)
#             p_location = p['location']
#             s = self.get_emission_point(tuple(p_location))
#             if s is None or s < -200 or s > self.length + 200:
#                 continue

#             # loop over all active PMTs in the mPMT
#             for i_pmt in range(simulated_event.npmt_per_mpmt):
#                 if not simulated_event.pmt_status[i_mpmt][i_pmt]:
#                     continue
#                 pmt = mpmt.pmts[i_pmt]
#                 p = pmt.get_placement(place_info, wcd)
#                 p_location = p['location']
#                 s = self.get_emission_point(tuple(p_location))
#                 if s is None or s < -pmt_radius or s > self.length + pmt_radius:
#                     continue
#                     # if close to edge of emitter track, calculate the fraction of PMT area that is illuminated
#                 frac_area = 1.
#                 if s < pmt_radius:
#                     frac_area = (s+pmt_radius) / (2.*pmt_radius)
#                 elif s > self.length - pmt_radius:
#                     frac_area = (self.length - s + pmt_radius) / (2.*pmt_radius)
#                 if frac_area <= 0.:
#                     continue

#                 # calculate the emission point
#                 ex = self.start_coord[0] + self.direction[0] * s
#                 ey = self.start_coord[1] + self.direction[1] * s
#                 ez = self.start_coord[2] + self.direction[2] * s
#                 e_location = [ex, ey, ez]

#                 to_pmt = np.array(p_location) - np.array(e_location)
#                 r = np.linalg.norm(to_pmt) + 1 # add 1 mm to avoid possible div by zero (see note in get_emission_point)

#                 direction_z = p['direction_z']
#                 cost = -1.*np.dot(to_pmt,direction_z)/r
#                 # ignore light hitting the back of the PMT
#                 if cost <= 0.:
#                     continue

#                 # corr is the simple geometric correction factor based on the distance and angle
#                 # r0 is the reference distance (1 m), cost0 is the reference costheta
#                 # note that Cherenkov light density scales as 1/r, not 1/r^2 (parallel rays from the emitter)
#                 r0 = 1000.
#                 cost0 = 1.0
#                 corr = r0/r*cost/cost0
                
#                 #apply an additional correction for the PMT angular response here (at some point)
                
#                 # expected number of PE from this emitter in this PMT
#                 mean_pe = self.intensity * corr * frac_area
                
#                 if mean_pe <= 0.001:
#                     continue
                    
#                 # actual number of PE from Poisson distribution
#                 n_pe = np.random.poisson(mean_pe)
                
#                 # time for light to reach the PMT
#                 t_light = r * self.n / self.c  # in ns
#                 # time for emitter to reach the emission point
#                 t_emitter = s / self.v  # in ns
#                 # total time of hit
#                 t_hit = self.starting_time + t_emitter + t_light
                
#                 # record the truth information about the hit in the simulated event
#                 simulated_event.expected_hit_times[i_mpmt][i_pmt].append(t_hit)
#                 simulated_event.expected_hit_pe[i_mpmt][i_pmt].append(mean_pe)
#                 simulated_event.true_hit_pe[i_mpmt][i_pmt].append(float(n_pe))

#                 costs.append(cost)
#                 mean_pes.append(mean_pe)

#         simulated_event.emitters.append(self)

#         return costs, mean_pes
   
    
    def get_correction_pos_dict(self,wcd):
    
        hall = Device.open_file('../../../../../Geometry/examples/wcte_bldg157.geo')
        #wcte = hall.wcds[0]

        # Get a dictionary to determine any corrections that need to be applied
        with open('/eos/user/j/jrimmer/SWAN_projects/data_checks/other_mpmt_info.dict', 'rb') as f:
            other_mpmt_info = pickle.load(f)

        place_info = 'design'

        # loop over all active mPMTs in the WCD and get positions of certain mPMTs that need corrections
        corr_mpmts = {'wut':[],'delam':[]}
        place_info = 'design'

        for i_mpmt in other_mpmt_info:

            mpmt = wcd.mpmts[i_mpmt]

            for i_pmt in range(19):

                pmt = mpmt.pmts[i_pmt]
                p = pmt.get_placement(place_info, wcd)['location']


                if other_mpmt_info[i_mpmt]['mpmt_site'] == 'WUT' and other_mpmt_info[i_mpmt]['mpmt_type'] == 'Ex-situ':
                    corr_mpmts['wut'].append([int(p[0]),int(p[1]),int(p[2])])

        corr_mpmts['wut'] = np.array(corr_mpmts['wut'])
        corr_mpmts['delam'] = np.array(corr_mpmts['delam'])
        
        return corr_mpmts
    
    
    def get_expected_pes_ts(
    self,
    wcd,
    s,                  # (N,) Cherenkov-cone emission points (fixed-angle)
    p_locations,        # (N,3) PMT positions
    direction_zs,       # (N,3) PMT pointing directions
    corr_pos,
    obs_pes             # For the purpose of getting the intensity right
):
        """
        Model B with masked cone-collapse:
        - Collapse geometry is computed only for PMTs with s > 0
        - s_b replaces s consistently for geometry, charge, and timing
        - PMTs with unphysical emission points are explicitly eliminated
        """
        pmt_radius = wcd.mpmts[0].pmts[0].get_properties('design')['size']/2
        # ================================================================
        # 1) Basic setup
        # ================================================================
        p_locations = np.asarray(p_locations, dtype=float)
        direction_zs = np.asarray(direction_zs, dtype=float)
        s = np.asarray(s, dtype=float)

        N = len(s)
        

        start_pos = np.asarray(self.start_coord, dtype=float)
        track_dir = np.asarray(self.direction, dtype=float)
        track_dir /= np.linalg.norm(track_dir)

        # ================================================================
        # 2) Allocate collapse outputs (FULL size, default zero)
        # ================================================================
        scale = np.zeros(N, dtype=float)
        s_b   = np.zeros(N, dtype=float)
        E_b = np.zeros(N, dtype=float)
        theta_c_b = np.zeros(N, dtype=float)

        # ================================================================
        # 3) Define which PMTs get collapse treatment
        #    
        # ================================================================
        
        #interp_E_init = np.interp(self.length, dist_travelled*10, init_energy) # Convert distance to mm
        main_idx = np.searchsorted(overall_distances, self.length)
        main_idx = np.clip(main_idx, 1, len(overall_distances) - 1)
        left = overall_distances[main_idx - 1]
        right = overall_distances[main_idx]
        main_idx -= (self.length - left) <= (right - self.length)
        interp_E_init = init_energy[main_idx][0]

        #print(interp_E_init)
        
        collapse_mask = s > -200
        
        select_pmts = False
        
        # If you only want to choose PMTs that seem likely to see light
        if (select_pmts):
            s_probe = np.array([0.05, 0.25, 0.5, 0.75, 0.95]) * self.length  # 5 points
            
            # emission points: (K,3)
            emit_probe = start_pos + np.outer(s_probe, track_dir)

            # vectors to PMTs: (N,K,3)
            d = p_locations[:, None, :] - emit_probe[None, :, :]

            r = np.linalg.norm(d, axis=2) + 1e-9
            cos_alpha = (d @ track_dir) / r   # (N,K)  (numpy uses matmul broadcasting here)
            cos_alpha = np.clip(cos_alpha, -1, 1)
            alpha = np.arccos(cos_alpha)      # (N,K)
            
            probe_energies = interp_E_init-0.2*s_probe # 0.2 MeV/mm
            print('probe energies',probe_energies)

            theta0 = theta_c_func(c_ang, energy, probe_energies)  # scalar
            print('test thetas',theta0)
            dalpha = np.abs(alpha - theta0)                      # (N,K)
            ring_mask = np.any(dalpha < np.deg2rad(6.0), axis=1) # 6° window; tune

            candidate_mask = collapse_mask & ring_mask
            idx = np.where(candidate_mask)[0]
            
        else:
            idx = np.where(collapse_mask)[0]

        
        #print('length',len(idx))

        # ================================================================
        # 4) Run expensive collapse solver ONLY on masked PMTs
        # ================================================================
        
       
        #print(interp_E_init)
        
        if idx.size > 0:
            scale_sub, s_b_sub, E_b_sub = find_scale_for_pmts(
                pmt_pos   = p_locations[idx],
                start_pos = start_pos,
                track_dir = track_dir,
                s_a_mm    = 0.001,
                s_max_mm  = self.length,
                #E_at_sa_MeV=(self.length / 10.0) * 2.0,  # ~2 MeV/cm
                #E_at_sa_MeV = interp_E_init,
                #dEdx_MeV_per_cm=2.0,
                theta_c_func=theta_c_func
            )
            
            #print('E_b_sub len',len(E_b_sub))
            scale[idx] = scale_sub
            s_b[idx]   = s_b_sub
            E_b[idx] = E_b_sub
            
            
            
            
            theta_c_b[idx] = theta_c_func(c_ang, energy_for_angle, E_b[idx])
            
            
        #print(scale_sub)
        # ================================================================
        # 5) Choose the effective emission point
        # ================================================================
        use_collapse = scale > 0.0
        s_eff = np.where(use_collapse, s_b, s)
        

        #print('LENGTH 1',len(s_eff))
        # ================================================================
        # 6) Explicitly eliminate unphysical emission points (s_eff < 0)
        # ================================================================
        
        
        
#         valid_s = s_eff >= 0.0
#         scale = scale * valid_s
       
        
#         s_eff = np.where(valid_s, s_eff, 0.0)
        
#         scale[s_eff is None] = 0.
        
        scale[s_eff < pmt_radius] = (s_eff[s_eff < pmt_radius] + pmt_radius) / (2.*pmt_radius) # uncomment for original
        
        
        valid_s = s_eff >= -pmt_radius #uncomment for original code
        #valid_s = s_eff >= -10*pmt_radius
        
        scale = scale * valid_s
        s_eff = np.where(valid_s, s_eff, 0.0)
        
        
        
        
        #scale[s_eff < -pmt_radius] = 0.
        
       
        

        # ================================================================
        # 7) Compute emission positions from s_eff
        # ================================================================
        e_pos = start_pos + np.outer(s_eff, track_dir)   # (N,3)

        # ================================================================
        # 8) Photon geometry
        # ================================================================
        to_pmt = p_locations - e_pos
        r = np.linalg.norm(to_pmt, axis=1) + 0.01  # +0.01 mm safety
        #print('r len',len(r))

        direction_zx = direction_zs[:, 0]
        direction_zy = direction_zs[:, 1]
        direction_zz = direction_zs[:, 2]

        cost = -(
            to_pmt[:, 0] * direction_zx +
            to_pmt[:, 1] * direction_zy +
            to_pmt[:, 2] * direction_zz
        ) / r

        # PMT must face emission point
        valid_cost = cost > 0.0
        scale = scale * valid_cost
        
#         def power_law(x, y0_fit, yinf, x50, n_fit):
#             # x must be >= 0; enforce numerical safety
#             x = np.clip(x, 0.0, None)
#             xn = x**n_fit
#             x50n = x50**n_fit

#             max_ = 0.967354918872639
#             norm = (y0_fit + (yinf - y0_fit) * (xn / (xn + x50n)))/max_


#             return norm
                
#         y0_fit   = 0.1209
#         #yinf = 41.3407 # plateau
#         yinf = 1.6397 # make it less steep at large cost
#         x50  = 0.9279    #(half-saturation)
#         n_fit    = 3.0777    #  (steepness)
        pwr_corr = self.power_law(cost)
        
        # ================================================================
        # 9) Geometric correction
        # ================================================================
        
        #max_cher_angle = 0.732 # maximum cherenkov angle for n = 1.344
        max_cher_angle = 0.732
        
        r0 = 1000.0
        #corr = r0 / r * cost
        #corr = r0/r*pwr_corr
        #old_corr = r0/r*pwr_corr
        corr = (r0*(np.sin(theta_c_b))**2)/(r*np.sin(max_cher_angle)**2)*pwr_corr
        
#         for i in range(len(corr[idx])):
#             print('corr',old_corr[idx][i])
#             print('new_corr',corr[idx][i])
#             print('scale',scale[idx][i])
#             print('theta_c',theta_c_b[idx][i])
#             print('PMT location',p_locations[idx][i])
#             print('')
        
        #corr = r0/(r**2)*pwr_corr
        corr[~valid_cost] = 0.0
        
        dist_from_stop = self.length - s_b
              

        wl_fac = self.wl_corr(dist_from_stop/10)
        #wl_fac = 1. # Try getting rid of end-of-track wavelength shifting
        
        # ================================================================
        # 10) Expected PE calculation
        # ================================================================
        mean_pes = self.intensity * corr * scale 
        mean_pes[mean_pes < 1e-3] = 0.0

        # ================================================================
        # 11) Optional PMT position correction
        # ================================================================
        if corr_pos is not None:
            p_int = p_locations.astype(int)
            matches = (p_int[:, None, :] == corr_pos['wut'][None, :, :]).all(axis=2)
            mask = matches.any(axis=1)
            mean_pes[mask] *= 0.5

        # ================================================================
        # 12) Timing (CONSISTENT with s_eff)
        # ================================================================
        t_light = r * self.n / self.c
        t_emitter = s_eff / self.v
        t_hits = self.starting_time + t_emitter + t_light
        
        
        # Final correction to get the correct intensities to match observation
        if np.mean(mean_pes) != 0:
            
            pe_scale_factor = np.mean(obs_pes)/np.mean(mean_pes)
            #pe_scale_factor = max(obs_pes)/max(mean_pes)
#             top20_obs = np.partition(obs_pes, -20)[-20:]
#             mean_top20_obs = np.mean(top20_obs)
            
#             top20_exp = np.partition(mean_pes, -20)[-20:]
#             mean_top20_exp = np.mean(top20_exp)
            
#             pe_scale_factor = mean_top20_obs/mean_top20_exp
            
            mean_pes = mean_pes*pe_scale_factor
            
        #print(np.mean(mean_pes))
        
        return mean_pes, t_hits



#     def get_expected_pes_ts(self, wcd, s, p_locations, direction_zs, corr_pos):
#         """ Use the same calculation as in emit() to get the expected number of PE and mean time for a list of
#         distance parameters s for pmts at p_locations and orientations direction_zs.
#         Use NumPy vectorization for speeding up the Likelihood calculation."""
#         pmt_radius = wcd.mpmts[0].pmts[0].get_properties('design')['size']/2.
#         s_final = 190 # Cherenkov angle begins to collapse for muons in last 190 mm of travel, adjust later

#         p_locations = np.array(p_locations)
#         p_int = p_locations.astype(int) # locations as integers for easy comparison
#         px = p_locations[:, 0]
#         py = p_locations[:, 1]
#         pz = p_locations[:, 2]
#         direction_zs = np.array(direction_zs)
#         direction_zx = direction_zs[:, 0]
#         direction_zy = direction_zs[:, 1]
#         direction_zz = direction_zs[:, 2]

#         ex = self.start_coord[0] + self.direction[0] * s
#         ey = self.start_coord[1] + self.direction[1] * s
#         ez = self.start_coord[2] + self.direction[2] * s
#         e_locations = np.array([ex, ey, ez]).T

#         to_pmt = p_locations - e_locations
#         r = np.linalg.norm(to_pmt, axis=1) +1 # add 1 mm to avoid possible div by zero (see note in get_emission_point)
#         cost = -(to_pmt[:, 0]*direction_zx + to_pmt[:, 1]*direction_zy + to_pmt[:, 2]*direction_zz)/r
#         #valid = cost > 0.
#         mean_pes = np.zeros(len(p_locations))
#         r0 = 1000.
#         cost0 = 1.0
#         corr = r0/r*cost/cost0
#         frac_area = np.ones(len(p_locations))

#         frac_area[s is None] = 0.
#         frac_area[s < -pmt_radius] = 0.
#         frac_area[s > self.length + pmt_radius] = 0.
#         frac_area[s < pmt_radius] = (s[s < pmt_radius] + pmt_radius) / (2.*pmt_radius)
        
#         #mask_final = (s > s_final) & (s<self.length)
        
#         #frac_area[mask_final]
        
#         frac_area[s > self.length - pmt_radius] = (self.length - s[s > self.length - pmt_radius] + pmt_radius) / (2. * pmt_radius)
#         frac_area[cost<=0.] = 0.

#         mean_pes = self.intensity * corr * frac_area
#         mean_pes[mean_pes < 0.001] = 0.
        
#         if corr_pos != None:
#             # Compare each positions row with each pos_target row
#             matches = (p_int[:, None, :] == corr_pos['wut'][None, :, :]).all(axis=2)

#             # Collapse over the pos_target axis → boolean mask for positions rows
#             mask = matches.any(axis=1)


#             #print(mask)

#             # Apply your operation (e.g., multiply matched indices by 0.5)
#             mean_pes = mean_pes.astype(float)
#             mean_pes[mask] *= 0.5
            
#         else:
#             mean_pes = mean_pes.astype(float)
            
#         # time for light to reach the PMT
#         t_light = r * self.n / self.c  # in ns
#         # time for emitter to reach the emission point
#         t_emitter = s / self.v  # in ns
#         # total time of hit
#         t_hits = self.starting_time + t_emitter + t_light

#         return mean_pes, t_hits

#     def get_expected_pes_ts(self, wcd, s, p_locations, direction_zs, corr_pos):

#         # --- unchanged setup ---
#         pmt_radius = wcd.mpmts[0].pmts[0].get_properties('design')['size'] / 2.

#         p_locations = np.array(p_locations)
#         p_int = p_locations.astype(int)
#         direction_zs = np.array(direction_zs)

#         ex = self.start_coord[0] + self.direction[0] * s
#         ey = self.start_coord[1] + self.direction[1] * s
#         ez = self.start_coord[2] + self.direction[2] * s
#         e_locations = np.column_stack((ex, ey, ez))

#         to_pmt = p_locations - e_locations
#         r = np.linalg.norm(to_pmt, axis=1) + 1.0

#         direction_zx = direction_zs[:, 0]
#         direction_zy = direction_zs[:, 1]
#         direction_zz = direction_zs[:, 2]

#         cost = -(to_pmt[:, 0]*direction_zx +
#                  to_pmt[:, 1]*direction_zy +
#                  to_pmt[:, 2]*direction_zz) / r

#         # --- distance/angle scaling ---
#         r0 = 1000.
#         corr = r0 / r * cost

#         # --- fractional track window (unchanged) ---
#         frac_area = np.ones(len(p_locations))
#         s_b = np.ones(len(p_locations))
        

#         #frac_area[s < 0] = 0.
#         #frac_area[s > self.length + pmt_radius] = 0.

#         #mask_start = s < pmt_radius
#         #frac_area[mask_start] = (s[mask_start] + pmt_radius) / (2. * pmt_radius)
#         start_pos = np.array([self.start_coord[0],self.start_coord[1],self.start_coord[2]])
#         track_dir = np.array([self.direction[0],self.direction[1],self.direction[2]])
#         s_max_mm = self.length
#         s_a_mm = 0.001

#         E_at_sa_MeV = (self.length/10.)*2 #Estimate initial energy as 2 x track length
#         dEdx_MeV_per_cm = 2


#         mask = s > 0.
#         frac_area[mask], s_b[mask] = find_scale_for_pmts(
#                             p_locations[mask],
#                             start_pos,
#                             track_dir,
#                             s_a_mm,
#                             s_max_mm,
#                             E_at_sa_MeV,
#                             dEdx_MeV_per_cm,
#                             theta_c_func
#                         )
        
#         mask2 = s<0
#         frac_area[mask2] = 0.
#         s_b[mask2] = 0.

#         #frac_area[mask_end] = (self.length - s[mask_end] + pmt_radius) / (2. * pmt_radius)


#         mean_pes = self.intensity * corr * frac_area 

#         mean_pes[mean_pes < 0.001] = 0.0

#         # --- optional position correction (unchanged) ---
#         if corr_pos is not None:
#             matches = (p_int[:, None, :] == corr_pos['wut'][None, :, :]).all(axis=2)
#             mask = matches.any(axis=1)
#             mean_pes[mask] *= 0.5

#         # --- timing (unchanged) ---
#         t_light = r * self.n / self.c
#         t_emitter = s_b / self.v
#         t_hits = self.starting_time + t_emitter + t_light

#         return mean_pes, t_hits

    @staticmethod
    def get_pmt_placements(event, wcd, place_info):
        p_locations = []
        direction_zs = []

        for i_mpmt in range(event.n_mpmt):
            if not event.mpmt_status[i_mpmt]:
                continue
            mpmt = wcd.mpmts[i_mpmt]
            for i_pmt in range(event.npmt_per_mpmt):
                if not event.pmt_status[i_mpmt][i_pmt]:
                    continue
                pmt = mpmt.pmts[i_pmt]
                p = pmt.get_placement(place_info, wcd)

                # Force copies (prevents any internal reuse/mutation issues)
                p_locations.append(np.array(p["location"], dtype=float).copy())
                direction_zs.append(np.array(p["direction_z"], dtype=float).copy())

        return np.asarray(p_locations, dtype=float), np.asarray(direction_zs, dtype=float)


#     @staticmethod
#     def get_pmt_placements(event, wcd, place_info):
#         """For faster likelihood calculation, prepare arrays of PMT locations and z directions.
#         Args:
#             event (Event): The event containing the PMT status information.
#             wcd (Geometry.wcd): The water Cherenkov detector with the PMT geometry.
#             place_info (string): specify which geometric info to use, e.g. 'design', 'est' (as-built)
#         Returns:
#             p_locations (list): A list of PMT locations for all active PMTs in the detector.
#             direction_zs (list): A list of PMT z directions for all active PMTs in the detector.
#             Note: the lists are indexed by a serial index over all active PMTs in the detector.
#             """
#         p_locations = []
#         direction_zs = []

#         # loop over all active mPMTs in the WCD
#         for i_mpmt in range(event.n_mpmt):
#             if not event.mpmt_status[i_mpmt]:
#                 continue
#             mpmt = wcd.mpmts[i_mpmt]
#             for i_pmt in range(event.npmt_per_mpmt):
#                 if not event.pmt_status[i_mpmt][i_pmt]:
#                     continue
#                 pmt = mpmt.pmts[i_pmt]
#                 p = pmt.get_placement(place_info, wcd)
#                 p_locations.append(p['location'])
#                 direction_zs.append(p['direction_z'])

#         return p_locations, direction_zs

    # Method to find intersection points of a cone with a cylinder "can" - for better event display visualization
    # This code was produced with the help of ChatGPT5, using the following two prompts:
    # (1)
    # Consider a straight line passing through the point (x0,y0,z0) and having direction cosines (cx,cy,cz).
    # Let this line be the axis of a cone having half-angle q, 0 < q < pi/2. The point (x0,y0,z0) is the apex of
    # the cone and is located inside a cylindrical can. The cylinder axis is along the y-axis with radius r. The top
    # and bottom endcaps of the can are normal to the y-axis and are located at y=ht and y=hb respectively with ht>hb.
    # To produce a drawing of the intersection of the cone and the cylindrical can, I require n intersection points of
    # lines equally spaced in azimuth around the cone axis. Write a python function that takes arguments
    # (x0, y0, z0, cx, cy, cz, q, r, ht, hb, n) and returns a list of n vectors (xi,yi,zi) that are points of
    # intersection of the cone and the cylindrical can.
    # (2)
    # Write this function only using numpy (and optionally scipy.spatial.transform). There is only one point of
    # intersection for the ray and the cylindrical can. If the intersection on the cylinder gives a value y < hb, then
    # the intersection must be on the bottom endcap. If the intersection on the cylinder gives a value y > ht, then the
    # intersection must be on the top endcap. There will be one and only one intersection for each ray. For convenience
    # the last vector in the returned list should repeat the first vector in the list.

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
