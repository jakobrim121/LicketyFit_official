from LicketyFit.event_displays_chatgpt import (
    plot_unified_surface,
    plot_photon_sky,
    plot_hough_circle_endcap,
    plot_hough_barrel_sinusoid
)

import numpy as np

class EventDisplay:
    """ The EventDisplay class is used to produce geometric visualizations of real and simulated events.

        Create the event display by specifying the detector geometry and the quantity (quantities) to be shown.
        Up to three quantities can be displayed (primary, secondary, tertiary). They are selected by specifying the
        attributes in the Event or SimulatedEvent object. Range limits can be specified for all quantities.
        The default minimum for the primary quantity is 0.

        The arguments, r, yb, and yt specify the cylindrical can surface (default to WCTE values)
    """


    def __init__(self, wcd, primary, primary_range=None,
                 secondary=None, secondary_range=None,
                 tertiary=None, tertiary_range=None,
                 r=1464., yb=-858., yt=1708.,
                 **kwargs):

        self.wcd = wcd
        self.primary = primary
        self.primary_range = primary_range
        self.secondary = secondary
        self.secondary_range = secondary_range
        self.tertiary = tertiary
        self.tertiary_range = tertiary_range
        self.r = r
        self.yb = yb
        self.yt = yt
        self.kwargs = kwargs
        self.no_hit_value = 0
        if self.primary_range is not None and self.primary_range[0] is not None:
            self.no_hit_value = self.primary_range[0]

    # Put locations and values into 1D numpy arrays for the chatgpt plotters
    def setup_chatgpt(self, event, choose_min=None, choose_max=None, labelled_mpmts=None):

        primary_source = None
        if self.primary is not None:
            primary_source = getattr(event, self.primary)
        secondary_source = None
        if self.secondary is not None:
            secondary_source = getattr(event, self.secondary)
        tertiary_source = None
        if self.tertiary is not None:
            tertiary_source = getattr(event, self.tertiary)
        choose_min_source = None
        if choose_min is not None:
            choose_min_source = getattr(event, choose_min)
        choose_max_source = None
        if choose_max is not None:
            choose_max_source = getattr(event, choose_max)
        if choose_min is not None and choose_max is not None:
            print("SETUP problem: only specify one of (choose_min, choose_max) to select from multiple hits")

        pmt_locations = []
        primary_values = []
        secondary_values = []
        tertiary_values = []
        mpmt_labels = []

        for i_mpmt in range(event.n_mpmt):
            if not event.mpmt_status[i_mpmt]:
                continue

            if labelled_mpmts is not None:
                if ((isinstance(labelled_mpmts, bool) and labelled_mpmts) or
                        (isinstance(labelled_mpmts, list) and i_mpmt in labelled_mpmts)):
                    p = self.wcd.mpmts[i_mpmt].get_placement('design', self.wcd)
                    mpmt_label = p['location'] + [str(i_mpmt)]
                    mpmt_labels.append(mpmt_label)

            for i_pmt in range(event.npmt_per_mpmt):
                if not event.pmt_status[i_mpmt][i_pmt]:
                    continue

                p = self.wcd.mpmts[i_mpmt].pmts[i_pmt].get_placement('design', self.wcd)
                pmt_locations.append(p['location'])

                primary_value = self.no_hit_value
                hit_choice = -1
                if self.primary is not None:
                    if len(primary_source[i_mpmt][i_pmt]) > 0:
                        hit_choice = 0
                        if len(primary_source[i_mpmt][i_pmt]) > 1:
                            if choose_min is not None:
                                hit_choice = np.argmin(choose_min_source[i_mpmt][i_pmt])
                            elif choose_max is not None:
                                hit_choice = np.argmax(choose_max_source[i_mpmt][i_pmt])
                        primary_value = primary_source[i_mpmt][i_pmt][hit_choice]

                        if self.primary_range is not None:
                            if self.primary_range[0] is not None:
                                if primary_value < self.primary_range[0]:
                                    primary_value = self.primary_range[0] + 1E-6
                            if self.primary_range[1] is not None:
                                if primary_value > self.primary_range[1]:
                                    primary_value = self.primary_range[1] - 1E-6

                primary_values.append(primary_value)

                if self.secondary is not None and hit_choice != -1:
                    secondary_value = secondary_source[i_mpmt][i_pmt][hit_choice]
                    if self.secondary_range is not None:
                        if self.secondary_range[0] is not None:
                            if secondary_value < self.secondary_range[0]:
                                secondary_value = self.secondary_range[0] + 1E-6
                        if self.secondary_range[1] is not None:
                            if secondary_value > self.secondary_range[1]:
                                secondary_value = self.secondary_range[1] - 1E-6
                else:
                    secondary_value = 1.

                secondary_values.append(secondary_value)

                if self.tertiary is not None and hit_choice != -1:
                    tertiary_value = tertiary_source[i_mpmt][i_pmt][hit_choice]
                    if self.tertiary_range is not None:
                        if self.tertiary_range[0] is not None:
                            if tertiary_value < self.tertiary_range[0]:
                                tertiary_value = self.tertiary_range[0] + 1E-6
                        if self.tertiary_range[1] is not None:
                            if tertiary_value > self.tertiary_range[1]:
                                tertiary_value = self.tertiary_range[1] - 1E-6
                else:
                    tertiary_value = 1.
                tertiary_values.append(tertiary_value)

        return np.array(pmt_locations), np.array(primary_values), np.array(secondary_values), np.array(tertiary_values), mpmt_labels


    def get_rollout(self, event, phi_cut=np.deg2rad(180), overlay_curves=None, labelled_mpmts=None,
                    choose_min=None, choose_max=None, **kwargs):

        """  Return the rollout display (showing barrel and endcaps separately)

        The primary secondary and tertiary are used to define the color, size, and alpha for dots at the PMT locations
        For PMTs that have no hit, the primary value is set equal to its lower range value, allowing the PMT location to be shown

        Arguments:
            phi_cut: The azimuthal angle that the barrel is cut at for the rollout display (default 180 degrees)
            overlay_curves: A list of curves to overlay on the display, each specified as a list of (x,y,z) points
            labelled_mpmts: If True, label all MPMTs. If a list of integers, label only those MPMTs.
            choose_min: For PMTs with multiple hits, the choose_min is the attribute for which the hit with the minimum
             of that attribute will be shown, otherwise the first hit is shown.
            choose_max: For PMTs with multiple hits, the choose_max is the attribute for which the hit with the maximum
             of that attribute will be shown, otherwise the first hit is shown.
             *** do not specify both choose_min and choose_max ***
        Returns:
            A matplotlib Figure object with the rollout display

        """
        data_missing = False
        for attr in [self.primary, self.secondary, self.tertiary]:
            if attr is not None and not hasattr(event, attr):
                print("Event does not have the attribute:", attr)
                data_missing = True
        if data_missing:
            return None

        sensors, color_vals, size_vals, alpha_vals, mpmt_labels = (
            self.setup_chatgpt(event, choose_min=choose_min, choose_max=choose_max, labelled_mpmts=labelled_mpmts))

        # adjust the sizes and alphas if secondary and tertiary quantity is not provided
        if self.secondary is None:
            data_mask = color_vals > self.no_hit_value
            size_vals[data_mask] = 0.2
            size_vals[~data_mask] = 0.02
        if self.tertiary is None:
            data_mask = color_vals > self.no_hit_value
            alpha_vals[data_mask] = 1.
            alpha_vals[~data_mask] = 0.3

        vmin_color = None
        vmax_color = None
        if self.primary_range is not None:
            vmin_color = self.primary_range[0]
            vmax_color = self.primary_range[1]

        return plot_unified_surface(
            sensors, self.r, self.yb, self.yt, phi_cut=phi_cut,
            values_color=color_vals, label_color=self.primary, vmin_color=vmin_color, vmax_color=vmax_color,
            values_size=size_vals, values_alpha=alpha_vals,
            overlay_curve_xyzs=overlay_curves, text_annotations=mpmt_labels, width_ratios=(4.5, 1.0)
        )
