import numpy as np
import scipy.stats as stats

class PMT:
    """
    A class to simulate the response of a PMT to photoelectrons (and for including dark noise).

    Attributes:
        single_pe_amp_mean (float): Mean of the single photoelectron amplitude response (in ADC counts).
        single_pe_amp_std (float): Standard deviation of the single photoelectron amplitude response (in ADC counts).
        single_pe_time_std (float): Standard deviation of the single photoelectron time response (in ns).
        separation_time (float): Minimum time separation to produce separate hits (in ns).
        amp_threshold (float): The amplitude threshold for a hit to be recorded (in ADC counts).
        noise_rate (float): The expected pe per ns to produce a single pe dark noise hit.
    """

    def __init__(self, single_pe_amp_mean, single_pe_amp_std, single_pe_time_std, separation_time, amp_threshold, noise_rate):
        if not(isinstance(single_pe_amp_mean, float) or single_pe_amp_mean <= 0):
            raise ValueError("single_pe_amp_mean must be a positive number")
        if not isinstance(single_pe_amp_std, float) or single_pe_amp_std <= 0:
            raise ValueError("single_pe_amp_std must be a positive number")
        if not isinstance(single_pe_time_std, float) or single_pe_time_std <= 0:
            raise ValueError("single_pe_time_std must be a positive number")
        if not isinstance(separation_time, (int, float)) or separation_time <= 0:
            raise ValueError("separation_time must be a positive number")
        if not isinstance(amp_threshold, (int, float)) or amp_threshold < 0:
            raise ValueError("amp_threshold must be a non-negative number")
        if not isinstance(noise_rate, (float)) or noise_rate < 0:
            raise ValueError("noise_rate must be a non-negative number")

        self.single_pe_amp_mean = float(single_pe_amp_mean)
        self.single_pe_amp_std = float(single_pe_amp_std)
        self.single_pe_time_std = float(single_pe_time_std)
        self.separation_time = float(separation_time)
        self.amp_threshold = float(amp_threshold)
        self.noise_rate = float(noise_rate)

        # probability of a 1 PE hit being below threshold
        self.prob01 = stats.norm.cdf(self.amp_threshold, loc=self.single_pe_amp_mean,
                                     scale=self.single_pe_amp_std)
        self.charge_response = self.precalculate_charge_response()

    def __repr__(self):
        return (f"PMT(single_pe_amp_mean={self.single_pe_amp_mean}, "
                f"single_pe_amp_std={self.single_pe_amp_std}, "
                f"single_pe_time_std={self.single_pe_time_std}, "
                f"separation_time={self.separation_time}, "
                f"amp_threshold={self.amp_threshold}, "
                f"noise_rate={self.noise_rate})")

    def precalculate_charge_response(self):
        """Pre-calculate the charge response for 1 to 8 PE to speed up the likelihood calculation later.
        If q/amp > 5, then a direct convolution (approximate Poisson by Gaussian) would be used instead.
        These are saved in np.arrays to be used in get_neg_log_likelihood_q_t.
        numpy vectorization is used to speed up the calculation in that method
        """
        charge_response = [] # list of np.arrays, the first element is for 1 PE, second for 2 PE, etc
        for npe in range(1, 9):
            # For npe PE, the charge response is a Gaussian with mean = npe*single_pe_amp_mean and std = single_pe_amp_std*sqrt(npe)
            # these are scaled by the single_pe_amp_mean
            mean = npe
            std = self.single_pe_amp_std * np.sqrt(npe) / self.single_pe_amp_mean
            # in what follows, ope is the observed PE = charge / single_pe_amp_mean
            # and ope10 is the bin from ope/10 to ope/10 + 0.1
            response = []
            for ope10 in range(0,50):
                ope_low = ope10 / 10.
                ope_high = ope_low + 0.1
                prob = 0.
                if ope_high > self.amp_threshold / self.single_pe_amp_mean:
                    # the threshold should be at a boundary of the ope10 bins
                    # Calculate the probability of observing ope PE given npe true PE
                    # This is a Gaussian with mean = npe and std = sqrt(npe), evaluated at ope
                    prob = stats.norm.cdf(ope_high, loc=mean, scale=std) - stats.norm.cdf(ope_low, loc=mean, scale=std)
                response.append(prob)

            charge_response.append(response)

        return np.array(charge_response)

    def add_noise(self, simulated_event):
        """Add noise hits to a simulated event

        Args:
            simulated_event (SimulatedEvent): The simulated event to which noise hits will be added.

        """
        # Find minimum and maximum expected hit times in the simulated event
        min_time = float('inf')
        max_time = float('-inf')
        for i_mpmt in range(simulated_event.n_mpmt):
            if not simulated_event.mpmt_status[i_mpmt]:
                continue
            for i_pmt in range(simulated_event.npmt_per_mpmt):
                if not simulated_event.pmt_status[i_mpmt][i_pmt]:
                    continue
                for exp_hit_time in simulated_event.expected_hit_times[i_mpmt][i_pmt]:
                    if exp_hit_time < min_time:
                        min_time = exp_hit_time
                    if exp_hit_time > max_time:
                        max_time = exp_hit_time

        #Expand the window of hit times by 10 ns on either side to account for time smearing
        min_time -= 10.
        max_time += 10.
        #expected number of noise hits in this time window per PMT
        expected_n_noise_hits = self.noise_rate * (max_time - min_time)

        for i_mpmt in range(simulated_event.n_mpmt):
            if not simulated_event.mpmt_status[i_mpmt]:
                continue
            for i_pmt in range(simulated_event.npmt_per_mpmt):
                if not simulated_event.pmt_status[i_mpmt][i_pmt]:
                    continue
                # Number of noise hits is Poisson distributed
                n_noise_hits = np.random.poisson(expected_n_noise_hits)
                for _ in range(n_noise_hits):
                    noise_time = np.random.uniform(min_time, max_time)
                    simulated_event.noise_hit_times[i_mpmt][i_pmt].append(noise_time)
                    simulated_event.noise_hit_pe[i_mpmt][i_pmt].append(1) # assume each noise hit is 1 PE

    def apply_response(self, simulated_event):
        """Apply the electronics response to a simulated event

        Args:
            simulated_event (SimulatedEvent): The simulated event to which the electronics response will be applied.

        """
        for i_mpmt in range(simulated_event.n_mpmt):
            if not simulated_event.mpmt_status[i_mpmt]:
                continue
            for i_pmt in range(simulated_event.npmt_per_mpmt):
                if not simulated_event.pmt_status[i_mpmt][i_pmt]:
                    continue
                # Hits that are too close together need to be merged into a single hit
                # First, reorder the expected hits by time (if there is more than one hit)
                all_hit_times = simulated_event.expected_hit_times[i_mpmt][i_pmt] + simulated_event.noise_hit_times[i_mpmt][i_pmt]
                all_hit_pe = simulated_event.true_hit_pe[i_mpmt][i_pmt] + simulated_event.noise_hit_pe[i_mpmt][i_pmt]
                if len(all_hit_times) == 0:
                    continue
                if len(all_hit_times) > 1:
                    sorted_indices = np.argsort(all_hit_times)
                    sorted_hit_times = [all_hit_times[i] for i in sorted_indices]
                    sorted_hit_pe = [all_hit_pe[i] for i in sorted_indices]
                    merged_hit_times = []
                    merged_hit_pe = []
                    current_hit_time = sorted_hit_times[0]
                    current_hit_pe = sorted_hit_pe[0]
                    for j in range(1, len(sorted_hit_times)):
                        if sorted_hit_times[j] - current_hit_time < self.separation_time:
                            # Merge hits that are too close together
                            # New hit time is the weighted average of the two hit times
                            total_pe = current_hit_pe + sorted_hit_pe[j]
                            current_hit_time = (current_hit_time * current_hit_pe + sorted_hit_times[j] * sorted_hit_pe[j]) / total_pe
                            current_hit_pe = total_pe
                        else:
                            merged_hit_times.append(current_hit_time)
                            merged_hit_pe.append(current_hit_pe)
                            current_hit_time = sorted_hit_times[j]
                            current_hit_pe = sorted_hit_pe[j]
                    # Append the last hit (be it merged or not)
                    merged_hit_times.append(current_hit_time)
                    merged_hit_pe.append(current_hit_pe)
                else:
                    merged_hit_times = all_hit_times
                    merged_hit_pe = all_hit_pe

                for k in range(len(merged_hit_times)):
                    # Note that we are using Poisson deviates for Cherenkov light and fixed 1 PE for noise hits
                    # It is therefore possible to have 0 true_pe
                    true_pe = merged_hit_pe[k]
                    if true_pe <= 0:
                        continue

                    # Amplitude is Gaussian distributed around the true number of PE*single_pe_amp_mean
                    # with a width of single_pe_amp_std*sqrt(true_pe)
                    amp = np.random.normal(true_pe*self.single_pe_amp_mean, self.single_pe_amp_std * np.sqrt(true_pe))
                    if amp > self.amp_threshold:
                        # Time is Gaussian distributed around the merged hit time
                        time = np.random.normal(merged_hit_times[k], self.single_pe_time_std/np.sqrt(true_pe))
                        simulated_event.hit_charges[i_mpmt][i_pmt].append(amp)
                        simulated_event.hit_times[i_mpmt][i_pmt].append(time)

    def get_neg_log_likelihood_npe(self,exp_pes, obs_pes):
        """ Get the negative log likelihood for the number of photoelectrons observed in a PMT
        This is just for testing, since n_pe is not typically observed directly.
        Use NumPy vectorization for speeding up the Likelihood calculation.

        Args:
            exp_pes (float numpy array): expected number of photoelectrons in each PMT
            obs_pes (int numpy array): observed number of photoelectrons in each PMT

        Returns:
            negative log likelihood value (float)
        """
        # no noise test
        mask = exp_pes > 0
        signal_nll = exp_pes[mask] - obs_pes[mask] * np.log(exp_pes[mask])
        mask2 = (exp_pes <= 0) & (obs_pes > 0)
        background_nll = - obs_pes[mask2] * np.log(1E-4)  # attributed to noise
        return np.sum(signal_nll) + np.sum(background_nll)

    def get_neg_log_likelihood_npe_t(self,exp_pes, obs_pes, exp_ts, obs_ts):
        """ Get the negative log likelihood for the number of photoelectrons and times observed in a PMT
        This is just for testing, since n_pe is not typically observed directly.
        Use NumPy vectorization for speeding up the Likelihood calculation.

        Args:
            exp_pes (float numpy array): expected number of photoelectrons in each PMT
            obs_pes (int numpy array): observed number of photoelectrons in each PMT
            exp_ts (float numpy array): expected time of photoelectrons in each PMT
            obs_ts (float numpy array): observed time of photoelectrons in each PMT (None if not available)

        Returns:
            negative log likelihood value (float)
        """
        # no noise test - npe
        mask = exp_pes > 0
        signal_nll = exp_pes[mask] - obs_pes[mask] * np.log(exp_pes[mask])
        mask2 = (exp_pes <= 0) & (obs_pes > 0)
        background_nll = - obs_pes[mask2] * np.log(1E-4)  # attributed to noise

        #negllt = np.zeros(len(exp_pes))
        mask_t = (exp_pes > 0) & (obs_pes > 0) & (obs_ts != None)  # hits with no expected pe are already counted above
        nll_t = np.array([])
        if np.any(mask_t):
            # For the time likelihood, we assume a Gaussian time response with std for single PE = self.single_pe_time_std
            sigma_t = self.single_pe_time_std/np.sqrt(obs_pes[mask_t])
            nll_t = 0.5 * ((obs_ts[mask_t] - exp_ts[mask_t]) / sigma_t)**2

        return np.sum(signal_nll) + np.sum(background_nll) + np.sum(nll_t)

    def get_neg_log_likelihood_q_t(self,exp_pes, obs_qs, exp_ts, obs_ts):
        """ Get the negative log likelihood for the charge and times observed in a PMT
        Use NumPy vectorization for speeding up the Likelihood calculation.

        Args:
            exp_pes (float numpy array): expected number of photoelectrons in each PMT
            obs_qs (int numpy array): observed hit charges in each PMT (in ADC counts)
            exp_ts (float numpy array): expected time of photoelectrons in each PMT
            obs_ts (float numpy array): observed time of photoelectrons in each PMT (None if not available)

        Returns:
            negative log likelihood value (float)
        """
        # no noise test - npe
        # convert observed charges to observed pes (naive)
        obs_pes = np.zeros(len(obs_qs))
        obs_pes[obs_qs != None] = obs_qs[obs_qs != None] / self.single_pe_amp_mean
        # if obs_pes > 5, we can approximate the convolution of Poisson and Gaussian by a Gaussian
        # if obs_pes <= 5, then we use the pre-calculated charge response

        mask0 = (exp_pes > 0) & (obs_pes == 0)
        no_signal_prob = np.exp(-exp_pes[mask0]) + self.prob01 * exp_pes[mask0] * np.exp(-exp_pes[mask0])
        no_signal_nll = - np.log(no_signal_prob + 1E-10)  # avoid log(0)

        mask = (exp_pes > 0) & (obs_pes <=5)
        #signal_nll = exp_pes[mask] - obs_pes[mask] * np.log(exp_pes[mask])
        n_pes = np.array([1,2,3,4,5,6,7,8])
        exp_pe = exp_pes[mask]
        prob_ns = np.exp(-exp_pe[:, None]) * exp_pe[:, None] ** n_pes[None, :] / \
                  np.array([np.math.factorial(n) for n in n_pes])[None, :]
        observed_ope10 = (obs_pes[mask] * 10.).astype(int)  # bin from ope/10 to ope/10 + 0.1
        pdfs_sums = np.sum(prob_ns * self.charge_response[:, observed_ope10], axis=1)
        signal_nllq = - np.log(pdfs_sums + 1E-10)  # avoid log(0)

        mask1 = (exp_pes > 0) & (obs_pes > 5)
        # Gaussian approximation of Poisson + Gaussian
        sigmas = np.sqrt(exp_pes[mask1] + self.single_pe_amp_std**2 * exp_pes[mask1])
        n_sigma = np.abs(obs_pes[mask1] - exp_pes[mask1]) / sigmas
        high_sigma = n_sigma >= 4.
        low_sigma = n_sigma < 4.
        high_sigma_nllq = - obs_pes[mask1][high_sigma] * np.log(1E-4)  # attributed to noise
        low_sigma_nllq = 0.5 * (obs_pes[mask1][low_sigma] - exp_pes[mask1][low_sigma]) ** 2 / sigmas[low_sigma] ** 2

        mask2 = (exp_pes <= 0) & (obs_pes > 0)
        background_nll = - obs_pes[mask2] * np.log(1E-4)  # attributed to noise

        #negllt = np.zeros(len(exp_pes))
        mask_t = (exp_pes > 0) & (obs_pes > 0) & (obs_ts != None)  # hits with no expected pe are already counted above
        high_sigma_nllt = np.array([])
        low_sigma_nllt = np.array([])
        if np.any(mask_t):
            # For the time likelihood, we assume a Gaussian time response with std for single PE = self.single_pe_time_std
            # For times outside 4 sigma, we attribute to noise as above
            sigma_t = self.single_pe_time_std/np.sqrt(obs_pes[mask_t])
            n_sigma = np.abs(obs_ts[mask_t] - exp_ts[mask_t]) / sigma_t
            high_sigma = n_sigma >= 4.
            low_sigma = n_sigma < 4.
            high_sigma_nllt = - obs_pes[mask_t][high_sigma] * np.log(1E-4)  # attributed to noise
            low_sigma_nllt = 0.5 * (obs_ts[mask_t][low_sigma] - exp_ts[mask_t][low_sigma])**2 / sigma_t[low_sigma]**2

        return (np.sum(no_signal_nll) + np.sum(signal_nllq) + np.sum(high_sigma_nllq) + np.sum(low_sigma_nllq) + np.sum(background_nll) +
                np.sum(high_sigma_nllt) + np.sum(low_sigma_nllt))