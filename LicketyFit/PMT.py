import math

import numpy as np
from numba import njit


@njit(cache=True)
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@njit(cache=True)
def _poisson_time_nll(exp_pes, obs_pes, exp_ts, obs_ts, single_pe_time_std):
    """
    Tight likelihood kernel for the npe+t fit.

    Compared with the original implementation, this avoids repeated boolean-mask
    allocations and works directly on contiguous float arrays.
    """
    n = exp_pes.size
    nll = 0.0
    log_noise = math.log(1e-4)

    for i in range(n):
        lam = exp_pes[i]
        obs = obs_pes[i]

        # Charge term
        if lam > 0.0:
            nll += lam - obs * math.log(lam)
        elif obs > 0.0:
            nll -= obs * log_noise

        # Timing term
        t_obs = obs_ts[i]
        if (lam > 0.0) and (obs > 0.0) and np.isfinite(t_obs):
            sigma_t = single_pe_time_std / math.sqrt(obs)
            dt = (t_obs - exp_ts[i]) / sigma_t
            nll += 0.5 * dt * dt

    return nll


class PMT:
    """
    PMT response model used by the fitter.

    The hot likelihood path is now handled by a compiled helper.  The public API
    is unchanged, so existing fit scripts can keep calling the same methods.
    """

    def __init__(
        self,
        single_pe_amp_mean,
        single_pe_amp_std,
        single_pe_time_std,
        separation_time,
        amp_threshold,
        noise_rate,
    ):
        if not isinstance(single_pe_amp_mean, (int, float)) or single_pe_amp_mean <= 0:
            raise ValueError("single_pe_amp_mean must be a positive number")
        if not isinstance(single_pe_amp_std, (int, float)) or single_pe_amp_std <= 0:
            raise ValueError("single_pe_amp_std must be a positive number")
        if not isinstance(single_pe_time_std, (int, float)) or single_pe_time_std <= 0:
            raise ValueError("single_pe_time_std must be a positive number")
        if not isinstance(separation_time, (int, float)) or separation_time <= 0:
            raise ValueError("separation_time must be a positive number")
        if not isinstance(amp_threshold, (int, float)) or amp_threshold < 0:
            raise ValueError("amp_threshold must be a non-negative number")
        if not isinstance(noise_rate, (int, float)) or noise_rate < 0:
            raise ValueError("noise_rate must be a non-negative number")

        self.single_pe_amp_mean = float(single_pe_amp_mean)
        self.single_pe_amp_std = float(single_pe_amp_std)
        self.single_pe_time_std = float(single_pe_time_std)
        self.separation_time = float(separation_time)
        self.amp_threshold = float(amp_threshold)
        self.noise_rate = float(noise_rate)

        # Probability that a single PE falls below threshold.
        z = (self.amp_threshold - self.single_pe_amp_mean) / self.single_pe_amp_std
        self.prob01 = _norm_cdf(z)

        # Precompute the small-PE charge response exactly once.
        self.charge_response = self.precalculate_charge_response()

    def __repr__(self):
        return (
            f"PMT(single_pe_amp_mean={self.single_pe_amp_mean}, "
            f"single_pe_amp_std={self.single_pe_amp_std}, "
            f"single_pe_time_std={self.single_pe_time_std}, "
            f"separation_time={self.separation_time}, "
            f"amp_threshold={self.amp_threshold}, "
            f"noise_rate={self.noise_rate})"
        )

    def precalculate_charge_response(self):
        """
        Precompute the small-PE charge response grid used by q+t fits.

        This method runs only during PMT construction, so clarity matters more
        than micro-optimizing every line here.
        """
        n_pes_max = 8
        n_bins = 50
        threshold_ope = self.amp_threshold / self.single_pe_amp_mean

        response = np.zeros((n_pes_max, n_bins), dtype=np.float64)

        for npe in range(1, n_pes_max + 1):
            mean = float(npe)
            std = self.single_pe_amp_std * np.sqrt(npe) / self.single_pe_amp_mean

            for ope10 in range(n_bins):
                ope_low = ope10 / 10.0
                ope_high = ope_low + 0.1

                if ope_high <= threshold_ope:
                    continue

                z_low = (ope_low - mean) / std
                z_high = (ope_high - mean) / std
                prob = _norm_cdf(z_high) - _norm_cdf(z_low)
                response[npe - 1, ope10] = prob

        return response

    def add_noise(self, simulated_event):
        """
        Add dark-noise hits to a simulated event.

        This is not part of the fitter hot path, so the implementation stays
        close to the original for readability.
        """
        min_time = float("inf")
        max_time = float("-inf")

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

        min_time -= 10.0
        max_time += 10.0
        expected_n_noise_hits = self.noise_rate * (max_time - min_time)

        for i_mpmt in range(simulated_event.n_mpmt):
            if not simulated_event.mpmt_status[i_mpmt]:
                continue
            for i_pmt in range(simulated_event.npmt_per_mpmt):
                if not simulated_event.pmt_status[i_mpmt][i_pmt]:
                    continue
                n_noise_hits = np.random.poisson(expected_n_noise_hits)
                for _ in range(n_noise_hits):
                    noise_time = np.random.uniform(min_time, max_time)
                    simulated_event.noise_hit_times[i_mpmt][i_pmt].append(noise_time)
                    simulated_event.noise_hit_pe[i_mpmt][i_pmt].append(1)

    def apply_response(self, simulated_event):
        """
        Apply the electronics response to a simulated event.

        This method is still mostly simulation-side code, so it is kept close to
        the original implementation.
        """
        for i_mpmt in range(simulated_event.n_mpmt):
            if not simulated_event.mpmt_status[i_mpmt]:
                continue
            for i_pmt in range(simulated_event.npmt_per_mpmt):
                if not simulated_event.pmt_status[i_mpmt][i_pmt]:
                    continue

                all_hit_times = (
                    simulated_event.expected_hit_times[i_mpmt][i_pmt]
                    + simulated_event.noise_hit_times[i_mpmt][i_pmt]
                )
                all_hit_pe = (
                    simulated_event.true_hit_pe[i_mpmt][i_pmt]
                    + simulated_event.noise_hit_pe[i_mpmt][i_pmt]
                )

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
                            total_pe = current_hit_pe + sorted_hit_pe[j]
                            current_hit_time = (
                                current_hit_time * current_hit_pe
                                + sorted_hit_times[j] * sorted_hit_pe[j]
                            ) / total_pe
                            current_hit_pe = total_pe
                        else:
                            merged_hit_times.append(current_hit_time)
                            merged_hit_pe.append(current_hit_pe)
                            current_hit_time = sorted_hit_times[j]
                            current_hit_pe = sorted_hit_pe[j]

                    merged_hit_times.append(current_hit_time)
                    merged_hit_pe.append(current_hit_pe)
                else:
                    merged_hit_times = all_hit_times
                    merged_hit_pe = all_hit_pe

                for k in range(len(merged_hit_times)):
                    true_pe = merged_hit_pe[k]
                    if true_pe <= 0:
                        continue

                    amp = np.random.normal(
                        true_pe * self.single_pe_amp_mean,
                        self.single_pe_amp_std * np.sqrt(true_pe),
                    )
                    if amp > self.amp_threshold:
                        time = np.random.normal(
                            merged_hit_times[k],
                            self.single_pe_time_std / np.sqrt(true_pe),
                        )
                        simulated_event.hit_charges[i_mpmt][i_pmt].append(amp)
                        simulated_event.hit_times[i_mpmt][i_pmt].append(time)

    def get_neg_log_likelihood_npe(self, exp_pes, obs_pes):
        exp_pes = np.asarray(exp_pes, dtype=np.float64)
        obs_pes = np.asarray(obs_pes, dtype=np.float64)

        mask = exp_pes > 0.0
        signal_nll = exp_pes[mask] - obs_pes[mask] * np.log(exp_pes[mask])

        mask2 = (exp_pes <= 0.0) & (obs_pes > 0.0)
        background_nll = -obs_pes[mask2] * np.log(1e-4)

        return float(np.sum(signal_nll) + np.sum(background_nll))

    def get_neg_log_likelihood_npe_t(self, exp_pes, obs_pes, exp_ts, obs_ts):
        exp_pes = np.asarray(exp_pes, dtype=np.float64)
        obs_pes = np.asarray(obs_pes, dtype=np.float64)
        exp_ts = np.asarray(exp_ts, dtype=np.float64)
        obs_ts = np.asarray(obs_ts, dtype=np.float64)

        return float(
            _poisson_time_nll(
                exp_pes,
                obs_pes,
                exp_ts,
                obs_ts,
                float(self.single_pe_time_std),
            )
        )

    def get_neg_log_likelihood_q_t(self, exp_pes, obs_qs, exp_ts, obs_ts):
        """
        Charge+time likelihood.  This path is typically not the bottleneck for
        your current fits, so it remains close to the original implementation.
        """
        exp_pes = np.asarray(exp_pes, dtype=np.float64)
        exp_ts = np.asarray(exp_ts, dtype=np.float64)
        obs_ts = np.asarray(obs_ts, dtype=np.float64)

        obs_qs = np.asarray(obs_qs)
        obs_pes = np.zeros(len(obs_qs), dtype=np.float64)

        valid_q = np.isfinite(obs_qs.astype(np.float64, copy=False))
        obs_pes[valid_q] = obs_qs[valid_q].astype(np.float64) / self.single_pe_amp_mean

        mask0 = (exp_pes > 0.0) & (obs_pes == 0.0)
        no_signal_prob = np.exp(-exp_pes[mask0]) + self.prob01 * exp_pes[mask0] * np.exp(-exp_pes[mask0])
        no_signal_nll = -np.log(no_signal_prob + 1e-10)

        mask = (exp_pes > 0.0) & (obs_pes <= 5.0)
        n_pes = np.arange(1, 9, dtype=np.float64)
        exp_pe = exp_pes[mask]
        prob_ns = (
            np.exp(-exp_pe[:, None])
            * exp_pe[:, None] ** n_pes[None, :]
            / np.array([math.factorial(int(n)) for n in n_pes], dtype=np.float64)[None, :]
        )
        observed_ope10 = np.clip((obs_pes[mask] * 10.0).astype(int), 0, self.charge_response.shape[1] - 1)
        pdfs_sums = np.sum(prob_ns * self.charge_response[:, observed_ope10].T, axis=1)
        signal_nllq = -np.log(pdfs_sums + 1e-10)

        mask1 = (exp_pes > 0.0) & (obs_pes > 5.0)
        sigmas = np.sqrt(exp_pes[mask1] + self.single_pe_amp_std**2 * exp_pes[mask1])
        n_sigma = np.abs(obs_pes[mask1] - exp_pes[mask1]) / sigmas
        high_sigma = n_sigma >= 4.0
        low_sigma = ~high_sigma
        high_sigma_nllq = -obs_pes[mask1][high_sigma] * np.log(1e-4)
        low_sigma_nllq = 0.5 * (obs_pes[mask1][low_sigma] - exp_pes[mask1][low_sigma]) ** 2 / sigmas[low_sigma] ** 2

        mask2 = (exp_pes <= 0.0) & (obs_pes > 0.0)
        background_nll = -obs_pes[mask2] * np.log(1e-4)

        mask_t = (exp_pes > 0.0) & (obs_pes > 0.0) & np.isfinite(obs_ts)
        high_sigma_nllt = np.array([], dtype=np.float64)
        low_sigma_nllt = np.array([], dtype=np.float64)
        if np.any(mask_t):
            sigma_t = self.single_pe_time_std / np.sqrt(obs_pes[mask_t])
            n_sigma_t = np.abs(obs_ts[mask_t] - exp_ts[mask_t]) / sigma_t
            high_sigma_t = n_sigma_t >= 4.0
            low_sigma_t = ~high_sigma_t
            high_sigma_nllt = -obs_pes[mask_t][high_sigma_t] * np.log(1e-4)
            low_sigma_nllt = 0.5 * (obs_ts[mask_t][low_sigma_t] - exp_ts[mask_t][low_sigma_t]) ** 2 / sigma_t[low_sigma_t] ** 2

        return float(
            np.sum(no_signal_nll)
            + np.sum(signal_nllq)
            + np.sum(high_sigma_nllq)
            + np.sum(low_sigma_nllq)
            + np.sum(background_nll)
            + np.sum(high_sigma_nllt)
            + np.sum(low_sigma_nllt)
        )
