// Exact native receiver kernel for LicketyFit molecular photon scattering.
//
// This implements the same PMT-facing, polarized phase, finite circular
// aperture, attenuation-LUT and arrival-time bin equations used by the Numba
// reference kernel. It changes execution only, not the optical quadrature.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" int lf_scatter_fused_selected(
    int nsel, int nn,
    const double* p, const double* n,
    const double* node_pos, const double* node_pol,
    const double* phase_a, const double* phase_b,
    const int8_t* node_ch, const double* node_b,
    const double* node_ng, const double* node_bt,
    double aperture_radius, double facing_width,
    int nbin, double tmin, double tmax,
    const double* response_lut, int n_response,
    const double* attenuation_lut, int n_attenuation,
    double attenuation_xmax,
    double* charge, double* rayleigh, double* raman,
    double* node_mu, double* node_mt,
    int nthreads
) {
    if (nsel < 0 || nn < 0 || nbin < 0 || n_response < 2 || n_attenuation < 2) {
        return 1;
    }
    const double dt = nbin > 0 ? (tmax - tmin) / static_cast<double>(nbin) : 1.0;
    const double a2 = aperture_radius * aperture_radius;
    if (nthreads < 1) nthreads = 1;

    // Keep the one-thread path completely outside an OpenMP parallel region.
    // OpenMP's `if(false)` form still enters the runtime on every FCN call;
    // after tens of thousands of tiny receiver calls that caused severe
    // long-run latency growth in a single-process fit.  The lambda below is
    // invoked either by a plain serial loop or by an actual OpenMP team.
    auto process_pmt = [&](int jj) {
        const double px = p[3 * jj];
        const double py = p[3 * jj + 1];
        const double pz = p[3 * jj + 2];
        const double nx = n[3 * jj];
        const double ny = n[3 * jj + 1];
        const double nz = n[3 * jj + 2];
        double acc = 0.0;
        double acc_ray = 0.0;
        double acc_ram = 0.0;
        // Timing output is stored as [bin, PMT].  Updating that strided global
        // array for every quadrature node needlessly bounces among cache lines.
        // Accumulate the same node sequence in a tiny PMT-local buffer, then
        // transpose once after the loop.  The arithmetic order within every
        // bin is unchanged, so this is bitwise equivalent to the reference.
        // Do not clear all 128 stack doubles for charge-only receiver calls
        // (nbin == 0), which are the dominant coherent-MCS workload.  For
        // timing calls initialize exactly the live prefix; the subsequent
        // node accumulation order and every returned value are unchanged.
        double local_mu_stack[64];
        double local_mt_stack[64];
        std::vector<double> local_mu_heap;
        std::vector<double> local_mt_heap;
        double* local_mu = local_mu_stack;
        double* local_mt = local_mt_stack;
        if (nbin > 64) {
            local_mu_heap.assign(static_cast<std::size_t>(nbin), 0.0);
            local_mt_heap.assign(static_cast<std::size_t>(nbin), 0.0);
            local_mu = local_mu_heap.data();
            local_mt = local_mt_heap.data();
        } else if (nbin > 0) {
            std::fill_n(local_mu_stack, nbin, 0.0);
            std::fill_n(local_mt_stack, nbin, 0.0);
        }

        for (int j = 0; j < nn; ++j) {
            const double dx = px - node_pos[3 * j];
            const double dy = py - node_pos[3 * j + 1];
            const double dz = pz - node_pos[3 * j + 2];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= 1.0e-12) continue;
            const double r = std::sqrt(r2);
            const double inv_r = 1.0 / r;
            const double kx = dx * inv_r;
            const double ky = dy * inv_r;
            const double kz = dz * inv_r;
            const double facing = -(nx * kx + ny * ky + nz * kz);

            double visibility;
            if (facing_width <= 0.0) {
                visibility = facing > 0.0 ? 1.0 : 0.0;
            } else if (facing <= -facing_width) {
                visibility = 0.0;
            } else if (facing >= facing_width) {
                visibility = 1.0;
            } else {
                const double u = (facing + facing_width) / (2.0 * facing_width);
                visibility = 3.0 * u * u - 2.0 * u * u * u;
            }
            if (visibility <= 0.0) continue;

            double pol_dot = node_pol[3 * j] * kx
                           + node_pol[3 * j + 1] * ky
                           + node_pol[3 * j + 2] * kz;
            pol_dot = std::max(-1.0, std::min(1.0, pol_dot));
            const double phase = phase_a[j] * (1.0 - pol_dot * pol_dot) + phase_b[j];

            // Algebraically identical to
            // 2*(1-r/sqrt(r^2+a^2))/a^2, but avoids subtractive cancellation.
            const double sr = std::sqrt(r2 + a2);
            const double omega_per_area = 2.0 / (sr * (sr + r));

            const double c = std::max(0.0, std::min(1.0, facing));
            const double fr = c * static_cast<double>(n_response - 1);
            int ir = static_cast<int>(fr);
            double response;
            if (ir >= n_response - 1) {
                response = response_lut[n_response - 1];
            } else {
                const double tr = fr - static_cast<double>(ir);
                response = response_lut[ir]
                         + tr * (response_lut[ir + 1] - response_lut[ir]);
            }

            const double xatt = node_b[j] * r;
            double attenuation;
            if (xatt >= attenuation_xmax) {
                attenuation = std::exp(-xatt);
            } else if (xatt <= 0.0) {
                attenuation = 1.0;
            } else {
                const double fe = xatt * static_cast<double>(n_attenuation - 1)
                                / attenuation_xmax;
                int ie = static_cast<int>(fe);
                if (ie >= n_attenuation - 1) {
                    attenuation = attenuation_lut[n_attenuation - 1];
                } else {
                    const double te = fe - static_cast<double>(ie);
                    attenuation = attenuation_lut[ie]
                                + te * (attenuation_lut[ie + 1] - attenuation_lut[ie]);
                }
            }

            const double amp = phase * omega_per_area * response
                             * visibility * attenuation;
            if (!(amp > 0.0) || !std::isfinite(amp)) continue;
            acc += amp;
            if (node_ch[j] == 0) acc_ray += amp;
            else acc_ram += amp;

            if (nbin > 0) {
                const double tt = node_bt[j] + node_ng[j] * r / 299.792458;
                int ib = static_cast<int>((tt - tmin) / dt);
                if (ib < 0) ib = 0;
                else if (ib >= nbin) ib = nbin - 1;
                local_mu[ib] += amp;
                local_mt[ib] += amp * tt;
            }
        }
        charge[jj] = acc;
        rayleigh[jj] = acc_ray;
        raman[jj] = acc_ram;
        for (int ib = 0; ib < nbin; ++ib) {
            const int out_index = ib * nsel + jj;
            node_mu[out_index] = local_mu[ib];
            node_mt[out_index] = local_mt[ib];
        }
    };

#ifdef _OPENMP
    if (nthreads > 1) {
#pragma omp parallel for schedule(static) num_threads(nthreads)
        for (int jj = 0; jj < nsel; ++jj) {
            process_pmt(jj);
        }
    } else {
        for (int jj = 0; jj < nsel; ++jj) {
            process_pmt(jj);
        }
    }
#else
    for (int jj = 0; jj < nsel; ++jj) {
        process_pmt(jj);
    }
#endif
    return 0;
}
