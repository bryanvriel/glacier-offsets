"""Why a vertical M2 tide looks exactly like a horizontal Msf velocity in fixed-UTC offsets.

Synthetic demo, numpy + matplotlib only. Run from anywhere:
  python docs/velocity-recovery/opus_checks/m2_msf_alias_demo.py [--out figure.png]

The idea in one line: Msf is the M2-S2 beat (f_Msf = f_S2 - f_M2 exactly). S2 is exactly two cycles
per solar day, so at a fixed UTC time it is "frozen". Sampling at the same UTC every 6 days (or any
whole number of days) therefore sees M2 advance by exactly the phase that Msf advances, and the sampled
M2 height traces a 14.765-day sinusoid. In the range channel, a pair difference of that height is
indistinguishable from the pair displacement produced by a horizontal Msf velocity. Azimuth has
almost no vertical sensitivity (0.002), so only azimuth separates the two.

Panels:
  A  continuous M2 height, the 08:02 UTC samples, and the 14.765-day curve through them
  B  stroboscope: the tidal curve vs time of day for successive acquisitions; the 08:02 sample
     catches it at a different phase each time
  C  pair displacements in range and azimuth: vertical M2 vs the equivalent horizontal Msf
  D  how the apparent Msf phase rotates with acquisition time of day (S1 vs NISAR windows)
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

M2_PERIOD = 12.4206012/24          # days
S2_PERIOD = 0.5                    # days
MSF_PERIOD = 1/(1/S2_PERIOD - 1/M2_PERIOD)   # 14.765294 days; f_Msf = f_S2 - f_M2
W_M2, W_MSF = 2*np.pi/M2_PERIOD, 2*np.pi/MSF_PERIOD

# Synthetic setup (time in days since 2000-01-01 00:00 UTC, as in the pipeline)
T0 = 6210 + (8 + 2/60)/24          # 2017-01-01 08:02 UTC
A_M2, G_M2 = 0.15, 40.0            # vertical M2 height amplitude (m) and phase (deg) at the source
S_RANGE = np.array([-0.48, 0.35, -0.80])     # range sensitivity, ENU
S_AZ = np.array([-0.59, -0.81, 0.002])       # azimuth sensitivity, ENU
FLOW = np.array([np.cos(np.radians(200)), np.sin(np.radians(200)), 0.0])   # along-flow unit vector


def m2_height(t, hour_shift=0.0):
    return A_M2*np.cos(W_M2*(t + hour_shift/24) - np.radians(G_M2))


def acquisitions(n=61, hour_shift=0.0, gaps=(7, 8, 23, 40)):
    """Every 6 days at the same UTC; a few missing scenes so some pairs are 12 days."""
    k = np.array([i for i in range(n) if i not in gaps])
    return T0 + 6*k + hour_shift/24


def msf_columns(t):
    # Displacement (position) basis of a velocity v = C cos(w t) + S sin(w t): integral of v.
    return np.stack([np.sin(W_MSF*t), -np.cos(W_MSF*t)], -1)/W_MSF


def equivalent_msf(t):
    """Least-squares horizontal Msf velocity (C, S) whose position matches the sampled M2 height."""
    a = np.column_stack([msf_columns(t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(a, m2_height(t), rcond=None)
    fit = a @ coef
    return coef[:2], np.max(np.abs(fit - m2_height(t)))


def main(out):
    t = acquisitions()
    (c, s), misfit = equivalent_msf(t)
    amp, phase = np.hypot(c, s), np.degrees(np.arctan2(s, c))
    step = (6/M2_PERIOD) % 1
    print(f'M2 cycles per 6 days: {6/M2_PERIOD:.4f} -> sampled phase steps by {step:.4f} cycle '
          f'({-(1-step)*360:.1f} deg) each acquisition')
    print(f'S2 cycles per 6 days: {6/S2_PERIOD:.1f} -> frozen at fixed UTC')
    print(f'Alias period 6/(1-{step:.4f}) = {6/(1-step):.4f} d; Msf period = {MSF_PERIOD:.4f} d')
    print(f'{A_M2:.3f} m of M2 height looks like Msf velocity amplitude {amp:.4f} m/day '
          f'({amp/A_M2:.4f} m/day per m), phase {phase:.1f} deg; max sample misfit {misfit:.1e} m')

    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))

    # A: continuous tide, samples and the alias curve
    a = ax[0, 0]
    win = t[t < T0 + 75]
    tt = np.linspace(T0 - 1, T0 + 75, 40000)
    a.plot(tt - T0, m2_height(tt), color='0.75', lw=0.5, label='continuous vertical M2 (12.42 h)')
    pos = msf_columns(tt) @ [c, s]
    off = np.mean(m2_height(t) - msf_columns(t) @ [c, s])
    a.plot(tt - T0, pos + off, 'C0--', lw=1.5, label=f'14.77-day curve: integrated Msf velocity, {amp:.3f} m/day')
    a.plot(win - T0, m2_height(win), 'o', color='C3', ms=6, label='height sampled at 08:02 UTC every 6 days')
    a.set(xlabel='days since 2017-01-01 08:02 UTC', ylabel='height or position (m)', ylim=(-0.27, 0.17),
          title='A  M2 sampled at 08:02 UTC traces a 14.77-day sinusoid')
    a.legend(loc='lower center', fontsize=8, ncol=2, framealpha=0.95)

    # B: stroboscope by time of day
    b = ax[0, 1]
    hours = np.linspace(0, 24, 600)
    for i, ti in enumerate(t[:6]):
        day = np.floor(ti)
        b.plot(hours, m2_height(day + hours/24), color=plt.cm.viridis(i/6), lw=1.2,
               label=f'scene {i} (day {ti - T0:.0f})')
        b.plot((ti - day)*24, m2_height(ti), 'o', color=plt.cm.viridis(i/6), ms=8, mec='k')
    b.axvline(8 + 2/60, color='C3', lw=1)
    b.text(8.3, 0.16, '08:02 UTC', color='C3')
    lag = (1 - step)*M2_PERIOD*24          # M2 high water arrives this much later after each 6 days
    b.set(xlabel='hour of day (UTC)', ylabel='vertical M2 height (m)', xlim=(0, 24), ylim=(-0.27, 0.19),
          title=f'B  High tide comes {lag:.2f} h later every 6 days')
    b.legend(fontsize=7, ncol=3, loc='lower center', framealpha=0.95)

    # C: pair displacements in range and azimuth
    p = ax[1, 0]
    early, late = t[:-1], t[1:]
    early2, late2 = t[:-2:2], t[2::2]          # some longer pairs too
    e = np.r_[early, early2]; l = np.r_[late, late2]
    dh = m2_height(l) - m2_height(e)
    dx = (msf_columns(l) - msf_columns(e)) @ [c, s]          # along-flow displacement for the fitted Msf
    # Scale the horizontal Msf so its range projection matches the vertical one (what a range-only fit sees)
    k = S_RANGE[2]/(S_RANGE @ FLOW)
    r_vert, r_hor = S_RANGE[2]*dh, (S_RANGE @ FLOW)*k*dx
    a_vert, a_hor = S_AZ[2]*dh, (S_AZ @ FLOW)*k*dx
    order = np.argsort(e)
    mid = ((e + l)/2 - T0)[order]
    p.plot(mid, r_vert[order], 'o', color='C3', ms=7, mfc='none', label='range: vertical M2')
    p.plot(mid, r_hor[order], '.', color='C0', ms=6, label=f'range: horizontal Msf ({abs(k)*amp:.3f} m/day along flow)')
    p.plot(mid, a_vert[order], 's', color='C3', ms=4, alpha=0.6, label='azimuth: vertical M2 (×0.002)')
    p.plot(mid, a_hor[order], '^', color='C0', ms=4, alpha=0.6, label='azimuth: same horizontal Msf')
    p.set(xlabel='pair midpoint (days since start)', ylabel='pair displacement (m)', ylim=(-0.95, 0.65),
          title='C  Pair offsets: range cannot tell them apart; azimuth can')
    p.legend(fontsize=8, loc='lower center', ncol=2, framealpha=0.95)
    print(f'range pair difference vertical-vs-horizontal: max {np.max(np.abs(r_vert - r_hor)):.1e} m '
          f'(signal rms {np.sqrt(np.mean(r_vert**2)):.3f} m); azimuth vertical rms {np.sqrt(np.mean(a_vert**2)):.1e} m')

    # D: alias phase vs acquisition time of day
    d = ax[1, 1]
    shifts = np.linspace(-8, 16, 97)          # hours relative to 08:02
    def alias_phase(h):
        cs = equivalent_msf(acquisitions(hour_shift=h))[0]
        return np.degrees(np.arctan2(cs[1], cs[0]))

    shifts = np.linspace(-8, 16, 241)          # hours relative to 08:02
    rel = (np.array([alias_phase(h) for h in shifts]) - phase + 180) % 360 - 180
    rate = np.polyfit(shifts, np.unwrap(np.radians(rel)), 1)[0]*180/np.pi
    d.plot(8 + 2/60 + shifts, rel, '.', color='k', ms=3)
    for lo, hi, name, col in [(8 + 2/60, 8 + 2/60, 'S1 track 38', 'C3'), (10 + 35/60, 10 + 51/60, 'NISAR asc', 'C2'),
                              (21 + 45/60, 22 + 1/60, 'NISAR desc', 'C1')]:
        r = [(alias_phase(x - 8 - 2/60) - phase + 180) % 360 - 180 for x in (lo, hi)]
        label = name if lo == hi else f'{name}: {r[0]:+.0f}° to {r[1]:+.0f}°'
        d.axvspan(lo - 0.08, hi + 0.08, color=col, alpha=0.4, label=label)
        print(f'{name} {lo:5.2f}-{hi:5.2f} UTC: alias phase rotated {r[0]:+.1f} to {r[1]:+.1f} deg relative to S1')
    d.set(xlabel='acquisition time of day (UTC)', ylabel='apparent Msf phase minus S1 phase (deg)',
          xlim=(0, 24), ylim=(-185, 185), yticks=np.arange(-180, 181, 60),
          title=f'D  Alias phase rotates {rate:.1f}°/h of acquisition time (= S2 rate)')
    d.legend(fontsize=8, loc='lower right')
    print(f'apparent Msf phase rotation: {rate:.2f} deg per hour of acquisition time')

    fig.suptitle('A vertical M2 tide sampled at a fixed UTC time aliases exactly onto Msf '
                 f'(Msf = S2 - M2; {amp/A_M2:.3f} m/day of Msf velocity per m of M2)', fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f'wrote {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_suffix('.png'))
    main(parser.parse_args().out)
