#!/usr/bin/env python3
"""
ROT RH / XI-QED SHARED LAW — THRESHOLD STRUCTURE AUDIT

Mechanism audit for the frozen Xi-QED two-scale shared law.

This script does NOT search formulas, coefficients, masses, or external Q scales.
It keeps the previously locked scales fixed:

  K2    = - d^2/dt^2 log Xi(1/2+i t)|_{t=0}
  A0    = 2*pi/K2
  Delta = alpha(0)^-1 - A0

  Q1 = (4/pi) * m_pi
  Q2 = sqrt(2*pi) * red(m_mu, m_pi),     red(a,b)=a*b/(a+b)

Baseline loop assignment:

  Q1 uses electron loop only.
  Q2 uses electron + muon loops.

Purpose:

  Decompose the exact spacelike one-loop QED vacuum-polarization signal into
  threshold/log/asymptotic pieces and ask where the match comes from:

    1. electron logarithmic part
    2. finite-mass threshold residual
    3. on-shell -5/3 constant
    4. exact Feynman-parameter kernel shape
    5. cancellation/agreement between Q1 and Q2

Important constraints:

  - no coefficient menu
  - no formula grammar search
  - no additive mixed-mass external Q scales
  - no fitting to alpha after the frozen law is declared
  - forbidden mixed scale is reported as diagnostic only

This is a numerical mechanism audit, not a proof and not a derivation of alpha.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from typing import Dict, Iterable, List, Tuple

import mpmath as mp
import numpy as np


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def banner(title: str) -> None:
    print("=" * 120)
    print(title)
    print("=" * 120)


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_json(path: str, obj) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_csv(path: str, rows: List[Dict]) -> None:
    ensure_parent(path)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    cols: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in cols:
                cols.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def f12(x: float) -> str:
    return f"{float(x):.12g}"


def safe_ratio(a: float, b: float) -> float:
    if b == 0.0:
        return float("inf")
    return float(a / b)


# -----------------------------------------------------------------------------
# Xi curvature invariant
# -----------------------------------------------------------------------------

def xi_completed(s: mp.mpf | mp.mpc) -> mp.mpf | mp.mpc:
    """Completed Riemann xi function xi(s)."""
    return mp.mpf("0.5") * s * (s - 1) * mp.power(mp.pi, -s / 2) * mp.gamma(s / 2) * mp.zeta(s)


def log_xi_critical_line(t: mp.mpf | mp.mpc) -> mp.mpf | mp.mpc:
    return mp.log(xi_completed(mp.mpf("0.5") + 1j * t))


def xi_invariants(dps: int) -> Dict[str, str]:
    mp.mp.dps = dps
    L2 = mp.diff(log_xi_critical_line, mp.mpf("0"), 2)
    L4 = mp.diff(log_xi_critical_line, mp.mpf("0"), 4)
    K2 = mp.re(-L2)
    L4r = mp.re(L4)
    A0 = 2 * mp.pi / K2
    R4 = L4r / (K2 * K2)
    return {
        "K2": mp.nstr(K2, 90),
        "A0": mp.nstr(A0, 90),
        "L4": mp.nstr(L4r, 90),
        "R4": mp.nstr(R4, 90),
    }


# -----------------------------------------------------------------------------
# Frozen scales and QED pieces
# -----------------------------------------------------------------------------

def red_mass(a: float, b: float) -> float:
    return (a * b) / (a + b)


def locked_scales(masses: Dict[str, float]) -> Dict[str, float]:
    return {
        "Q1_MeV": (4.0 / math.pi) * masses["pi"],
        "Q2_MeV": math.sqrt(2.0 * math.pi) * red_mass(masses["mu"], masses["pi"]),
    }


def gauss_unit(n: int) -> Tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return (0.5 * (x + 1.0)).astype(np.float64), (0.5 * w).astype(np.float64)


def exact_spacelike_piece(Q: float, m: float, u: np.ndarray, w: np.ndarray) -> float:
    """
    Exact one-loop spacelike on-shell inverse-alpha shift for one charged lepton:

      Delta alpha^{-1}(Q) = (2/pi) int_0^1 dx x(1-x) log(1 + Q^2 x(1-x)/m^2)

    This is the baseline convention used in the locked-law audit.
    """
    z = u * (1.0 - u)
    return float((2.0 / math.pi) * np.dot(w, z * np.log1p(((Q / m) ** 2) * z)))


def asymptotic_minus_5over3_piece(Q: float, m: float) -> float:
    """
    Large-Q/m asymptotic piece:

      (1/(3*pi)) * [log(Q^2/m^2) - 5/3]
    """
    return float((1.0 / (3.0 * math.pi)) * (math.log((Q / m) ** 2) - 5.0 / 3.0))


def asymptotic_log_piece(Q: float, m: float) -> float:
    """
    Pure high-energy logarithmic piece:

      (1/(3*pi)) * log(Q^2/m^2)
    """
    return float((1.0 / (3.0 * math.pi)) * math.log((Q / m) ** 2))


def asymptotic_constant_piece() -> float:
    """
    The on-shell high-energy constant contribution per lepton:

      -(1/(3*pi)) * 5/3 = -5/(9*pi)
    """
    return float(-5.0 / (9.0 * math.pi))


def small_q_series_piece(Q: float, m: float) -> float:
    """
    Low-Q expansion through O((Q/m)^6):

      (2/pi) * [r/30 - r^2/280 + r^3/1890],  r=(Q/m)^2

    This is diagnostic only and is not valid for the present electron channel.
    """
    r = (Q / m) ** 2
    return float((2.0 / math.pi) * (r / 30.0 - (r * r) / 280.0 + (r ** 3) / 1890.0))


def kernel_moment_stats(Q: float, m: float, u: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    """
    Summarize the exact integrand shape for one lepton.
    These are diagnostics for the Feynman-parameter threshold structure.
    """
    z = u * (1.0 - u)
    r = (Q / m) ** 2
    y = z * np.log1p(r * z)
    integ = w * y
    total = float(np.sum(integ))
    if total == 0.0:
        return {
            "x_mean_weighted": float("nan"),
            "x_std_weighted": float("nan"),
            "central_80_mass_width": float("nan"),
            "frac_from_x_0p25_to_0p75": float("nan"),
            "frac_from_x_0p40_to_0p60": float("nan"),
            "integrand_peak_x": float(u[int(np.argmax(y))]),
        }
    p = integ / total
    mean = float(np.sum(p * u))
    std = float(math.sqrt(max(0.0, np.sum(p * (u - mean) ** 2))))
    cdf = np.cumsum(p)
    x10 = float(np.interp(0.10, cdf, u))
    x90 = float(np.interp(0.90, cdf, u))
    frac_25_75 = float(np.sum(integ[(u >= 0.25) & (u <= 0.75)]) / total)
    frac_40_60 = float(np.sum(integ[(u >= 0.40) & (u <= 0.60)]) / total)
    return {
        "x_mean_weighted": mean,
        "x_std_weighted": std,
        "central_80_mass_width": x90 - x10,
        "frac_from_x_0p25_to_0p75": frac_25_75,
        "frac_from_x_0p40_to_0p60": frac_40_60,
        "integrand_peak_x": float(u[int(np.argmax(y))]),
    }


def one_lepton_decomposition(Q: float, m: float, u: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    exact = exact_spacelike_piece(Q, m, u, w)
    log_only = asymptotic_log_piece(Q, m)
    constant = asymptotic_constant_piece()
    asymp = log_only + constant
    residual = exact - asymp
    smallq = small_q_series_piece(Q, m)
    r = Q / m
    out = {
        "Q_over_m": r,
        "threshold_2m_over_Q": safe_ratio(2.0 * m, Q),
        "exact": exact,
        "asymptotic_log_only": log_only,
        "asymptotic_constant_minus_5over3": constant,
        "asymptotic_minus_5over3": asymp,
        "finite_mass_residual_exact_minus_asymp": residual,
        "small_q_series_diag": smallq,
        "residual_over_exact": safe_ratio(residual, exact),
        "asymp_over_exact": safe_ratio(asymp, exact),
        "log_over_exact": safe_ratio(log_only, exact),
        "constant_over_exact": safe_ratio(constant, exact),
        "smallq_over_exact": safe_ratio(smallq, exact),
    }
    out.update(kernel_moment_stats(Q, m, u, w))
    return out


def sum_parts(parts: Iterable[float]) -> float:
    return float(sum(float(x) for x in parts))


def rel_err(pred: float, target: float) -> float:
    return abs(float(pred) - float(target)) / abs(float(target))


def pair_metrics(delta1: float, delta2: float, target: float) -> Dict[str, float]:
    r1 = rel_err(delta1, target)
    r2 = rel_err(delta2, target)
    return {
        "delta1": float(delta1),
        "delta2": float(delta2),
        "rel1": r1,
        "rel2": r2,
        "joint_max_rel": max(r1, r2),
        "joint_rms_rel": math.sqrt(0.5 * (r1 * r1 + r2 * r2)),
        "joint_mean_rel": 0.5 * (r1 + r2),
        "prediction_agreement_rel": abs(float(delta1) - float(delta2)) / abs(float(target)),
    }


def score_from_mode(
    mode: str,
    Q: float,
    loop_labels: List[str],
    masses: Dict[str, float],
    u: np.ndarray,
    w: np.ndarray,
) -> float:
    vals = []
    for lab in loop_labels:
        decomp = one_lepton_decomposition(Q, masses[lab], u, w)
        if mode == "exact":
            vals.append(decomp["exact"])
        elif mode == "asymp_minus_5over3":
            vals.append(decomp["asymptotic_minus_5over3"])
        elif mode == "log_only":
            vals.append(decomp["asymptotic_log_only"])
        elif mode == "constant_only":
            vals.append(decomp["asymptotic_constant_minus_5over3"])
        elif mode == "finite_mass_residual":
            vals.append(decomp["finite_mass_residual_exact_minus_asymp"])
        elif mode == "exact_minus_log_only":
            vals.append(decomp["exact"] - decomp["asymptotic_log_only"])
        elif mode == "small_q_series_diag":
            vals.append(decomp["small_q_series_diag"])
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return sum_parts(vals)


def hybrid_score(
    Q: float,
    loop_labels: List[str],
    masses: Dict[str, float],
    u: np.ndarray,
    w: np.ndarray,
    per_lepton_mode: Dict[str, str],
) -> float:
    vals = []
    for lab in loop_labels:
        vals.append(score_from_mode(per_lepton_mode.get(lab, "exact"), Q, [lab], masses, u, w))
    return sum_parts(vals)


# -----------------------------------------------------------------------------
# Protocol
# -----------------------------------------------------------------------------

def protocol_dict(args: argparse.Namespace) -> Dict:
    return {
        "audit": "rot_rh_xi_qed_threshold_structure_audit",
        "version": "1.0",
        "locked_law": {
            "A0": "2*pi/K2",
            "K2": "-d2 log Xi(1/2+i*t) at t=0",
            "Delta": "alpha(0)^-1 - A0",
            "Q1": "(4/pi)*m_pi_charged",
            "Q2": "sqrt(2*pi)*red(m_mu,m_pi_charged)",
            "red(a,b)": "a*b/(a+b)",
            "baseline_loops": {"Q1": ["e"], "Q2": ["e", "mu"]},
        },
        "constraints": [
            "no coefficient menu",
            "no formula grammar search",
            "no additive mixed-mass external Q scales",
            "no fitting to alpha after locked law",
            "forbidden mixed scale only diagnostic",
        ],
        "qed_kernel": {
            "exact_spacelike_one_lepton": "(2/pi) int_0^1 dx x(1-x) log(1 + Q^2 x(1-x)/m^2)",
            "asymptotic_minus_5over3": "(1/(3*pi))*(log(Q^2/m^2)-5/3)",
            "asymptotic_log_only": "(1/(3*pi))*log(Q^2/m^2)",
            "finite_mass_residual": "exact_spacelike - asymptotic_minus_5over3",
        },
        "inputs": {
            "dps": args.dps,
            "alpha0_inv": args.alpha0_inv,
            "me_MeV": args.me_MeV,
            "mmu_MeV": args.mmu_MeV,
            "mtau_MeV": args.mtau_MeV,
            "mpi_charged_MeV": args.mpi_charged_MeV,
            "quad_n": args.quad_n,
            "quad_n_list": args.quad_n_list,
            "interesting_rel_err": args.interesting_rel_err,
            "joint_rel_err": args.joint_rel_err,
            "threshold_residual_ratio_min": args.threshold_residual_ratio_min,
        },
    }


def protocol_hash(proto: Dict) -> str:
    s = json.dumps(proto, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Main audit
# -----------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    proto = protocol_dict(args)
    phash = protocol_hash(proto)

    if args.protocol_only:
        out = f"{args.out_prefix}_threshold_protocol.json"
        write_json(out, {"protocol_hash": phash, "protocol": proto})
        banner("PROTOCOL ONLY")
        print(f"protocol hash : {phash}")
        print(f"file written  : {out}")
        return 0

    masses = {
        "e": float(args.me_MeV),
        "mu": float(args.mmu_MeV),
        "tau": float(args.mtau_MeV),
        "pi": float(args.mpi_charged_MeV),
    }

    banner("ROT RH / XI-QED SHARED LAW — THRESHOLD STRUCTURE AUDIT")
    print(f"time                    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"dps                     : {args.dps}")
    print(f"alpha(0)^-1             : {float(args.alpha0_inv):.15f}")
    print(f"masses MeV              : e={masses['e']}, mu={masses['mu']}, tau={masses['tau']}, pi_ch={masses['pi']}")
    print("frozen law              : Q1=(4/pi)m_pi, Q2=sqrt(2pi)*red(mu,pi)")
    print("baseline loops          : Q1=e ; Q2=e+mu")
    print("search                  : DISABLED — mechanism audit only")
    print(f"protocol hash           : {phash}")
    print(f"out_prefix              : {args.out_prefix}")
    print("-" * 120)

    banner("COMPUTING XI CURVATURE COUNT")
    inv = xi_invariants(args.dps)
    K2 = float(inv["K2"])
    A0 = float(inv["A0"])
    R4 = float(inv["R4"])
    delta_needed = float(args.alpha0_inv) - A0
    rel_gap = delta_needed / float(args.alpha0_inv)
    print(f"K2 = -d2 log Xi(0)       : {K2:.18g}")
    print(f"A0 = 2*pi/K2             : {A0:.15f}")
    print(f"R4 = L4/K2^2             : {R4:.18g}")
    print(f"delta alpha^-1 - A0      : {delta_needed:.16g}")
    print(f"relative gap             : {rel_gap:.15g}")

    scales = locked_scales(masses)
    Q1 = scales["Q1_MeV"]
    Q2 = scales["Q2_MeV"]
    u, w = gauss_unit(args.quad_n)

    # Lepton-by-lepton decompositions.
    component_rows: List[Dict] = []
    for channel, Q, loops in [
        ("Q1", Q1, ["e"]),
        ("Q2", Q2, ["e", "mu"]),
    ]:
        for lab in loops:
            d = one_lepton_decomposition(Q, masses[lab], u, w)
            row = {
                "channel": channel,
                "Q_MeV": Q,
                "lepton": lab,
                "m_MeV": masses[lab],
            }
            row.update(d)
            component_rows.append(row)

    baseline_delta1 = score_from_mode("exact", Q1, ["e"], masses, u, w)
    baseline_delta2 = score_from_mode("exact", Q2, ["e", "mu"], masses, u, w)
    base_metrics = pair_metrics(baseline_delta1, baseline_delta2, delta_needed)

    banner("BASELINE EXACT SPACELIKE SCORE")
    print(f"Q1=(4/pi)m_pi            : {Q1:.12f} MeV")
    print(f"Q2=sqrt2pi*red(mu,pi)    : {Q2:.12f} MeV")
    print(f"Delta1 exact e           : {baseline_delta1:.13g} rel={base_metrics['rel1']:.6e}")
    print(f"Delta2 exact e+mu        : {baseline_delta2:.13g} rel={base_metrics['rel2']:.6e}")
    print(f"joint max/rms/mean rel   : {base_metrics['joint_max_rel']:.6e} / {base_metrics['joint_rms_rel']:.6e} / {base_metrics['joint_mean_rel']:.6e}")
    print(f"prediction agreement rel : {base_metrics['prediction_agreement_rel']:.6e}")

    banner("LEPTON-BY-LEPTON THRESHOLD DECOMPOSITION")
    for row in component_rows:
        print(
            f"{row['channel']} {row['lepton']:>2s}: "
            f"Q/m={row['Q_over_m']:.6g} 2m/Q={row['threshold_2m_over_Q']:.6g} "
            f"exact={row['exact']:.12g} log={row['asymptotic_log_only']:.12g} "
            f"const={row['asymptotic_constant_minus_5over3']:.12g} "
            f"asymp={row['asymptotic_minus_5over3']:.12g} "
            f"resid={row['finite_mass_residual_exact_minus_asymp']:.12g} "
            f"resid/exact={row['residual_over_exact']:.6g}"
        )

    # Channel totals by pieces.
    mode_rows: List[Dict] = []
    modes = [
        "exact",
        "asymp_minus_5over3",
        "log_only",
        "constant_only",
        "finite_mass_residual",
        "exact_minus_log_only",
        "small_q_series_diag",
    ]
    for mode in modes:
        d1 = score_from_mode(mode, Q1, ["e"], masses, u, w)
        d2 = score_from_mode(mode, Q2, ["e", "mu"], masses, u, w)
        m = pair_metrics(d1, d2, delta_needed)
        row = {"mode": mode, "diag": mode == "small_q_series_diag"}
        row.update(m)
        mode_rows.append(row)

    mode_rows_sorted = sorted(mode_rows, key=lambda r: (r["diag"], r["joint_max_rel"]))

    banner("PIECE-ONLY CHANNEL SCORES")
    print("Rows ask: if this piece alone tried to explain the Xi-alpha gap, how close would it be?")
    for i, row in enumerate(mode_rows_sorted, 1):
        tag = " DIAG" if row["diag"] else ""
        print(
            f"{i:02d}. joint={row['joint_max_rel']:.6e} rms={row['joint_rms_rel']:.6e} "
            f"rel1={row['rel1']:.3e} rel2={row['rel2']:.3e} "
            f"d1={row['delta1']:.12g} d2={row['delta2']:.12g} mode={row['mode']}{tag}"
        )

    # Hybrid ablations to isolate which finite-mass pieces matter.
    # For Q2, electron is highly relativistic but muon is near threshold-ish.
    hybrid_specs = [
        ("baseline_all_exact", {"e": "exact", "mu": "exact"}),
        ("all_asymp_minus_5over3", {"e": "asymp_minus_5over3", "mu": "asymp_minus_5over3"}),
        ("all_log_only", {"e": "log_only", "mu": "log_only"}),
        ("electron_exact_mu_asymp", {"e": "exact", "mu": "asymp_minus_5over3"}),
        ("electron_asymp_mu_exact", {"e": "asymp_minus_5over3", "mu": "exact"}),
        ("electron_log_mu_exact", {"e": "log_only", "mu": "exact"}),
        ("electron_exact_mu_log", {"e": "exact", "mu": "log_only"}),
        ("electron_exact_mu_residual_only", {"e": "exact", "mu": "finite_mass_residual"}),
        ("electron_log_mu_residual", {"e": "log_only", "mu": "finite_mass_residual"}),
        ("electron_asymp_mu_residual", {"e": "asymp_minus_5over3", "mu": "finite_mass_residual"}),
    ]

    hybrid_rows: List[Dict] = []
    for name, pmode in hybrid_specs:
        # Q1 only has electron; Q2 has electron+muon.
        d1 = hybrid_score(Q1, ["e"], masses, u, w, pmode)
        d2 = hybrid_score(Q2, ["e", "mu"], masses, u, w, pmode)
        m = pair_metrics(d1, d2, delta_needed)
        row = {"hybrid": name, "electron_mode": pmode.get("e"), "muon_mode": pmode.get("mu", "n/a")}
        row.update(m)
        hybrid_rows.append(row)

    hybrid_rows_sorted = sorted(hybrid_rows, key=lambda r: r["joint_max_rel"])

    banner("HYBRID ABLATION SCORES")
    print("Rows replace exact pieces by asymptotic/log/residual pieces without changing Q1,Q2.")
    for i, row in enumerate(hybrid_rows_sorted, 1):
        print(
            f"{i:02d}. joint={row['joint_max_rel']:.6e} rms={row['joint_rms_rel']:.6e} "
            f"rel1={row['rel1']:.3e} rel2={row['rel2']:.3e} agree={row['prediction_agreement_rel']:.3e} "
            f"hybrid={row['hybrid']} e={row['electron_mode']} mu={row['muon_mode']}"
        )

    # Muon threshold residual influence: compare baseline to Q2 with muon asymptotic.
    d2_mu_asymp = hybrid_score(Q2, ["e", "mu"], masses, u, w, {"e": "exact", "mu": "asymp_minus_5over3"})
    d2_mu_log = hybrid_score(Q2, ["e", "mu"], masses, u, w, {"e": "exact", "mu": "log_only"})
    d2_mu_resid_only = hybrid_score(Q2, ["e", "mu"], masses, u, w, {"e": "exact", "mu": "finite_mass_residual"})
    mu_exact = one_lepton_decomposition(Q2, masses["mu"], u, w)["exact"]
    mu_asymp = one_lepton_decomposition(Q2, masses["mu"], u, w)["asymptotic_minus_5over3"]
    mu_resid = one_lepton_decomposition(Q2, masses["mu"], u, w)["finite_mass_residual_exact_minus_asymp"]

    mechanism = {
        "baseline_joint_max_rel": base_metrics["joint_max_rel"],
        "baseline_rel1": base_metrics["rel1"],
        "baseline_rel2": base_metrics["rel2"],
        "q2_mu_exact_contribution": mu_exact,
        "q2_mu_asymptotic_minus_5over3": mu_asymp,
        "q2_mu_finite_mass_residual": mu_resid,
        "q2_mu_residual_over_exact": safe_ratio(mu_resid, mu_exact),
        "q2_with_mu_asymp_rel": rel_err(d2_mu_asymp, delta_needed),
        "q2_with_mu_log_rel": rel_err(d2_mu_log, delta_needed),
        "q2_with_mu_residual_only_rel": rel_err(d2_mu_resid_only, delta_needed),
        "exact_vs_mu_asymp_q2_degradation": safe_ratio(rel_err(d2_mu_asymp, delta_needed), base_metrics["rel2"]),
        "exact_vs_mu_log_q2_degradation": safe_ratio(rel_err(d2_mu_log, delta_needed), base_metrics["rel2"]),
    }

    banner("THRESHOLD MECHANISM DIAGNOSTIC")
    print(f"Q2 mu exact contribution          : {mechanism['q2_mu_exact_contribution']:.12g}")
    print(f"Q2 mu asymp -5/3 contribution     : {mechanism['q2_mu_asymptotic_minus_5over3']:.12g}")
    print(f"Q2 mu finite-mass residual        : {mechanism['q2_mu_finite_mass_residual']:.12g}")
    print(f"Q2 mu residual/exact              : {mechanism['q2_mu_residual_over_exact']:.6g}")
    print(f"Q2 rel if mu exact -> mu asymp    : {mechanism['q2_with_mu_asymp_rel']:.6e}")
    print(f"Q2 rel if mu exact -> mu log      : {mechanism['q2_with_mu_log_rel']:.6e}")
    print(f"Q2 rel if mu exact -> resid only  : {mechanism['q2_with_mu_residual_only_rel']:.6e}")
    print(f"Q2 degradation exact->mu asymp    : {mechanism['exact_vs_mu_asymp_q2_degradation']:.6g}x")
    print(f"Q2 degradation exact->mu log      : {mechanism['exact_vs_mu_log_q2_degradation']:.6g}x")

    # Quadrature stability for exact and residual pieces.
    stab_rows: List[Dict] = []
    for n in [int(x.strip()) for x in args.quad_n_list.split(",") if x.strip()]:
        uu, ww = gauss_unit(n)
        b1 = score_from_mode("exact", Q1, ["e"], masses, uu, ww)
        b2 = score_from_mode("exact", Q2, ["e", "mu"], masses, uu, ww)
        q2_mu_dec = one_lepton_decomposition(Q2, masses["mu"], uu, ww)
        mm = pair_metrics(b1, b2, delta_needed)
        stab_rows.append({
            "quad_n": n,
            "delta1_exact": b1,
            "delta2_exact": b2,
            "rel1": mm["rel1"],
            "rel2": mm["rel2"],
            "joint_max_rel": mm["joint_max_rel"],
            "prediction_agreement_rel": mm["prediction_agreement_rel"],
            "q2_mu_exact": q2_mu_dec["exact"],
            "q2_mu_residual": q2_mu_dec["finite_mass_residual_exact_minus_asymp"],
        })

    banner("QUADRATURE STABILITY — EXACT AND MUON RESIDUAL")
    for r in stab_rows:
        print(
            f"quad={r['quad_n']:4d} rel1={r['rel1']:.12e} rel2={r['rel2']:.12e} "
            f"joint={r['joint_max_rel']:.12e} q2_mu_resid={r['q2_mu_residual']:.12g}"
        )
    stability_span = max(r["joint_max_rel"] for r in stab_rows) - min(r["joint_max_rel"] for r in stab_rows)

    # Forbidden mixed diagnostic, exact only.
    Q_forbidden = (masses["pi"] + masses["e"]) * (4.0 / math.pi)
    forbidden_delta = score_from_mode("exact", Q_forbidden, ["e"], masses, u, w)
    forbidden_rel = rel_err(forbidden_delta, delta_needed)

    # Decision logic: this audit is not about p-values; it asks whether exact threshold structure is essential.
    baseline_interesting = base_metrics["joint_max_rel"] <= args.joint_rel_err
    exact_stable = stability_span <= 1e-7
    mu_threshold_material = abs(mechanism["q2_mu_residual_over_exact"]) >= args.threshold_residual_ratio_min
    asymp_degrades = mechanism["exact_vs_mu_asymp_q2_degradation"] >= args.asymp_degradation_min
    exact_best_hybrid = hybrid_rows_sorted[0]["hybrid"] == "baseline_all_exact"

    if baseline_interesting and exact_stable and mu_threshold_material and asymp_degrades and exact_best_hybrid:
        flag = "XI_QED_THRESHOLD_STRUCTURE_EXACT_KERNEL_ESSENTIAL"
    elif baseline_interesting and exact_stable and asymp_degrades:
        flag = "XI_QED_THRESHOLD_STRUCTURE_EXACT_KERNEL_FAVORED"
    elif baseline_interesting:
        flag = "XI_QED_THRESHOLD_STRUCTURE_BASELINE_INTERESTING_BUT_MECHANISM_WEAK"
    else:
        flag = "XI_QED_THRESHOLD_STRUCTURE_NOT_INTERESTING"

    summary = {
        "global_flag": flag,
        "protocol_hash": phash,
        "K2": K2,
        "A0": A0,
        "R4": R4,
        "delta_needed": delta_needed,
        "relative_gap": rel_gap,
        "Q1_MeV": Q1,
        "Q2_MeV": Q2,
        "baseline": base_metrics,
        "mechanism": mechanism,
        "stability_span": stability_span,
        "forbidden_mixed_diagnostic": {
            "Q_MeV": Q_forbidden,
            "delta_exact_e_loop": forbidden_delta,
            "rel": forbidden_rel,
            "status": "diagnostic_only_forbidden_additive_mixed_mass_external_scale",
        },
        "passes": {
            "baseline_interesting": baseline_interesting,
            "exact_stable": exact_stable,
            "mu_threshold_material": mu_threshold_material,
            "asymp_degrades": asymp_degrades,
            "exact_best_hybrid": exact_best_hybrid,
        },
    }

    write_json(f"{args.out_prefix}_threshold_protocol.json", {"protocol_hash": phash, "protocol": proto})
    write_csv(f"{args.out_prefix}_threshold_component_decomposition.csv", component_rows)
    write_csv(f"{args.out_prefix}_threshold_piece_scores.csv", mode_rows_sorted)
    write_csv(f"{args.out_prefix}_threshold_hybrid_ablations.csv", hybrid_rows_sorted)
    write_csv(f"{args.out_prefix}_threshold_stability.csv", stab_rows)
    write_json(f"{args.out_prefix}_threshold_summary.json", summary)
    write_json(f"{args.out_prefix}_threshold_meta.json", {
        "argv": sys.argv,
        "runtime_seconds": time.time() - t0,
        "python": sys.version,
        "numpy": np.__version__,
        "mpmath": mp.__version__,
    })

    banner("XI-QED THRESHOLD STRUCTURE SUMMARY")
    print(f"global flag                         : {flag}")
    print(f"protocol hash                       : {phash}")
    print(f"A0                                  : {A0:.15f}")
    print(f"delta needed                        : {delta_needed:.16g}")
    print(f"Q1,Q2                               : {Q1:.12f} MeV, {Q2:.12f} MeV")
    print(f"baseline rel1, rel2                 : {base_metrics['rel1']:.12e}, {base_metrics['rel2']:.12e}")
    print(f"baseline joint max/rms              : {base_metrics['joint_max_rel']:.12e}, {base_metrics['joint_rms_rel']:.12e}")
    print(f"Q2 mu residual/exact                : {mechanism['q2_mu_residual_over_exact']:.12g}")
    print(f"Q2 exact->mu-asymp degradation      : {mechanism['exact_vs_mu_asymp_q2_degradation']:.12g}x")
    print(f"Q2 exact->mu-log degradation        : {mechanism['exact_vs_mu_log_q2_degradation']:.12g}x")
    print(f"best hybrid                         : {hybrid_rows_sorted[0]['hybrid']}")
    print(f"forbidden mixed diagnostic rel      : {forbidden_rel:.12e}")
    print(f"stability span                      : {stability_span:.3e}")
    print(
        "passes baseline/stability/mu_threshold/asymp_degrades/exact_best: "
        f"{baseline_interesting} / {exact_stable} / {mu_threshold_material} / {asymp_degrades} / {exact_best_hybrid}"
    )
    print("-" * 120)
    print("Files written")
    for suffix in [
        "threshold_protocol.json",
        "threshold_component_decomposition.csv",
        "threshold_piece_scores.csv",
        "threshold_hybrid_ablations.csv",
        "threshold_stability.csv",
        "threshold_summary.json",
        "threshold_meta.json",
    ]:
        print(f"  {args.out_prefix}_{suffix}")
    banner(f"Total runtime: {time.time() - t0:.2f}s")
    print(f"AUDIT FLAG: {flag}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Frozen Xi-QED shared-law threshold-structure mechanism audit. No formula search."
    )
    p.add_argument("--dps", type=int, default=100)
    p.add_argument("--alpha0-inv", dest="alpha0_inv", type=float, default=137.035999177)
    p.add_argument("--me-MeV", dest="me_MeV", type=float, default=0.51099895000)
    p.add_argument("--mmu-MeV", dest="mmu_MeV", type=float, default=105.6583755)
    p.add_argument("--mtau-MeV", dest="mtau_MeV", type=float, default=1776.86)
    p.add_argument("--mpi-charged-MeV", dest="mpi_charged_MeV", type=float, default=139.57039)
    p.add_argument("--quad-n", type=int, default=256)
    p.add_argument("--quad-n-list", type=str, default="128,256,512")
    p.add_argument("--interesting-rel-err", type=float, default=1e-3)
    p.add_argument("--joint-rel-err", type=float, default=1e-3)
    p.add_argument("--threshold-residual-ratio-min", type=float, default=0.05,
                   help="Minimum |finite-mass residual/exact| for calling the threshold residual material.")
    p.add_argument("--asymp-degradation-min", type=float, default=25.0,
                   help="Minimum degradation factor when replacing exact muon kernel by asymptotic kernel.")
    p.add_argument("--protocol-only", action="store_true")
    p.add_argument("--out-prefix", type=str, default="xi_qed_threshold_structure")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
