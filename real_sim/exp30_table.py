"""Exp 30 report table + figure: completion / clean completion with 95% seed-bootstrap CIs, from exp30_final_matrix.json."""

from __future__ import annotations

import json

from common import OUT, np, plt

ROWS = ["nominal", "latency_dr", "full_dr", "hn_thrust_lat", "motor_sysid", "residual_ft", "oracle", "classical_expert",
        "student_f3_clean", "student_f3_mid", "student_f3_hard", "student_clean", "student_mid", "student_hard"]
CELLS = ["source_l2f", "target_nominal", "target_perturbed"]


def ci(vals, n_boot=10_000, seed=0):
    v = np.asarray(vals, float)
    b = np.random.default_rng(seed).choice(v, (n_boot, len(v))).mean(1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    agg = json.load(open(f"{OUT}/exp30_final_matrix.json"))["agg"]
    rows = [r for r in ROWS if r in agg]
    for course in ["figure8", "ellipse3d"]:
        print(f"\n**{course}** — completion / clean completion [95% CI] / median lap s\n")
        print("| Method | " + " | ".join(CELLS) + " |\n|---|" + "---|" * len(CELLS))
        for r in rows:
            out = []
            for c in CELLS:
                m = agg[r][course][c]
                cm, lo, hi = ci(m["clean_completion"]["values"])
                lap = m["median_lap_s"]["mean"]
                out.append(f"{m['completion']['mean']:.2f} / {cm:.2f} [{lo:.2f}, {hi:.2f}] / "
                           + ("—" if not np.isfinite(lap) else f"{lap:.2f}"))
            print(f"| `{r}` | " + " | ".join(out) + " |")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    x = np.arange(len(rows))
    for k, c in enumerate(["target_nominal", "target_perturbed"]):
        st = [ci(agg[r]["ellipse3d"][c]["clean_completion"]["values"]) for r in rows]
        ax[k].bar(x - 0.2, [agg[r]["ellipse3d"][c]["completion"]["mean"] for r in rows], 0.4, label="completion",
                  color="#9db4d6")
        ax[k].bar(x + 0.2, [s[0] for s in st], 0.4, yerr=[[s[0] - s[1] for s in st], [s[2] - s[0] for s in st]],
                  capsize=2, label="clean completion (95% CI)", color="#2f5d9e")
        ax[k].set_xticks(x, rows, rotation=45, ha="right", fontsize=8)
        ax[k].set_title(f"held-out ellipse3d, {c} (5 seeds × 30 eps)")
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp30_final_matrix.png", dpi=150)


if __name__ == "__main__":
    main()
