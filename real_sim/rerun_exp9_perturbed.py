"""Re-run Exp 9 residual_ft on the perturbed target (kf x 1.15) after the Step-0 scripted + lag-filtered SysID fix."""

import json
import multiprocessing as mp

import exp9_residual_finetune as E9
from common import OUT, mean_std, np, plt, save_json

if __name__ == "__main__":
    jobs = [("residual_ft", "perturbed", s) for s in E9.SEEDS]
    with mp.get_context("spawn").Pool(3) as pool:
        outs = pool.map(E9.job, jobs)
    with open(f"{OUT}/exp9_residual_finetune.json") as f:
        res = json.load(f)
    res["agg"]["perturbed"]["residual_ft"] = {m: mean_std([x[3][m] for x in outs]) for m in outs[0][3]}
    res["calibration"]["perturbed"] = [x[4] for x in outs]
    res["note_sysid_fix"] = (
        "perturbed residual_ft re-run with Step-0 multi-sine scripted excitation + lag-filtered thrust SysID"
    )
    save_json("exp9_residual_finetune.json", res)

    print(
        "perturbed residual_ft eff_err:",
        f"{res['agg']['perturbed']['residual_ft']['eff_err']['mean']:.3f} ± "
        f"{res['agg']['perturbed']['residual_ft']['eff_err']['std']:.3f}",
        "| calib:",
        res["calibration"]["perturbed"],
    )

    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(E9.ARMS))
    for j, t in enumerate(E9.TARGETS):
        ax.bar(
            x + (j - 1.5) * 0.2,
            [res["agg"][t][a]["eff_err"]["mean"] for a in E9.ARMS],
            0.2,
            yerr=[res["agg"][t][a]["eff_err"]["std"] for a in E9.ARMS],
            capsize=2,
            label=f"target: {t}",
        )
    ax.set_xticks(x, E9.ARMS)
    ax.set_ylabel("true hold error in target [m]")
    ax.set_title(f"Residual fine-tuning vs DR (equal {res['ft_steps'] // 1000}k source steps)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/exp9_residual_finetune.png", dpi=140)
