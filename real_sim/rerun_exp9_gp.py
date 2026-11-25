"""Re-run Exp 9 residual_ft for the sensor-bias targets after the GP-input fix; merges into the Exp 9 JSON."""

import json
import multiprocessing as mp

import exp9_residual_finetune as E9
from common import OUT, mean_std, save_json

if __name__ == "__main__":
    jobs = [("residual_ft", t, s) for t in ["biased", "noisy"] for s in E9.SEEDS]
    with mp.get_context("spawn").Pool(6) as pool:
        outs = pool.map(E9.job, jobs)
    res = json.load(open(f"{OUT}/exp9_residual_finetune.json"))
    for t in ["biased", "noisy"]:
        o = [x for x in outs if x[1] == t]
        res["agg"][t]["residual_ft"] = {m: mean_std([x[3][m] for x in o]) for m in o[0][3]}
        res["calibration"][t] = [x[4] for x in o]
        print(t, "residual_ft eff_err", res["agg"][t]["residual_ft"]["eff_err"]["mean"], "±",
              res["agg"][t]["residual_ft"]["eff_err"]["std"], "| calib", res["calibration"][t])
    res["note_gp_fix"] = "biased/noisy residual_ft re-run with GP fit on ground-truth pose (rerun_exp9_gp.py)"
    save_json("exp9_residual_finetune.json", res)
