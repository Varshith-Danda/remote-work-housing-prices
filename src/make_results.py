"""Run the analyses added for the final results submission and save tables and figures.

    .venv/bin/python src/make_results.py          # about one minute

Tables go to outputs/tables/r_*.csv, figures to outputs/figures/r_*.png, report copies to report/figures/r_*.jpg.
Builds on the methodology results (outputs/tables/m_*.csv), which come from src/make_method_results.py.
"""
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

import methods as M  # noqa: E402
import results as R  # noqa: E402
import viz  # noqa: E402
import viz_results as VR  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
F, T, RF = ROOT / "outputs" / "figures", ROOT / "outputs" / "tables", ROOT / "report" / "figures"
viz.set_style()


def save(fig, name):
    fig.savefig(F / f"{name}.png")
    plt.close(fig)
    im = Image.open(F / f"{name}.png").convert("RGB")
    if im.width > 1800:
        im = im.resize((1800, round(im.height * 1800 / im.width)), Image.LANCZOS)
    im.save(RF / f"{name}.jpg", quality=90, optimize=True, subsampling=0)


gd, xd = M.load()
rt, rs = R.rent_event_study(gd)
rt.to_csv(T / "r_rent_event_study.csv", index=False)
rs.to_csv(T / "r_rent_event_summary.csv", index=False)
ft, fs_ = R.flag_free_event_study(gd)
ft.to_csv(T / "r_flagfree_event_study.csv", index=False)
fs_.to_csv(T / "r_flagfree_event_summary.csv", index=False)
ex = R.exposure_specs(xd)
ex.to_csv(T / "r_exposure_specs.csv", index=False)
# the same specifications without ZIP codes whose work-from-home estimate is unreliable (margin of error over half the estimate)
R.exposure_specs(xd[~xd["unreliable"]]).to_csv(T / "r_exposure_specs_reliable.csv", index=False)
cd = R.cluster_determinants(gd, xd)
cd.to_csv(T / "r_cluster_determinants.csv", index=False)

full = pd.read_csv(T / "m_event_summary.csv")
save(VR.summary_figure(full, fs_), "r_1_summary")
save(VR.rent_figure(rt), "r_2_rent")
save(VR.exposure_figure(ex), "r_3_exposure")
save(VR.cluster_determinants_figure(cd), "r_4_clusters")
print("done")
