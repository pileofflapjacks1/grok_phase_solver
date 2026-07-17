"""
Streamlit scientist GUI for grok_phase_solver.

Run via ``gps-gui`` or ``streamlit run …/gui/app.py``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

# Streamlit is an optional dependency (extras: gui)
import streamlit as st

from grok_phase_solver.gui.backend import METHODS, demo_paths, run_solve_job, zip_outdir


def _page_config() -> None:
    st.set_page_config(
        page_title="gps-solve GUI",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _sidebar_params() -> dict:
    st.sidebar.header("Phasing options")
    method = st.sidebar.selectbox(
        "Method",
        METHODS,
        index=0,
        help="auto → ensemble on easy data; hard uses prior/CF. "
        "Partial seeds force partial_phaseed when provided.",
    )
    n_iter = st.sidebar.slider("Iterations (CF / polish)", 20, 200, 80, 10)
    n_starts = st.sidebar.slider("Multistart trials", 1, 6, 2)
    n_extend = st.sidebar.slider("PhaSeed extend cycles", 4, 30, 12)
    n_peaks = st.sidebar.slider("Max peaks", 10, 80, 40, 5)
    d_min = st.sidebar.number_input(
        "d_min cutoff (Å, 0 = auto)",
        min_value=0.0,
        max_value=3.0,
        value=0.0,
        step=0.05,
    )
    seed = st.sidebar.number_input("Random seed", min_value=0, value=0, step=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Hard path / HA")
    patterson_ha = st.sidebar.checkbox(
        "Patterson HA heuristic seed",
        value=False,
        help="Single-dataset heavy-atom trial (weak). Prefer known φ or fragment .res.",
    )
    ha_element = st.sidebar.text_input("HA element", value="Br")
    n_ha = st.sidebar.number_input("N HA sites", min_value=1, max_value=8, value=1)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Open physics/AI phasing assistant · MIT · "
        "Not a general protein ab initio solver."
    )
    return {
        "method": method,
        "n_iter": int(n_iter),
        "n_starts": int(n_starts),
        "n_extend": int(n_extend),
        "n_peaks": int(n_peaks),
        "d_min": float(d_min) if d_min and d_min > 0 else None,
        "seed": int(seed),
        "patterson_ha": bool(patterson_ha),
        "ha_element": ha_element.strip() or "Br",
        "n_ha": int(n_ha),
    }


def _load_demo_choice() -> tuple[bytes | None, bytes | None, bytes | None, str]:
    demos = demo_paths()
    choice = st.selectbox(
        "Or load a packaged demo",
        [
            "(none — use uploads)",
            "Easy demo (ensemble path)",
            "Hard + 30% oracle phases",
            "Hard + fragment .res seed",
        ],
    )
    hkl_b = ins_b = seed_b = None
    seed_kind = ""
    if choice.startswith("Easy") and demos["easy_hkl"].exists():
        hkl_b = demos["easy_hkl"].read_bytes()
        ins_b = demos["easy_ins"].read_bytes() if demos["easy_ins"].exists() else None
        st.info("Loaded **examples/demo_solve** — good for `auto` / `ensemble`.")
    elif choice.startswith("Hard + 30%") and demos["hard_hkl"].exists():
        hkl_b = demos["hard_hkl"].read_bytes()
        ins_b = demos["hard_ins"].read_bytes() if demos["hard_ins"].exists() else None
        if demos["hard_seed_csv"].exists():
            seed_b = demos["hard_seed_csv"].read_bytes()
            seed_kind = "csv"
        st.info("Loaded hard demo + **known_phases_30pct.csv** (partial_phaseed).")
    elif choice.startswith("Hard + fragment") and demos["hard_hkl"].exists():
        hkl_b = demos["hard_hkl"].read_bytes()
        ins_b = demos["hard_ins"].read_bytes() if demos["hard_ins"].exists() else None
        if demos["hard_fragment_res"].exists():
            seed_b = demos["hard_fragment_res"].read_bytes()
            seed_kind = "res"
        st.info("Loaded hard demo + **fragment.res** (Fcalc seed).")
    return hkl_b, ins_b, seed_b, seed_kind


def main() -> None:
    _page_config()
    st.title("gps-solve — crystallographic phasing")
    st.markdown(
        "Upload experimental **HKL** (+ **INS** or cell/SG), choose a method, "
        "and export density + **trial.res** for Olex2 / SHELXL. "
        "Same engine as the `gps-solve` CLI."
    )

    params = _sidebar_params()

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.subheader("1. Data")
        up_hkl = st.file_uploader("Reflection file (.hkl / CIF HKL / .mtz)", type=None)
        up_ins = st.file_uploader("SHELX .ins / .res (recommended)", type=None)
        cell = st.text_input(
            "Cell a,b,c,α,β,γ (if no .ins)",
            placeholder="9.75,8.89,7.57,90,112.7,90",
        )
        sg = st.text_input("Space group (if no .ins)", placeholder="P 1 21/c 1")

        demo_hkl, demo_ins, demo_seed, demo_seed_kind = _load_demo_choice()

    with col_r:
        st.subheader("2. Optional seeds (hard path)")
        up_seed_csv = st.file_uploader(
            "Phase seed CSV (h,k,l,phase_deg)", type=["csv", "txt"]
        )
        up_seed_res = st.file_uploader(
            "Fragment / SHELXS model (.res)", type=["res", "ins"]
        )
        up_seed_peaks = st.file_uploader(
            "Prior peaks.csv", type=["csv"]
        )
        st.caption(
            "If any seed is provided, hard-path extension is used "
            "(`partial_phaseed`). Oracle bar: ~30% strong |E| within ~20°."
        )

    st.subheader("3. Run")
    run = st.button("Phase structure", type="primary", use_container_width=True)

    if not run:
        st.markdown(
            """
### Decision tree (short)

| Situation | Method |
|-----------|--------|
| Default / easy high-res | `auto` or `ensemble` |
| Hard + known φ | `partial_phaseed` + phase CSV |
| Hard + SHELXS fragment | upload `.res` seed |
| External classical | `shelxs` / `shelxs+shelxe` (local binaries) |

Outputs always include `report.md`, `density_slice.png`, `peaks.csv`, **`trial.res`**.
"""
        )
        return

    # Resolve inputs
    hkl_bytes = None
    hkl_name = "data.hkl"
    if up_hkl is not None:
        hkl_bytes = up_hkl.getvalue()
        hkl_name = up_hkl.name or hkl_name
    elif demo_hkl is not None:
        hkl_bytes = demo_hkl
        hkl_name = "demo.hkl"

    if hkl_bytes is None:
        st.error("Please upload an HKL file or select a packaged demo.")
        return

    ins_bytes = up_ins.getvalue() if up_ins is not None else demo_ins
    ins_name = up_ins.name if up_ins is not None else "data.ins"

    seed_csv = up_seed_csv.getvalue() if up_seed_csv is not None else None
    seed_res = up_seed_res.getvalue() if up_seed_res is not None else None
    seed_peaks = up_seed_peaks.getvalue() if up_seed_peaks is not None else None
    if demo_seed is not None and demo_seed_kind == "csv" and seed_csv is None:
        seed_csv = demo_seed
    if demo_seed is not None and demo_seed_kind == "res" and seed_res is None:
        seed_res = demo_seed

    method = params["method"]
    if (seed_csv or seed_res or seed_peaks) and method == "auto":
        method = "partial_phaseed"

    cell_str = cell.strip() or None
    sg_str = sg.strip() or None
    if ins_bytes is None and cell_str is None:
        st.error("Provide a .ins file **or** unit cell (a,b,c,α,β,γ).")
        return

    with st.spinner("Phasing… (classical / hybrid methods can take seconds to minutes)"):
        try:
            work = Path(tempfile.mkdtemp(prefix="gps_gui_"))
            summary = run_solve_job(
                work,
                hkl_bytes=hkl_bytes,
                hkl_name=hkl_name,
                ins_bytes=ins_bytes,
                ins_name=ins_name or "data.ins",
                cell=cell_str,
                space_group=sg_str,
                method=method,
                d_min=params["d_min"],
                n_iter=params["n_iter"],
                n_starts=params["n_starts"],
                n_extend=params["n_extend"],
                n_peaks=params["n_peaks"],
                seed=params["seed"],
                phase_seed_csv_bytes=seed_csv,
                phase_seed_res_bytes=seed_res,
                seed_peaks_csv_bytes=seed_peaks,
                patterson_ha=params["patterson_ha"],
                ha_element=params["ha_element"],
                n_ha=params["n_ha"],
                verbose=False,
            )
            st.session_state["last_summary"] = summary
            st.session_state["last_work"] = str(work)
        except Exception as e:
            st.exception(e)
            return

    _render_results(summary)


def _render_results(summary: dict) -> None:
    st.success(f"Done — method **`{summary.get('method')}`**")
    d = summary.get("diagnostics") or {}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reflections", summary.get("n_reflections", "—"))
    m2.metric("Free FOM", f"{d.get('free_fom_composite', float('nan')):.3f}"
              if d.get("free_fom_composite") is not None else "—")
    m3.metric("Peaks", summary.get("n_peaks", "—"))
    m4.metric("Space group", summary.get("space_group") or "—")

    if d.get("seed_size_meets_bar") is not None:
        st.info(
            f"Seed quality (truth-free): strong-|E| coverage "
            f"**{100 * float(d.get('seed_frac_strong') or 0):.0f}%** · "
            f"size bar: **{'OK' if d.get('seed_size_meets_bar') else 'BELOW'}** · "
            f"source: `{d.get('seed_kind') or d.get('seed_source') or '—'}`"
        )

    warns = summary.get("warnings") or []
    if warns:
        with st.expander("Warnings", expanded=True):
            for w in warns:
                st.warning(w)

    out = Path(summary["out_dir"])
    tabs = st.tabs(["Density", "Peaks", "Report", "Downloads"])

    with tabs[0]:
        png = out / "density_slice.png"
        if png.exists():
            st.image(str(png), caption="Central density slice", use_container_width=True)
        else:
            st.write("No density_slice.png (matplotlib may be unavailable).")

    with tabs[1]:
        peaks = summary.get("peaks") or []
        if peaks:
            st.dataframe(peaks, use_container_width=True)
        else:
            st.write("No peaks above threshold.")

    with tabs[2]:
        st.markdown(summary.get("report_md") or "_No report._")

    with tabs[3]:
        st.markdown("Download individual files or a zip of the full export folder.")
        for name in (
            "trial.res",
            "report.md",
            "phases.csv",
            "peaks.csv",
            "density_slice.png",
            "solve_summary.json",
            "mapped_seed.csv",
        ):
            p = out / name
            if p.exists():
                st.download_button(
                    label=f"Download {name}",
                    data=p.read_bytes(),
                    file_name=name,
                    key=f"dl_{name}",
                )
        zpath = out.parent / "gps_solve_out.zip"
        try:
            zip_outdir(out, zpath)
            st.download_button(
                label="Download all (zip)",
                data=zpath.read_bytes(),
                file_name="gps_solve_out.zip",
                mime="application/zip",
                key="dl_zip",
            )
        except Exception as e:
            st.caption(f"Zip failed: {e}")

        st.caption(f"Server work dir: `{summary.get('out_dir')}`")


if __name__ == "__main__":
    main()
