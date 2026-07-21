"""
Streamlit scientist GUI for grok_phase_solver (polished).

Run via ``gps-gui`` or ``streamlit run …/gui/app.py``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from grok_phase_solver.gui.backend import (
    METHODS,
    WIZARD_SCENARIOS,
    demo_paths,
    format_user_error,
    parse_cell_string,
    resolve_wizard,
    run_solve_job,
    zip_outdir,
)


def _page_config() -> None:
    st.set_page_config(
        page_title="gps-solve GUI",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _init_state() -> None:
    defaults = {
        "last_summary": None,
        "last_work": None,
        "job_inputs": None,  # bytes + params for retry
        "status_log": [],
        "run_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _sidebar() -> dict:
    st.sidebar.header("Scenario")
    keys = list(WIZARD_SCENARIOS.keys())
    labels = [WIZARD_SCENARIOS[k]["label"] for k in keys]
    idx = st.sidebar.selectbox(
        "What describes your data?",
        range(len(keys)),
        format_func=lambda i: labels[i],
        help="Picks a sensible method and defaults. Use Advanced to override.",
    )
    scenario = keys[idx]
    st.sidebar.caption(WIZARD_SCENARIOS[scenario]["help"])

    with st.sidebar.expander("Advanced options", expanded=(scenario == "advanced")):
        method_adv = st.selectbox(
            "Method override",
            METHODS,
            index=0,
            help="Used when scenario is Advanced, or as base for auto.",
        )
        n_iter = st.slider("Iterations (CF / polish)", 20, 200, 80, 10)
        n_starts = st.slider("Multistart trials", 1, 6, 2)
        n_extend = st.slider("PhaSeed extend cycles", 4, 30, 12)
        n_peaks = st.slider("Max peaks", 10, 80, 40, 5)
        d_min = st.number_input(
            "d_min cutoff (Å, 0 = auto)",
            min_value=0.0,
            max_value=3.0,
            value=0.0,
            step=0.05,
        )
        seed = st.number_input("Random seed", min_value=0, value=0, step=1)
        ai_dm_hybrid = st.checkbox(
            "DM+AI hybrid tangent (Carrozzini)",
            value=False,
            help="Modified tangent weights AI phases as a priori info (λ=0.5).",
        )
        low_res_path = st.checkbox(
            "Low-res / large-Vol EDM path",
            value=False,
            help="More solvent flatten + longer seed anneal (hybrid-friendly).",
        )
        seed_quality_filter = st.checkbox(
            "Warn on Class 0 seed quality",
            value=True,
            help="Carrozzini-style Class 0/1 predictor diagnostics.",
        )
        show_log = st.checkbox("Capture solver log", value=True)
        verbose = st.checkbox("Verbose solver prints", value=False)

    wiz = resolve_wizard(scenario, method_adv)
    # Wizard supplies method/starts; advanced sliders still apply for n_iter unless
    # user stays on defaults — use max of wizard suggestion when wizard sets starts
    method = wiz["method"]
    if scenario != "advanced":
        # Prefer wizard n_starts when not advanced
        n_starts_use = int(wiz.get("n_starts", n_starts))
        n_iter_use = int(wiz.get("n_iter", n_iter))
    else:
        n_starts_use = int(n_starts)
        n_iter_use = int(n_iter)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Hard path / HA")
    patterson_ha = st.sidebar.checkbox(
        "Patterson HA heuristic seed",
        value=bool(wiz.get("patterson_ha", False)),
        help="Single-dataset heavy-atom trial (weak). Prefer known φ or fragment .res.",
    )
    ha_element = st.sidebar.text_input("HA element", value="Br")
    n_ha = st.sidebar.number_input("N HA sites", min_value=1, max_value=8, value=1)

    st.sidebar.markdown("---")
    if st.session_state.get("last_work"):
        st.sidebar.caption(f"Last work dir:\n`{st.session_state['last_work']}`")
    st.sidebar.caption(
        "Open physics/AI phasing · MIT · Not a general protein ab initio solver."
    )
    return {
        "scenario": scenario,
        "method": method,
        "n_iter": n_iter_use,
        "n_starts": n_starts_use,
        "n_extend": int(n_extend),
        "n_peaks": int(n_peaks),
        "d_min": float(d_min) if d_min and d_min > 0 else None,
        "seed": int(seed),
        "patterson_ha": bool(patterson_ha),
        "ha_element": ha_element.strip() or "Br",
        "n_ha": int(n_ha),
        "show_log": show_log,
        "verbose": verbose,
        "ai_dm_hybrid": bool(ai_dm_hybrid),
        "low_res_path": bool(low_res_path),
        "seed_quality_filter": bool(seed_quality_filter),
        "wizard_label": wiz.get("label", ""),
    }


def _load_demo_choice() -> tuple[bytes | None, bytes | None, bytes | None, str, str]:
    demos = demo_paths()
    choice = st.selectbox(
        "Packaged demo",
        [
            "(none — use uploads)",
            "Easy demo (ensemble path)",
            "Hard + 30% oracle phases",
            "Hard + fragment .res seed",
        ],
    )
    hkl_b = ins_b = seed_b = None
    seed_kind = ""
    note = ""
    if choice.startswith("Easy") and demos["easy_hkl"].exists():
        hkl_b = demos["easy_hkl"].read_bytes()
        ins_b = demos["easy_ins"].read_bytes() if demos["easy_ins"].exists() else None
        note = "Loaded **examples/demo_solve** — good for auto / ensemble."
    elif choice.startswith("Hard + 30%") and demos["hard_hkl"].exists():
        hkl_b = demos["hard_hkl"].read_bytes()
        ins_b = demos["hard_ins"].read_bytes() if demos["hard_ins"].exists() else None
        if demos["hard_seed_csv"].exists():
            seed_b = demos["hard_seed_csv"].read_bytes()
            seed_kind = "csv"
        note = "Loaded hard demo + **known_phases_30pct.csv** (partial_phaseed)."
    elif choice.startswith("Hard + fragment") and demos["hard_hkl"].exists():
        hkl_b = demos["hard_hkl"].read_bytes()
        ins_b = demos["hard_ins"].read_bytes() if demos["hard_ins"].exists() else None
        if demos["hard_fragment_res"].exists():
            seed_b = demos["hard_fragment_res"].read_bytes()
            seed_kind = "res"
        note = "Loaded hard demo + **fragment.res** (Fcalc seed)."
    return hkl_b, ins_b, seed_b, seed_kind, note


def _collect_inputs(params: dict) -> dict | None:
    """Build job kwargs from widgets; None if validation fails."""
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.subheader("1. Data")
        up_hkl = st.file_uploader(
            "Reflection file",
            type=["hkl", "mtz", "cif", "txt"],
            help="SHELX .hkl, MTZ, or CIF-style HKL from COD",
        )
        up_ins = st.file_uploader(
            "SHELX .ins / .res (recommended)",
            type=["ins", "res"],
        )
        cell_raw = st.text_area(
            "Cell (if no .ins)",
            height=68,
            placeholder="9.75, 8.89, 7.57, 90, 112.7, 90\n"
            "or: CELL 0.71073 9.75 8.89 7.57 90 112.7 90",
            help="Comma/space separated a b c α β γ, or a full SHELX CELL line.",
        )
        sg = st.text_input("Space group (if no .ins)", placeholder="P 1 21/c 1")

        demo_hkl, demo_ins, demo_seed, demo_seed_kind, demo_note = _load_demo_choice()
        if demo_note:
            st.info(demo_note)

        # Live cell parse feedback
        if cell_raw.strip():
            try:
                cell_csv, wl, notes = parse_cell_string(cell_raw)
                st.caption(f"Parsed cell: `{cell_csv}`" + (f" · λ={wl} Å" if wl else ""))
                for n in notes:
                    if not n.startswith("__SG__"):
                        st.caption(f"· {n}")
            except ValueError as e:
                st.warning(str(e))

    with col_r:
        st.subheader("2. Optional seeds (hard path)")
        up_seed_csv = st.file_uploader(
            "Phase seed CSV (h,k,l,phase_deg)", type=["csv", "txt"]
        )
        up_seed_res = st.file_uploader(
            "Fragment / SHELXS model (.res)", type=["res", "ins"]
        )
        up_seed_peaks = st.file_uploader("Prior peaks.csv", type=["csv"])
        st.caption(
            "Any seed → hard-path extension (`partial_phaseed`). "
            "Oracle bar: ~30% of strong |E| within ~20° of truth."
        )
        if params["scenario"] in ("have_phases", "have_fragment", "have_ha"):
            st.info(
                f"Scenario **{params['wizard_label']}** expects a matching seed "
                f"(or Patterson HA). Method → `{params['method']}`."
            )

    # Resolve bytes
    hkl_bytes = None
    hkl_name = "data.hkl"
    if up_hkl is not None:
        hkl_bytes = up_hkl.getvalue()
        hkl_name = up_hkl.name or hkl_name
    elif demo_hkl is not None:
        hkl_bytes = demo_hkl
        hkl_name = "demo.hkl"

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
    if (seed_csv or seed_res or seed_peaks or params["patterson_ha"]) and method == "auto":
        method = "partial_phaseed"

    cell_str = None
    sg_str = sg.strip() or None
    if cell_raw.strip():
        try:
            cell_str, _, notes = parse_cell_string(cell_raw)
            for n in notes:
                if n.startswith("__SG__:") and not sg_str:
                    sg_str = n.split(":", 1)[1]
        except ValueError as e:
            st.error(format_user_error(e))
            return None

    return {
        "hkl_bytes": hkl_bytes,
        "hkl_name": hkl_name,
        "ins_bytes": ins_bytes,
        "ins_name": ins_name or "data.ins",
        "cell": cell_str,
        "space_group": sg_str,
        "method": method,
        "phase_seed_csv_bytes": seed_csv,
        "phase_seed_res_bytes": seed_res,
        "seed_peaks_csv_bytes": seed_peaks,
        "patterson_ha": params["patterson_ha"],
        "ha_element": params["ha_element"],
        "n_ha": params["n_ha"],
        "n_iter": params["n_iter"],
        "n_starts": params["n_starts"],
        "n_extend": params["n_extend"],
        "n_peaks": params["n_peaks"],
        "d_min": params["d_min"],
        "seed": params["seed"],
        "verbose": params["verbose"],
        "capture_log": params["show_log"],
        "dm_ai_weight": 0.5 if params.get("ai_dm_hybrid") else 0.0,
        "low_res_path": bool(params.get("low_res_path")),
        "seed_quality_filter": bool(params.get("seed_quality_filter")),
    }


def _run_job(job: dict, *, status) -> dict | None:
    if job.get("hkl_bytes") is None:
        st.error("Please upload an HKL/MTZ file or select a packaged demo.")
        return None
    if job.get("ins_bytes") is None and not job.get("cell"):
        st.error("Provide a **.ins** file **or** unit cell (a,b,c,α,β,γ / CELL line).")
        return None

    log_lines: list[str] = []

    def progress(msg: str) -> None:
        log_lines.append(msg)
        status.markdown("**Status:** " + " → ".join(log_lines[-4:]))

    try:
        work = Path(tempfile.mkdtemp(prefix="gps_gui_"))
        summary = run_solve_job(work, progress=progress, **{
            k: v
            for k, v in job.items()
            if k
            in {
                "hkl_bytes",
                "hkl_name",
                "ins_bytes",
                "ins_name",
                "cell",
                "space_group",
                "method",
                "phase_seed_csv_bytes",
                "phase_seed_res_bytes",
                "seed_peaks_csv_bytes",
                "patterson_ha",
                "ha_element",
                "n_ha",
                "n_iter",
                "n_starts",
                "n_extend",
                "n_peaks",
                "d_min",
                "seed",
                "verbose",
                "capture_log",
                "dm_ai_weight",
                "low_res_path",
                "seed_quality_filter",
            }
        })
        # Persist inputs for peak-retry (drop large seeds except we re-read peaks from disk)
        st.session_state["job_inputs"] = {
            "hkl_bytes": job["hkl_bytes"],
            "hkl_name": job["hkl_name"],
            "ins_bytes": job.get("ins_bytes"),
            "ins_name": job.get("ins_name"),
            "cell": job.get("cell"),
            "space_group": job.get("space_group"),
            "n_iter": job.get("n_iter"),
            "n_starts": job.get("n_starts"),
            "n_extend": job.get("n_extend"),
            "n_peaks": job.get("n_peaks"),
            "d_min": job.get("d_min"),
            "seed": job.get("seed"),
        }
        st.session_state["last_summary"] = summary
        st.session_state["last_work"] = summary.get("work_dir")
        st.session_state["run_error"] = None
        st.session_state["status_log"] = log_lines
        return summary
    except Exception as e:
        st.session_state["run_error"] = format_user_error(e)
        st.error(st.session_state["run_error"])
        with st.expander("Technical details"):
            st.exception(e)
        return None


def _retry_with_peaks(params: dict, status) -> dict | None:
    """Re-run partial_phaseed using peaks.csv from the last solve."""
    summary = st.session_state.get("last_summary")
    inputs = st.session_state.get("job_inputs")
    if not summary or not inputs:
        st.warning("No previous solve to retry.")
        return None
    out = Path(summary["out_dir"])
    peaks = out / "peaks.csv"
    if not peaks.exists():
        st.error("No peaks.csv from the last run — cannot seed.")
        return None
    job = {
        **inputs,
        "method": "partial_phaseed",
        "seed_peaks_csv_bytes": peaks.read_bytes(),
        "phase_seed_csv_bytes": None,
        "phase_seed_res_bytes": None,
        "patterson_ha": False,
        "ha_element": params.get("ha_element", "Br"),
        "n_ha": params.get("n_ha", 1),
        "verbose": params.get("verbose", False),
        "capture_log": params.get("show_log", True),
    }
    st.info("Retrying with **peaks as fragment seed** (`partial_phaseed`)…")
    return _run_job(job, status=status)


def _render_results(summary: dict, params: dict) -> None:
    st.success(
        f"Done — method **`{summary.get('method')}`** · "
        f"{summary.get('n_reflections')} reflections · "
        f"{summary.get('n_peaks')} peaks"
    )
    d = summary.get("diagnostics") or {}

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Reflections", summary.get("n_reflections", "—"))
    fom = d.get("free_fom_composite")
    m2.metric("Free FOM", f"{float(fom):.3f}" if fom is not None else "—")
    m3.metric("Peaks", summary.get("n_peaks", "—"))
    m4.metric("Space group", summary.get("space_group") or "—")
    m5.metric("d_min (Å)", f"{summary.get('d_min'):.2f}" if summary.get("d_min") else "—")

    # Quality hints
    for h in summary.get("hints") or []:
        if "low" in h.lower() or "below" in h.lower() or "fail" in h.lower():
            st.warning(h)
        else:
            st.info(h)

    # Carrozzini-style / partial seed quality panel
    from grok_phase_solver.gui.backend import format_seed_quality_panel

    sq_panel = format_seed_quality_panel(d)
    if sq_panel.get("has_quality") or d.get("seed_size_meets_bar") is not None:
        st.markdown("##### Seed quality")
        c1, c2, c3, c4 = st.columns(4)
        if sq_panel.get("predicted_class") is not None:
            c1.metric("Predicted class", sq_panel["predicted_class"])
            p_succ = sq_panel.get("success_probability")
            c2.metric(
                "P(success)",
                f"{float(p_succ):.2f}" if p_succ is not None else "—",
            )
            mpe = sq_panel.get("predicted_mpe_deg")
            c3.metric("Est. seed MPE°", f"{float(mpe):.0f}" if mpe is not None else "—")
            feats = sq_panel.get("features") or {}
            c4.metric("max |E|", f"{feats.get('max_W', float('nan')):.2f}" if feats.get("max_W") is not None else "—")
        if d.get("seed_size_meets_bar") is not None:
            st.info(
                f"Partial-φ size bar: strong-|E| coverage "
                f"**{100 * float(d.get('seed_frac_strong') or 0):.0f}%** · "
                f"size: **{'OK' if d.get('seed_size_meets_bar') else 'BELOW'}** · "
                f"source: `{d.get('seed_kind') or d.get('seed_source') or '—'}`"
            )
        if sq_panel.get("warning"):
            st.warning(sq_panel["warning"])
        if sq_panel.get("recommend_fallback"):
            st.warning(
                "Class 0 seed — prefer fragment/HA partial-φ or enable DM+AI hybrid."
            )

    # One-click hard path recovery
    fom_v = float(fom) if fom is not None else 0.0
    can_retry = (Path(summary["out_dir"]) / "peaks.csv").exists()
    if can_retry and (
        fom_v < 0.60
        or summary.get("method")
        in ("charge_flipping", "strong_prior_phaseed", "hard_p1_phaseed", "ensemble")
    ):
        st.markdown("##### Next step if the map looks poor")
        if st.button(
            "Retry with peaks as seed (partial_phaseed)",
            type="secondary",
            key="retry_peaks",
        ):
            status = st.empty()
            with st.spinner("Re-phasing from density peaks…"):
                new_sum = _retry_with_peaks(params, status)
            if new_sum:
                st.rerun()

    warns = summary.get("warnings") or []
    if warns:
        with st.expander(f"Warnings ({len(warns)})", expanded=len(warns) <= 3):
            for w in warns:
                st.warning(w)

    out = Path(summary["out_dir"])
    tabs = st.tabs(
        ["Density", "Peaks", "Report", "SHELXL handoff", "Log", "Downloads"]
    )

    with tabs[0]:
        png = out / "density_slice.png"
        if png.exists():
            st.image(str(png), caption="Central density slice", use_container_width=True)
        else:
            st.write("No density_slice.png (matplotlib may be unavailable).")

    with tabs[1]:
        peaks = summary.get("peaks") or []
        if peaks:
            st.dataframe(peaks, use_container_width=True, hide_index=True)
        else:
            st.write("No peaks above threshold.")

    with tabs[2]:
        st.markdown(summary.get("report_md") or "_No report._")

    with tabs[3]:
        st.markdown(
            "Copy this into a terminal after downloading **trial.res** "
            "(or use files in the work directory)."
        )
        st.code(summary.get("shelxl_snippet") or "", language="bash")
        st.markdown(
            """
**Olex2 / ShelXle**

1. Download `trial.res` and your experimental `.hkl`.
2. Open the `.res` in Olex2 — assign elements to Q peaks.
3. Refine with SHELXL (or Olex2’s refinement).

gps-solve **phases** only; R-factors and ADPs come from refinement.
"""
        )

    with tabs[4]:
        log = summary.get("log") or ""
        if log.strip():
            st.text(log[-12000:])
        else:
            st.caption("No captured log (enable “Capture solver log” in Advanced).")
        if st.session_state.get("status_log"):
            st.caption("GUI steps: " + " → ".join(st.session_state["status_log"]))

    with tabs[5]:
        st.markdown("Download individual files or a zip of the full export folder.")
        cols = st.columns(3)
        names = [
            "trial.res",
            "report.md",
            "phases.csv",
            "peaks.csv",
            "density_slice.png",
            "solve_summary.json",
            "mapped_seed.csv",
        ]
        for i, name in enumerate(names):
            p = out / name
            if p.exists():
                with cols[i % 3]:
                    st.download_button(
                        label=name,
                        data=p.read_bytes(),
                        file_name=name,
                        key=f"dl_{name}_{summary.get('work_dir', '')[-8:]}",
                    )
        zpath = out.parent / "gps_solve_out.zip"
        try:
            zip_outdir(out, zpath)
            st.download_button(
                label="Download all (zip)",
                data=zpath.read_bytes(),
                file_name="gps_solve_out.zip",
                mime="application/zip",
                key=f"dl_zip_{summary.get('work_dir', '')[-8:]}",
            )
        except Exception as e:
            st.caption(f"Zip failed: {e}")

        st.caption(
            f"Work dir: `{summary.get('work_dir')}` · Out: `{summary.get('out_dir')}`"
        )


def main() -> None:
    _page_config()
    _init_state()

    st.title("gps-solve — crystallographic phasing")
    st.markdown(
        "Upload experimental **HKL** (+ **INS** or cell), pick a **scenario**, "
        "and export density + **trial.res** for Olex2 / SHELXL. "
        "Same engine as the `gps-solve` CLI."
    )

    params = _sidebar()
    job = _collect_inputs(params)

    st.subheader("3. Run")
    c1, c2 = st.columns([2, 1])
    with c1:
        run = st.button("Phase structure", type="primary", use_container_width=True)
    with c2:
        clear = st.button("Clear last results", use_container_width=True)
    if clear:
        st.session_state["last_summary"] = None
        st.session_state["last_work"] = None
        st.session_state["job_inputs"] = None
        st.session_state["run_error"] = None
        st.rerun()

    status = st.empty()

    if run:
        if job is None:
            st.stop()
        with st.spinner(
            f"Phasing with `{job.get('method')}`… "
            "(seconds to a few minutes depending on method)"
        ):
            summary = _run_job(job, status=status)
        if summary:
            _render_results(summary, params)
            return

    # Show previous results if available
    if st.session_state.get("last_summary"):
        st.markdown("---")
        st.subheader("Last results")
        _render_results(st.session_state["last_summary"], params)
    else:
        st.markdown(
            """
### Quick guide

| Scenario | What to do |
|----------|------------|
| Easy / high-res | Scenario **Easy** or **auto** — no seed needed |
| Hard + known φ | Upload phase CSV |
| Hard + SHELXS fragment | Upload `.res` model |
| Map failed once | **Retry with peaks as seed** on the results page |

Outputs always include `report.md`, `density_slice.png`, `peaks.csv`, **`trial.res`**.
"""
        )


if __name__ == "__main__":
    main()
