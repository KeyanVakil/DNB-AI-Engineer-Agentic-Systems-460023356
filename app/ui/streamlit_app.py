"""FinSight Agents — Streamlit demo UI."""

from __future__ import annotations

import os
import time

import httpx
import streamlit as st

API_URL = os.environ.get("FINSIGHT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="FinSight Agents", layout="wide")
st.title("FinSight Agents")

page = st.sidebar.radio(
    "Navigation",
    ["Run", "Reports", "Trace", "Reviews", "Bench", "Drift"],
)


def _api(method: str, path: str, **kwargs) -> dict:
    try:
        resp = httpx.request(method, f"{API_URL}{path}", timeout=60, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return {}


if page == "Run":
    st.header("Generate Insight Report")
    data = _api("GET", "/customers")
    customers = data.get("items", [])
    if not customers:
        st.warning("No customers found. Seed the database first.")
    else:
        options = {f"{c['name']} ({c['id']})": c["id"] for c in customers}
        choice = st.selectbox("Customer", list(options.keys()))
        customer_id = options[choice]

        if st.button("Generate Report"):
            resp = _api("POST", "/runs", json={"customer_id": customer_id})
            run_id = resp.get("run_id")
            if run_id:
                st.info(f"Run started: `{run_id}`")
                with st.spinner("Waiting for completion…"):
                    for _ in range(120):
                        time.sleep(1)
                        r = _api("GET", f"/runs/{run_id}")
                        if r.get("status") in ("succeeded", "failed"):
                            break
                if r.get("status") == "succeeded":
                    st.success("Report ready!")
                    report = r.get("report", {})
                    st.markdown(report.get("markdown", ""))
                    st.json(report.get("json", {}))
                    evals = r.get("evals", {})
                    if evals:
                        st.subheader("Evaluations")
                        st.json(evals)
                else:
                    st.error(f"Run failed: {r.get('error')}")

elif page == "Reports":
    st.header("All Reports")
    data = _api("GET", "/runs", params={"status": "succeeded", "limit": 20})
    for item in data.get("items", []):
        with st.expander(f"{item['run_id']} — {item['customer_id']}"):
            r = _api("GET", f"/runs/{item['run_id']}")
            st.markdown(r.get("report", {}).get("markdown", ""))

elif page == "Trace":
    st.header("Agent Trace")
    run_id = st.text_input("Run ID")
    if run_id and st.button("Load Trace"):
        tree = _api("GET", f"/runs/{run_id}/trace")
        st.json(tree)

elif page == "Reviews":
    st.header("Human Review Queue")
    queue = _api("GET", "/reviews/queue")
    for item in queue.get("items", []):
        with st.expander(f"Run {item['run_id']}"):
            score = st.slider("Score", 1, 5, 3, key=item["run_id"])
            approved = st.checkbox("Approved", key=f"a-{item['run_id']}")
            notes = st.text_area("Notes", key=f"n-{item['run_id']}")
            if st.button("Submit", key=f"s-{item['run_id']}"):
                _api(
                    "POST",
                    f"/runs/{item['run_id']}/reviews",
                    json={"reviewer": "demo-user", "score": score, "approved": approved, "notes": notes},
                )
                st.success("Review submitted")

elif page == "Bench":
    st.header("Model Benchmarking")
    st.info("Benchmark variants defined in configs/models.yaml against the golden dataset.")
    if st.button("Run Bench"):
        resp = _api(
            "POST",
            "/bench",
            json={
                "config_path": "configs/models.yaml",
                "dataset_path": "app/eval/datasets/golden.jsonl",
                "wait": True,
            },
        )
        st.json(resp)

elif page == "Drift":
    st.header("Drift Events")
    metric_filter = st.selectbox("Metric", ["all", "judge_score"])
    params = {} if metric_filter == "all" else {"metric": metric_filter}
    data = _api("GET", "/drift", params=params)
    items = data.get("items", [])
    if not items:
        st.info("No drift events recorded yet.")
    else:
        for evt in items:
            badge = {"info": "🟢", "warn": "🟡", "alert": "🔴"}.get(evt["severity"], "⚪")
            st.write(f"{badge} **{evt['metric']}** — PSI {evt.get('psi', 'N/A')} — {evt['created_at']}")
