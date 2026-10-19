"""Customer Segmentation (K-Means on RFM) — Streamlit application.

Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.config import RFM_FEATURES, RFM_LABELS
from src.data import load_rfm
from src.predict import ensure_trained, load_metrics, load_model, range_warnings, score_customer, validate_customer
from src.segment import NAMING_RULES, assign_segments, make_pipeline, profile, segment_names

PLOT_MARGIN = dict(l=10, r=10, t=10, b=10)
SEGMENT_COLORS = {
    "Champions": "#059669", "Loyal customers": "#34D399", "Regular mid-value": "#3B82F6",
    "New or occasional": "#F59E0B", "At-risk valuable": "#EF4444", "Lapsed low-value": "#9CA3AF",
}
SEGMENT_ICONS = {  # Material Symbols bundled with Streamlit (Apache-2.0); no image files needed
    "Champions": "workspace_premium", "Loyal customers": "favorite", "Regular mid-value": "autorenew",
    "New or occasional": "person_add", "At-risk valuable": "warning", "Lapsed low-value": "bedtime",
}
SHORT = {"recency": "Recency", "frequency": "Frequency", "monetary": "Monetary"}
PRESETS = {
    "Big recent spender": dict(recency=5, frequency=25, monetary=8000.0),
    "Steady regular": dict(recency=45, frequency=4, monetary=1200.0),
    "New customer": dict(recency=10, frequency=1, monetary=300.0),
    "One-off, long ago": dict(recency=250, frequency=1, monetary=150.0),
}


def color_of(name: str) -> str:
    return SEGMENT_COLORS[name.split(" (tier")[0]]


@st.cache_resource(show_spinner="Preparing the segmentation model…")
def get_model_and_metrics():
    ensure_trained()
    return load_model(), load_metrics()


@st.cache_data(show_spinner="Loading customers…")
def get_segmented(_model, trained_at: str) -> pd.DataFrame:
    rfm = load_rfm()
    names = segment_names(_model)
    seg = assign_segments(_model, rfm)
    return rfm.assign(segment=[names[s] for s in seg])


@st.cache_data(show_spinner="Fitting K-Means…")
def explore_k(k: int) -> tuple[pd.DataFrame, list[dict]]:
    rfm = load_rfm()
    model = make_pipeline(k).fit(rfm[RFM_FEATURES])
    names = segment_names(model)
    seg = assign_segments(model, rfm)
    return rfm.assign(segment=[names[s] for s in seg]), profile(rfm, seg, names)


def apply_preset(name: str) -> None:
    for k, v in PRESETS[name].items():
        st.session_state[f"in_{k}"] = v


def log_scatter(df: pd.DataFrame, height: int = 460, highlight: dict | None = None) -> go.Figure:
    fig = px.scatter(df, x="frequency", y="monetary", color="segment", log_x=True, log_y=True,
                     color_discrete_map={s: color_of(s) for s in df["segment"].unique()},
                     hover_data={"CustomerID": True, "recency": True}, opacity=0.6,
                     labels={"frequency": RFM_LABELS["frequency"], "monetary": RFM_LABELS["monetary"]})
    fig.update_traces(marker=dict(size=5))
    if highlight:
        fig.add_trace(go.Scatter(x=[highlight["frequency"]], y=[highlight["monetary"]], mode="markers",
                                 name="This customer", marker=dict(symbol="star", size=22, color="#DB2777",
                                                                   line=dict(width=1.5, color="white"))))
    fig.update_layout(height=height, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)", legend_title=None,
                      legend=dict(orientation="h", y=-0.2))
    return fig


# --------------------------------------------------------------------------- #
# Segments tab
# --------------------------------------------------------------------------- #
def render_segments(df: pd.DataFrame, metrics: dict) -> None:
    data = metrics["data"]
    k = st.columns(4)
    k[0].metric("Customers", f"{data['n_customers']:,}", border=True)
    k[1].metric("Orders", f"{data['n_invoices']:,}", border=True)
    k[2].metric("Revenue", f"£{data['total_revenue'] / 1e6:.2f}M", border=True)
    k[3].metric("Segments (k)", metrics["chosen_k"], border=True)
    st.caption(f"Transactions from {data['date_range'][0][:10]} to {data['date_range'][1][:10]}. "
               f"Recency is measured from {data['snapshot_date']}.")

    st.markdown("##### The three behaviours every customer is measured on")
    r1, r2, r3 = st.columns(3)
    med = metrics["overall_median"]
    r1.markdown(f":material/schedule: **Recency**: days since the last order. Lower is better. "
                f"Typical customer: {med['recency']:.0f} days.")
    r2.markdown(f":material/repeat: **Frequency**: number of separate orders. Higher is better. "
                f"Typical customer: {med['frequency']:.0f} orders.")
    r3.markdown(f":material/payments: **Monetary**: total amount spent. Higher is better. "
                f"Typical customer: £{med['monetary']:,.0f}.")

    st.subheader("Segment profiles")
    cols = st.columns(len(metrics["segments"]))
    for col, s in zip(cols, metrics["segments"]):
        with col:
            m = s["median"]
            st.markdown(f"### :material/{SEGMENT_ICONS[s['name'].split(' (tier')[0]]}:")
            ui.result_card(
                f"{s['customer_share']:.0%} of customers · {s['revenue_share']:.0%} of revenue",
                s["name"],
                f"{s['description']} Typical: last order {m['recency']:.0f} days ago, "
                f"{m['frequency']:.0f} orders, £{m['monetary']:,.0f} spent.",
                color_of(s["name"]),
            )
            with st.expander("Suggested action"):
                st.write(s["action"])
                st.caption("A hypothesis to test (e.g. with an A/B test). The data shows past behaviour, "
                           "not the effect of any campaign.")

    st.markdown("**How each segment's typical customer compares with the typical customer overall**")
    med = metrics["overall_median"]
    rows = []
    for s in metrics["segments"]:
        m = s["median"]
        rows.append({
            "Segment": s["name"],
            "Last order (days ago)": f"{m['recency']:.0f}  ({m['recency'] / med['recency']:.1f}×)",
            "Orders": f"{m['frequency']:.0f}  ({m['frequency'] / med['frequency']:.1f}×)",
            "Spend": f"£{m['monetary']:,.0f}  ({m['monetary'] / med['monetary']:.1f}×)",
            "Share of customers": f"{s['customer_share']:.1%}",
            "Share of revenue": f"{s['revenue_share']:.1%}",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(f"Medians; the multiple in brackets compares with the median customer (last order "
               f"{med['recency']:.0f} days ago, {med['frequency']:.0f} orders, £{med['monetary']:,.0f}). For "
               "recency a *smaller* multiple is better.")

    a, b = st.columns([1, 1.3], gap="large")
    with a:
        st.markdown("**Share of customers vs share of revenue**")
        seg = pd.DataFrame(metrics["segments"])
        long = seg.melt(id_vars="name", value_vars=["customer_share", "revenue_share"], var_name="measure")
        long["measure"] = long["measure"].map({"customer_share": "Customers", "revenue_share": "Revenue"})
        fig = px.bar(long, x="name", y="value", color="measure", barmode="group", text=long["value"].map("{:.0%}".format),
                     color_discrete_map={"Customers": "#A7F3D0", "Revenue": "#059669"})
        fig.update_layout(height=380, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)", legend_title=None,
                          yaxis=dict(tickformat=".0%", title=None), xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        top = metrics["segments"][0]
        st.caption(f"**{top['name']}** are {top['customer_share']:.0%} of customers but generate "
                   f"{top['revenue_share']:.0%} of revenue: a strongly concentrated customer base.")
    with b:
        st.markdown("**Segment fingerprints (cluster centres, standardised)**")
        rows = []
        for c in metrics["centroids"]:
            for f, z in zip(RFM_FEATURES, c["z"]):
                rows.append({"segment": c["name"], "feature": SHORT[f], "z": z})
        fig = px.line(pd.DataFrame(rows), x="feature", y="z", color="segment", markers=True,
                      color_discrete_map={c["name"]: color_of(c["name"]) for c in metrics["centroids"]})
        fig.add_hline(y=0, line_dash="dot", line_color="#6B7280", annotation_text="average customer")
        fig.update_layout(height=380, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)", legend_title=None,
                          yaxis_title="Std. deviations from average (log scale)", xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption("For recency, *below* zero is good (bought more recently than average). For frequency and "
                   "monetary, above zero is good.")

    st.markdown("**All customers, coloured by segment**")
    c, d = st.columns([1.4, 1], gap="large")
    with c:
        fig = px.scatter_3d(df, x="recency", y="frequency", z="monetary", color="segment", log_x=True,
                            log_y=True, log_z=True, opacity=0.6,
                            color_discrete_map={s: color_of(s) for s in df["segment"].unique()},
                            hover_data={"CustomerID": True})
        fig.update_traces(marker=dict(size=2.5))
        fig.update_layout(height=520, margin=PLOT_MARGIN, legend_title=None,
                          scene=dict(xaxis_title="Recency (days)", yaxis_title="Orders", zaxis_title="Spend (£)"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Drag to rotate. Axes are logarithmic, matching the log transform the model uses.")
    with d:
        feature = st.selectbox("Distribution by segment", RFM_FEATURES, format_func=RFM_LABELS.get)
        fig = px.box(df, x="segment", y=feature, color="segment", log_y=True, points=False,
                     color_discrete_map={s: color_of(s) for s in df["segment"].unique()},
                     category_orders={"segment": [s["name"] for s in metrics["segments"]]})
        fig.update_layout(height=430, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                          xaxis_title=None, yaxis_title=RFM_LABELS[feature])
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Export a segment for a campaign**")
    e1, e2 = st.columns([2, 1])
    choice = e1.selectbox("Segment", [s["name"] for s in metrics["segments"]], label_visibility="collapsed")
    subset = df[df["segment"] == choice].sort_values("monetary", ascending=False)
    e2.download_button(f"Download {len(subset):,} customers (CSV)", subset.to_csv(index=False).encode("utf-8"),
                       file_name=f"segment_{choice.lower().replace(' ', '_')}.csv", mime="text/csv",
                       width="stretch")


# --------------------------------------------------------------------------- #
# Score a customer tab
# --------------------------------------------------------------------------- #
def render_score(model, df: pd.DataFrame, metrics: dict) -> None:
    st.markdown("Enter a customer's purchase history to see which segment they belong to.")
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        cols = st.columns(2)
        for i, name in enumerate(PRESETS):
            cols[i % 2].button(name, on_click=apply_preset, args=(name,), width="stretch", key=f"preset_{i}")
        values = {
            "recency": st.number_input("Days since last purchase", 1, 3650, step=1, key="in_recency"),
            "frequency": st.number_input("Number of orders", 1, 10_000, step=1, key="in_frequency"),
            "monetary": st.number_input("Total spend (£)", 0.01, 10_000_000.0, step=50.0, key="in_monetary"),
        }
    with right:
        errors = validate_customer(values)
        if errors:
            for e in errors:
                st.error(e)
            return
        r = score_customer(model, values)
        seg = metrics["segments"][r["rank"]]
        ui.result_card("Assigned segment", r["name"], seg["description"], color_of(r["name"]))
        st.markdown(f"**Suggested action** *(hypothesis to test)*: {seg['action']}")
        for w in range_warnings(values, metrics["ranges"]):
            st.warning(f"**Outside the observed data:** {w}")
        dist = pd.DataFrame({"segment": list(r["distances"]), "distance": list(r["distances"].values())})
        fig = px.bar(dist.sort_values("distance"), x="distance", y="segment", orientation="h",
                     color="segment", color_discrete_map={s: color_of(s) for s in dist["segment"]})
        fig.update_layout(height=220, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                          xaxis_title="Distance to segment centre (shorter = more similar)", yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption("K-Means assigns a customer to the nearest centre, measured after the log transform "
                   "and standardisation. Similar distances mean the customer is a borderline case.")
    st.plotly_chart(log_scatter(df, 420, values), width="stretch")


# --------------------------------------------------------------------------- #
# Choosing k tab
# --------------------------------------------------------------------------- #
def render_choosing_k(metrics: dict) -> None:
    ev = pd.DataFrame(metrics["k_evaluation"])
    k = metrics["chosen_k"]
    st.subheader(f"Why {k} segments?")
    st.info(metrics["k_rule"])
    charts = [("inertia", "Inertia (within-cluster sum of squares) — look for the 'elbow'"),
              ("silhouette", "Silhouette score — higher is better"),
              ("davies_bouldin", "Davies–Bouldin index — lower is better")]
    cols = st.columns(3)
    for col, (metric, title) in zip(cols, charts):
        with col:
            st.markdown(f"**{title}**")
            fig = px.line(ev, x="k", y=metric, markers=True, color_discrete_sequence=["#059669"])
            fig.add_vline(x=k, line_dash="dash", line_color="#64748B", annotation_text=f"k = {k}")
            fig.update_layout(height=260, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)", yaxis_title=None)
            st.plotly_chart(fig, width="stretch")
    st.dataframe(ev.rename(columns={"davies_bouldin": "Davies-Bouldin", "calinski_harabasz": "Calinski-Harabasz"})
                 .style.format({"inertia": "{:.1f}", "silhouette": "{:.4f}", "Davies-Bouldin": "{:.3f}",
                                "Calinski-Harabasz": "{:.1f}"}), width="stretch", hide_index=True)
    s, ss = metrics["stability"], metrics["subsample_stability"]
    st.markdown("**Stability checks** (adjusted Rand index: 1.0 = identical assignments, 0 = chance agreement)")
    s1, s2 = st.columns(2)
    s1.metric("Different random starting points", f"{s['mean_ari']:.3f}", border=True,
              help=f"{s['n_seeds']} refits on all customers with new random seeds; minimum {s['min_ari']:.3f}.")
    s2.metric(f"Different {ss['fraction']:.0%} samples of customers", f"{ss['mean_ari']:.3f}", border=True,
              help=f"{ss['n_subsamples']} refits, each on a random {ss['fraction']:.0%} of customers; "
                   f"minimum {ss['min_ari']:.3f}.")
    st.caption("The first check only shows the result does not depend on K-Means' random initialisation. The "
               "second is stronger: it shows the segments survive changes to *which customers* are in the data. "
               f"Both are high, though the second is lower (minimum {ss['min_ari']:.2f}), as expected for "
               "overlapping groups.")
    st.markdown(
        f"**Honest reading:** silhouette scores of about {metrics['chosen']['silhouette']:.2f} mean the "
        "segments are real but **overlapping**. Customer behaviour is a continuum, not neatly separated groups. "
        "Customers near a boundary could reasonably belong to either neighbouring segment."
    )

    st.divider()
    st.markdown("#### Explore other values of k")
    k_try = st.slider("Number of segments", 2, 8, k, key="k_explore")
    seg_df, prof = explore_k(k_try)
    table = pd.DataFrame([{"Segment": p["name"], "Customers": p["customers"],
                           "Customer share": f"{p['customer_share']:.1%}", "Revenue share": f"{p['revenue_share']:.1%}",
                           "Median recency (days)": p["median"]["recency"], "Median orders": p["median"]["frequency"],
                           "Median spend (£)": round(p["median"]["monetary"])} for p in prof])
    a, b = st.columns([1, 1.2], gap="large")
    with a:
        st.dataframe(table, width="stretch", hide_index=True)
        if k_try != k:
            st.caption(f"Exploratory view only. The rest of the app uses the chosen k = {k}.")
    with b:
        st.plotly_chart(log_scatter(seg_df, 380), width="stretch")


# --------------------------------------------------------------------------- #
# How it works tab
# --------------------------------------------------------------------------- #
def render_how_it_works(metrics: dict) -> None:
    st.subheader("From transactions to segments")
    st.markdown(
        """
1. **Clean** 541,909 raw transaction lines (see *Data*): remove lines without a customer, cancellations,
   non-positive quantities or prices, non-product entries, and duplicates.
2. **Aggregate to one row per customer (RFM)**:
   - **Recency**: days since the customer's last purchase (lower = more recent)
   - **Frequency**: number of distinct orders
   - **Monetary**: total spend in £
3. **Log-transform** (`log(1 + x)`): spend ranges from £4 to £279,000, so without the log a handful of
   wholesale buyers would dominate every distance calculation.
4. **Standardise** each feature to mean 0 and standard deviation 1, so days, orders and pounds count equally.
5. **K-Means** groups customers so that each one is close to its segment's centre.
6. **Rank and name** segments from their centres using the explicit rules below.
        """
    )
    st.markdown("**K-Means** minimises the total squared distance between each customer and their cluster centre:")
    st.latex(r"\min_{C_1..C_k}\ \sum_{j=1}^{k}\ \sum_{x \in C_j} \lVert x - \mu_j \rVert^2")
    st.markdown("It alternates two steps until nothing changes: **(1) assign** every customer to the nearest "
                "centre; **(2) update** every centre to the mean of its customers. Because the result depends on "
                "the random starting centres, scikit-learn runs it 20 times (`n_init=20`) and keeps the best run.")
    st.markdown("#### How segments are named")
    st.markdown("`z` = position of the segment centre in standard deviations from the average customer; "
                "`value = (z_F + z_M) / 2`. Names follow from the centre's position; they are not chosen by hand.")
    st.dataframe(pd.DataFrame(NAMING_RULES, columns=["Recency rule", "Value rule", "Name"]),
                 width="stretch", hide_index=True)
    st.markdown("#### What this analysis can and cannot tell you")
    st.markdown(
        """
- ✅ Which customers behave similarly in **when**, **how often** and **how much** they buy.
- ⚠️ RFM ignores *what* customers buy, their margins, and why they behave as they do.
- ⚠️ Segments describe the past year (Dec 2010 – Dec 2011) of one UK online gift retailer; many customers
  are wholesalers. Patterns may not transfer to other businesses.
- ⚠️ K-Means assumes roughly round, similarly sized clusters, and always produces k groups, even when
  the underlying behaviour is a continuum.
- ⚠️ The suggested actions are reasonable starting hypotheses. Their effect has **not** been measured.
        """
    )


# --------------------------------------------------------------------------- #
# Data tab
# --------------------------------------------------------------------------- #
def render_data(df: pd.DataFrame, metrics: dict) -> None:
    data = metrics["data"]
    st.subheader("UCI Online Retail dataset")
    st.markdown(
        "All transactions of a UK-based online retailer of giftware between 1 Dec 2010 and 9 Dec 2011; many "
        "customers are wholesalers. Source: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning "
        "Repository. https://doi.org/10.24432/C5BW33, licensed CC BY 4.0."
    )
    a, b = st.columns([1.2, 1], gap="large")
    with a:
        st.markdown("**Cleaning steps**")
        steps = pd.DataFrame(data["cleaning_steps"]).rename(columns={"step": "Step", "rows": "Rows left",
                                                                      "removed": "Removed"})
        st.dataframe(steps, width="stretch", hide_index=True)
        st.caption(f"{data['n_customers']:,} customers remain; {data['uk_customer_share']:.0%} are based in the UK.")
    with b:
        st.markdown("**Why the log transform? Spend before and after**")
        view = st.radio("Scale", ["Raw £", "log(1 + £)"], horizontal=True, label_visibility="collapsed")
        x = df["monetary"] if view == "Raw £" else np.log1p(df["monetary"])
        fig = px.histogram(x=x, nbins=60, color_discrete_sequence=["#059669"])
        fig.update_layout(height=280, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)", xaxis_title=view,
                          yaxis_title="Customers")
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Customer-level RFM table**")
    st.dataframe(df, width="stretch", height=300, hide_index=True)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")
    ui.inject_css()
    try:
        model, metrics = get_model_and_metrics()
        df = get_segmented(model, metrics["trained_at_utc"])
    except Exception as exc:
        st.error(f"The model or data could not be loaded: {exc}")
        st.info("Run `python -m src.train` from the project folder, then reload this page.")
        st.stop()

    for k, v in PRESETS["Steady regular"].items():
        st.session_state.setdefault(f"in_{k}", v)

    ui.hero(
        "Customer Segmentation",
        f"K-Means clustering of {metrics['data']['n_customers']:,} real online-retail customers by how "
        "recently, how often and how much they buy, with profiles and suggested actions for every segment.",
        ["RFM + K-Means", f"k = {metrics['chosen_k']} · silhouette {metrics['chosen']['silhouette']:.2f}",
         f"£{metrics['data']['total_revenue'] / 1e6:.1f}M of real transactions"],
    )
    tabs = st.tabs(["👥 Segments", "🎯 Score a customer", "📐 Choosing k", "🧠 How it works", "🗂️ Data"])
    with tabs[0]:
        render_segments(df, metrics)
    with tabs[1]:
        render_score(model, df, metrics)
    with tabs[2]:
        render_choosing_k(metrics)
    with tabs[3]:
        render_how_it_works(metrics)
    with tabs[4]:
        render_data(df, metrics)
    ui.footer("Customer Segmentation · Arya Kumar · Educational machine-learning project")


main()
