"""
UP ARSENIC PREDICTION
SAMPLING-AWARE SPATIOTEMPORAL DASHBOARD
"""

from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="UP Arsenic Spatiotemporal Dashboard",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "docs"
    / "hotspot_analysis"
    / "sampling_aware_hotspot_change_2023_2024.csv"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌍 Uttar Pradesh Arsenic Spatiotemporal Dashboard"
)

st.markdown(
    """
    ### Sampling-Aware Groundwater Arsenic Hotspot Analysis

    This dashboard compares observed arsenic hotspot status
    between 2023 and 2024 while explicitly distinguishing
    unsampled spatial cells from observed non-hotspots.
    """
)

st.info(
    """
    **Research Note:** This analysis uses the project's
    selected arsenic hotspot threshold and spatial grid.
    A missing observation is not treated as evidence that
    arsenic was absent.
    """
)


# ============================================================
# FILE CHECK
# ============================================================

if not DATA_FILE.exists():

    st.error(
        "Sampling-aware analysis file was not found."
    )

    st.code(
        str(DATA_FILE)
    )

    st.info(
        "Run hotspot_sampling_aware.py first."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_FILE
    )

    return df


try:

    df = load_data()

except Exception as error:

    st.error(
        f"Unable to load analysis data:\n\n{error}"
    )

    st.stop()


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "Grid_Lat",
    "Grid_Lon",
    "Sampled_2023",
    "Sampled_2024",
    "Hotspot_2023",
    "Hotspot_2024",
    "Change_Type"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "Missing required columns:"
    )

    for col in missing_columns:
        st.write(f"- `{col}`")

    st.stop()


# ============================================================
# BASIC CLEANING
# ============================================================

for col in [
    "Grid_Lat",
    "Grid_Lon",
    "Sampled_2023",
    "Sampled_2024",
    "Hotspot_2023",
    "Hotspot_2024"
]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df = df.dropna(
    subset=[
        "Grid_Lat",
        "Grid_Lon"
    ]
)


# ============================================================
# BOTH-YEAR COMPARABLE CELLS
# ============================================================

both_sampled = df[
    (df["Sampled_2023"] == 1)
    &
    (df["Sampled_2024"] == 1)
].copy()


# ============================================================
# METRICS
# ============================================================

total_cells = len(df)

sampled_2023 = int(
    (df["Sampled_2023"] == 1).sum()
)

sampled_2024 = int(
    (df["Sampled_2024"] == 1).sum()
)

both_years = len(
    both_sampled
)


persistent = both_sampled[
    (both_sampled["Hotspot_2023"] == 1)
    &
    (both_sampled["Hotspot_2024"] == 1)
]

expansion = both_sampled[
    (both_sampled["Hotspot_2023"] == 0)
    &
    (both_sampled["Hotspot_2024"] == 1)
]

contraction = both_sampled[
    (both_sampled["Hotspot_2023"] == 1)
    &
    (both_sampled["Hotspot_2024"] == 0)
]

observed_non_hotspot = both_sampled[
    (both_sampled["Hotspot_2023"] == 0)
    &
    (both_sampled["Hotspot_2024"] == 0)
]


persistent_count = len(
    persistent
)

expansion_count = len(
    expansion
)

contraction_count = len(
    contraction
)

non_hotspot_count = len(
    observed_non_hotspot
)


# ============================================================
# 2023 HOTSPOTS AMONG COMPARABLE CELLS
# ============================================================

hotspot_2023_comparable = int(
    (both_sampled["Hotspot_2023"] == 1).sum()
)

hotspot_2024_comparable = int(
    (both_sampled["Hotspot_2024"] == 1).sum()
)


if hotspot_2023_comparable > 0:

    persistence_rate = (
        persistent_count
        / hotspot_2023_comparable
        * 100
    )

else:

    persistence_rate = 0.0


# ============================================================
# NET CHANGE
# ============================================================

net_change = (
    hotspot_2024_comparable
    - hotspot_2023_comparable
)


# ============================================================
# TITLE
# ============================================================

st.header(
    "📊 Sampling-Aware Summary"
)


# ============================================================
# KPI ROW 1
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Spatial Cells",
        total_cells
    )

with col2:

    st.metric(
        "Sampled in Both Years",
        both_years
    )

with col3:

    st.metric(
        "2023 Comparable Hotspots",
        hotspot_2023_comparable
    )

with col4:

    st.metric(
        "2024 Comparable Hotspots",
        hotspot_2024_comparable,
        delta=net_change
    )


# ============================================================
# KPI ROW 2
# ============================================================

st.subheader(
    "🔥 Hotspot Dynamics"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Persistent",
        persistent_count
    )

with col2:

    st.metric(
        "Expansion",
        expansion_count
    )

with col3:

    st.metric(
        "Contraction",
        contraction_count
    )

with col4:

    st.metric(
        "Persistence Rate",
        f"{persistence_rate:.2f}%"
    )


# ============================================================
# SAMPLING COVERAGE
# ============================================================

st.header(
    "🧪 Sampling Coverage"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "2023 Sampled Cells",
        sampled_2023
    )

with col2:

    st.metric(
        "2024 Sampled Cells",
        sampled_2024
    )

with col3:

    st.metric(
        "Both-Year Sampled Cells",
        both_years
    )


st.progress(
    min(
        both_years / total_cells,
        1.0
    )
)

st.caption(
    f"{both_years} of {total_cells} spatial cells "
    f"have observations in both years."
)


# ============================================================
# CHANGE DISTRIBUTION
# ============================================================

st.header(
    "📈 Sampling-Aware Change Classification"
)

change_order = [
    "Persistent hotspot",
    "Expansion",
    "Contraction",
    "Observed non-hotspot",
    "Newly sampled hotspot",
    "Newly sampled non-hotspot",
    "No 2024 observation",
    "Not sampled"
]

change_counts = (
    df["Change_Type"]
    .value_counts()
    .reindex(
        change_order,
        fill_value=0
    )
)


fig, ax = plt.subplots(
    figsize=(10, 6)
)

change_counts.plot(
    kind="bar",
    ax=ax
)

ax.set_title(
    "Spatial Cells by Sampling-Aware Classification"
)

ax.set_xlabel(
    "Classification"
)

ax.set_ylabel(
    "Number of Spatial Cells"
)

ax.tick_params(
    axis="x",
    rotation=45
)

plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# MAP FUNCTION
# ============================================================

def plot_categories(
    dataframe,
    title
):

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    categories = [
        "Persistent hotspot",
        "Expansion",
        "Contraction",
        "Observed non-hotspot",
        "Newly sampled hotspot",
        "Newly sampled non-hotspot",
        "No 2024 observation"
    ]

    for category in categories:

        subset = dataframe[
            dataframe["Change_Type"] == category
        ]

        if len(subset) == 0:
            continue

        ax.scatter(
            subset["Grid_Lon"],
            subset["Grid_Lat"],
            s=35,
            alpha=0.75,
            label=category
        )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="best",
        fontsize=8
    )

    plt.tight_layout()

    return fig


# ============================================================
# SPATIAL MAP
# ============================================================

st.header(
    "🗺️ Sampling-Aware Spatial Change Map"
)

fig = plot_categories(
    df,
    "Uttar Pradesh Arsenic Hotspot Change: 2023 → 2024"
)

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# COMPARABLE CELLS ONLY
# ============================================================

st.header(
    "🔬 Year-to-Year Comparable Cells"
)

st.write(
    """
    The following analysis includes only grid cells with
    observations in both 2023 and 2024.
    """
)

comparable_counts = pd.Series(
    {
        "Persistent hotspot": persistent_count,
        "Expansion": expansion_count,
        "Contraction": contraction_count,
        "Observed non-hotspot": non_hotspot_count
    }
)


fig, ax = plt.subplots(
    figsize=(8, 5)
)

comparable_counts.plot(
    kind="bar",
    ax=ax
)

ax.set_title(
    "Comparable Spatial Cells"
)

ax.set_xlabel(
    "Classification"
)

ax.set_ylabel(
    "Number of Cells"
)

ax.tick_params(
    axis="x",
    rotation=0
)

plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# PERSISTENT HOTSPOTS
# ============================================================

st.header(
    "📍 Persistent Hotspots"
)

if persistent_count > 0:

    persistent_display = persistent[
        [
            "Grid_Lat",
            "Grid_Lon",
            "Observations_2023",
            "Observations_2024",
            "Max_As_2023",
            "Max_As_2024",
            "Mean_As_2023",
            "Mean_As_2024"
        ]
    ].copy()

    st.dataframe(
        persistent_display.sort_values(
            [
                "Grid_Lat",
                "Grid_Lon"
            ]
        ),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No persistent hotspots were identified."
    )


# ============================================================
# EXPANSION
# ============================================================

st.header(
    "🔴 Expansion Cells"
)

if expansion_count > 0:

    expansion_display = expansion[
        [
            "Grid_Lat",
            "Grid_Lon",
            "Max_As_2023",
            "Max_As_2024",
            "Mean_As_2023",
            "Mean_As_2024"
        ]
    ].copy()

    st.dataframe(
        expansion_display.sort_values(
            [
                "Grid_Lat",
                "Grid_Lon"
            ]
        ),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No expansion cells identified."
    )


# ============================================================
# CONTRACTION
# ============================================================

st.header(
    "🔵 Contraction Cells"
)

if contraction_count > 0:

    contraction_display = contraction[
        [
            "Grid_Lat",
            "Grid_Lon",
            "Max_As_2023",
            "Max_As_2024",
            "Mean_As_2023",
            "Mean_As_2024"
        ]
    ].copy()

    st.dataframe(
        contraction_display.sort_values(
            [
                "Grid_Lat",
                "Grid_Lon"
            ]
        ),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No contraction cells identified."
    )


# ============================================================
# SAMPLING-AWARE INTERPRETATION
# ============================================================

st.header(
    "🔬 Research Summary"
)

st.markdown(
    f"""
### Key findings

- The analysis contains **{total_cells} spatial grid cells**.
- **{sampled_2023}** cells were sampled in 2023.
- **{sampled_2024}** cells were sampled in 2024.
- **{both_years}** cells were sampled in both years and
  therefore provide the directly comparable spatial subset.
- Within this comparable subset, **{hotspot_2023_comparable}**
  cells were hotspots in 2023.
- **{hotspot_2024_comparable}** cells were hotspots in 2024.
- **{persistent_count}** cells remained hotspots in both years.
- **{expansion_count}** cells changed from non-hotspot to hotspot.
- **{contraction_count}** cells changed from hotspot to non-hotspot.
- The persistence rate relative to the comparable 2023
  hotspot cells is **{persistence_rate:.2f}%**.
"""
)


# ============================================================
# METHODOLOGICAL WARNING
# ============================================================

with st.expander(
    "⚠️ Important Methodological Limitations"
):

    st.markdown(
        """
        **1. Sampling coverage**

        A spatial cell without a 2024 observation is not
        automatically considered a contraction. It is reported
        separately as having no 2024 observation.

        **2. Temporal comparability**

        The available 2024 dataset is identified as
        Pre-Monsoon, while the 2023 dataset does not specify
        a comparable season in the combined temporal file.

        **3. Spatial aggregation**

        The comparison uses the project's selected spatial grid.
        Different grid sizes can produce different hotspot
        classifications.

        **4. Interpretation**

        Expansion and contraction describe observed spatial
        classification changes. They do not establish the
        environmental cause of those changes.
        """
    )


# ============================================================
# COMPLETE DATA
# ============================================================

with st.expander(
    "🔎 View Complete Sampling-Aware Dataset"
):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DOWNLOAD
# ============================================================

st.header(
    "⬇️ Download Results"
)

csv_data = df.to_csv(
    index=False
).encode(
    "utf-8"
)

st.download_button(
    label="Download Sampling-Aware Hotspot Analysis",
    data=csv_data,
    file_name=(
        "UP_sampling_aware_"
        "hotspot_analysis_2023_2024.csv"
    ),
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "UP Arsenic Prediction Project | "
    "Sampling-Aware Spatiotemporal Research Dashboard"
)