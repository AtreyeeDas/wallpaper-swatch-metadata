import json
import os
import pandas as pd
import streamlit as st
from openai import OpenAI

from options_loader import load_options
from db import get_engine, upsert_swatch, fetch_all
from llm import extract_metadata, generate_description, validate_categorical

st.set_page_config(page_title="Swatch Metadata Extractor", layout="wide")
st.title("Wallpaper Swatch Metadata Extractor")

# ---------- Load options ----------
@st.cache_data
def _cached_options():
    return load_options()

color_opts, design_opts, theme_opts = _cached_options()
color_set, design_set, theme_set = set(color_opts), set(design_opts), set(theme_opts)

# ---------- Secrets ----------
# For local: put in .streamlit/secrets.toml
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
DATABASE_URL = st.secrets.get("DATABASE_URL", "")

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

if not DATABASE_URL:
    st.error("Missing DATABASE_URL in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
engine = get_engine(DATABASE_URL)

# ---------- Session state ----------
def _init_state():
    st.session_state.setdefault("draft_meta", None)        # last extracted + editable draft
    st.session_state.setdefault("accepted_meta", None)     # frozen snapshot after Accept
    st.session_state.setdefault("meta_dirty", False)       # if user edits after accept
    st.session_state.setdefault("description", "")         # generated/editable text
    st.session_state.setdefault("desc_based_on_meta", None)# snapshot used for current description

_init_state()

# ---------- Sidebar: Download anytime ----------
st.sidebar.header("Dataset")
rows = fetch_all(engine)
if rows:
    df = pd.DataFrame(rows)

    # Convert secondary_colors JSON string -> readable string for CSV
    def pretty_secondary(x):
        try:
            arr = json.loads(x) if isinstance(x, str) else []
            if isinstance(arr, list):
                return ", ".join(arr)
        except Exception:
            pass
        return str(x) if x is not None else ""

    if "secondary_colors" in df.columns:
        df["secondary_colors"] = df["secondary_colors"].apply(pretty_secondary)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name="swatch_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )
    with st.sidebar.expander("Preview latest rows"):
        st.dataframe(df.head(25), use_container_width=True)
else:
    st.sidebar.info("No saved records yet.")

st.divider()

# ---------- Main: Upload + ID ----------
col1, col2 = st.columns([2, 2])
with col1:
    uploaded = st.file_uploader("Upload swatch image", type=["png", "jpg", "jpeg", "webp"])
with col2:
    swatch_id_input = st.text_input(
        "Swatch ID (optional). If blank, image filename is used.",
        placeholder="e.g., SWATCH_0001",
    )

if not uploaded:
    st.info("Upload a swatch image to begin.")
    st.stop()

image_bytes = uploaded.read()
filename = uploaded.name
default_swatch_id = os.path.splitext(filename)[0]
swatch_id = swatch_id_input.strip() if swatch_id_input.strip() else default_swatch_id

st.caption(f"Using **swatch_id** = `{swatch_id}`  |  filename = `{filename}`")

# --- Show uploaded swatch preview (Step 1 context) ---
st.subheader("Uploaded swatch (Preview)")
st.image(image_bytes, caption=f"{filename}  |  swatch_id: {swatch_id}", use_container_width=True)
st.divider()


# ---------- Extract / Regenerate metadata ----------
cA, cB = st.columns([1, 3])
with cA:
    if st.button("🔎 Extract / Regenerate metadata", use_container_width=True):
        with st.spinner("Extracting metadata from image..."):
            try:
                meta = extract_metadata(client, image_bytes, color_opts, design_opts, theme_opts)
                # Ensure required keys exist
                meta.setdefault("primary_color", "")
                meta.setdefault("secondary_colors", [])
                meta.setdefault("design_style", "")
                meta.setdefault("theme", "")
                meta.setdefault("suitable_for", "")
                st.session_state.draft_meta = meta
                st.session_state.accepted_meta = None
                st.session_state.meta_dirty = False
                st.session_state.description = ""
                st.session_state.desc_based_on_meta = None
            except Exception as e:
                st.error(f"Metadata extraction failed: {e}")

# If no draft yet, force user to extract first
if st.session_state.draft_meta is None:
    st.warning("Click **Extract / Regenerate metadata** to generate the initial metadata draft.")
    st.stop()

draft = st.session_state.draft_meta

# ---------- Show validation warnings (but allow user correction) ----------
errs = validate_categorical(draft, color_set, design_set, theme_set)
if errs:
    st.warning("LLM output needs review (you can correct using dropdowns):\n- " + "\n- ".join(errs))

st.subheader("Step 1 — Review / Edit metadata (options-restricted)")

# ---------- Editable controls (restricted to options) ----------
# We always render UI using allowed options so final data is safe.

def _mark_dirty():
    # If user edits after accepting, mark dirty and invalidate accepted snapshot + description
    if st.session_state.accepted_meta is not None:
        st.session_state.meta_dirty = True
        st.session_state.accepted_meta = None
        st.session_state.description = ""
        st.session_state.desc_based_on_meta = None

# Determine default indices safely
def safe_index(options, value):
    try:
        return options.index(value)
    except Exception:
        return 0

e1, e2, e3, e4, e5 = st.columns([1, 2, 1, 1, 2])

with e1:
    primary = st.selectbox(
        "Primary color",
        options=color_opts,
        index=safe_index(color_opts, draft.get("primary_color")),
        on_change=_mark_dirty,
    )

with e2:
    secondary_default = [x for x in (draft.get("secondary_colors") or []) if x in color_set]
    secondary = st.multiselect(
        "Secondary colors (add/remove)",
        options=color_opts,
        default=secondary_default,
        on_change=_mark_dirty,
    )

with e3:
    design_style = st.selectbox(
        "Design style",
        options=design_opts,
        index=safe_index(design_opts, draft.get("design_style")),
        on_change=_mark_dirty,
    )

with e4:
    theme = st.selectbox(
        "Theme",
        options=theme_opts,
        index=safe_index(theme_opts, draft.get("theme")),
        on_change=_mark_dirty,
    )

with e5:
    suitable_for = st.text_input(
        "Suitable for (free text)",
        value=str(draft.get("suitable_for") or ""),
        on_change=_mark_dirty,
    )

final_meta = {
    "primary_color": primary,
    "secondary_colors": secondary,
    "design_style": design_style,
    "theme": theme,
    "suitable_for": suitable_for,
}

b1, b2 = st.columns([1, 1])

with b1:
    if st.button("✅ Accept metadata", use_container_width=True):
        st.session_state.accepted_meta = final_meta
        st.session_state.meta_dirty = False
        st.success("Metadata accepted. You can now generate the description.")

with b2:
    if st.session_state.meta_dirty:
        st.warning("Metadata changed after acceptance. Please Accept again to generate description.")

st.divider()

# ---------- Step 2: Description generation (ONLY after accept) ----------
st.subheader("Step 2 — Generate description (based ONLY on accepted metadata)")

if st.session_state.accepted_meta is None:
    st.info("Accept the metadata first to enable description generation.")
    st.stop()

accepted = st.session_state.accepted_meta

d1, d2 = st.columns([1, 1])

with d1:
    if st.button("✍️ Generate / Regenerate description", use_container_width=True):
        with st.spinner("Generating description from accepted metadata..."):
            try:
                desc = generate_description(client, accepted)
                st.session_state.description = desc
                st.session_state.desc_based_on_meta = json.dumps(accepted, sort_keys=True)
            except Exception as e:
                st.error(f"Description generation failed: {e}")

with d2:
    st.caption("Description will be generated strictly from the accepted metadata snapshot.")

# Description editor
st.session_state.description = st.text_area(
    "Description (editable before saving)",
    value=st.session_state.description,
    height=120,
)

# Save button
if st.button("💾 Save (overwrite by swatch_id)", type="primary", use_container_width=True):
    if not st.session_state.description.strip():
        st.error("Description is empty. Generate or write a description before saving.")
        st.stop()

    # Safety: ensure description is based on current accepted snapshot
    current_sig = json.dumps(accepted, sort_keys=True)
    if st.session_state.desc_based_on_meta is not None and st.session_state.desc_based_on_meta != current_sig:
        st.warning("Metadata changed since last description generation. Please regenerate description.")
        st.stop()

    with st.spinner("Saving to database..."):
        try:
            upsert_swatch(engine, {
                "swatch_id": swatch_id,
                "primary_color": accepted["primary_color"],
                "secondary_colors": accepted["secondary_colors"],
                "design_style": accepted["design_style"],
                "theme": accepted["theme"],
                "suitable_for": accepted.get("suitable_for", ""),
                "description": st.session_state.description.strip(),
                "image_filename": filename,
            })
            st.success("Saved ✅ (If swatch_id existed, it was overwritten.)")
        except Exception as e:
            st.error(f"Save failed: {e}")
