import pandas as pd

def load_options(
    color_csv_path: str = "color_options.csv",
    design_csv_path: str = "design_options.csv",
    theme_csv_path: str = "Theme_options.csv",
):
    """
    Loads allowed categorical options from CSVs.

    Expected columns:
      color_options.csv:  'GENERIC NAMES'
      design_options.csv: 'DESIGN STYLE'
      Theme_options.csv:  'THEMES'
    """
    color_df = pd.read_csv(color_csv_path)
    design_df = pd.read_csv(design_csv_path)
    theme_df = pd.read_csv(theme_csv_path)

    if "GENERIC NAMES" not in color_df.columns:
        raise ValueError("color_options.csv must contain column 'GENERIC NAMES'")
    if "DESIGN STYLE" not in design_df.columns:
        raise ValueError("design_options.csv must contain column 'DESIGN STYLE'")
    if "THEMES" not in theme_df.columns:
        raise ValueError("Theme_options.csv must contain column 'THEMES'")

    colors = (
        color_df["GENERIC NAMES"]
        .dropna()
        .astype(str)
        .map(lambda s: s.strip())
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )

    designs = (
        design_df["DESIGN STYLE"]
        .dropna()
        .astype(str)
        .map(lambda s: s.strip())
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )

    themes = (
        theme_df["THEMES"]
        .dropna()
        .astype(str)
        .map(lambda s: s.strip())
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )

    # Optional: stable sorting for UI consistency
    colors.sort()
    designs.sort()
    themes.sort()

    return colors, designs, themes
