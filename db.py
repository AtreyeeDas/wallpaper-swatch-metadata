import json
from typing import Any, Dict, List
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

def get_engine(database_url: str) -> Engine:
    """
    database_url example:
      postgresql+psycopg2://user:pass@host:5432/dbname?sslmode=require
    """
    return create_engine(database_url, pool_pre_ping=True)

def upsert_swatch(engine: Engine, row: Dict[str, Any]) -> None:
    """
    Upsert (overwrite on swatch_id conflict).
    secondary_colors is stored as JSON string in DB.
    """
    secondary_json = json.dumps(row["secondary_colors"], ensure_ascii=False)

    sql = text("""
    insert into swatch_metadata
      (swatch_id, primary_color, secondary_colors, design_style, theme,
       suitable_for, description, image_filename, updated_at)
    values
      (:swatch_id, :primary_color, :secondary_colors, :design_style, :theme,
       :suitable_for, :description, :image_filename, now())
    on conflict (swatch_id) do update set
      primary_color = excluded.primary_color,
      secondary_colors = excluded.secondary_colors,
      design_style = excluded.design_style,
      theme = excluded.theme,
      suitable_for = excluded.suitable_for,
      description = excluded.description,
      image_filename = excluded.image_filename,
      updated_at = now();
    """)

    with engine.begin() as conn:
        conn.execute(sql, {
            "swatch_id": row["swatch_id"],
            "primary_color": row["primary_color"],
            "secondary_colors": secondary_json,
            "design_style": row["design_style"],
            "theme": row["theme"],
            "suitable_for": row.get("suitable_for"),
            "description": row.get("description"),
            "image_filename": row.get("image_filename"),
        })

def fetch_all(engine: Engine) -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(
            text("select * from swatch_metadata order by updated_at desc")
        ).mappings().all()
    return [dict(r) for r in rows]
