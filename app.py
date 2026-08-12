import base64
import html
from io import BytesIO
import math
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "conxo.xlsx"
ASSETS_DIR = BASE_DIR / "assets"
TEAMS_ASSETS_DIR = ASSETS_DIR / "Equipos"
FONTS_DIR = ASSETS_DIR / "fonts"

TEAM_NAME = "Conxo Santiago B"
COMPETITION_NAME = "JUVENIL - PREFERENTE FUTGAL (GRUPO 2)"
TIME_BANDS = [
    (0, 15, "0-15"),
    (15, 30, "15-30"),
    (30, 45, "30-45"),
    (45, 60, "45-60"),
    (60, 75, "60-75"),
    (75, 90, "75-90"),
]
EVENT_ORDER = [
    "Apertura de marcador",
    "Igualar marcador",
    "Ponerse por delante",
    "Reducir distancia",
    "Ampliar Ventaja",
    "Victoria",
]

PLOT_CARD_BG = "#f8fbfd"
PLOT_PAPER_BG = "rgba(248,251,253,0)"
PLOT_GRID = "#dfe9f1"
PLOT_AXIS = "#90a2b5"
PLOT_LINE = "#10364d"

OFFENSE_SCALE = [
    [0.0, "#eaf6ff"],
    [0.2, "#bcdff5"],
    [0.4, "#7cc4e8"],
    [0.6, "#3aa5db"],
    [0.8, "#1477b8"],
    [1.0, "#0b4f7a"],
]
DEFENSE_SCALE = [
    [0.0, "#fff3e8"],
    [0.2, "#ffd2a8"],
    [0.4, "#ffae70"],
    [0.6, "#f77b45"],
    [0.8, "#d4572a"],
    [1.0, "#9f3413"],
]
DIFF_SCALE = [
    [0.0, "#b54848"],
    [0.25, "#e58d7c"],
    [0.5, "#f3f3f3"],
    [0.75, "#74b9d6"],
    [1.0, "#1f6f8b"],
]
NAV_OPTIONS = ["General", "Equipo", "Plantilla"]
NAV_LABELS = {
    "General": "🏠 General",
    "Equipo": "📈 Equipo",
    "Plantilla": "🧩 Plantilla",
}


def _find_time_band(minute: float) -> str:
    if pd.isna(minute):
        return "Sin minuto"
    minute_value = float(minute)
    if minute_value > 90:
        minute_value = 90
    for start, end, label in TIME_BANDS:
        if start <= minute_value < end:
            return label
    return "75-90"


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _normalize_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return "".join(char.lower() for char in text if char.isalnum())


def _resolve_asset_path(*preferred_parts: str, fallback_contains: list[str] | None = None) -> Path | None:
    preferred = ASSETS_DIR.joinpath(*preferred_parts)
    if preferred.exists():
        return preferred
    if fallback_contains:
        normalized_needles = [_normalize_key(part) for part in fallback_contains]
        for candidate in ASSETS_DIR.rglob("*"):
            if not candidate.is_file():
                continue
            normalized_candidate = _normalize_key(candidate.as_posix())
            if all(needle in normalized_candidate for needle in normalized_needles):
                return candidate
    return None


LOGO_PATH = _resolve_asset_path(
    "MCode Analytics",
    "MCODE Sport Analytics.png",
    fallback_contains=["mcode", "sport", "analytics"],
)
CREST_PATH = _resolve_asset_path(
    "escudo",
    "conxo_hd.png",
    fallback_contains=["escudo", "conxo", "hd"],
)
FEDERATION_LOGO_PATH = _resolve_asset_path(
    "Federación",
    "real-federacion-galega-de-futbol-logo-png_seeklogo-486963-1021954568.png",
    fallback_contains=["federacion", "galega", "futbol", "logo"],
)


@st.cache_data(show_spinner=False)
def build_team_crest_map():
    crest_map = {}
    if not TEAMS_ASSETS_DIR.exists():
        return crest_map
    files = [p for p in TEAMS_ASSETS_DIR.iterdir() if p.is_file()]
    files.sort(key=lambda p: (p.suffix.lower() != ".png", p.name.lower()))
    for file_path in files:
        key = _normalize_key(file_path.stem)
        crest_map.setdefault(key, file_path)
    return crest_map


def _path_to_data_uri(path: Path | None) -> str | None:
    if path is None or not Path(path).exists():
        return None
    suffix = Path(path).suffix.lower().replace(".", "") or "png"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/{suffix};base64,{encoded}"


def _load_pdf_font(size: int, bold: bool = False):
    font_candidates = []
    if bold:
        font_candidates.extend(
            [
                FONTS_DIR / "DejaVuSans-Bold.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            ]
        )
    else:
        font_candidates.extend(
            [
                FONTS_DIR / "DejaVuSans.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ]
        )
    for candidate in font_candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            try:
                return ImageFont.truetype(str(candidate_path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_vertical_gradient(image: Image.Image, top_color: tuple[int, int, int], bottom_color: tuple[int, int, int]):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for y in range(height):
        ratio = y / max(height - 1, 1)
        color = tuple(
            int(top_color[i] * (1 - ratio) + bottom_color[i] * ratio)
            for i in range(3)
        )
        draw.line([(0, y), (width, y)], fill=color)


def _wrap_text_for_width(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_lines(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str], font, fill, line_gap: int = 6):
    x, y = xy
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_height = bbox[3] - bbox[1]
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height + line_gap
    return y


def _pct_color_hex(value: float) -> str:
    if value >= 95:
        return "#10364d"
    if value >= 85:
        return "#1b5977"
    if value >= 70:
        return "#2f7f9f"
    if value >= 50:
        return "#d8b24d"
    return "#d95f59"


def _draw_metric_card_pdf(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    value: str,
    accent: str = "#d8b24d",
    value_fill: str = "#10364d",
):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=26, fill="#ffffff", outline="#d9e7ef", width=2)
    draw.rounded_rectangle((x1, y1, x2, y1 + 8), radius=26, fill=accent)
    card_width = x2 - x1
    max_label_width = card_width - 26
    label_font = _load_pdf_font(24, bold=True)
    label_lines = _wrap_text_for_width(draw, label, label_font, max_label_width)
    for font_size in [22, 20, 18, 16]:
        if len(label_lines) <= 2:
            break
        label_font = _load_pdf_font(font_size, bold=True)
        label_lines = _wrap_text_for_width(draw, label, label_font, max_label_width)

    label_bbox = draw.textbbox((0, 0), "Ag", font=label_font)
    label_line_height = label_bbox[3] - label_bbox[1]
    label_gap = 2
    label_block_height = len(label_lines) * label_line_height + max(0, len(label_lines) - 1) * label_gap
    label_y = y1 + 18
    current_y = label_y
    for line in label_lines:
        line_bbox = draw.textbbox((0, 0), line, font=label_font)
        draw.text(
            (x1 + (card_width - (line_bbox[2] - line_bbox[0])) / 2, current_y),
            line,
            font=label_font,
            fill="#6b7c8f",
        )
        current_y += label_line_height + label_gap

    value_font = _load_pdf_font(40, bold=True)
    value_bbox = draw.textbbox((0, 0), value, font=value_font)
    max_value_width = card_width - 20
    for font_size in [38, 36, 34, 32, 30]:
        if value_bbox[2] - value_bbox[0] <= max_value_width:
            break
        value_font = _load_pdf_font(font_size, bold=True)
        value_bbox = draw.textbbox((0, 0), value, font=value_font)

    value_y = max(y1 + 56, label_y + label_block_height + 8)
    draw.text(
        (x1 + (card_width - (value_bbox[2] - value_bbox[0])) / 2, value_y),
        value,
        font=value_font,
        fill=value_fill,
    )


def _draw_radar_pdf(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    labels: list[str],
    values: list[float],
):
    cx, cy = center
    grid_color = "#dce8f1"
    secondary_grid_color = "#eef4f8"
    axis_color = "#d3e0e9"
    label_font = _load_pdf_font(10, bold=True)
    value_font = _load_pdf_font(12, bold=True)
    levels = [20, 40, 60, 80, 100]
    points: list[tuple[float, float]] = []
    total = len(labels)
    start_angle = -math.pi / 2

    # Soft halo so the radar feels lifted without drawing a heavy ring.
    draw.ellipse(
        (cx - radius - 18, cy - radius - 18, cx + radius + 18, cy + radius + 18),
        outline="#edf4f8",
        width=14,
    )
    draw.ellipse(
        (cx - radius - 6, cy - radius - 6, cx + radius + 6, cy + radius + 6),
        outline="#f6fafc",
        width=10,
    )
    draw.ellipse(
        (cx - radius - 2, cy - radius - 2, cx + radius + 2, cy + radius + 2),
        outline="#d9e9f2",
        width=1,
    )

    for level in range(10, 101, 10):
        level_radius = radius * (level / 100.0)
        ellipse_box = (
            cx - level_radius,
            cy - level_radius,
            cx + level_radius,
            cy + level_radius,
        )
        draw.ellipse(
            ellipse_box,
            outline=grid_color if level in levels else secondary_grid_color,
            width=1,
        )
        if level in levels[:-1]:
            tick_text = str(level)
            tick_bbox = draw.textbbox((0, 0), tick_text, font=value_font)
            draw.text(
                (cx + 12, cy - level_radius - (tick_bbox[3] - tick_bbox[1]) / 2),
                tick_text,
                font=value_font,
                fill="#88a1b4",
            )

    for idx, (label, value) in enumerate(zip(labels, values)):
        angle = start_angle + (2 * math.pi * idx / total)
        end_x = cx + math.cos(angle) * radius
        end_y = cy + math.sin(angle) * radius
        draw.line((cx, cy, end_x, end_y), fill=axis_color, width=1)

        value_radius = radius * (value / 100.0)
        point = (
            cx + math.cos(angle) * value_radius,
            cy + math.sin(angle) * value_radius,
        )
        points.append(point)

        label_distance = radius + 30
        label_x = cx + math.cos(angle) * label_distance
        label_y = cy + math.sin(angle) * label_distance
        value_text = f"{value:.1f}%"
        label_text = str(label).replace("% ", "").replace(" / ", "/")
        combined_label = f"{label_text} {value_text}"
        label_bbox = draw.textbbox((0, 0), combined_label, font=label_font)
        text_width = label_bbox[2] - label_bbox[0]
        text_height = label_bbox[3] - label_bbox[1]
        if abs(math.cos(angle)) < 0.25:
            text_pos = (
                label_x - text_width / 2,
                label_y - text_height / 2 + (-16 if math.sin(angle) < 0 else 16),
            )
        elif math.cos(angle) > 0:
            text_pos = (
                label_x + 34,
                label_y - text_height / 2,
            )
        else:
            text_pos = (
                label_x - text_width - 34,
                label_y - text_height / 2,
            )
        pad_x = 7
        pad_y = 4
        draw.rounded_rectangle(
            (
                text_pos[0] - pad_x,
                text_pos[1] - pad_y,
                text_pos[0] + text_width + pad_x,
                text_pos[1] + text_height + pad_y,
            ),
            radius=8,
            fill="#f8fbfd",
        )
        draw.text(text_pos, combined_label, font=label_font, fill=_pct_color_hex(value))

    shadow_points = [(x + 4, y + 5) for x, y in points]
    draw.polygon(shadow_points, fill=(215, 229, 238))
    draw.polygon(points, fill=(78, 129, 164), outline="#245a77", width=2)
    for point, value in zip(points, values):
        color = _pct_color_hex(value)
        draw.ellipse((point[0] - 7, point[1] - 7, point[0] + 7, point[1] + 7), fill=color, outline="#245a77", width=2)


def build_player_report_pdf(
    player_name: str,
    team_name: str,
    season_label: str,
    crest_path: Path | None,
    brand_logo_path: Path | None,
    coach_name: str,
    designer_name: str,
    dorsal_value: str,
    posicion_global: str,
    posicion_especifica: str,
    fecha_nacimiento: str,
    edad_value: str,
    metric_groups: list[list[tuple[str, str]]],
    radar_labels: list[str],
    radar_values: list[float],
    impact_metrics: list[tuple[str, str, str]],
    comments_text: str,
) -> bytes:
    page_size = (1754, 1240)
    cover = Image.new("RGB", page_size, "#f7fbfe")
    _draw_vertical_gradient(cover, (248, 251, 253), (229, 240, 247))
    draw = ImageDraw.Draw(cover)

    title_font = _load_pdf_font(88, bold=True)
    subtitle_font = _load_pdf_font(42, bold=True)
    body_font = _load_pdf_font(34, bold=False)
    small_font = _load_pdf_font(28, bold=False)

    draw.ellipse((80, 120, 540, 580), fill="#d8b24d")
    draw.ellipse((330, 70, 930, 670), fill="#1b5977")
    draw.rounded_rectangle((90, 120, 1280, 980), radius=48, fill="#10364d")
    draw.rounded_rectangle((90, 760, 1280, 980), radius=48, fill="#1b5977")

    draw.text((150, 180), "INFORME DE JUGADOR", font=subtitle_font, fill="#d8b24d")
    name_lines = _wrap_text_for_width(draw, player_name.upper(), title_font, 990)
    y_cursor = 290
    for line in name_lines:
        draw.text((150, y_cursor), line, font=title_font, fill="#ffffff")
        y_cursor += 96

    draw.text((150, 680), team_name, font=subtitle_font, fill="#f8fbfd")
    draw.text((150, 760), f"Temporada {season_label}", font=subtitle_font, fill="#d8b24d")

    if crest_path and crest_path.exists():
        crest = Image.open(crest_path).convert("RGBA")
        crest.thumbnail((220, 220))
        badge = Image.new("RGBA", (280, 280), (0, 0, 0, 0))
        badge_draw = ImageDraw.Draw(badge)
        badge_draw.ellipse((0, 0, 280, 280), fill=(255, 255, 255, 240), outline=(217, 231, 239, 255), width=6)
        badge_draw.ellipse((18, 18, 262, 262), fill=(248, 251, 253, 255))
        badge.alpha_composite(crest, ((280 - crest.width) // 2, (280 - crest.height) // 2))
        cover.paste(badge, (1370, 110), badge)

    footer_x = 1120
    draw.text((footer_x, 1000), f"Entrenador: {coach_name}", font=body_font, fill="#10364d")
    draw.text((footer_x, 1050), f"Informe diseñado por: {designer_name}", font=small_font, fill="#10364d")
    if brand_logo_path and brand_logo_path.exists():
        brand_logo = Image.open(brand_logo_path).convert("RGBA")
        brand_logo.thumbnail((58, 58))
        cover.paste(brand_logo, (1620, 1112), brand_logo)

    detail = Image.new("RGB", page_size, "#f8fbfd")
    _draw_vertical_gradient(detail, (248, 251, 253), (236, 245, 250))
    ddraw = ImageDraw.Draw(detail)
    section_title = _load_pdf_font(52, bold=True)
    h1 = _load_pdf_font(38, bold=True)
    h2 = _load_pdf_font(26, bold=True)
    value_font_big = _load_pdf_font(38, bold=True)
    body_small = _load_pdf_font(24, bold=False)

    ddraw.rounded_rectangle((54, 42, 1700, 170), radius=36, fill="#10364d")
    ddraw.text((90, 72), player_name.upper(), font=section_title, fill="#ffffff")
    ddraw.text((90, 126), f"Dorsal {dorsal_value} · {team_name} · Temporada {season_label}", font=body_small, fill="#d8e7ef")
    if crest_path and crest_path.exists():
        crest = Image.open(crest_path).convert("RGBA")
        crest.thumbnail((110, 110))
        detail.paste(crest, (1540, 52), crest)
    if brand_logo_path and brand_logo_path.exists():
        brand_logo = Image.open(brand_logo_path).convert("RGBA")
        brand_logo.thumbnail((110, 110))
        detail.paste(brand_logo, (1410, 52), brand_logo)

    footer_small_font = _load_pdf_font(24, bold=False)
    top_meta_y = 184
    designer_text = f"Informe diseñado por: {designer_name}"
    coach_text = f"Entrenador: {coach_name}"
    detail_width = 1700
    ddraw.text((86, top_meta_y), designer_text, font=footer_small_font, fill="#6b7c8f")
    coach_bbox = ddraw.textbbox((0, 0), coach_text, font=footer_small_font)
    ddraw.text((detail_width - (coach_bbox[2] - coach_bbox[0]) - 24, top_meta_y), coach_text, font=footer_small_font, fill="#6b7c8f")

    ddraw.rounded_rectangle((54, 210, 630, 650), radius=28, fill="#ffffff", outline="#d9e7ef", width=2)
    ddraw.text((86, 244), "Ficha del jugador", font=h1, fill="#10364d")
    bio_y = 314
    bio_step = 118
    bio_pairs = [
        ("Posición general", posicion_global),
        ("Posición específica", posicion_especifica),
    ]
    for label, value in bio_pairs:
        ddraw.text((86, bio_y), label, font=h2, fill="#6b7c8f")
        ddraw.text((86, bio_y + 38), value, font=value_font_big, fill="#10364d")
        bio_y += bio_step

    left_col_x = 86
    right_col_x = 334
    ddraw.text((left_col_x, bio_y), "Año de nacimiento", font=h2, fill="#6b7c8f")
    ddraw.text((left_col_x, bio_y + 38), fecha_nacimiento, font=value_font_big, fill="#10364d")
    ddraw.text((right_col_x, bio_y), "Edad", font=h2, fill="#6b7c8f")
    ddraw.text((right_col_x, bio_y + 38), edad_value, font=value_font_big, fill="#10364d")

    ddraw.rounded_rectangle((670, 210, 1700, 650), radius=28, fill="#ffffff", outline="#d9e7ef", width=2)
    ddraw.text((710, 244), "Radar de disponibilidad y participación", font=h1, fill="#10364d")
    _draw_radar_pdf(ddraw, (1185, 482), 120, radar_labels, radar_values)

    stat_cards = metric_groups[0] + metric_groups[1] + metric_groups[2]
    card_cols = 7
    card_w = 220
    card_h = 112
    start_x = 54
    start_y = 690
    gap_x = 14
    gap_y = 14
    for idx, (label, value) in enumerate(stat_cards):
        row = idx // card_cols
        col = idx % card_cols
        x1 = start_x + col * (card_w + gap_x)
        y1 = start_y + row * (card_h + gap_y)
        _draw_metric_card_pdf(ddraw, (x1, y1, x1 + card_w, y1 + card_h), label, value)

    if impact_metrics:
        impact_top = 948
        impact_width = 530
        impact_gap = 22
        impact_x_positions = [54, 54 + impact_width + impact_gap, 54 + 2 * (impact_width + impact_gap)]
        for idx, (label, value, color) in enumerate(impact_metrics):
            _draw_metric_card_pdf(
                ddraw,
                (impact_x_positions[idx], impact_top, impact_x_positions[idx] + impact_width, impact_top + 130),
                label,
                value,
                accent=color,
                value_fill=color,
            )
        comments_top = 1086
    else:
        comments_top = 946
    comments_bottom = 1210
    ddraw.rounded_rectangle((54, comments_top, 1700, comments_bottom), radius=24, fill="#ffffff", outline="#d9e7ef", width=2)
    ddraw.text((86, comments_top + 26), "Comentarios del entrenador", font=h1, fill="#10364d")
    comments_font = _load_pdf_font(24, bold=False)
    comments = comments_text.strip() if str(comments_text).strip() else "Sin comentarios registrados."
    comment_lines = _wrap_text_for_width(ddraw, comments, comments_font, 1540)
    line_bbox = ddraw.textbbox((0, 0), "Ag", font=comments_font)
    line_height = (line_bbox[3] - line_bbox[1]) + 8
    text_start_y = comments_top + 74
    max_lines_second_page = max(1, int((comments_bottom - text_start_y - 20) / line_height))
    second_page_lines = comment_lines[:max_lines_second_page]
    remaining_lines = comment_lines[max_lines_second_page:]
    _draw_lines(ddraw, (86, text_start_y), second_page_lines, comments_font, "#10364d", line_gap=8)

    pdf_pages = [cover.convert("RGB"), detail.convert("RGB")]
    if remaining_lines:
        extra_page = Image.new("RGB", page_size, "#f8fbfd")
        _draw_vertical_gradient(extra_page, (248, 251, 253), (236, 245, 250))
        extra_draw = ImageDraw.Draw(extra_page)
        extra_draw.rounded_rectangle((54, 54, 1700, 1186), radius=28, fill="#ffffff", outline="#d9e7ef", width=2)
        extra_draw.text((86, 90), "Comentarios del entrenador (continuación)", font=h1, fill="#10364d")
        _draw_lines(extra_draw, (86, 156), remaining_lines, comments_font, "#10364d", line_gap=8)
        pdf_pages.append(extra_page.convert("RGB"))

    output = BytesIO()
    pdf_pages[0].save(
        output,
        format="PDF",
        save_all=True,
        append_images=pdf_pages[1:],
        resolution=150.0,
    )
    output.seek(0)
    return output.getvalue()


def _display_player_name(name: str) -> str:
    raw_name = str(name).strip()
    if not raw_name:
        return "-"
    if "," in raw_name:
        last_name, first_name = [part.strip() for part in raw_name.split(",", 1)]
        ordered = f"{first_name} {last_name}".strip()
    else:
        ordered = raw_name
    return ordered.title()


def _safe_profile_value(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "Pendiente"
    return str(value).strip()


def _first_existing_column(df: pd.DataFrame, candidates: list[str], default=np.nan) -> pd.Series:
    for candidate in candidates:
        if candidate in df.columns:
            return df[candidate]
    return pd.Series([default] * len(df), index=df.index)


def _sync_streamlit_text_snapshot(source_key: str, target_key: str):
    st.session_state[target_key] = st.session_state.get(source_key, "")


def _minutes_cell_style(value: int, *, is_total: bool = False, total_max: int = 0) -> str:
    value = int(value)
    if is_total:
        if total_max <= 0:
            return "background:#f2f6f9;color:#10364d;"
        ratio = value / total_max
        if ratio >= 0.72:
            return "background:#dff1df;color:#1e6b39;"
        if ratio >= 0.45:
            return "background:#f4ebc7;color:#8b6610;"
        return "background:#eef3f7;color:#10364d;"
    if value <= 0:
        return "background:#f8e0e0;color:#a33e3e;"
    if value < 40:
        return "background:#f8dfcf;color:#b55d24;"
    if value < 60:
        return "background:#f5edc8;color:#8b6610;"
    return "background:#dff1df;color:#1e6b39;"


def _change_cell_style(value: int, *, max_value: int = 0, summary: bool = False) -> str:
    value = int(value)
    if value <= 0:
        return "background:#f7fafc;color:#90a2b5;"
    if summary:
        ratio = value / max(max_value, 1)
        if ratio >= 0.72:
            return "background:#10364d;color:#ffffff;"
        if ratio >= 0.42:
            return "background:#d8b24d;color:#10364d;"
        return "background:#edf4f9;color:#10364d;"
    ratio = value / max(max_value, 1)
    if ratio >= 0.66:
        return "background:#10364d;color:#ffffff;"
    if ratio >= 0.33:
        return "background:#8cc4d7;color:#10364d;"
    return "background:#edf4f9;color:#10364d;"


def build_minutes_scroll_table(minutes_matrix: pd.DataFrame, total_minutes: pd.Series) -> str:
    total_max = int(total_minutes.max()) if len(total_minutes) else 0
    headers = "".join(
        f'<th class="scroll-table-head scroll-table-col-head">{html.escape(str(col))}</th>'
        for col in minutes_matrix.columns.tolist()
    )
    rows = []
    for row_label, row_values in minutes_matrix.iterrows():
        total_value = int(total_minutes.loc[row_label])
        cells = [
            f'<td class="scroll-table-cell scroll-table-sticky-player">{html.escape(str(row_label))}</td>',
            (
                f'<td class="scroll-table-cell scroll-table-sticky-total scroll-table-number" '
                f'style="{_minutes_cell_style(total_value, is_total=True, total_max=total_max)}">{total_value}</td>'
            ),
        ]
        for value in row_values.tolist():
            style = _minutes_cell_style(int(value))
            cells.append(
                f'<td class="scroll-table-cell scroll-table-number" style="{style}">{int(value)}</td>'
            )
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<div class="scroll-table-shell">'
        '<div class="scroll-table-kicker">Minutos disputados por jornada</div>'
        '<div class="scroll-table-wrap">'
        '<table class="scroll-table scroll-table--minutes">'
        '<thead><tr>'
        '<th class="scroll-table-head scroll-table-sticky-player">Jugador</th>'
        '<th class="scroll-table-head scroll-table-sticky-total">Total</th>'
        f"{headers}"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div></div>"
    )


def build_change_scroll_table(summary_df: pd.DataFrame, change_matrix: pd.DataFrame) -> str:
    summary_max = int(summary_df.values.max()) if summary_df.size else 0
    matrix_max = int(change_matrix.values.max()) if change_matrix.size else 0
    matrix_headers = []
    for player_name in change_matrix.columns.tolist():
        matrix_headers.append(
            '<th class="scroll-table-head scroll-table-col-head scroll-table-player-head">'
            f'<span>{html.escape(str(player_name))}</span></th>'
        )

    rows = []
    for row_label in summary_df.index.tolist():
        cells = [
            f'<td class="scroll-table-cell scroll-table-sticky-player"><div class="scroll-table-player-name">{html.escape(str(row_label))}</div></td>'
        ]
        summary_values = [int(value) for value in summary_df.loc[row_label].tolist()]
        sticky_classes = [
            "scroll-table-sticky-summary-1",
            "scroll-table-sticky-summary-2",
            "scroll-table-sticky-summary-3",
        ]
        for sticky_class, value in zip(sticky_classes, summary_values):
            style = _change_cell_style(int(value), max_value=summary_max, summary=True)
            cells.append(
                f'<td class="scroll-table-cell scroll-table-number {sticky_class}" style="{style}">{int(value)}</td>'
            )
        matrix_values = change_matrix.loc[row_label].tolist()
        for value in matrix_values:
            style = _change_cell_style(int(value), max_value=matrix_max, summary=False)
            text = str(int(value)) if int(value) > 0 else ""
            cells.append(
                f'<td class="scroll-table-cell scroll-table-number" style="{style}">{text}</td>'
            )
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        '<div class="scroll-table-shell">'
        '<div class="scroll-table-kicker">Matriz de cambios</div>'
        '<div class="scroll-table-wrap">'
        '<table class="scroll-table scroll-table--changes">'
        '<thead><tr>'
        '<th class="scroll-table-head scroll-table-sticky-player">Jugador</th>'
        '<th class="scroll-table-head scroll-table-sticky-summary-1 scroll-table-summary-head"><span>Total</span></th>'
        '<th class="scroll-table-head scroll-table-sticky-summary-2 scroll-table-summary-head"><span>Titular completo</span></th>'
        '<th class="scroll-table-head scroll-table-sticky-summary-3 scroll-table-summary-head"><span>Entró de suplente</span></th>'
        f"{''.join(matrix_headers)}"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div></div>"
    )


def build_player_percentage_svg(items: list[tuple[str, float, str]]) -> str:
    width = 880
    bar_width = 470
    start_x = 250
    top_y = 26
    row_gap = 52
    height = top_y + row_gap * len(items) + 12
    rows = []
    for idx, (label, value, color) in enumerate(items):
        clamped = max(0.0, min(float(value), 100.0))
        y = top_y + idx * row_gap
        rows.append(
            f"""
            <text x="22" y="{y + 17}" font-size="15" font-weight="700" fill="#10364d">{label}</text>
            <rect x="{start_x}" y="{y}" width="{bar_width}" height="18" rx="9" fill="#eaf1f6"/>
            <rect x="{start_x}" y="{y}" width="{bar_width * clamped / 100:.1f}" height="18" rx="9" fill="{color}"/>
            <text x="{start_x + bar_width + 18}" y="{y + 14}" font-size="15" font-weight="800" fill="#10364d">{clamped:.1f}%</text>
            """
        )
    return (
        '<div class="player-svg-card" style="margin:0;padding:0;background:transparent;">'
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        'preserveAspectRatio="xMinYMin meet" style="width:100%;height:auto;display:block;background:transparent;">'
        f'{"".join(rows)}'
        "</svg></div>"
    )


@st.cache_data(show_spinner=False)
def load_data():
    general = pd.read_excel(DATA_PATH, sheet_name="General")
    goals = pd.read_excel(DATA_PATH, sheet_name="Goles Favor Contra")
    yellows = pd.read_excel(DATA_PATH, sheet_name="Tarjetas Amarillas")
    reds = pd.read_excel(DATA_PATH, sheet_name="Tarjetas Rojas")
    squad = pd.read_excel(DATA_PATH, sheet_name="Plantilla")
    teams = pd.read_excel(DATA_PATH, sheet_name="Equipos")

    general = general.loc[:, ~general.columns.astype(str).str.startswith("Unnamed")]
    goals = goals.loc[:, ~goals.columns.astype(str).str.startswith("Unnamed")]
    yellows = yellows.loc[:, ~yellows.columns.astype(str).str.startswith("Unnamed")]
    teams = teams.loc[:, ~teams.columns.astype(str).str.startswith("Columna")]

    general["Fecha"] = pd.to_datetime(general["Fecha"], errors="coerce")
    general["Jornada"] = pd.to_numeric(general["Jornada"], errors="coerce")
    goals["Jornada"] = pd.to_numeric(goals["Jornada"], errors="coerce")
    yellows["Jornada"] = pd.to_numeric(yellows["Jornada"], errors="coerce")
    reds["Jornada"] = pd.to_numeric(reds["Jornada"], errors="coerce")
    teams["Jornada"] = pd.to_numeric(teams["Jornada"], errors="coerce")

    numeric_general_cols = [
        "Goles a Favor_Eq_local",
        "Gol a Favor_Eq_Visitante",
        "Dorsal",
        "Convocado",
        "Partido Jugado",
        "Titular",
        "Minutos Jugados",
        "Minutos Sustituto",
        "Goles",
        "Goles a Favor_Jug",
        "Goles en Contra_Jug",
        "Resultado Jug_Eq",
        "Puntos x minutos",
        "Puntos Jug_Campo",
    ]
    for col in numeric_general_cols:
        if col in general.columns:
            general[col] = _safe_numeric(general[col])

    numeric_goal_cols = ["Gol", "Penalti", "Propia Puerta", "Minuto"]
    for col in numeric_goal_cols:
        if col in goals.columns:
            goals[col] = _safe_numeric(goals[col])

    for col in ["Tarjeta Amarilla", "Minuto 1a Amarilla", "2a Tarjeta Amarilla", "Minuto 2a Amarilla"]:
        if col in yellows.columns:
            yellows[col] = _safe_numeric(yellows[col])
    for col in ["Tarjeta Roja", "Minuto Tarjeta Roja"]:
        if col in reds.columns:
            reds[col] = _safe_numeric(reds[col])

    numeric_team_cols = [
        "Posicion",
        "Puntos",
        "Partidos Jugados",
        "P Ganados",
        "P Empatados",
        "P Perdidos",
        "Goles Fav",
        "Goles Contr",
        "Dif Goles",
    ]
    for col in numeric_team_cols:
        if col in teams.columns:
            teams[col] = _safe_numeric(teams[col])

    team_name = (
        general["Equipo_Jug"].dropna().astype(str).str.strip().replace("", np.nan).dropna().iloc[0]
        if "Equipo_Jug" in general.columns and not general["Equipo_Jug"].dropna().empty
        else TEAM_NAME
    )

    match_cols = [
        "Jornada",
        "Fecha",
        "Eq.Local",
        "Eq.Visitante",
        "Partido",
        "Estadio",
        "Local/Visitante",
        "Resultado",
        "Goles a Favor_Eq_local",
        "Gol a Favor_Eq_Visitante",
        "Marcador",
    ]
    matches = (
        general[match_cols]
        .dropna(subset=["Jornada"])
        .drop_duplicates(subset=["Jornada"])
        .sort_values("Jornada")
        .reset_index(drop=True)
    )
    matches["Conxo Es Local"] = (
        matches["Local/Visitante"].astype(str).str.strip().str.lower().eq("local")
    )
    matches["GF"] = np.where(
        matches["Conxo Es Local"],
        matches["Goles a Favor_Eq_local"],
        matches["Gol a Favor_Eq_Visitante"],
    ).astype(int)
    matches["GC"] = np.where(
        matches["Conxo Es Local"],
        matches["Gol a Favor_Eq_Visitante"],
        matches["Goles a Favor_Eq_local"],
    ).astype(int)
    # Jornada a jornada: diferencia de goles del Conxo en ese partido.
    matches["Dif Goles Partido"] = (matches["GF"] - matches["GC"]).astype(int)
    matches["Dif Goles Acumulada"] = matches["Dif Goles Partido"].cumsum()
    matches["Rival"] = np.where(
        matches["Conxo Es Local"], matches["Eq.Visitante"], matches["Eq.Local"]
    )

    team_table = (
        teams[teams["Equipo"].astype(str).str.strip().eq(team_name)]
        .sort_values("Jornada")
        .reset_index(drop=True)
    )

    current_round = int(matches["Jornada"].max())
    current_match = matches.loc[matches["Jornada"].eq(current_round)].iloc[0]
    current_team_row = team_table.loc[team_table["Jornada"].eq(current_round)].iloc[0]

    general_players = general[general["Nombre"].notna()].copy()
    player_summary = (
        general_players.groupby("Nombre", as_index=False)
        .agg(
            Convocados=("Convocado", "sum"),
            Titulares=("Titular", "sum"),
            Jugados=("Partido Jugado", "sum"),
            Minutos=("Minutos Jugados", "sum"),
            Goles=("Goles", "sum"),
        )
        .sort_values(["Minutos", "Titulares", "Nombre"], ascending=[False, False, True])
    )
    squad_clean = pd.DataFrame(
        {
            "Nombre": _first_existing_column(squad, ["Nombre"], ""),
            "Dorsal": _first_existing_column(squad, ["Dorsal", "ID"], np.nan),
            "Alias": _first_existing_column(squad, ["Alias"], np.nan),
            "Posicion Especifica": _first_existing_column(squad, ["Posicion Especifica"], np.nan),
            "Posicion Global": _first_existing_column(squad, ["Posicion Global"], np.nan),
            "Fecha Nacimiento": _first_existing_column(
                squad, ["Fecha Nacimiento", "Fecha de Nacimiento"], np.nan
            ),
            "Años": _first_existing_column(squad, ["Años", "Anos", "Ano", "Año"], np.nan),
        }
    )
    squad_clean["Nombre"] = squad_clean["Nombre"].astype(str).str.strip()
    player_summary["Nombre"] = player_summary["Nombre"].astype(str).str.strip()
    squad_clean = (
        squad_clean[squad_clean["Nombre"].ne("")]
        .drop_duplicates(subset=["Nombre"], keep="first")
        .reset_index(drop=True)
    )
    current_year = pd.Timestamp.now().year
    birth_year_numeric = pd.to_numeric(squad_clean["Años"], errors="coerce")
    squad_clean["Edad"] = np.where(
        birth_year_numeric.notna(),
        current_year - birth_year_numeric,
        np.nan,
    )
    player_summary = squad_clean.merge(player_summary, on="Nombre", how="left")
    for metric in ["Convocados", "Titulares", "Jugados", "Minutos", "Goles"]:
        player_summary[metric] = player_summary[metric].fillna(0).astype(int)
    if "Nombre" in yellows.columns and "Tarjeta Amarilla" in yellows.columns:
        yellow_summary = (
            yellows.assign(Nombre=yellows["Nombre"].astype(str).str.strip())
            .groupby("Nombre", as_index=False)["Tarjeta Amarilla"]
            .sum()
            .rename(columns={"Tarjeta Amarilla": "Amarillas"})
        )
        player_summary = player_summary.merge(yellow_summary, on="Nombre", how="left")
    else:
        player_summary["Amarillas"] = 0
    if "Nombre" in reds.columns and "Tarjeta Roja" in reds.columns:
        red_summary = (
            reds.assign(Nombre=reds["Nombre"].astype(str).str.strip())
            .groupby("Nombre", as_index=False)["Tarjeta Roja"]
            .sum()
            .rename(columns={"Tarjeta Roja": "Rojas"})
        )
        player_summary = player_summary.merge(red_summary, on="Nombre", how="left")
    else:
        player_summary["Rojas"] = 0
    for metric in ["Amarillas", "Rojas"]:
        player_summary[metric] = player_summary[metric].fillna(0).astype(int)
    player_summary["% Minutos Convocado"] = np.where(
        player_summary["Convocados"] > 0,
        (player_summary["Minutos"] / (player_summary["Convocados"] * 90.0)) * 100,
        0,
    ).round(1)
    player_summary["Dorsal"] = player_summary["Dorsal"].fillna(0).astype(int).replace(0, np.nan)
    player_summary = player_summary.sort_values(["Minutos", "Nombre"], ascending=[False, True]).reset_index(drop=True)

    goals["Es Conxo"] = goals["Equipo Marca"].astype(str).str.strip().eq(team_name)
    goals["Franja"] = goals["Minuto"].apply(_find_time_band)
    goals["Tipo"] = pd.Categorical(goals["Tipo"], categories=EVENT_ORDER, ordered=True)

    grouped = (
        goals.dropna(subset=["Tipo"])
        .groupby(["Es Conxo", "Franja", "Tipo"], as_index=False)["Gol"]
        .sum()
        .rename(columns={"Gol": "Total"})
    )
    offense = grouped[grouped["Es Conxo"]].copy()
    defense = grouped[~grouped["Es Conxo"]].copy()

    return {
        "team_name": team_name,
        "general": general,
        "goals": goals,
        "yellows": yellows,
        "reds": reds,
        "squad": squad,
        "teams": teams,
        "matches": matches,
        "team_table": team_table,
        "current_round": current_round,
        "current_match": current_match,
        "current_team_row": current_team_row,
        "player_summary": player_summary,
        "offense": offense,
        "defense": defense,
    }


def build_bubble_matrix(df: pd.DataFrame, title: str, colorscale, reverse: bool = False):
    chart_type_labels = {
        "Apertura de marcador": "Apertura<br>marcador",
        "Igualar marcador": "Igualar<br>marcador",
        "Ponerse por delante": "Ponerse por<br>delante",
        "Reducir distancia": "Reducir<br>distancia",
        "Ampliar Ventaja": "Ampliar<br>ventaja",
        "Victoria": "Victoria",
    }
    frame = pd.DataFrame(
        [(band[2], event) for band in TIME_BANDS for event in EVENT_ORDER],
        columns=["Franja", "Tipo"],
    ).merge(df[["Franja", "Tipo", "Total"]], on=["Franja", "Tipo"], how="left")
    frame["Total"] = frame["Total"].fillna(0)
    frame["TipoLabel"] = frame["Tipo"].map(chart_type_labels).fillna(frame["Tipo"])
    frame["BubbleText"] = frame["Total"].apply(lambda value: str(int(value)) if int(value) >= 3 else "")

    fig = px.scatter(
        frame,
        x="Franja",
        y="TipoLabel",
        size="Total",
        color="Total",
        size_max=26,
        color_continuous_scale=colorscale,
        text="BubbleText",
    )
    fig.update_traces(
        textposition="middle center",
        textfont=dict(size=9),
        marker=dict(line=dict(color="#17324d", width=0.8)),
        hovertemplate="<b>%{customdata[0]}</b><br>Franja: %{x}<br>Goles: %{marker.color}<extra></extra>",
        customdata=np.stack([frame["Tipo"]], axis=-1),
    )
    fig.update_layout(
        title=title,
        xaxis_title="Franja de tiempo",
        yaxis_title="Tipo de acción",
        template="plotly_white",
        margin=dict(l=20, r=20, t=56, b=26),
        height=600,
        coloraxis_colorbar_title="Goles",
        plot_bgcolor=PLOT_CARD_BG,
        paper_bgcolor=PLOT_PAPER_BG,
        font=dict(color=PLOT_LINE),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=PLOT_GRID,
        zeroline=False,
        linecolor=PLOT_AXIS,
        tickfont=dict(size=11),
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=PLOT_GRID,
        zeroline=False,
        linecolor=PLOT_AXIS,
        tickfont=dict(size=11),
        automargin=True,
    )
    if reverse:
        fig.update_yaxes(autorange="reversed")
    return fig


def build_diff_heatmap(offense: pd.DataFrame, defense: pd.DataFrame):
    base = pd.DataFrame(
        [(band[2], event) for band in TIME_BANDS for event in EVENT_ORDER],
        columns=["Franja", "Tipo"],
    )
    merged = (
        base.merge(
            offense[["Franja", "Tipo", "Total"]].rename(columns={"Total": "Ofensiva"}),
            on=["Franja", "Tipo"],
            how="left",
        )
        .merge(
            defense[["Franja", "Tipo", "Total"]].rename(columns={"Total": "Defensiva"}),
            on=["Franja", "Tipo"],
            how="left",
        )
        .fillna(0)
    )
    merged["Diferencia"] = merged["Ofensiva"] - merged["Defensiva"]
    pivot = merged.pivot(index="Tipo", columns="Franja", values="Diferencia").reindex(EVENT_ORDER)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=DIFF_SCALE,
            zmid=0,
            text=pivot.values.astype(int),
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b><br>Franja: %{x}<br>Diferencia: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Diferencia Ofensiva - Defensiva",
        xaxis_title="Franja de tiempo",
        yaxis_title="Tipo de acción",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=430,
        plot_bgcolor=PLOT_CARD_BG,
        paper_bgcolor=PLOT_PAPER_BG,
        font=dict(color=PLOT_LINE),
    )
    fig.update_xaxes(showgrid=False, linecolor=PLOT_AXIS)
    fig.update_yaxes(showgrid=False, linecolor=PLOT_AXIS)
    return fig


def render_header(team_name: str):
    crest_uri = _path_to_data_uri(CREST_PATH) if CREST_PATH.exists() else None
    header_html = f"""
    <div class="app-hero-shell">
        <div class="app-hero-banner">
            <div class="app-hero-copy">
                <div class="app-hero-kicker">Panel de análisis de rendimiento</div>
                <div class="app-hero-title">Conxo Analytics</div>
                <div class="app-hero-subtitle">Panel local de rendimiento para {team_name}</div>
            </div>
        </div>
        <div class="app-hero-crest-wrap">
            <div class="app-hero-crest-ring"></div>
            <div class="app-hero-crest-core">
                {f'<img src="{crest_uri}" alt="Escudo Conxo">' if crest_uri else ""}
            </div>
            <div class="app-hero-crest-gloss"></div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    logo_uri = _path_to_data_uri(LOGO_PATH) if LOGO_PATH.exists() else None
    federation_uri = (
        _path_to_data_uri(FEDERATION_LOGO_PATH)
        if FEDERATION_LOGO_PATH and FEDERATION_LOGO_PATH.exists()
        else None
    )
    sidebar_branding_html = f"""
    <div class="sidebar-brand-shell">
        <div class="sidebar-logo-row">
            <div class="sidebar-logo-badge">
                {f'<img src="{logo_uri}" alt="MCode Analytics">' if logo_uri else ""}
            </div>
            <div class="sidebar-logo-badge sidebar-logo-badge--crest">
                {f'<img src="{crest_uri}" alt="Conxo Santiago B">' if crest_uri else ""}
            </div>
        </div>
        <div class="sidebar-brand-title">MCode Analytics x Conxo Santiago B</div>
        <div class="sidebar-brand-subtitle">Análisis de rendimiento y seguimiento competitivo</div>
        <div class="sidebar-competition-card">
            <div class="sidebar-competition-logo">
                {f'<img src="{federation_uri}" alt="Real Federación Galega de Fútbol">' if federation_uri else ""}
            </div>
            <div class="sidebar-competition-meta">
                <div class="sidebar-competition-label">Competición oficial</div>
                <div class="sidebar-competition-name">{COMPETITION_NAME}</div>
                <div class="sidebar-competition-org">Real Federación Galega de Fútbol</div>
            </div>
        </div>
    </div>
    """
    with st.sidebar:
        st.html(sidebar_branding_html)


def render_sidebar_navigation():
    current_value = st.session_state.get("section", "General")
    st.sidebar.markdown('<div class="sidebar-nav-title">Secciones</div>', unsafe_allow_html=True)
    selected = st.sidebar.pills(
        "Secciones",
        NAV_OPTIONS,
        default=current_value,
        format_func=lambda option: NAV_LABELS[option],
        label_visibility="collapsed",
        width="stretch",
        key="section_pills",
    )
    st.session_state["section"] = selected or current_value
    st.sidebar.markdown(
        '<div class="sidebar-copyright">🧠 Diseñado por: © Ramón Codesido</div>',
        unsafe_allow_html=True,
    )
    return selected or current_value


def render_stat_cards(items, columns=5, card_class="stat-card"):
    rows = [items[i : i + columns] for i in range(0, len(items), columns)]
    for row in rows:
        cols = st.columns(len(row))
        for col, (label, value) in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div class="stat-card-label">{label}</div>
                        <div class="stat-card-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_page_section_title(title):
    st.markdown(
        f'<div class="page-section-title">{str(title).upper()}</div>',
        unsafe_allow_html=True,
    )


def render_subsection_title(title, icon=""):
    icon_html = f'<span class="page-subsection-icon">{icon}</span>' if icon else ""
    st.markdown(
        f'<div class="page-subsection-title">{icon_html}<span>{title}</span></div>',
        unsafe_allow_html=True,
    )


def render_people_panel(title, icon, players, empty_text, grid_columns=1, variant="starter"):
    render_subsection_title(title, icon)
    if not players:
        st.markdown(f'<div class="people-empty">{empty_text}</div>', unsafe_allow_html=True)
        return
    number_class = "player-chip-number bench" if variant == "bench" else "player-chip-number"
    rows = [players[i : i + grid_columns] for i in range(0, len(players), grid_columns)]
    for row in rows:
        cols = st.columns(len(row))
        for col, player in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class="player-chip">
                        <div class="{number_class}">{player["dorsal"]}</div>
                        <div class="player-chip-body">
                            <div class="player-chip-name">{player["name"]}</div>
                            <div class="player-chip-meta">Minutos jugados: {player["minutes"]}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_goal_panel(title, icon, goals_list, empty_text, variant="conxo"):
    render_subsection_title(title, icon)
    if not goals_list:
        st.markdown(f'<div class="people-empty">{empty_text}</div>', unsafe_allow_html=True)
        return
    accent_class = "goal-chip-minute rival" if variant == "rival" else "goal-chip-minute"
    for goal in goals_list:
        st.markdown(
            f"""
            <div class="goal-chip">
                <div class="{accent_class}">{goal["minute"]}'</div>
                <div class="goal-chip-body">
                    <div class="goal-chip-name">{goal["name"]}</div>
                    <div class="goal-chip-type">{goal["type"]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_event_panel(title, icon, events_list, empty_text, variant="yellow"):
    render_subsection_title(title, icon)
    if not events_list:
        st.markdown(f'<div class="people-empty">{empty_text}</div>', unsafe_allow_html=True)
        return
    badge_class = f"goal-chip-minute {variant}"
    for event in events_list:
        st.markdown(
            f"""
            <div class="goal-chip compact-event-chip">
                <div class="{badge_class}">{event["minute"]}'</div>
                <div class="goal-chip-body">
                    <div class="goal-chip-name">{event["name"]}</div>
                    <div class="goal-chip-type">{event["subtitle"]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_change_panel(changes_list):
    render_subsection_title("Cambios", "🔁")
    if not changes_list:
        st.markdown('<div class="people-empty">Sin cambios registrados en esta jornada</div>', unsafe_allow_html=True)
        return
    for change in changes_list:
        st.markdown(
            f"""
            <div class="change-chip">
                <div class="change-chip-minute">{change["minute"]}'</div>
                <div class="change-chip-body">
                    <div class="change-chip-out">⬅️ Sale: {change["out_number"]} · {change["out_name"]}</div>
                    <div class="change-chip-in">➡️ Entra: {change["in_number"]} · {change["in_name"]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_match_timeline(events, rival_name):
    render_subsection_title("Timeline del partido", "🕒")
    if not events:
        st.markdown('<div class="people-empty">Sin acciones registradas para esta jornada</div>', unsafe_allow_html=True)
        return

    timeline_df = pd.DataFrame(events).copy()
    crest_map = build_team_crest_map()
    conxo_crest_uri = _path_to_data_uri(CREST_PATH) if CREST_PATH and Path(CREST_PATH).exists() else None
    rival_crest_path = crest_map.get(_normalize_key(rival_name))
    rival_crest_uri = _path_to_data_uri(rival_crest_path) if rival_crest_path and Path(rival_crest_path).exists() else None

    action_meta = {
        "goal": {"label": "Gol", "icon": "⚽", "stem": "#2f9e44"},
        "yellow": {"label": "Tarjeta amarilla", "icon": "🟨", "stem": "#d4a72c"},
        "red": {"label": "Tarjeta roja", "icon": "🟥", "stem": "#c55252"},
        "change": {"label": "Cambio", "icon": "🔁", "stem": "#4d7e62"},
    }

    def assign_levels(frame, threshold=6):
        levels = []
        last_minute_by_level = []
        for minute in frame["minute"].tolist():
            placed = False
            for idx, last_minute in enumerate(last_minute_by_level):
                if minute - last_minute >= threshold:
                    levels.append(idx)
                    last_minute_by_level[idx] = minute
                    placed = True
                    break
            if not placed:
                levels.append(len(last_minute_by_level))
                last_minute_by_level.append(minute)
        return levels

    own_df = timeline_df[timeline_df["is_conxo"]].sort_values("minute").copy()
    rival_df = timeline_df[~timeline_df["is_conxo"]].sort_values("minute").copy()
    if not own_df.empty:
        own_df["level"] = assign_levels(own_df)
    if not rival_df.empty:
        rival_df["level"] = assign_levels(rival_df)

    max_minute = max(90, int(timeline_df["minute"].max()))
    width = 1840
    height = 760
    margin_left = 190
    margin_right = 250
    axis_y = 340
    top_base = 235
    bottom_base = 455
    level_step = 82
    usable_width = width - margin_left - margin_right
    team_label_x = margin_left - 58
    crest_x = margin_left - 48
    crest_size = 26

    def x_pos(minute):
        return margin_left + (float(minute) / float(max_minute)) * usable_width

    svg_parts = [
        f'<svg class="timeline-match-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMinYMin meet">',
        '<style>'
        '.axis{stroke:#8ea3b3;stroke-width:3;}'
        '.tick{stroke:#dbe5ed;stroke-width:2;}'
        '.label{fill:#10364d;font:800 24px Arial, sans-serif;}'
        '.sub{fill:#6b7c8f;font:700 18px Arial, sans-serif;}'
        '.team{fill:#5b6f82;font:800 28px Arial, sans-serif;}'
        '.eventTag{fill:#10364d;font:800 21px Arial, sans-serif;}'
        '.eventDot{stroke:#f8fbfd;stroke-width:2.5;}'
        '</style>',
        f'<line x1="{margin_left}" y1="{axis_y}" x2="{width-margin_right}" y2="{axis_y}" class="axis" />',
        f'<text x="{team_label_x}" y="{top_base+10}" text-anchor="end" class="team">Conxo</text>',
        f'<text x="{team_label_x}" y="{bottom_base+10}" text-anchor="end" class="team">{rival_name}</text>',
    ]
    if conxo_crest_uri:
        svg_parts.append(
            f'<image href="{conxo_crest_uri}" x="{crest_x}" y="{top_base-11}" width="{crest_size}" height="{crest_size}" preserveAspectRatio="xMidYMid meet" />'
        )
    if rival_crest_uri:
        svg_parts.append(
            f'<image href="{rival_crest_uri}" x="{crest_x}" y="{bottom_base-11}" width="{crest_size}" height="{crest_size}" preserveAspectRatio="xMidYMid meet" />'
        )

    for tick in [0, 15, 30, 45, 60, 75, 90]:
        xpos = x_pos(tick)
        svg_parts.append(f'<line x1="{xpos}" y1="{axis_y-6}" x2="{xpos}" y2="{axis_y+6}" class="tick" />')
        svg_parts.append(f'<text x="{xpos}" y="{axis_y+34}" text-anchor="middle" class="sub">{tick}</text>')

    if max_minute > 90:
        xpos = x_pos(max_minute)
        svg_parts.append(f'<line x1="{xpos}" y1="{axis_y-6}" x2="{xpos}" y2="{axis_y+6}" class="tick" />')
        svg_parts.append(f'<text x="{xpos}" y="{axis_y+34}" text-anchor="middle" class="sub">{max_minute}</text>')

    def add_event_block(row, is_conxo):
        minute = int(row["minute"])
        event_kind = str(row["kind"])
        event_meta = action_meta.get(event_kind, action_meta["goal"])
        level = int(row["level"])
        xpos = x_pos(minute)
        if event_kind == "goal":
            stem_color = "#2f9e44" if is_conxo else "#c55252"
        else:
            stem_color = event_meta["stem"] if is_conxo else "#b54747"
        vertical_offset = 64 + (level * 56)
        label_text = f"{minute}' {event_meta['icon']}"

        if is_conxo:
            vertical_end_y = axis_y - vertical_offset
            svg_parts.append(f'<line x1="{xpos}" y1="{axis_y}" x2="{xpos}" y2="{vertical_end_y}" stroke="{stem_color}" stroke-width="2"/>')
            svg_parts.append(f'<circle cx="{xpos}" cy="{axis_y}" r="5" fill="{stem_color}" class="eventDot"/>')
            svg_parts.append(
                f'<text x="{xpos}" y="{vertical_end_y-10}" text-anchor="middle" fill="{stem_color}" '
                f'transform="rotate(-90 {xpos} {vertical_end_y-8})" class="eventTag">{label_text}</text>'
            )
        else:
            vertical_end_y = axis_y + vertical_offset
            svg_parts.append(f'<line x1="{xpos}" y1="{axis_y}" x2="{xpos}" y2="{vertical_end_y}" stroke="{stem_color}" stroke-width="2"/>')
            svg_parts.append(f'<circle cx="{xpos}" cy="{axis_y}" r="5" fill="{stem_color}" class="eventDot"/>')
            svg_parts.append(
                f'<text x="{xpos}" y="{vertical_end_y+10}" text-anchor="middle" fill="{stem_color}" '
                f'transform="rotate(90 {xpos} {vertical_end_y+8})" class="eventTag">{label_text}</text>'
            )

    for _, row in own_df.iterrows():
        add_event_block(row, True)
    for _, row in rival_df.iterrows():
        add_event_block(row, False)

    svg_parts.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" class="label">Minuto de partido</text>')
    svg_parts.append('</svg>')
    st.markdown(
        f'<div class="timeline-svg-wrap">{"".join(svg_parts)}</div>',
        unsafe_allow_html=True,
    )

    descriptive_rows = []
    for _, row in timeline_df.sort_values(["minute", "sort_order"], ascending=[False, False]).iterrows():
        kind = str(row["kind"])
        event_meta = action_meta.get(kind, action_meta["goal"])
        minute = int(row["minute"])
        row_classes = ["timeline-log-row"]
        detail_class = "timeline-log-name"
        if kind == "goal":
            detail_class = "timeline-log-name timeline-log-name--goal-conxo" if bool(row.get("is_conxo", False)) else "timeline-log-name timeline-log-name--goal-rival"
            crest_uri = conxo_crest_uri if bool(row.get("is_conxo", False)) else rival_crest_uri
            crest_html = (
                f'<img src="{crest_uri}" class="timeline-log-crest" alt="Escudo">'
                if crest_uri
                else ""
            )
            detail_html = f'{crest_html}<span class="{detail_class}">{row["name"]}</span>'
        elif kind in {"yellow", "red"}:
            detail_class = "timeline-log-name timeline-log-name--yellow" if kind == "yellow" else "timeline-log-name timeline-log-name--red"
            detail_html = f'<span class="{detail_class}">{row["name"]}</span>'
        elif kind == "change":
            detail_html = (
                f'<span class="timeline-log-name timeline-log-name--in">↙ Entra {row["in_name"]}</span>'
                f'<span class="timeline-log-sep">·</span>'
                f'<span class="timeline-log-name timeline-log-name--out">↗ Sale {row["out_name"]}</span>'
            )
        else:
            detail_html = f'<span class="timeline-log-name">{row.get("name", "")}</span>'

        descriptive_rows.append(
            f"""
            <div class="{' '.join(row_classes)}">
                <div class="timeline-log-minute">Min. {minute}</div>
                <div class="timeline-log-action">
                    <span class="timeline-log-icon">{event_meta["icon"]}</span>
                    <span class="timeline-log-label">{row["action"]}</span>
                </div>
                <div class="timeline-log-detail">{detail_html}</div>
            </div>
            """
        )

    render_subsection_title("Timeline descriptivo", "🧾")
    st.html(f'<div class="timeline-log-card">{"".join(descriptive_rows)}</div>')


def render_general(data):
    render_page_section_title("General")
    matches = data["matches"]
    team_table = data["team_table"]
    current_match = data["current_match"]
    current_team_row = data["current_team_row"]
    team_name = data["team_name"]
    general = data["general"]
    goals = data["goals"]
    yellows = data["yellows"]
    reds = data["reds"]
    crest_map = build_team_crest_map()

    selected_round = int(st.session_state.get("general_round", int(current_match["Jornada"])))
    match_row = matches[matches["Jornada"].eq(selected_round)].iloc[0]
    team_row = team_table[team_table["Jornada"].eq(selected_round)].iloc[0]

    render_subsection_title("Clasificación del Conxo")
    snapshot_cards = [
        ("Jornada", int(team_row["Jornada"])),
        ("Posición", int(team_row["Posicion"])),
        ("Puntos", int(team_row["Puntos"])),
        ("Jugados", int(team_row["Partidos Jugados"])),
        ("Ganados", int(team_row["P Ganados"])),
        ("Empatados", int(team_row["P Empatados"])),
        ("Perdidos", int(team_row["P Perdidos"])),
        ("Goles Fav", int(team_row["Goles Fav"])),
        ("Goles Contr", int(team_row["Goles Contr"])),
        ("Dif Goles", int(team_row["Dif Goles"])),
    ]
    render_stat_cards(snapshot_cards, columns=len(snapshot_cards), card_class="stat-card compact-card")

    jornadas = matches["Jornada"].dropna().astype(int).tolist()
    jornada_labels = {}
    for jornada in jornadas:
        row = matches[matches["Jornada"].eq(jornada)].iloc[0]
        jornada_labels[jornada] = f"J{jornada} · {str(row['Eq.Local']).strip()} vs {str(row['Eq.Visitante']).strip()}"
    render_subsection_title("Navegación por jornada")
    selected_round = st.selectbox(
        "Selecciona jornada",
        jornadas,
        index=jornadas.index(selected_round) if selected_round in jornadas else len(jornadas) - 1,
        key="general_round",
        format_func=lambda jornada: jornada_labels[jornada],
    )

    match_row = matches[matches["Jornada"].eq(selected_round)].iloc[0]
    team_row = team_table[team_table["Jornada"].eq(selected_round)].iloc[0]
    rival_crest = crest_map.get(_normalize_key(match_row["Rival"]))
    is_conxo_local = str(match_row["Eq.Local"]).strip() == team_name
    left_team_name = str(match_row["Eq.Local"]).strip()
    right_team_name = str(match_row["Eq.Visitante"]).strip()
    left_crest = CREST_PATH if is_conxo_local else rival_crest
    right_crest = rival_crest if is_conxo_local else CREST_PATH

    render_subsection_title("Ficha técnica de la jornada")
    top_wrap_left, top_left, top_center, top_right, top_wrap_right = st.columns([0.6, 0.7, 2.8, 0.7, 0.6])
    with top_left:
        if left_crest and Path(left_crest).exists():
            st.image(str(left_crest), width=82)
        st.markdown(f'<div class="crest-team-name">{left_team_name}</div>', unsafe_allow_html=True)
    with top_center:
        st.markdown(
            f"""
            <div class="match-hero">
                <div class="match-hero-inner">
                    <div class="match-hero-round">Jornada {int(match_row['Jornada'])}</div>
                    <div class="match-hero-score">{match_row['Marcador']}</div>
                    <div class="match-hero-fixture">{left_team_name} vs {right_team_name}</div>
                    <div class="match-hero-status">{match_row['Resultado']} · {match_row['Local/Visitante']} · {match_row['Estadio']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        if right_crest and Path(right_crest).exists():
            st.image(str(right_crest), width=82)
        st.markdown(f'<div class="crest-team-name">{right_team_name}</div>', unsafe_allow_html=True)
    detail_1, detail_2, detail_3, detail_4 = st.columns(4)
    technical_cards = [
        ("📍 Rival", match_row["Rival"]),
        ("🏟️ Estadio", match_row["Estadio"]),
        ("🧭 Condición", match_row["Local/Visitante"]),
        ("📅 Fecha", match_row["Fecha"].strftime("%d/%m/%Y") if pd.notna(match_row["Fecha"]) else "-"),
    ]
    for col, item in zip([detail_1, detail_2, detail_3, detail_4], technical_cards):
        with col:
            render_stat_cards([item], columns=1, card_class="stat-card detail-card")

    team_round = general[general["Jornada"].eq(selected_round)].copy()
    team_round["Dorsal"] = _safe_numeric(team_round["Dorsal"]).astype(int)
    name_to_dorsal = (
        team_round.dropna(subset=["Nombre"])
        .drop_duplicates(subset=["Nombre"])
        .set_index("Nombre")["Dorsal"]
        .to_dict()
    )
    starters = (
        team_round[team_round["Titular"].eq(1)][["Dorsal", "Nombre", "Minutos Jugados"]]
        .dropna(subset=["Nombre"])
        .sort_values("Dorsal")
        .rename(columns={"Dorsal": "dorsal", "Nombre": "name", "Minutos Jugados": "minutes"})
        .to_dict(orient="records")
    )
    bench = (
        team_round[team_round["Convocado"].eq(1) & team_round["Titular"].eq(0)][["Dorsal", "Nombre", "Minutos Jugados"]]
        .dropna(subset=["Nombre"])
        .sort_values("Dorsal")
        .rename(columns={"Dorsal": "dorsal", "Nombre": "name", "Minutos Jugados": "minutes"})
        .to_dict(orient="records")
    )
    changes_list = (
        team_round[
            team_round["Sustituido Por:"].notna() & team_round["Sustituido Por:"].ne(0)
        ][["Dorsal", "Nombre", "Sustituido Por:", "Minutos Jugados"]]
        .sort_values("Minutos Jugados")
        .rename(
            columns={
                "Dorsal": "out_number",
                "Nombre": "out_name",
                "Sustituido Por:": "in_name",
                "Minutos Jugados": "minute",
            }
        )
        .to_dict(orient="records")
    )
    for change in changes_list:
        change["in_number"] = int(name_to_dorsal.get(change["in_name"], 0))
        change["minute"] = int(change["minute"])
    for player_group in (starters, bench):
        for player in player_group:
            player["dorsal"] = int(player["dorsal"])
            player["minutes"] = int(player["minutes"])
    scorers = goals[
        goals["Jornada"].eq(selected_round) & goals["Equipo Marca"].astype(str).str.strip().eq(team_name)
    ][["Nombre", "Minuto", "Tipo"]].copy()
    scorers_list = (
        scorers.sort_values("Minuto")
        .rename(columns={"Nombre": "name", "Minuto": "minute", "Tipo": "type"})
        .to_dict(orient="records")
    )
    rival_scorers = goals[
        goals["Jornada"].eq(selected_round) & goals["Equipo Marca"].astype(str).str.strip().eq(match_row["Rival"])
    ][["Nombre", "Minuto", "Tipo"]].copy()
    rival_scorers_list = (
        rival_scorers.sort_values("Minuto")
        .rename(columns={"Nombre": "name", "Minuto": "minute", "Tipo": "type"})
        .to_dict(orient="records")
    )
    yellows_round = yellows[
        yellows["Jornada"].eq(selected_round) & yellows["Nombre"].notna()
    ][["Nombre", "Minuto 1a Amarilla", "2a Tarjeta Amarilla"]].copy()
    yellows_list = (
        yellows_round.sort_values("Minuto 1a Amarilla")
        .rename(columns={"Nombre": "name", "Minuto 1a Amarilla": "minute"})
        .assign(
            subtitle=lambda df: np.where(
                df["2a Tarjeta Amarilla"].fillna(0).gt(0),
                "Doble amarilla",
                "Tarjeta amarilla",
            )
        )[["name", "minute", "subtitle"]]
        .to_dict(orient="records")
    )
    reds_round = reds[
        reds["Jornada"].eq(selected_round) & reds["Nombre"].notna()
    ][["Nombre", "Minuto Tarjeta Roja", "Roja Directa"]].copy()
    reds_list = (
        reds_round.sort_values("Minuto Tarjeta Roja")
        .rename(columns={"Nombre": "name", "Minuto Tarjeta Roja": "minute"})
        .assign(
            subtitle=lambda df: np.where(
                df["Roja Directa"].astype(str).str.strip().str.lower().eq("si"),
                "Roja directa",
                "Tarjeta roja",
            )
        )[["name", "minute", "subtitle"]]
        .to_dict(orient="records")
    )
    for goals_group in (scorers_list, rival_scorers_list):
        for goal in goals_group:
            goal["minute"] = int(goal["minute"])
    for card_group in (yellows_list, reds_list):
        for card in card_group:
            card["minute"] = int(card["minute"])
    timeline_events = []
    for goal in scorers_list:
        timeline_events.append(
            {
                "minute": int(goal["minute"]),
                "kind": "goal",
                "action": "Gol",
                "name": str(goal["name"]),
                "is_conxo": True,
                "sort_order": 4,
            }
        )
    for goal in rival_scorers_list:
        timeline_events.append(
            {
                "minute": int(goal["minute"]),
                "kind": "goal",
                "action": "Gol",
                "name": str(goal["name"]),
                "is_conxo": False,
                "sort_order": 4,
            }
        )
    for card in yellows_list:
        timeline_events.append(
            {
                "minute": int(card["minute"]),
                "kind": "yellow",
                "action": "Tarjeta amarilla",
                "name": str(card["name"]),
                "is_conxo": True,
                "sort_order": 3,
            }
        )
    for card in reds_list:
        timeline_events.append(
            {
                "minute": int(card["minute"]),
                "kind": "red",
                "action": "Tarjeta roja",
                "name": str(card["name"]),
                "is_conxo": True,
                "sort_order": 2,
            }
        )
    for change in changes_list:
        timeline_events.append(
            {
                "minute": int(change["minute"]),
                "kind": "change",
                "action": "Cambio",
                "out_name": str(change["out_name"]),
                "in_name": str(change["in_name"]),
                "is_conxo": True,
                "sort_order": 1,
            }
        )

    bottom_1, bottom_2, bottom_3 = st.columns([1, 1, 1])
    with bottom_1:
        render_people_panel("Once titular", "🧱", starters, "Sin once cargado en esta jornada", grid_columns=2, variant="starter")
    with bottom_2:
        render_people_panel("Banquillo", "🪑", bench, "Sin suplentes cargados en esta jornada", variant="bench")
    with bottom_3:
        render_change_panel(changes_list)

    st.markdown('<div class="goals-section-spacer"></div>', unsafe_allow_html=True)
    goals_col_1, goals_col_2, goals_col_3, goals_col_4 = st.columns([1, 1, 1, 1])
    with goals_col_1:
        render_goal_panel("Goles del Conxo", "⚽", scorers_list, "Sin goles del Conxo en esta jornada")
    with goals_col_2:
        render_goal_panel(
            f"Goles de {match_row['Rival']}",
            "🥅",
            rival_scorers_list,
            "Sin goles del rival en esta jornada",
            variant="rival",
        )
    with goals_col_3:
        render_event_panel("Amarillas", "🟨", yellows_list, "Sin amarillas en esta jornada", variant="yellow")
    with goals_col_4:
        render_event_panel("Rojas", "🟥", reds_list, "Sin rojas en esta jornada", variant="red")

    render_match_timeline(timeline_events, match_row["Rival"])


def render_equipo(data):
    render_page_section_title("Equipo")
    matches = data["matches"]
    team_table = data["team_table"]
    teams = data["teams"]
    team_name = data["team_name"]
    offense = data["offense"]
    defense = data["defense"]
    crest_map = build_team_crest_map()

    top_col, bottom_col = st.tabs(["Evolución", "Producción Ofensiva / Defensiva"])

    with top_col:
        crest_uri = None
        if CREST_PATH.exists():
            crest_uri = "data:image/png;base64," + base64.b64encode(CREST_PATH.read_bytes()).decode("ascii")

        fig_position = go.Figure()
        fig_position.add_scatter(
            x=team_table["Jornada"],
            y=team_table["Posicion"],
            mode="lines",
            name="Posición",
            line=dict(color="#0b4f7a", width=4),
            hovertemplate="Jornada %{x}<br>Posición %{y}<extra></extra>",
        )
        fig_position.add_scatter(
            x=team_table["Jornada"],
            y=team_table["Posicion"],
            mode="markers",
            marker=dict(size=8, color="#0b4f7a", opacity=0.16),
            showlegend=False,
            hovertemplate="Jornada %{x}<br>Posición %{y}<extra></extra>",
        )
        if crest_uri:
            for _, row in team_table.iterrows():
                fig_position.add_layout_image(
                    dict(
                        source=crest_uri,
                        xref="x",
                        yref="y",
                        x=float(row["Jornada"]),
                        y=float(row["Posicion"]),
                        xanchor="center",
                        yanchor="middle",
                        sizex=0.46,
                        sizey=0.34,
                        sizing="contain",
                        layer="above",
                    )
                )
        fig_position.update_layout(
            title="Evolución de la posición del Conxo",
            xaxis_title="Jornada",
            yaxis_title="Posición",
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            height=380,
            plot_bgcolor=PLOT_CARD_BG,
            paper_bgcolor=PLOT_PAPER_BG,
            font=dict(color=PLOT_LINE),
        )
        fig_position.update_xaxes(
            tickmode="linear",
            tick0=1,
            dtick=1,
            range=[0.5, max(30, int(team_table["Jornada"].max())) + 0.5],
            showgrid=False,
            linecolor=PLOT_AXIS,
        )
        fig_position.update_yaxes(
            autorange="reversed",
            dtick=1,
            showgrid=True,
            gridcolor=PLOT_GRID,
            zeroline=False,
            linecolor=PLOT_AXIS,
        )
        st.plotly_chart(fig_position, use_container_width=True)

        combo = go.Figure()
        combo.add_bar(
            x=matches["Jornada"],
            y=matches["Dif Goles Partido"],
            name="Dif. goles por jornada",
            marker_color=np.where(matches["Dif Goles Partido"] >= 0, "#2b93c9", "#d95f59"),
        )
        combo.add_scatter(
            x=matches["Jornada"],
            y=matches["Dif Goles Acumulada"],
            mode="lines+text",
            name="Dif. goles acumulada",
            line=dict(color="#17324d", width=4),
            text=matches["Dif Goles Acumulada"].astype(int).astype(str),
            textposition="top center",
            textfont=dict(color="#17324d", size=11),
            yaxis="y2",
        )
        y1_min = int(min(matches["Dif Goles Partido"].min(), 0))
        y1_max = int(max(matches["Dif Goles Partido"].max(), 0))
        if y1_min == y1_max:
            y1_max = y1_min + 1
        zero_ratio = (0 - y1_min) / (y1_max - y1_min)

        y2_min = int(min(matches["Dif Goles Acumulada"].min(), 0))
        y2_max_data = int(max(matches["Dif Goles Acumulada"].max(), 0))
        if 0 < zero_ratio < 1 and y2_min < 0:
            aligned_y2_max = y2_min + ((0 - y2_min) / zero_ratio)
        else:
            aligned_y2_max = y2_max_data
        y2_max = max(y2_max_data, aligned_y2_max)

        combo.update_layout(
            title="Diferencia de goles: jornada y acumulado",
            xaxis_title="Jornada",
            yaxis=dict(
                title="GF - GC en la jornada",
                range=[y1_min - 0.5, y1_max + 0.5],
                zeroline=True,
                zerolinecolor="#cfdbe5",
                gridcolor=PLOT_GRID,
            ),
            yaxis2=dict(
                title="Acumulado de GF - GC",
                overlaying="y",
                side="right",
                range=[y2_min - 2, y2_max + 2],
                zeroline=True,
                zerolinecolor="#cfdbe5",
            ),
            template="plotly_white",
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=20, r=20, t=60, b=20),
            height=420,
            plot_bgcolor=PLOT_CARD_BG,
            paper_bgcolor=PLOT_PAPER_BG,
            font=dict(color=PLOT_LINE),
        )
        combo.update_xaxes(
            tickmode="linear",
            tick0=1,
            dtick=1,
            range=[0.5, max(30, int(matches["Jornada"].max())) + 0.5],
            showgrid=False,
            linecolor=PLOT_AXIS,
        )
        if CREST_PATH.exists():
            combo_crest_uri = _path_to_data_uri(CREST_PATH)
            for _, row in matches.iterrows():
                combo.add_layout_image(
                    dict(
                        source=combo_crest_uri,
                        xref="x",
                        yref="y2",
                        x=float(row["Jornada"]),
                        y=float(row["Dif Goles Acumulada"]),
                        xanchor="center",
                        yanchor="middle",
                        sizex=0.62,
                        sizey=3.3,
                        sizing="contain",
                        layer="above",
                    )
                )
        st.plotly_chart(combo, use_container_width=True)

        render_subsection_title("Contexto competitivo por jornada")
        jornada_focus_options = [
            (
                int(row["Jornada"]),
                f"J{int(row['Jornada'])} · {row['Eq.Local']} vs {row['Eq.Visitante']}",
            )
            for _, row in matches.sort_values("Jornada").iterrows()
        ]
        jornada_focus_map = {label: jornada for jornada, label in jornada_focus_options}
        default_focus_label = jornada_focus_options[-1][1]
        selected_focus_label = st.selectbox(
            "Selecciona jornada de análisis",
            [label for _, label in jornada_focus_options],
            index=len(jornada_focus_options) - 1,
            key="equipo_focus_round",
        )
        selected_focus_round = jornada_focus_map[selected_focus_label]
        round_table = (
            teams[teams["Jornada"].eq(selected_focus_round)][
                [
                    "Equipo",
                    "Posicion",
                    "Partidos Jugados",
                    "Goles Fav",
                    "Goles Contr",
                    "Dif Goles",
                    "Puntos",
                ]
            ]
            .copy()
            .sort_values("Posicion")
            .reset_index(drop=True)
        )
        avg_gf = round_table["Goles Fav"].mean()
        avg_gc = round_table["Goles Contr"].mean()
        x_min = max(0, float(round_table["Goles Contr"].min()) - 1.5)
        x_max = float(round_table["Goles Contr"].max()) + 1.5
        y_min = max(0, float(round_table["Goles Fav"].min()) - 1.5)
        y_max = float(round_table["Goles Fav"].max()) + 1.5
        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)
        crest_size_x = x_span * 0.055
        crest_size_y = y_span * 0.055
        round_table["Escudo"] = round_table["Equipo"].map(lambda name: crest_map.get(_normalize_key(name)))
        round_table.loc[
            round_table["Equipo"].astype(str).str.strip().eq(team_name),
            "Escudo",
        ] = CREST_PATH

        scatter_fig = go.Figure()
        scatter_fig.add_trace(
            go.Scatter(
                x=round_table["Goles Contr"],
                y=round_table["Goles Fav"],
                mode="markers",
                marker=dict(size=2, color="rgba(0,0,0,0)", line=dict(width=0)),
                text=round_table["Equipo"],
                customdata=np.stack(
                    [
                        round_table["Posicion"].astype(int),
                        round_table["Dif Goles"].astype(int),
                        round_table["Puntos"].astype(int),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Posición: %{customdata[0]}<br>"
                    "GF: %{y}<br>"
                    "GC: %{x}<br>"
                    "Dif.: %{customdata[1]}<br>"
                    "Puntos: %{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        for _, row in round_table.iterrows():
            crest_uri = _path_to_data_uri(row["Escudo"])
            if crest_uri:
                scatter_fig.add_layout_image(
                    dict(
                        source=crest_uri,
                        xref="x",
                        yref="y",
                        x=float(row["Goles Contr"]),
                        y=float(row["Goles Fav"]),
                        xanchor="center",
                        yanchor="middle",
                        sizex=crest_size_x,
                        sizey=crest_size_y,
                        sizing="contain",
                        layer="above",
                        opacity=1,
                    )
                )
            scatter_fig.add_annotation(
                x=float(row["Goles Contr"]),
                y=float(row["Goles Fav"]) - 0.72,
                text=f"{int(row['Posicion'])}",
                showarrow=False,
                font=dict(size=10, color="#10364d"),
                align="center",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.86)",
                bordercolor="rgba(217,231,239,0.9)",
                borderwidth=1,
                borderpad=2,
            )
        scatter_fig.add_hline(y=avg_gf, line_dash="dash", line_color="#a9b4be")
        scatter_fig.add_vline(x=avg_gc, line_dash="dash", line_color="#a9b4be")
        scatter_fig.add_annotation(
            x=float(round_table["Goles Contr"].min()),
            y=float(avg_gf),
            text="Promedio",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=10, color="#6b7c8f"),
        )
        scatter_fig.add_annotation(
            x=float(avg_gc),
            y=float(round_table["Goles Fav"].min()),
            text="Promedio",
            showarrow=False,
            textangle=-90,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=10, color="#6b7c8f"),
        )
        scatter_fig.update_layout(
            title="Goles a favor vs goles en contra",
            xaxis_title="Goles en contra",
            yaxis_title="Goles a favor",
            template="plotly_white",
            margin=dict(l=20, r=20, t=55, b=32),
            height=560,
            plot_bgcolor=PLOT_CARD_BG,
            paper_bgcolor=PLOT_PAPER_BG,
            font=dict(color=PLOT_LINE),
        )
        scatter_fig.update_xaxes(
            range=[x_min, x_max],
            showgrid=True,
            gridcolor=PLOT_GRID,
            zeroline=False,
            linecolor=PLOT_AXIS,
        )
        scatter_fig.update_yaxes(
            range=[y_min, y_max],
            showgrid=True,
            gridcolor=PLOT_GRID,
            zeroline=False,
            linecolor=PLOT_AXIS,
        )

        bars_df = round_table.sort_values(["Dif Goles", "Equipo"], ascending=[False, True]).copy()
        max_abs_diff = max(abs(int(bars_df["Dif Goles"].min())), abs(int(bars_df["Dif Goles"].max())), 1)
        crest_gap = max(4.0, round(max_abs_diff * 0.11, 1))
        bar_colors = []
        for _, row in bars_df.iterrows():
            if str(row["Equipo"]).strip() == team_name:
                bar_colors.append("#d8b24d")
            elif row["Dif Goles"] > 0:
                bar_colors.append("#4f9b5b")
            elif row["Dif Goles"] < 0:
                bar_colors.append("#e5764c")
            else:
                bar_colors.append("#d8c85b")
        bars_df = bars_df.reset_index(drop=True)
        bars_df["y_pos"] = list(range(len(bars_df)))
        bars_df["bar_base"] = bars_df["Dif Goles"].apply(lambda value: crest_gap if value >= 0 else -crest_gap)
        bar_fig = go.Figure()
        bar_fig.add_bar(
            x=bars_df["Dif Goles"],
            base=bars_df["bar_base"],
            y=bars_df["y_pos"],
            orientation="h",
            marker_color=bar_colors,
            text=bars_df["Dif Goles"].astype(int).astype(str),
            textposition="outside",
            customdata=np.stack(
                [
                    bars_df["Equipo"],
                    bars_df["Posicion"].astype(int),
                    bars_df["Goles Fav"].astype(int),
                    bars_df["Goles Contr"].astype(int),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Posición: %{customdata[1]}<br>"
                "GF: %{customdata[2]}<br>"
                "GC: %{customdata[3]}<br>"
                "Dif. goles: %{x}<extra></extra>"
            ),
            showlegend=False,
        )
        for _, row in bars_df.iterrows():
            crest_uri = _path_to_data_uri(row["Escudo"])
            if crest_uri:
                bar_fig.add_layout_image(
                    dict(
                        source=crest_uri,
                        xref="x",
                        yref="y",
                        x=0,
                        y=float(row["y_pos"]),
                        xanchor="center",
                        yanchor="middle",
                        sizex=crest_gap * 1.45,
                        sizey=0.9,
                        sizing="contain",
                        layer="above",
                        opacity=1,
                    )
                )
        bar_fig.update_layout(
            title="Diferencia de goles",
            xaxis_title="GF - GC",
            yaxis_title="",
            template="plotly_white",
            margin=dict(l=20, r=20, t=55, b=26),
            height=560,
            plot_bgcolor=PLOT_CARD_BG,
            paper_bgcolor=PLOT_PAPER_BG,
            font=dict(color=PLOT_LINE),
        )
        bar_fig.update_xaxes(
            range=[-max_abs_diff - crest_gap - 4, max_abs_diff + crest_gap + 4],
            zeroline=True,
            zerolinecolor="#b9c7d3",
            zerolinewidth=2,
            showgrid=False,
            linecolor=PLOT_AXIS,
        )
        bar_fig.update_yaxes(
            autorange="reversed",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            linecolor=PLOT_AXIS,
        )

        context_left, context_right = st.columns([1.35, 1])
        with context_left:
            st.plotly_chart(scatter_fig, use_container_width=True)
        with context_right:
            st.plotly_chart(bar_fig, use_container_width=True)

        render_subsection_title("Clasificación")
        classification_rows = []
        for _, row in round_table.sort_values("Posicion").iterrows():
            crest_uri = _path_to_data_uri(row["Escudo"])
            crest_html = (
                f'<img src="{crest_uri}" style="width:28px;height:28px;object-fit:contain;">'
                if crest_uri
                else '<div style="width:28px;height:28px;border-radius:50%;background:#eaf1f6;"></div>'
            )
            row_class = "classification-row conxo" if str(row["Equipo"]).strip() == team_name else "classification-row"
            classification_rows.append(
                f"""
<div class="{row_class}">
    <div class="classification-pos">{int(row["Posicion"])}</div>
    <div class="classification-team">
        <div class="classification-crest">{crest_html}</div>
        <div class="classification-name">{row["Equipo"]}</div>
    </div>
    <div class="classification-stat">{int(row["Puntos"])}</div>
    <div class="classification-stat">{int(row["Partidos Jugados"])}</div>
    <div class="classification-stat">{int(row["Goles Fav"])}</div>
    <div class="classification-stat">{int(row["Goles Contr"])}</div>
    <div class="classification-stat">{int(row["Dif Goles"])}</div>
</div>
                """
            )
        classification_html = (
            '<div class="classification-card">'
            '<div class="classification-header">'
            '<div class="classification-pos">Pos</div>'
            '<div class="classification-team">Equipo</div>'
            '<div class="classification-stat">Pts</div>'
            '<div class="classification-stat">PJ</div>'
            '<div class="classification-stat">GF</div>'
            '<div class="classification-stat">GC</div>'
            '<div class="classification-stat">DG</div>'
            "</div>"
            + "".join(classification_rows)
            + "</div>"
        )
        st.markdown(
            classification_html,
            unsafe_allow_html=True,
        )

    with bottom_col:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                build_bubble_matrix(offense, "Producción ofensiva del Conxo", OFFENSE_SCALE),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                build_bubble_matrix(defense, "Producción defensiva del Conxo", DEFENSE_SCALE),
                use_container_width=True,
            )

        st.plotly_chart(build_diff_heatmap(offense, defense), use_container_width=True)


def render_plantilla(data):
    render_page_section_title("Plantilla")
    players = data["player_summary"].copy()
    general_tab, individual_tab = st.tabs(["General", "Individual"])

    with general_tab:
        ranking_title_style = ""
        if CREST_PATH.exists():
            ranking_crest_uri = "data:image/png;base64," + base64.b64encode(CREST_PATH.read_bytes()).decode("ascii")
            ranking_title_style = (
                f"background-image:linear-gradient(90deg, rgba(16,54,77,0.88) 0%, rgba(27,89,119,0.84) 100%), url('{ranking_crest_uri}');"
                "background-repeat:no-repeat;"
                "background-position:left top, right 12px center;"
                "background-size:auto, 78px;"
            )
        chart_players = players[
            ["Nombre", "Convocados", "Jugados", "Titulares", "Minutos"]
        ].copy()
        chart_players = chart_players.sort_values(
            ["Minutos", "Jugados", "Titulares", "Nombre"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        fig_players = go.Figure()
        fig_players.add_bar(
            x=chart_players["Convocados"],
            y=chart_players["Nombre"],
            name="Convocado",
            orientation="h",
            marker_color="rgba(216, 178, 77, 0.45)",
            marker_line=dict(color="#d8b24d", width=1),
            width=0.84,
            hovertemplate="%{y}<br>Convocado: %{x}<extra></extra>",
        )
        fig_players.add_bar(
            x=chart_players["Jugados"],
            y=chart_players["Nombre"],
            name="Jugado",
            orientation="h",
            marker_color="rgba(74, 150, 200, 0.88)",
            marker_line=dict(color="#4a96c8", width=0.5),
            width=0.58,
            hovertemplate="%{y}<br>Jugado: %{x}<extra></extra>",
        )
        fig_players.add_bar(
            x=chart_players["Titulares"],
            y=chart_players["Nombre"],
            name="Titular",
            orientation="h",
            marker_color="#10364d",
            width=0.34,
            hovertemplate="%{y}<br>Titular: %{x}<extra></extra>",
        )
        fig_players.add_scatter(
            x=chart_players["Convocados"] + 0.35,
            y=chart_players["Nombre"],
            mode="text",
            text=[
                f"T {int(t)} · J {int(j)} · C {int(c)}"
                for t, j, c in zip(chart_players["Titulares"], chart_players["Jugados"], chart_players["Convocados"])
            ],
            textposition="middle right",
            textfont=dict(color="#10364d", size=11, family="Arial"),
            showlegend=False,
            hoverinfo="skip",
        )
        fig_players.update_layout(
            title="Disponibilidad y participación de la plantilla",
            xaxis_title="Número de partidos",
            yaxis_title="Jugador",
            template="plotly_white",
            legend_title_text="",
            margin=dict(l=20, r=200, t=60, b=20),
            height=max(560, len(chart_players) * 28),
            bargap=0.28,
            barmode="overlay",
        )
        fig_players.update_xaxes(
            dtick=1,
            range=[0, max(int(chart_players["Convocados"].max()) + 4, 8)],
            showgrid=True,
            gridcolor="#edf3f7",
        )
        fig_players.update_yaxes(categoryorder="array", categoryarray=chart_players["Nombre"][::-1].tolist())
        st.plotly_chart(fig_players, use_container_width=True)

        general_players = data["general"][["Nombre", "Jornada", "Minutos Jugados"]].copy()
        general_players["Nombre"] = general_players["Nombre"].astype(str).str.strip()
        general_players["Jornada"] = pd.to_numeric(general_players["Jornada"], errors="coerce")
        general_players["Minutos Jugados"] = pd.to_numeric(general_players["Minutos Jugados"], errors="coerce").fillna(0)
        general_players = general_players[general_players["Nombre"].ne("")]

        jornadas = sorted(data["matches"]["Jornada"].dropna().astype(int).unique().tolist())
        player_order = chart_players["Nombre"].astype(str).tolist()
        player_order_ranked = [f"{idx + 1}. {name}" for idx, name in enumerate(player_order)]
        minutes_matrix = (
            general_players.groupby(["Nombre", "Jornada"], as_index=False)["Minutos Jugados"]
            .sum()
            .pivot(index="Nombre", columns="Jornada", values="Minutos Jugados")
            .reindex(index=player_order, columns=jornadas)
            .fillna(0)
            .astype(int)
        )
        minutes_matrix.index = player_order_ranked
        total_minutes = minutes_matrix.sum(axis=1).astype(int)

        colorscale_minutes = [
            [0.0, "#c74c4c"],
            [0.001, "#c74c4c"],
            [1 / 90, "#e98b39"],
            [40 / 90, "#e98b39"],
            [40 / 90, "#e6c84f"],
            [60 / 90, "#e6c84f"],
            [60 / 90, "#5fae67"],
            [1.0, "#5fae67"],
        ]
        minutes_matrix.columns = [f"J{j}" for j in jornadas]
        render_subsection_title("Minutos disputados por jornada")
        st.markdown(build_minutes_scroll_table(minutes_matrix, total_minutes), unsafe_allow_html=True)

        render_subsection_title("Matriz de cambios")
        change_players = data["general"][
            ["Jornada", "Nombre", "Titular", "Minutos Jugados", "Sustituido Por:"]
        ].copy()
        change_players["Nombre"] = change_players["Nombre"].astype(str).str.strip()
        change_players["Titular"] = pd.to_numeric(change_players["Titular"], errors="coerce").fillna(0).astype(int)
        change_players["Minutos Jugados"] = pd.to_numeric(change_players["Minutos Jugados"], errors="coerce").fillna(0).astype(int)
        change_players["Sustituido Por:"] = change_players["Sustituido Por:"].fillna("").astype(str).str.strip()
        change_players = change_players[change_players["Nombre"].ne("")]
        change_players["Entrante"] = change_players["Sustituido Por:"].where(
            ~change_players["Sustituido Por:"].isin(["", "0", "nan", "None"]),
            "",
        )

        player_order_plain = chart_players["Nombre"].astype(str).tolist()
        player_labels_ranked = [f"{idx + 1}. {name}" for idx, name in enumerate(player_order_plain)]
        row_label_map = dict(zip(player_order_plain, player_labels_ranked))

        def _format_matrix_row_label(label_text: str) -> str:
            parts = str(label_text).split(". ", 1)
            if len(parts) == 2:
                prefix, rest = parts
                return f"<b>{prefix}.</b>&nbsp;{rest.replace(' ', '&nbsp;')}"
            return str(label_text).replace(" ", "&nbsp;")

        def _format_matrix_col_label(label_text: str) -> str:
            return str(label_text).replace(" ", "&nbsp;")

        played_mask = change_players["Minutos Jugados"] > 0
        starter_mask = (change_players["Titular"] == 1) & played_mask
        subbed_off_mask = starter_mask & change_players["Entrante"].ne("")
        sub_appearance_mask = (change_players["Titular"] == 0) & played_mask
        full_starter_mask = starter_mask & ~subbed_off_mask

        summary_df = pd.DataFrame(
            {
                "Total": change_players[played_mask].groupby("Nombre").size(),
                "Titular completo": change_players[full_starter_mask].groupby("Nombre").size(),
                "Entró de suplente": change_players[sub_appearance_mask].groupby("Nombre").size(),
            }
        ).reindex(player_order_plain).fillna(0).astype(int)
        summary_df.index = player_labels_ranked

        change_matrix = pd.DataFrame(0, index=player_order_plain, columns=player_order_plain, dtype=int)
        for _, row in change_players[subbed_off_mask].iterrows():
            out_name = row["Nombre"]
            in_name = row["Entrante"]
            if out_name in change_matrix.index and in_name in change_matrix.columns:
                change_matrix.loc[out_name, in_name] += 1

        change_matrix = change_matrix.reindex(index=player_order_plain, columns=player_order_plain).fillna(0).astype(int)
        change_matrix.index = player_labels_ranked
        diagonal_mask = np.full((len(player_order_plain), len(player_order_plain)), np.nan)
        np.fill_diagonal(diagonal_mask, 1)

        st.markdown(build_change_scroll_table(summary_df, change_matrix), unsafe_allow_html=True)

        change_pairs = (
            change_players[subbed_off_mask]
            .groupby(["Nombre", "Entrante"], dropna=False)
            .size()
            .reset_index(name="Cambios")
            .sort_values(["Cambios", "Nombre", "Entrante"], ascending=[False, True, True])
            .head(3)
        )
        top_change_minutes = (
            change_players[subbed_off_mask]
            .groupby("Minutos Jugados", dropna=False)
            .size()
            .reset_index(name="Frecuencia")
            .sort_values(["Frecuencia", "Minutos Jugados"], ascending=[False, True])
            .head(3)
        )

        top_changes_items = []
        for idx, (_, row) in enumerate(change_pairs.iterrows(), start=1):
            top_changes_items.append(
                (
                    f'<div class="matrix-insight-item">'
                    f'<div class="matrix-insight-rank">{idx}</div>'
                    f'<div class="matrix-insight-body">'
                    f'<div class="matrix-insight-main">{row["Nombre"]} → {row["Entrante"]}</div>'
                    f'<div class="matrix-insight-sub">Cambio repetido en distintas jornadas</div>'
                    f'</div>'
                    f'<div class="matrix-insight-count">{int(row["Cambios"])}x</div>'
                    f'</div>'
                )
            )
        if not top_changes_items:
            top_changes_items.append('<div class="people-empty">Sin datos disponibles</div>')

        top_minutes_items = []
        for idx, (_, row) in enumerate(top_change_minutes.iterrows(), start=1):
            minute_value = int(row["Minutos Jugados"])
            top_minutes_items.append(
                (
                    f'<div class="matrix-insight-item">'
                    f'<div class="matrix-insight-rank">{idx}</div>'
                    f'<div class="matrix-insight-body">'
                    f'<div class="matrix-insight-main">Minuto {minute_value}\'</div>'
                    f'<div class="matrix-insight-sub">Momento con más sustituciones acumuladas</div>'
                    f'</div>'
                    f'<div class="matrix-insight-count">{int(row["Frecuencia"])}x</div>'
                    f'</div>'
                )
            )
        if not top_minutes_items:
            top_minutes_items.append('<div class="people-empty">Sin datos disponibles</div>')

        st.markdown(
            (
                '<div class="matrix-insight-grid">'
                '<div class="matrix-insight-card">'
                '<div class="matrix-insight-card-title">Top 3 de cambios realizados</div>'
                f'<div class="matrix-insight-list">{"".join(top_changes_items)}</div>'
                '</div>'
                '<div class="matrix-insight-card">'
                '<div class="matrix-insight-card-title">Top 3 de minutos con más cambios</div>'
                f'<div class="matrix-insight-list">{"".join(top_minutes_items)}</div>'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

        render_subsection_title("Ranking")
        ranking_config = [
            ("Partidos jugados", "Jugados", "{:.0f}"),
            ("Minutos jugados", "Minutos", "{:.0f}"),
            ("% minutos convocado", "% Minutos Convocado", "{:.1f}%"),
            ("Goles", "Goles", "{:.0f}"),
            ("Tarjetas amarillas", "Amarillas", "{:.0f}"),
            ("Tarjetas rojas", "Rojas", "{:.0f}"),
        ]
        ranking_cols = st.columns(3)
        for idx, (title, metric, fmt) in enumerate(ranking_config):
            ranking_df = players[["Nombre", metric]].copy()
            ranking_df = ranking_df.sort_values([metric, "Nombre"], ascending=[False, True]).reset_index(drop=True)
            ranking_df = ranking_df[ranking_df[metric].fillna(0).gt(0)] if metric in ["Goles", "Amarillas", "Rojas"] else ranking_df
            metric_total = float(players[metric].fillna(0).sum()) if metric in ["Goles", "Amarillas", "Rojas"] else 0.0
            items_html = []
            for pos, (_, row) in enumerate(ranking_df.head(12).iterrows(), start=1):
                value = fmt.format(float(row[metric]))
                pct_html = ""
                if metric in ["Goles", "Amarillas", "Rojas"] and metric_total > 0:
                    pct = (float(row[metric]) / metric_total) * 100
                    pct_html = f'<span class="ranking-share">{pct:.0f}%</span>'
                items_html.append(
                    f'<div class="ranking-item"><div class="ranking-pos">{pos}</div><div class="ranking-body"><div class="ranking-name">{row["Nombre"]}</div></div><div class="ranking-value-wrap">{pct_html}<div class="ranking-value">{value}</div></div></div>'
                )
            if not items_html:
                items_html.append('<div class="people-empty">Sin datos disponibles</div>')
            with ranking_cols[idx % 3]:
                st.markdown(
                    (
                        f'<div class="ranking-card">'
                        f'<div class="ranking-card-title" style="{ranking_title_style}">{title}</div>'
                        f'<div class="ranking-card-list">{"".join(items_html)}</div>'
                        f'</div>'
                    ),
                    unsafe_allow_html=True,
                )

    with individual_tab:
        player_names = players["Nombre"].dropna().astype(str).tolist()
        selected_player = st.selectbox("Selecciona jugador", player_names)
        player_row = players[players["Nombre"].eq(selected_player)].iloc[0]
        squad_source = data["squad"].copy()
        squad_source["Nombre"] = squad_source["Nombre"].astype(str).str.strip()
        squad_row = squad_source[squad_source["Nombre"].eq(selected_player)].head(1)
        squad_info = squad_row.iloc[0] if not squad_row.empty else pd.Series(dtype=object)

        display_name = _display_player_name(selected_player)
        dorsal_value = (
            str(int(player_row["Dorsal"]))
            if pd.notna(player_row["Dorsal"]) and int(float(player_row["Dorsal"])) > 0
            else "--"
        )
        posicion_global = _safe_profile_value(
            squad_info["Posicion Global"] if "Posicion Global" in squad_info.index else player_row.get("Posicion Global", np.nan)
        )
        posicion_especifica = _safe_profile_value(
            squad_info["Posicion Especifica"] if "Posicion Especifica" in squad_info.index else player_row.get("Posicion Especifica", np.nan)
        )
        birth_year_value = pd.to_numeric(
            squad_info["Años"] if "Años" in squad_info.index else squad_info.get("Ano", np.nan),
            errors="coerce",
        )
        if pd.notna(birth_year_value):
            fecha_nacimiento = str(int(birth_year_value))
        else:
            fecha_nacimiento = _safe_profile_value(
                squad_info["Fecha Nacimiento"] if "Fecha Nacimiento" in squad_info.index else squad_info.get("Fecha de Nacimiento", np.nan)
            )

        edad_raw = squad_info["Edad"] if "Edad" in squad_info.index else np.nan
        if pd.notna(edad_raw):
            edad_value = str(int(float(edad_raw)))
        elif pd.notna(birth_year_value):
            edad_value = str(int(pd.Timestamp.now().year - float(birth_year_value)))
        else:
            edad_value = "Pendiente"

        convocados = int(player_row["Convocados"])
        jugados = int(player_row["Jugados"])
        titulares = int(player_row["Titulares"])
        minutos = int(player_row["Minutos"])
        goles = int(player_row["Goles"])
        amarillas = int(player_row["Amarillas"])
        rojas = int(player_row["Rojas"])

        pct_convocados_liga = round((convocados / 30.0) * 100, 1)
        pct_jugados_convocado = round((jugados / convocados) * 100, 1) if convocados > 0 else 0.0
        pct_titular_convocado = round((titulares / convocados) * 100, 1) if convocados > 0 else 0.0
        pct_titular_jugado = round((titulares / jugados) * 100, 1) if jugados > 0 else 0.0
        pct_minutos_convocado = round((minutos / (convocados * 90.0)) * 100, 1) if convocados > 0 else 0.0
        pct_minutos_liga = round((minutos / (30.0 * 90.0)) * 100, 1) if minutos > 0 else 0.0
        goles_90 = round((goles / minutos) * 90, 2) if minutos > 0 else 0.0
        ta_90 = round((amarillas / minutos) * 90, 2) if minutos > 0 else 0.0
        tr_90 = round((rojas / minutos) * 90, 2) if minutos > 0 else 0.0

        player_match_presence = data["general"][["Nombre", "Jornada", "Minutos Jugados"]].copy()
        player_match_presence["Nombre"] = player_match_presence["Nombre"].astype(str).str.strip()
        player_match_presence["Jornada"] = pd.to_numeric(player_match_presence["Jornada"], errors="coerce")
        player_match_presence["Minutos Jugados"] = pd.to_numeric(
            player_match_presence["Minutos Jugados"], errors="coerce"
        ).fillna(0)
        player_match_presence = player_match_presence[
            player_match_presence["Nombre"].eq(selected_player) & player_match_presence["Minutos Jugados"].gt(0)
        ]
        jornadas_participadas = sorted(
            player_match_presence["Jornada"].dropna().astype(int).unique().tolist()
        )
        player_match_stats = data["matches"][
            data["matches"]["Jornada"].astype(int).isin(jornadas_participadas)
        ].copy()
        avg_gf_with_player = round(float(player_match_stats["GF"].mean()), 2) if not player_match_stats.empty else 0.0
        avg_gc_with_player = round(float(player_match_stats["GC"].mean()), 2) if not player_match_stats.empty else 0.0
        avg_diff_with_player = (
            round(float(player_match_stats["Dif Goles Partido"].mean()), 2) if not player_match_stats.empty else 0.0
        )

        st.markdown(
            f"""
            <div class="player-profile-shell">
                <div class="player-top-grid">
                    <div class="player-photo-box">
                        <div class="player-photo-icon">🖼️</div>
                        <div>Foto del jugador</div>
                        <div style="font-size:0.82rem; font-weight:600; margin-top:0.25rem;">Pendiente de incorporar</div>
                    </div>
                    <div class="player-bio-card">
                        <div class="player-bio-dorsal">{dorsal_value}</div>
                        <div class="player-bio-name">{display_name}</div>
                        <div class="player-bio-grid">
                            <div class="player-bio-field">
                                <div class="player-bio-label">Posición general</div>
                                <div class="player-bio-value">{posicion_global}</div>
                            </div>
                            <div class="player-bio-field">
                                <div class="player-bio-label">Posición específica</div>
                                <div class="player-bio-value">{posicion_especifica}</div>
                            </div>
                            <div class="player-bio-field">
                                <div class="player-bio-label">Fecha de nacimiento</div>
                                <div class="player-bio-value">{fecha_nacimiento}</div>
                            </div>
                            <div class="player-bio-field">
                                <div class="player-bio-label">Edad</div>
                                <div class="player-bio-value">{edad_value}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_groups = [
            [
                ("Convocado", f"{convocados}"),
                ("Jugados", f"{jugados}"),
                ("Titular", f"{titulares}"),
                ("Minutos", f"{minutos}"),
            ],
            [
                ("% Convocados / liga", f"{pct_convocados_liga:.1f}%"),
                ("% Titular / convocado", f"{pct_titular_convocado:.1f}%"),
                ("% Jugados / convocado", f"{pct_jugados_convocado:.1f}%"),
                ("% Minutos / convocado", f"{pct_minutos_convocado:.1f}%"),
            ],
            [
                ("Goles", f"{goles}"),
                ("Goles / 90", f"{goles_90:.2f}"),
                ("Tarjetas Amarillas", f"{amarillas}"),
                ("TA / 90", f"{ta_90:.2f}"),
                ("Tarjetas Rojas", f"{rojas}"),
                ("TR / 90", f"{tr_90:.2f}"),
            ],
        ]

        for group in metric_groups:
            st.markdown(
                '<div class="player-metric-grid" style="grid-template-columns: repeat('
                + str(len(group))
                + ', minmax(0, 1fr));">'
                + "".join(
                    [
                        f'<div class="player-metric-card"><div class="player-metric-label">{label}</div><div class="player-metric-value">{value}</div></div>'
                        for label, value in group
                    ]
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        def _pct_color(value: float) -> str:
            if value >= 95:
                return "#10364d"
            if value >= 85:
                return "#1b5977"
            if value >= 70:
                return "#2f7f9f"
            if value >= 50:
                return "#d8b24d"
            return "#d95f59"

        radar_labels = [
            "% Convocados / liga",
            "% Titular / convocado",
            "% Titular / jugado",
            "% Jugados / convocado",
            "% Minutos / convocado",
            "% Minutos / liga",
        ]
        radar_values = [
            pct_convocados_liga,
            pct_titular_convocado,
            pct_titular_jugado,
            pct_jugados_convocado,
            pct_minutos_convocado,
            pct_minutos_liga,
        ]
        radar_labels_closed = radar_labels + [radar_labels[0]]
        radar_values_closed = radar_values + [radar_values[0]]
        radar_colors = [_pct_color(v) for v in radar_values]

        radar_fig = go.Figure()
        radar_fig.add_trace(
            go.Scatterpolar(
                r=radar_values_closed,
                theta=radar_labels_closed,
                mode="lines",
                fill="toself",
                fillcolor="rgba(16, 54, 77, 0.18)",
                line=dict(color="#10364d", width=3),
                hovertemplate="%{theta}<br>%{r:.1f}%<extra></extra>",
                name="Perfil porcentual",
            )
        )
        for label, value, color in zip(radar_labels, radar_values, radar_colors):
            radar_fig.add_trace(
                go.Scatterpolar(
                    r=[value],
                    theta=[label],
                    mode="markers+text",
                    marker=dict(size=10, color=color, line=dict(color="#10364d", width=1.2)),
                    text=[f"{value:.1f}%"],
                    textposition="top center",
                    textfont=dict(size=11, color=color, family="Arial Black"),
                    hovertemplate=f"{label}<br>{value:.1f}%<extra></extra>",
                    showlegend=False,
                )
            )
        radar_fig.update_layout(
            template="plotly_white",
            margin=dict(l=40, r=40, t=40, b=25),
            height=460,
            showlegend=False,
            polar=dict(
                bgcolor="rgba(248,251,253,0.0)",
                radialaxis=dict(
                    range=[0, 100],
                    tickvals=[20, 40, 60, 80, 100],
                    tickfont=dict(size=10, color="#7b8da0"),
                    gridcolor="#dfe9f1",
                    linecolor="#dfe9f1",
                    angle=90,
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color="#10364d", family="Arial Black"),
                    gridcolor="#edf3f7",
                    linecolor="#dfe9f1",
                ),
            ),
            paper_bgcolor="rgba(248,251,253,0)",
            plot_bgcolor="rgba(248,251,253,0)",
        )
        st.plotly_chart(
            radar_fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

        gf_value_color = "#2f9e44"
        gc_value_color = "#d95f59"
        diff_value_color = "#2f9e44" if avg_diff_with_player > 0 else "#d95f59" if avg_diff_with_player < 0 else "#10364d"

        st.markdown(
            """
            <div class="player-impact-grid">
                <div class="player-impact-card" style="--impact-accent: #2f9e44;">
                    <div class="player-impact-label">GF medio con participación</div>
                    <div class="player-impact-value" style="color:"""
            + gf_value_color
            + """;">"""
            + f"{avg_gf_with_player:.2f}"
            + """</div>
                </div>
                <div class="player-impact-card" style="--impact-accent: #d95f59;">
                    <div class="player-impact-label">GC medio con participación</div>
                    <div class="player-impact-value" style="color:"""
            + gc_value_color
            + """;">"""
            + f"{avg_gc_with_player:.2f}"
            + """</div>
                </div>
                <div class="player-impact-card" style="--impact-accent: """
            + diff_value_color
            + """;">
                    <div class="player-impact-label">Dif. media con participación</div>
                    <div class="player-impact-value" style="color:"""
            + diff_value_color
            + """;">"""
            + f"{avg_diff_with_player:.2f}"
            + """</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        comments_key = f"player_comments_{selected_player}"
        comments_pdf_key = f"{comments_key}_pdf_snapshot"
        player_comments = st.text_area(
            "Comentarios del jugador",
            key=comments_key,
            height=140,
            placeholder="Espacio reservado para observaciones técnicas, evolución, puntos fuertes y aspectos a mejorar.",
        )
        st.session_state[comments_pdf_key] = player_comments
        player_comments_for_pdf = st.session_state.get(comments_pdf_key, "")
        include_impact_pdf = st.checkbox(
            "Incluir en el PDF el bloque de GF, GC y diferencia con el jugador en el campo",
            value=True,
            key=f"include_impact_pdf_{selected_player}",
        )
        pdf_bytes = build_player_report_pdf(
            player_name=display_name,
            team_name="CD Conxo Santiago Juvenil B",
            season_label="25-26",
            crest_path=CREST_PATH,
            brand_logo_path=LOGO_PATH,
            coach_name="Simon Goodey",
            designer_name="Ramón Codesido",
            dorsal_value=dorsal_value,
            posicion_global=posicion_global,
            posicion_especifica=posicion_especifica,
            fecha_nacimiento=fecha_nacimiento,
            edad_value=edad_value,
            metric_groups=metric_groups,
            radar_labels=radar_labels,
            radar_values=radar_values,
            impact_metrics=(
                [
                    ("GF medio con participación", f"{avg_gf_with_player:.2f}", "#2f9e44"),
                    ("GC medio con participación", f"{avg_gc_with_player:.2f}", "#d95f59"),
                    (
                        "Dif. media con participación",
                        f"{avg_diff_with_player:.2f}",
                        "#2f9e44" if avg_diff_with_player > 0 else "#d95f59" if avg_diff_with_player < 0 else "#10364d",
                    ),
                ]
                if include_impact_pdf
                else []
            ),
            comments_text=player_comments_for_pdf,
        )
        st.download_button(
            "Imprimir PDF",
            data=pdf_bytes,
            file_name=f"informe_jugador_{_normalize_key(display_name)}.pdf",
            mime="application/pdf",
            key=f"player_pdf_{selected_player}",
            use_container_width=False,
            type="primary",
        )


def main():
    st.set_page_config(page_title="Conxo Analytics", page_icon="⚽", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        section.main > div.block-container {
            max-width: 1440px;
        }
        section.main div[data-testid="stHorizontalBlock"] {
            gap: 0.95rem;
        }
        section.main div[data-testid="column"] {
            min-width: 0;
        }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(255, 214, 102, 0.14), transparent 24%),
                linear-gradient(180deg, #fffdf7 0%, #f8fbfd 100%);
        }
        .app-hero-shell {
            position: relative;
            margin: 0.15rem 0 1.4rem;
            padding-right: 8.8rem;
        }
        .page-section-title {
            display: flex;
            align-items: center;
            min-height: 3.6rem;
            margin: 0.35rem 0 1.05rem;
            color: #10364d;
            font-size: clamp(2rem, 3vw, 2.35rem);
            font-weight: 900;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            line-height: 1;
        }
        .page-subsection-title {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin: 1rem 0 0.9rem;
        }
        .page-subsection-title span {
            color: #10364d;
            font-size: 1.15rem;
            font-weight: 850;
            line-height: 1.1;
            white-space: nowrap;
        }
        .page-subsection-icon {
            font-size: 1.05rem;
            line-height: 1;
            transform: translateY(-1px);
        }
        .page-subsection-title::after {
            content: "";
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, rgba(16, 54, 77, 0.18) 0%, rgba(16, 54, 77, 0.05) 100%);
            transform: translateY(1px);
        }
        .app-hero-banner {
            position: relative;
            overflow: visible;
            background:
                radial-gradient(circle at 18% 18%, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0) 22%),
                linear-gradient(135deg, #10364d 0%, #1a5876 55%, #236b88 100%);
            border: 1px solid rgba(16, 54, 77, 0.08);
            border-radius: 28px;
            min-height: 164px;
            box-shadow:
                0 24px 44px rgba(16, 54, 77, 0.14),
                inset 0 1px 0 rgba(255,255,255,0.22);
            padding: 1.45rem 11rem 1.35rem 1.6rem;
        }
        .app-hero-banner::after {
            content: "";
            position: absolute;
            inset: auto 1.3rem 1rem 1.3rem;
            height: 1px;
            background: linear-gradient(90deg, rgba(255,255,255,0.0) 0%, rgba(255,255,255,0.24) 16%, rgba(255,255,255,0.10) 100%);
        }
        .app-hero-copy {
            position: relative;
            z-index: 2;
            max-width: 900px;
        }
        .app-hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.34rem 0.72rem;
            margin-bottom: 0.8rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.18);
            color: rgba(248, 251, 253, 0.84);
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            backdrop-filter: blur(6px);
        }
        .app-hero-title {
            color: #f8fbfd;
            font-size: clamp(2.5rem, 4vw, 3.9rem);
            line-height: 0.96;
            font-weight: 900;
            letter-spacing: -0.04em;
            margin-bottom: 0.6rem;
            text-shadow: 0 10px 24px rgba(5, 19, 27, 0.22);
        }
        .app-hero-subtitle {
            color: rgba(248, 251, 253, 0.84);
            font-size: 1.02rem;
            font-weight: 600;
            line-height: 1.45;
            max-width: 700px;
        }
        .app-hero-crest-wrap {
            position: absolute;
            right: 1.15rem;
            top: 2.25rem;
            width: 148px;
            height: 148px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 4;
            filter: drop-shadow(0 18px 28px rgba(10, 38, 53, 0.22));
        }
        .app-hero-crest-ring {
            position: absolute;
            inset: 0;
            border-radius: 50%;
            background:
                linear-gradient(145deg, rgba(255,255,255,0.95) 0%, rgba(224,233,239,0.95) 35%, rgba(192,206,217,0.95) 100%);
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.95),
                inset 0 -10px 18px rgba(16, 54, 77, 0.10),
                0 6px 20px rgba(16, 54, 77, 0.14);
        }
        .app-hero-crest-core {
            position: absolute;
            inset: 10px;
            border-radius: 50%;
            background:
                radial-gradient(circle at 30% 24%, rgba(255,255,255,0.96) 0%, rgba(247,250,252,0.94) 34%, rgba(225,235,242,0.92) 100%);
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.98),
                inset 0 -8px 14px rgba(16, 54, 77, 0.08);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .app-hero-crest-core img {
            width: 76%;
            height: 76%;
            object-fit: contain;
            display: block;
            transform: translateY(6px);
        }
        .app-hero-crest-gloss {
            position: absolute;
            top: 14px;
            left: 20px;
            width: 72px;
            height: 34px;
            border-radius: 999px;
            background: linear-gradient(180deg, rgba(255,255,255,0.72) 0%, rgba(255,255,255,0.08) 100%);
            filter: blur(0.4px);
            opacity: 0.95;
            transform: rotate(-12deg);
            pointer-events: none;
        }
        @media (max-width: 980px) {
            .app-hero-shell {
                padding-right: 0;
                padding-bottom: 1rem;
            }
            .page-section-title {
                font-size: 1.7rem;
                min-height: 3rem;
            }
            .page-subsection-title {
                align-items: flex-start;
                flex-wrap: wrap;
            }
            .page-subsection-title::after {
                flex-basis: 100%;
            }
            .page-subsection-title span {
                white-space: normal;
            }
            .matrix-insight-grid {
                grid-template-columns: 1fr;
            }
            .app-hero-banner {
                padding: 1.3rem 1.25rem 1.2rem 1.25rem;
                min-height: 170px;
            }
            .app-hero-crest-wrap {
                position: relative;
                right: auto;
                top: auto;
                margin: -1.2rem auto 0;
                width: 132px;
                height: 132px;
            }
            .app-hero-copy {
                text-align: left;
                max-width: none;
            }
        }
        @media (max-width: 1120px) {
            section.main div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
                gap: 0.8rem;
            }
            section.main div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                flex: 1 1 320px;
                min-width: min(100%, 320px);
            }
            .page-section-title {
                letter-spacing: 0.04em;
            }
            .stat-card {
                min-height: 108px;
            }
            .stat-card-value {
                font-size: 1.7rem;
            }
            .player-top-grid {
                grid-template-columns: 1fr;
            }
            .player-photo-box {
                min-height: 170px;
            }
            .player-metric-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .player-impact-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .matrix-insight-grid {
                grid-template-columns: 1fr;
            }
            .timeline-log-row {
                grid-template-columns: 102px 1fr;
                align-items: start;
            }
            .timeline-log-detail {
                grid-column: 1 / -1;
                padding-top: 0.2rem;
            }
            .classification-card {
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
            }
            .classification-header,
            .classification-row {
                min-width: 880px;
            }
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #fffdf6 0%, #f7fbfd 100%);
            border: 1px solid #d9e7ef;
            padding: 0.8rem 1rem;
            border-radius: 16px;
        }
        .stat-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbfd 100%);
            border: 1px solid #d9e7ef;
            border-radius: 18px;
            padding: 1rem 1rem 1.1rem 1rem;
            min-height: 118px;
            box-shadow: 0 10px 28px rgba(16, 54, 77, 0.08);
            margin-bottom: 0.8rem;
            position: relative;
            overflow: hidden;
        }
        .stat-card::before {
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            height: 5px;
            width: 100%;
            background: linear-gradient(90deg, #0f5d7a 0%, #caa84b 100%);
        }
        .stat-card-label {
            font-size: 0.95rem;
            color: #66788d;
            font-weight: 600;
            margin-bottom: 0.85rem;
        }
        .stat-card-value {
            font-size: 2rem;
            line-height: 1;
            font-weight: 700;
            color: #10364d;
        }
        div[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top right, rgba(216,178,77,0.18) 0%, rgba(216,178,77,0) 24%),
                linear-gradient(180deg, #0d2d40 0%, #123d56 45%, #184c69 100%);
        }
        div[data-testid="stSidebar"] * {
            color: #f3f7fa;
        }
        .sidebar-brand-shell {
            margin: 0.35rem 0 1rem;
            padding: 0.95rem 0.9rem 1rem;
            border-radius: 24px;
            background:
                radial-gradient(circle at 18% 15%, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0.02) 28%),
                linear-gradient(180deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.04) 100%);
            border: 1px solid rgba(255,255,255,0.11);
            box-shadow:
                0 18px 28px rgba(5, 22, 31, 0.18),
                inset 0 1px 0 rgba(255,255,255,0.08);
            backdrop-filter: blur(10px);
        }
        .sidebar-logo-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.8rem;
            align-items: center;
            margin-bottom: 0.9rem;
        }
        .sidebar-logo-badge {
            position: relative;
            min-height: 92px;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(233,240,245,0.96) 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow:
                0 10px 18px rgba(8, 29, 41, 0.16),
                inset 0 1px 0 rgba(255,255,255,0.9);
            overflow: hidden;
        }
        .sidebar-logo-badge::after {
            content: "";
            position: absolute;
            top: 8px;
            left: 12px;
            width: 42px;
            height: 16px;
            border-radius: 999px;
            background: linear-gradient(180deg, rgba(255,255,255,0.82) 0%, rgba(255,255,255,0.08) 100%);
            transform: rotate(-15deg);
            pointer-events: none;
        }
        .sidebar-logo-badge--crest {
            transform: translateY(4px);
        }
        .sidebar-logo-badge img {
            width: 72%;
            max-height: 72px;
            object-fit: contain;
            display: block;
        }
        .sidebar-brand-title {
            color: #111111;
            font-size: 0.94rem;
            line-height: 1.2;
            font-weight: 900;
            margin-bottom: 0.3rem;
            letter-spacing: -0.02em;
            white-space: nowrap;
        }
        .sidebar-brand-subtitle {
            color: #111111;
            font-size: 0.82rem;
            line-height: 1.45;
            margin-bottom: 0.95rem;
            font-weight: 700;
        }
        .sidebar-competition-card {
            display: grid;
            grid-template-columns: 62px minmax(0, 1fr);
            gap: 0.75rem;
            align-items: center;
            padding: 0.75rem;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(248,251,253,0.95) 0%, rgba(230,238,244,0.92) 100%);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.82);
        }
        .sidebar-competition-logo {
            width: 62px;
            height: 62px;
            border-radius: 18px;
            background: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 10px 18px rgba(16, 54, 77, 0.10);
        }
        .sidebar-competition-logo img {
            width: 76%;
            height: 76%;
            object-fit: contain;
            display: block;
        }
        .sidebar-competition-meta {
            min-width: 0;
        }
        .sidebar-competition-label {
            color: #587389;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.28rem;
        }
        .sidebar-competition-name {
            color: #10364d;
            font-size: 0.88rem;
            font-weight: 900;
            line-height: 1.2;
            margin-bottom: 0.22rem;
        }
        .sidebar-competition-org {
            color: #5e7386;
            font-size: 0.74rem;
            font-weight: 700;
            line-height: 1.3;
        }
        .sidebar-nav-title {
            margin: 0.2rem 0 0.55rem;
            color: rgba(243, 247, 250, 0.72);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        div[data-testid="stSidebar"] [data-testid="stPills"] {
            margin-top: 0.15rem;
        }
        div[data-testid="stSidebar"] [data-testid="stPills"] > div {
            display: grid !important;
            grid-template-columns: 1fr !important;
            gap: 0.42rem;
        }
        div[data-testid="stSidebar"] [data-testid="stPills"] button {
            width: 100%;
            justify-content: flex-start;
            border-radius: 18px;
            padding: 1rem 1.05rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.07) 0%, rgba(255,255,255,0.035) 100%);
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow:
                0 10px 18px rgba(6, 24, 34, 0.10),
                inset 0 1px 0 rgba(255,255,255,0.06);
            transition: background 120ms ease, transform 120ms ease, color 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
            color: #f3f7fa !important;
            font-size: 1.2rem;
            font-weight: 900;
            line-height: 1.08;
            min-height: 66px;
            letter-spacing: -0.01em;
        }
        div[data-testid="stSidebar"] [data-testid="stPills"] button:hover {
            background: linear-gradient(180deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.05) 100%);
            transform: translateX(2px);
            color: #ffffff !important;
            border-color: rgba(255,255,255,0.16);
        }
        div[data-testid="stSidebar"] [data-testid="stPills"] button[aria-pressed="true"] {
            background: linear-gradient(180deg, rgba(246,248,251,0.98) 0%, rgba(224,231,239,0.97) 100%);
            border-color: rgba(255,255,255,0.08);
            box-shadow:
                inset 5px 0 0 #caa84b,
                0 14px 22px rgba(4, 18, 26, 0.12);
            color: #10364d !important;
        }
        .sidebar-copyright {
            margin-top: 1rem;
            padding: 0.9rem 0.95rem 0;
            border-top: 1px solid rgba(255,255,255,0.10);
            color: #111111;
            font-size: 0.95rem;
            font-weight: 800;
            line-height: 1.35;
        }
        div[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {
            color: rgba(243, 247, 250, 0.88);
            font-weight: 700;
        }
        div[data-testid="stSelectbox"] label[data-testid="stWidgetLabel"] {
            color: #10364d;
            font-weight: 700;
            font-size: 1rem;
            letter-spacing: 0.01em;
        }
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background: linear-gradient(180deg, #edf4fb 0%, #dfeaf5 100%);
            border: 1px solid #bdd4e5;
            border-radius: 16px;
            min-height: 58px;
            box-shadow: 0 10px 24px rgba(16, 54, 77, 0.08);
        }
        div[data-testid="stSelectbox"] [data-baseweb="select"] span {
            color: #10364d;
            font-size: 1.1rem;
            font-weight: 900;
        }
        .match-hero {
            background: linear-gradient(135deg, #10364d 0%, #1b5977 100%);
            border-radius: 20px;
            padding: 1.2rem 1.25rem;
            color: #f8fbfd;
            box-shadow: 0 18px 30px rgba(16, 54, 77, 0.14);
            min-height: 170px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        .match-hero-inner {
            width: 100%;
        }
        .match-hero-round {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: rgba(248, 251, 253, 0.72);
            margin-bottom: 0.45rem;
        }
        .match-hero-score {
            font-size: 2.8rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 0.45rem;
        }
        .match-hero-fixture {
            font-size: 1.18rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .match-hero-status {
            font-size: 0.92rem;
            color: rgba(248, 251, 253, 0.82);
        }
        .crest-team-name {
            margin-top: 0.45rem;
            text-align: center;
            color: #10364d;
            font-weight: 800;
            font-size: 0.92rem;
            line-height: 1.2;
            min-height: 2.3rem;
            display: flex;
            align-items: flex-start;
            justify-content: center;
        }
        .player-chip, .goal-chip {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbfd 100%);
            border: 1px solid #d9e7ef;
            border-radius: 16px;
            padding: 0.7rem 0.85rem;
            margin-bottom: 0.55rem;
        }
        .player-chip-number, .goal-chip-minute {
            width: 42px;
            height: 42px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background: linear-gradient(135deg, #caa84b 0%, #edd289 100%);
            color: #10364d;
            font-weight: 800;
            font-size: 1rem;
            flex-shrink: 0;
        }
        .player-chip-number.bench {
            background: linear-gradient(135deg, #88a9bf 0%, #c2d7e5 100%);
            color: #10364d;
        }
        .goal-chip-minute.rival {
            background: linear-gradient(135deg, #c95d5d 0%, #e69b9b 100%);
            color: #fff7f7;
        }
        .goal-chip-minute.yellow {
            background: linear-gradient(135deg, #f0c94f 0%, #ffe189 100%);
            color: #10364d;
        }
        .goal-chip-minute.red {
            background: linear-gradient(135deg, #b83e3e 0%, #e07777 100%);
            color: #fff7f7;
        }
        .compact-event-chip {
            padding: 0.65rem 0.75rem;
        }
        .compact-event-chip .goal-chip-name {
            font-size: 0.92rem;
        }
        .compact-event-chip .goal-chip-type {
            font-size: 0.8rem;
        }
        .player-chip-body {
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
        }
        .player-chip-name, .goal-chip-name {
            color: #10364d;
            font-weight: 700;
            line-height: 1.2;
            font-size: 0.96rem;
        }
        .player-chip-meta {
            color: #6b7c8f;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .goal-chip-body {
            display: flex;
            flex-direction: column;
            gap: 0.1rem;
        }
        .goal-chip-type {
            font-size: 0.84rem;
            color: #6b7c8f;
            font-weight: 600;
        }
        .change-chip {
            display: flex;
            align-items: flex-start;
            gap: 0.85rem;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbfd 100%);
            border: 1px solid #d9e7ef;
            border-radius: 16px;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.55rem;
        }
        .change-chip-minute {
            width: 42px;
            height: 42px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background: linear-gradient(135deg, #10364d 0%, #1b5977 100%);
            color: #f8fbfd;
            font-weight: 800;
            flex-shrink: 0;
        }
        .change-chip-body {
            display: flex;
            flex-direction: column;
            gap: 0.18rem;
        }
        .change-chip-out, .change-chip-in {
            font-weight: 700;
            line-height: 1.25;
            font-size: 0.96rem;
        }
        .change-chip-out {
            color: #b54747;
        }
        .change-chip-in {
            color: #2f8c55;
        }
        .people-empty {
            border: 1px dashed #c9dbe7;
            border-radius: 16px;
            padding: 0.9rem 1rem;
            color: #6b7c8f;
            background: rgba(248, 251, 253, 0.9);
        }
        .stat-card.compact-card {
            min-height: 96px;
            padding: 0.9rem 0.75rem;
            background: linear-gradient(180deg, #fcfeff 0%, #f1f6fb 100%);
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .stat-card.compact-card .stat-card-label {
            font-size: 0.86rem;
            margin-bottom: 0.38rem;
            text-align: center;
        }
        .stat-card.compact-card .stat-card-value {
            font-size: 1.2rem;
            line-height: 1.05;
            word-break: break-word;
            text-align: center;
        }
        .stat-card.detail-card {
            min-height: 112px;
        }
        .stat-card.detail-card .stat-card-value {
            font-size: 0.92rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .goals-section-spacer {
            height: 0.8rem;
        }
        .ranking-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbfd 100%);
            border: 1px solid #d9e7ef;
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 10px 28px rgba(16, 54, 77, 0.08);
            margin-bottom: 0.85rem;
            background-blend-mode: lighten;
        }
        .ranking-card-title {
            padding: 0.85rem 1rem;
            background: linear-gradient(90deg, #10364d 0%, #1b5977 100%);
            color: #f8fbfd;
            font-size: 0.98rem;
            font-weight: 800;
            letter-spacing: 0.01em;
        }
        .ranking-card-list {
            padding: 0.55rem 0.7rem 0.75rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(248,251,253,0.94) 100%);
        }
        .ranking-item {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.58rem 0.25rem;
            border-bottom: 1px solid #ebf2f7;
        }
        .ranking-item:last-child {
            border-bottom: none;
        }
        .ranking-pos {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #d8b24d 0%, #edd289 100%);
            color: #10364d;
            font-size: 0.84rem;
            font-weight: 800;
            flex-shrink: 0;
        }
        .ranking-body {
            flex: 1;
            min-width: 0;
        }
        .ranking-name {
            color: #10364d;
            font-size: 0.92rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .ranking-value {
            color: #10364d;
            font-size: 0.92rem;
            font-weight: 800;
            white-space: nowrap;
        }
        .ranking-value-wrap {
            display: flex;
            align-items: baseline;
            gap: 0.45rem;
            white-space: nowrap;
            margin-left: auto;
        }
        .ranking-share {
            color: #6b7f92;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.01em;
        }
        .matrix-insight-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin: 0.9rem 0 1.25rem;
        }
        .matrix-insight-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbfd 100%);
            border: 1px solid #d9e7ef;
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 10px 28px rgba(16, 54, 77, 0.08);
        }
        .matrix-insight-card-title {
            padding: 0.82rem 1rem;
            background: linear-gradient(90deg, #10364d 0%, #1b5977 68%, #d8b24d 100%);
            color: #f8fbfd;
            font-size: 0.98rem;
            font-weight: 850;
            letter-spacing: 0.01em;
        }
        .matrix-insight-list {
            padding: 0.65rem 0.8rem 0.8rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.94) 0%, rgba(248,251,253,0.97) 100%);
        }
        .matrix-insight-item {
            display: flex;
            align-items: center;
            gap: 0.78rem;
            padding: 0.7rem 0.15rem;
            border-bottom: 1px solid #ebf2f7;
        }
        .matrix-insight-item:last-child {
            border-bottom: none;
        }
        .matrix-insight-rank {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #d8b24d 0%, #f0daa0 100%);
            color: #10364d;
            font-size: 0.88rem;
            font-weight: 850;
            flex-shrink: 0;
        }
        .matrix-insight-body {
            flex: 1;
            min-width: 0;
        }
        .matrix-insight-main {
            color: #10364d;
            font-size: 0.95rem;
            font-weight: 800;
            line-height: 1.2;
        }
        .matrix-insight-sub {
            margin-top: 0.16rem;
            color: #6b7f92;
            font-size: 0.82rem;
            font-weight: 600;
            line-height: 1.25;
        }
        .matrix-insight-count {
            color: #10364d;
            font-size: 1rem;
            font-weight: 900;
            white-space: nowrap;
        }
        .timeline-svg-wrap {
            width: 100%;
            overflow-x: auto;
            padding-bottom: 0.45rem;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: thin;
        }
        .timeline-match-svg {
            display: block;
            width: max(100%, 1280px);
            min-width: 1280px;
            height: auto;
        }
        .timeline-log-card {
            display: flex;
            flex-direction: column;
            gap: 0.55rem;
            padding: 0.2rem 0 0.35rem;
        }
        .timeline-log-row {
            display: grid;
            grid-template-columns: 120px 220px 1fr;
            gap: 0.85rem;
            align-items: center;
            background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,251,253,0.98) 100%);
            border: 1px solid #d9e7ef;
            border-radius: 16px;
            padding: 0.8rem 0.95rem;
            box-shadow: 0 10px 20px rgba(16, 54, 77, 0.05);
        }
        .timeline-log-minute {
            color: #10364d;
            font-size: 0.92rem;
            font-weight: 900;
            white-space: nowrap;
        }
        .timeline-log-action {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            color: #10364d;
            font-size: 0.92rem;
            font-weight: 800;
            white-space: nowrap;
        }
        .timeline-log-icon {
            font-size: 1rem;
            line-height: 1;
        }
        .timeline-log-label {
            color: #10364d;
            font-weight: 800;
        }
        .timeline-log-detail {
            color: #4f6477;
            font-size: 0.9rem;
            font-weight: 700;
            line-height: 1.35;
            display: flex;
            align-items: center;
            gap: 0.48rem;
            flex-wrap: wrap;
        }
        .timeline-log-name {
            color: #10364d;
            font-weight: 800;
        }
        .timeline-log-crest {
            width: 18px;
            height: 18px;
            object-fit: contain;
            flex-shrink: 0;
            filter: drop-shadow(0 1px 1px rgba(16, 54, 77, 0.12));
        }
        .timeline-log-name--in {
            color: #2f9e44;
        }
        .timeline-log-name--out {
            color: #d95f59;
        }
        .timeline-log-name--goal-conxo {
            color: #2f9e44;
        }
        .timeline-log-name--goal-rival {
            color: #c55252;
        }
        .timeline-log-name--yellow {
            color: #b88909;
        }
        .timeline-log-name--red {
            color: #c55252;
        }
        .timeline-log-sep {
            color: #8ba0b2;
            margin: 0 0.28rem;
            font-weight: 700;
        }
        .scroll-table-shell {
            background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,251,253,0.98) 100%);
            border: 1px solid #d9e7ef;
            border-radius: 22px;
            box-shadow: 0 16px 32px rgba(16, 54, 77, 0.07);
            padding: 0.95rem;
            margin: 0.75rem 0 1rem;
        }
        .scroll-table-kicker {
            color: #10364d;
            font-size: 1rem;
            font-weight: 850;
            margin-bottom: 0.75rem;
        }
        .scroll-table-wrap {
            overflow-x: auto;
            overflow-y: auto;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: thin;
            padding-bottom: 0.2rem;
            max-height: min(70vh, 620px);
            border: 1px solid #d9e7ef;
            border-radius: 20px;
            background: #ffffff;
        }
        .scroll-table {
            border-collapse: separate;
            border-spacing: 0;
            min-width: max-content;
            width: max-content;
        }
        .scroll-table-head,
        .scroll-table-cell {
            padding: 0.62rem 0.7rem;
            border-right: 1px solid #e6eef4;
            border-bottom: 1px solid #e6eef4;
            text-align: center;
            font-size: 0.85rem;
            line-height: 1.15;
            white-space: nowrap;
            background: #ffffff;
        }
        .scroll-table thead .scroll-table-head {
            position: sticky;
            top: 0;
            z-index: 9;
            background: linear-gradient(180deg, #10364d 0%, #1b5977 100%);
            color: #f8fbfd;
            font-weight: 800;
            box-shadow: inset 0 -1px 0 rgba(255, 255, 255, 0.08), 0 4px 10px rgba(16, 54, 77, 0.12);
        }
        .scroll-table-col-head {
            min-width: 70px;
        }
        .scroll-table-number {
            font-weight: 800;
        }
        .scroll-table-sticky-player {
            position: sticky;
            left: 0;
            z-index: 8;
            min-width: 240px;
            max-width: 240px;
            text-align: left;
            font-weight: 800;
            color: #10364d;
            background: linear-gradient(180deg, #fdfefe 0%, #f3f8fb 100%);
        }
        .scroll-table thead .scroll-table-sticky-player {
            z-index: 12;
        }
        .scroll-table-sticky-total {
            position: sticky;
            left: 240px;
            z-index: 8;
            min-width: 88px;
            background: #ffffff;
        }
        .scroll-table thead .scroll-table-sticky-total {
            z-index: 11;
        }
        .scroll-table-sticky-summary-1,
        .scroll-table-sticky-summary-2,
        .scroll-table-sticky-summary-3 {
            position: sticky;
            z-index: 8;
            min-width: 112px;
            background: #ffffff;
        }
        .scroll-table-sticky-summary-1 { left: 240px; }
        .scroll-table-sticky-summary-2 { left: 352px; }
        .scroll-table-sticky-summary-3 { left: 464px; }
        .scroll-table thead .scroll-table-sticky-summary-1,
        .scroll-table thead .scroll-table-sticky-summary-2,
        .scroll-table thead .scroll-table-sticky-summary-3 {
            z-index: 11;
        }
        .scroll-table-player-name {
            font-weight: 700;
            color: #10364d;
        }
        .scroll-table--changes .scroll-table-sticky-player {
            overflow: hidden;
        }
        .scroll-table--changes .scroll-table-player-name {
            display: block;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .scroll-table--changes .scroll-table-sticky-summary-1,
        .scroll-table--changes .scroll-table-sticky-summary-2,
        .scroll-table--changes .scroll-table-sticky-summary-3 {
            min-width: 76px;
            width: 76px;
            max-width: 76px;
        }
        .scroll-table--changes .scroll-table-sticky-summary-1 { left: 240px; }
        .scroll-table--changes .scroll-table-sticky-summary-2 { left: 316px; }
        .scroll-table--changes .scroll-table-sticky-summary-3 { left: 392px; }
        .scroll-table-summary-head {
            padding: 0;
            vertical-align: bottom;
        }
        .scroll-table-summary-head span {
            display: flex;
            align-items: flex-end;
            justify-content: center;
            min-height: 120px;
            padding: 0.45rem 0.2rem 0.55rem;
            font-size: 0.68rem;
            line-height: 1.05;
            writing-mode: vertical-rl;
            transform: rotate(180deg);
            text-align: left;
        }
        .scroll-table-player-head {
            min-width: 84px;
            width: 84px;
            max-width: 84px;
            padding: 0;
            vertical-align: bottom;
        }
        .scroll-table-player-head span {
            display: flex;
            align-items: flex-end;
            justify-content: center;
            min-height: 150px;
            padding: 0.45rem 0.2rem 0.55rem;
            font-size: 0.72rem;
            line-height: 1.05;
            writing-mode: vertical-rl;
            transform: rotate(180deg);
            text-align: left;
        }
        .player-profile-shell {
            background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,251,253,0.98) 100%);
            border: 1px solid #d9e7ef;
            border-radius: 24px;
            padding: 1.2rem;
            box-shadow: 0 16px 36px rgba(16, 54, 77, 0.08);
            margin-bottom: 1rem;
        }
        .player-top-grid {
            display: grid;
            grid-template-columns: 180px 1fr;
            gap: 1rem;
            align-items: stretch;
        }
        .player-photo-box {
            min-height: 220px;
            border-radius: 22px;
            border: 1px dashed #bdd4e5;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(242,247,251,0.95) 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            color: #6b7c8f;
            font-weight: 700;
            text-align: center;
            padding: 1rem;
        }
        .player-photo-icon {
            font-size: 2rem;
            margin-bottom: 0.45rem;
        }
        .player-bio-card {
            background: linear-gradient(135deg, #10364d 0%, #1b5977 100%);
            color: #f8fbfd;
            border-radius: 22px;
            padding: 1.1rem 1.15rem;
            box-shadow: 0 16px 32px rgba(16, 54, 77, 0.14);
        }
        .player-bio-dorsal {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 52px;
            height: 52px;
            padding: 0 0.9rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #d8b24d 0%, #edd289 100%);
            color: #10364d;
            font-size: 1.4rem;
            font-weight: 900;
            margin-bottom: 0.8rem;
        }
        .player-bio-name {
            font-size: 1.9rem;
            font-weight: 900;
            line-height: 1.08;
            margin-bottom: 1rem;
        }
        .player-bio-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem 1rem;
        }
        .player-bio-field {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 0.72rem 0.78rem;
        }
        .player-bio-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: rgba(248,251,253,0.72);
            margin-bottom: 0.28rem;
            font-weight: 700;
        }
        .player-bio-value {
            font-size: 1rem;
            color: #f8fbfd;
            font-weight: 800;
            line-height: 1.2;
        }
        .player-metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 1rem;
        }
        .player-metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #f4f9fc 100%);
            border: 1px solid #d9e7ef;
            border-radius: 18px;
            padding: 0.95rem 0.9rem;
            box-shadow: 0 10px 24px rgba(16, 54, 77, 0.06);
            text-align: center;
            min-height: 96px;
            position: relative;
            overflow: hidden;
        }
        .player-metric-card::before {
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            height: 4px;
            width: 100%;
            background: linear-gradient(90deg, #0f5d7a 0%, #d8b24d 100%);
        }
        .player-metric-label {
            color: #6b7c8f;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
            line-height: 1.2;
        }
        .player-metric-value {
            color: #10364d;
            font-size: 1.5rem;
            font-weight: 900;
            line-height: 1.05;
        }
        .player-svg-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbfd 100%);
            border: 1px solid #d9e7ef;
            border-radius: 20px;
            padding: 0.9rem 1rem;
            box-shadow: 0 10px 24px rgba(16, 54, 77, 0.06);
            margin-top: 1rem;
            margin-bottom: 0.9rem;
            overflow-x: auto;
        }
        .player-svg-card svg {
            width: 100%;
            min-width: 760px;
            height: auto;
            display: block;
        }
        .player-radar-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbfd 100%);
            border: 1px solid #d9e7ef;
            border-radius: 22px;
            padding: 0.75rem 0.9rem 0.4rem 0.9rem;
            box-shadow: 0 10px 24px rgba(16, 54, 77, 0.06);
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .player-impact-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 0.25rem;
            margin-bottom: 1rem;
        }
        .player-impact-card {
            background: linear-gradient(180deg, #ffffff 0%, #f4f9fc 100%);
            border: 1px solid #d9e7ef;
            border-radius: 18px;
            padding: 1rem 0.95rem;
            box-shadow: 0 10px 24px rgba(16, 54, 77, 0.06);
            text-align: center;
            position: relative;
            overflow: hidden;
            min-height: 104px;
        }
        .player-impact-card::before {
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            height: 4px;
            width: 100%;
            background: linear-gradient(90deg, var(--impact-accent, #0f5d7a) 0%, rgba(216, 178, 77, 0.92) 100%);
        }
        .player-impact-label {
            color: #6b7c8f;
            font-size: 0.84rem;
            font-weight: 800;
            line-height: 1.25;
            margin-bottom: 0.48rem;
        }
        .player-impact-value {
            color: #10364d;
            font-size: 1.5rem;
            font-weight: 900;
            line-height: 1.05;
        }
        div[data-testid="stDownloadButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #b92f2f 0%, #e14b4b 100%);
            color: #ffffff;
            border: 1px solid rgba(126, 20, 20, 0.55);
            border-radius: 14px;
            font-weight: 800;
            padding: 0.62rem 1.15rem;
            box-shadow: 0 12px 24px rgba(185, 47, 47, 0.18);
        }
        div[data-testid="stDownloadButton"] > button[kind="primary"]:hover {
            border-color: rgba(126, 20, 20, 0.7);
            color: #ffffff;
            background: linear-gradient(135deg, #a72828 0%, #d84040 100%);
        }
        .classification-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbfd 100%);
            border: 1px solid #d9e7ef;
            border-radius: 20px;
            box-shadow: 0 10px 24px rgba(16, 54, 77, 0.06);
            overflow: hidden;
            margin-top: 0.7rem;
            margin-bottom: 1rem;
        }
        .classification-header, .classification-row {
            display: grid;
            grid-template-columns: 64px minmax(280px, 1.6fr) repeat(5, minmax(70px, 0.55fr));
            gap: 0.5rem;
            align-items: center;
            padding: 0.78rem 1rem;
        }
        .classification-header {
            background: linear-gradient(90deg, #10364d 0%, #1b5977 100%);
            color: #f8fbfd;
            font-size: 0.86rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .classification-row {
            border-bottom: 1px solid #ebf2f7;
            color: #10364d;
            background: rgba(255,255,255,0.92);
        }
        .classification-row:last-child {
            border-bottom: none;
        }
        .classification-row.conxo {
            background: linear-gradient(90deg, rgba(216,178,77,0.14) 0%, rgba(255,255,255,0.95) 55%);
        }
        .classification-pos {
            font-weight: 800;
            text-align: center;
        }
        .classification-team {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            min-width: 0;
        }
        .classification-crest {
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .classification-name {
            font-weight: 700;
            line-height: 1.15;
        }
        .classification-stat {
            text-align: center;
            font-weight: 700;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-top: 1rem;
                padding-bottom: 1.3rem;
            }
            section.main > div.block-container {
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }
            section.main div[data-testid="stHorizontalBlock"] {
                gap: 0.72rem;
            }
            section.main div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                flex: 1 1 100%;
                min-width: 100%;
            }
            div[data-testid="stTabs"] [data-baseweb="tab-list"] {
                flex-wrap: wrap;
                gap: 0.45rem;
            }
            div[data-testid="stTabs"] [data-baseweb="tab"] {
                padding-left: 0.8rem;
                padding-right: 0.8rem;
                min-height: 44px;
            }
            .page-section-title {
                font-size: 1.45rem;
                line-height: 1.05;
                letter-spacing: 0.03em;
                margin-bottom: 0.8rem;
            }
            .page-subsection-title span {
                font-size: 1rem;
            }
            .app-hero-banner {
                border-radius: 22px;
                min-height: auto;
                padding: 1rem 1rem 1.1rem;
            }
            .app-hero-kicker {
                font-size: 0.7rem;
                padding: 0.3rem 0.6rem;
            }
            .app-hero-title {
                font-size: clamp(2rem, 10vw, 2.7rem);
                line-height: 0.98;
            }
            .app-hero-subtitle {
                font-size: 0.92rem;
                line-height: 1.4;
            }
            .app-hero-crest-wrap {
                width: 108px;
                height: 108px;
                margin-top: -0.75rem;
            }
            .stat-card,
            .stat-card.compact-card,
            .stat-card.detail-card {
                min-height: auto;
                padding: 0.9rem 0.9rem 1rem;
                border-radius: 16px;
            }
            .stat-card-label,
            .stat-card.compact-card .stat-card-label {
                font-size: 0.88rem;
                margin-bottom: 0.6rem;
            }
            .stat-card-value,
            .stat-card.compact-card .stat-card-value {
                font-size: 1.45rem;
                line-height: 1.05;
                word-break: break-word;
            }
            .stat-card.detail-card .stat-card-value {
                white-space: normal;
                overflow: visible;
                text-overflow: unset;
                line-height: 1.25;
            }
            .match-hero {
                min-height: auto;
                padding: 1rem 0.9rem;
                border-radius: 18px;
            }
            .match-hero-score {
                font-size: 2.2rem;
            }
            .match-hero-fixture {
                font-size: 1rem;
                line-height: 1.25;
            }
            .match-hero-status {
                font-size: 0.86rem;
                line-height: 1.35;
            }
            .player-chip,
            .goal-chip,
            .change-chip {
                border-radius: 16px;
                padding: 0.75rem;
            }
            .player-chip-name,
            .goal-chip-name {
                font-size: 0.95rem;
            }
            .goal-chip-type,
            .player-chip-meta,
            .change-chip-out,
            .change-chip-in {
                font-size: 0.82rem;
                line-height: 1.35;
            }
            .timeline-svg-wrap {
                margin: 0 -0.2rem;
                padding-bottom: 0.45rem;
            }
            .timeline-match-svg {
                width: 1500px;
                min-width: 1500px;
            }
            .timeline-log-card {
                gap: 0.7rem;
            }
            .timeline-log-row {
                grid-template-columns: 1fr;
                gap: 0.3rem;
                padding: 0.78rem 0.82rem;
            }
            .timeline-log-minute,
            .timeline-log-action,
            .timeline-log-detail {
                white-space: normal;
            }
            .timeline-log-action {
                font-size: 0.88rem;
            }
            .timeline-log-detail {
                font-size: 0.84rem;
                padding-top: 0.08rem;
            }
            .scroll-table-shell {
                padding: 0.8rem;
                border-radius: 18px;
            }
            .scroll-table-kicker {
                font-size: 0.95rem;
            }
            .scroll-table-head,
            .scroll-table-cell {
                padding: 0.54rem 0.56rem;
                font-size: 0.78rem;
            }
            .scroll-table-wrap {
                max-height: min(64vh, 520px);
            }
            .scroll-table-sticky-player {
                min-width: 190px;
                max-width: 190px;
            }
            .scroll-table-sticky-total {
                left: 190px;
                min-width: 78px;
            }
            .scroll-table-sticky-summary-1 { left: 190px; min-width: 94px; }
            .scroll-table-sticky-summary-2 { left: 284px; min-width: 94px; }
            .scroll-table-sticky-summary-3 { left: 378px; min-width: 94px; }
            .scroll-table--changes .scroll-table-sticky-summary-1,
            .scroll-table--changes .scroll-table-sticky-summary-2,
            .scroll-table--changes .scroll-table-sticky-summary-3 {
                min-width: 62px;
                width: 62px;
                max-width: 62px;
            }
            .scroll-table--changes .scroll-table-sticky-summary-1 { left: 190px; }
            .scroll-table--changes .scroll-table-sticky-summary-2 { left: 252px; }
            .scroll-table--changes .scroll-table-sticky-summary-3 { left: 314px; }
            .scroll-table-summary-head span {
                min-height: 108px;
                font-size: 0.62rem;
            }
            .scroll-table--changes .scroll-table-sticky-player {
                min-width: 176px;
                max-width: 176px;
            }
            .scroll-table--changes .scroll-table-player-name {
                font-size: 0.76rem;
            }
            .scroll-table--changes .scroll-table-sticky-summary-1,
            .scroll-table--changes .scroll-table-sticky-summary-2,
            .scroll-table--changes .scroll-table-sticky-summary-3 {
                position: static;
                left: auto;
                min-width: 72px;
                width: 72px;
                max-width: 72px;
            }
            .scroll-table--changes thead .scroll-table-sticky-summary-1,
            .scroll-table--changes thead .scroll-table-sticky-summary-2,
            .scroll-table--changes thead .scroll-table-sticky-summary-3 {
                position: sticky;
                top: 0;
                left: auto;
                z-index: 10;
            }
            .scroll-table-col-head {
                min-width: 62px;
            }
            .scroll-table-player-head {
                min-width: 72px;
                width: 72px;
                max-width: 72px;
            }
            .scroll-table-player-head span {
                min-height: 132px;
                font-size: 0.68rem;
            }
            .player-profile-shell {
                border-radius: 20px;
                padding: 0.9rem;
            }
            .player-photo-box {
                min-height: 140px;
            }
            .player-bio-card {
                border-radius: 18px;
                padding: 0.95rem;
            }
            .player-bio-dorsal {
                min-width: 46px;
                height: 46px;
                font-size: 1.2rem;
                margin-bottom: 0.65rem;
            }
            .player-bio-name {
                font-size: 1.55rem;
                margin-bottom: 0.8rem;
            }
            .player-bio-grid {
                grid-template-columns: 1fr;
                gap: 0.65rem;
            }
            .player-metric-grid,
            .player-impact-grid {
                grid-template-columns: 1fr;
                gap: 0.7rem;
            }
            .player-metric-card,
            .player-impact-card {
                min-height: auto;
                padding: 0.85rem 0.8rem 0.92rem;
            }
            .player-metric-value,
            .player-impact-value {
                font-size: 1.35rem;
            }
            .player-svg-card {
                padding: 0.8rem 0.75rem;
                border-radius: 18px;
            }
            .player-svg-card svg {
                min-width: 540px;
            }
            .player-radar-card {
                padding: 0.65rem 0.65rem 0.2rem;
                border-radius: 18px;
            }
            .ranking-card-title,
            .matrix-insight-card-title {
                font-size: 0.9rem;
                padding: 0.78rem 0.88rem;
            }
            .ranking-item,
            .matrix-insight-item {
                align-items: flex-start;
            }
            .ranking-value-wrap {
                flex-wrap: wrap;
                justify-content: flex-end;
                row-gap: 0.12rem;
            }
            .matrix-insight-main {
                font-size: 0.9rem;
            }
            .matrix-insight-sub {
                font-size: 0.78rem;
            }
            div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
                min-height: 52px;
                border-radius: 14px;
            }
            div[data-testid="stSelectbox"] [data-baseweb="select"] span {
                font-size: 1rem;
            }
            div[data-testid="stTextArea"] textarea {
                min-height: 130px;
            }
            div[data-testid="stDownloadButton"] > button[kind="primary"] {
                width: 100%;
                min-height: 48px;
            }
        }
        @media (max-width: 480px) {
            .page-section-title {
                font-size: 1.28rem;
            }
            .app-hero-title {
                font-size: 1.8rem;
            }
            .app-hero-subtitle {
                font-size: 0.87rem;
            }
            .player-bio-name {
                font-size: 1.35rem;
            }
            .stat-card-value,
            .stat-card.compact-card .stat-card-value,
            .player-metric-value,
            .player-impact-value {
                font-size: 1.22rem;
            }
            .timeline-log-minute {
                font-size: 0.84rem;
            }
            .scroll-table-sticky-player {
                min-width: 168px;
                max-width: 168px;
            }
            .scroll-table-wrap {
                max-height: min(58vh, 460px);
            }
            .scroll-table-sticky-total {
                left: 168px;
            }
            .scroll-table--changes .scroll-table-sticky-player {
                min-width: 158px;
                max-width: 158px;
            }
            .scroll-table--changes .scroll-table-player-name {
                font-size: 0.72rem;
            }
            .scroll-table--changes .scroll-table-sticky-summary-1,
            .scroll-table--changes .scroll-table-sticky-summary-2,
            .scroll-table--changes .scroll-table-sticky-summary-3 {
                position: static;
                left: auto;
                min-width: 68px;
                width: 68px;
                max-width: 68px;
            }
            .scroll-table--changes thead .scroll-table-sticky-summary-1,
            .scroll-table--changes thead .scroll-table-sticky-summary-2,
            .scroll-table--changes thead .scroll-table-sticky-summary-3 {
                position: sticky;
                top: 0;
                left: auto;
                z-index: 10;
            }
            .scroll-table-sticky-summary-1 { left: 168px; }
            .scroll-table-sticky-summary-2 { left: 262px; }
            .scroll-table-sticky-summary-3 { left: 356px; }
            .scroll-table-summary-head span {
                min-height: 102px;
                font-size: 0.58rem;
            }
            .scroll-table-player-head {
                min-width: 68px;
                width: 68px;
                max-width: 68px;
            }
            .scroll-table-player-head span {
                min-height: 118px;
                font-size: 0.64rem;
            }
            .timeline-match-svg {
                width: 1360px;
                min-width: 1360px;
            }
            .player-svg-card svg {
                min-width: 500px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    data = load_data()
    render_header(data["team_name"])
    if "section" not in st.session_state:
        st.session_state["section"] = "General"
    section = render_sidebar_navigation()

    if section == "General":
        render_general(data)
    elif section == "Equipo":
        render_equipo(data)
    else:
        render_plantilla(data)


if __name__ == "__main__":
    main()
