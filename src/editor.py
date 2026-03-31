"""
Corpography Template Editor
Entry point: python src/editor.py

A standalone pygame utility for authoring pose template JSON files.
Press a letter key to open that letter's editor canvas, drag joint circles
to position them, right-click a joint to cycle its weight, optionally seed
positions from a live webcam pose, then Save.
"""
from __future__ import annotations

import enum
import math
import os
import sys

import pygame

from core.editor_model import EditorModel
from core.scoring import DEFAULT_D_MAX, score_pose_detail, score_pose_from_pts
from core.templates import (
    MP_INDEX_TO_NAME, TEMPLATE_LANDMARK_NAMES, template_path,
    alphabet_path, load_alphabet,
)
from pose import VISIBILITY_THRESHOLD
from ui.display import draw_skeleton
from utils import resource_path

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

WINDOW_W, WINDOW_H = 1280, 960
CANVAS_W = 940           # left region: letter bg + joints
PANEL_X  = CANVAS_W     # right region starts here
PANEL_W  = WINDOW_W - CANVAS_W
FPS = 30

JOINT_RADIUS   = 14
JOINT_ACTIVE_R = 18

# Joint colors by weight
_COLOR_W_FULL    = (0, 220, 0)
_COLOR_W_HALF    = (255, 165, 0)
_COLOR_W_IGNORE  = (120, 120, 120)
_COLOR_W_ACTIVE  = (255, 200, 0)
_COLOR_W_WHITE   = (255, 255, 255)

# Skeleton connections — ordered by visual clarity (top → bottom)
EDITOR_CONNECTIONS: tuple[tuple[str, str], ...] = (
    ("LEFT_SHOULDER",  "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER",  "LEFT_ELBOW"),
    ("LEFT_ELBOW",     "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW",    "RIGHT_WRIST"),
    ("LEFT_SHOULDER",  "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP",       "RIGHT_HIP"),
    ("LEFT_HIP",       "LEFT_KNEE"),
    ("RIGHT_HIP",      "RIGHT_KNEE"),
    ("LEFT_KNEE",      "LEFT_ANKLE"),
    ("RIGHT_KNEE",     "RIGHT_ANKLE"),
)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class EditorState(enum.Enum):
    LETTER_SELECT  = "letter_select"
    EDITING        = "editing"
    SCORING        = "scoring"
    SCORING_MANUAL = "scoring_manual"
    SAVE_CONFIRM   = "save_confirm"


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _norm_to_canvas(nx: float, ny: float) -> tuple[int, int]:
    return (int(nx * CANVAS_W), int(ny * WINDOW_H))


def _canvas_to_norm(cx: int, cy: int) -> tuple[float, float]:
    return (cx / CANVAS_W, cy / WINDOW_H)


def _pixel_to_norm(px: int, py: int) -> tuple[float, float]:
    """Convert raw screen pixel (x, y) to normalized canvas coords, clamped."""
    nx = max(0.0, min(1.0, px / CANVAS_W))
    ny = max(0.0, min(1.0, py / WINDOW_H))
    return nx, ny


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _connection_color(w1: float, w2: float) -> tuple[int, int, int]:
    avg = (w1 + w2) / 2.0
    if avg >= 0.75:
        return (220, 220, 220)
    if avg >= 0.25:
        return (200, 130, 0)
    return (80, 80, 80)


def _draw_skeleton_on_canvas(surface: pygame.Surface, model: EditorModel) -> None:
    lm = model.template.landmarks
    # Connections first (under joints)
    for a, b in EDITOR_CONNECTIONS:
        if a not in lm or b not in lm:
            continue
        ea, eb = lm[a], lm[b]
        color = _connection_color(ea.weight, eb.weight)
        pa = _norm_to_canvas(ea.x, ea.y)
        pb = _norm_to_canvas(eb.x, eb.y)
        pygame.draw.line(surface, color, pa, pb, 2)

    # Joints
    for name, entry in lm.items():
        cx, cy = _norm_to_canvas(entry.x, entry.y)
        is_active = name == model.dragging
        r = JOINT_ACTIVE_R if is_active else JOINT_RADIUS

        if is_active:
            pygame.draw.circle(surface, _COLOR_W_ACTIVE, (cx, cy), r)
            pygame.draw.circle(surface, _COLOR_W_WHITE, (cx, cy), r, 2)
        elif entry.weight >= 0.75:
            pygame.draw.circle(surface, _COLOR_W_FULL, (cx, cy), r)
            pygame.draw.circle(surface, _COLOR_W_WHITE, (cx, cy), r, 2)
        elif entry.weight >= 0.25:
            pygame.draw.circle(surface, _COLOR_W_HALF, (cx, cy), r)
            pygame.draw.circle(surface, _COLOR_W_WHITE, (cx, cy), r, 2)
        else:
            # weight ≈ 0 — hollow circle
            pygame.draw.circle(surface, (30, 30, 30), (cx, cy), r)
            pygame.draw.circle(surface, _COLOR_W_IGNORE, (cx, cy), r, 2)


def _draw_ghost_skeleton_on_canvas(
    surface: pygame.Surface, model: EditorModel, alpha: int = 80
) -> None:
    """Draw semi-transparent cyan skeleton for RECORDING overlay."""
    ghost = pygame.Surface((CANVAS_W, WINDOW_H), pygame.SRCALPHA)
    lm = model.template.landmarks
    for a, b in EDITOR_CONNECTIONS:
        if a not in lm or b not in lm:
            continue
        pa = _norm_to_canvas(lm[a].x, lm[a].y)
        pb = _norm_to_canvas(lm[b].x, lm[b].y)
        pygame.draw.line(ghost, (0, 220, 220, alpha), pa, pb, 2)
    for name, entry in lm.items():
        cx, cy = _norm_to_canvas(entry.x, entry.y)
        pygame.draw.circle(ghost, (0, 220, 220, alpha), (cx, cy), JOINT_RADIUS)
    surface.blit(ghost, (0, 0))


def _draw_per_joint_scores(
    surface: pygame.Surface,
    per_joint: dict[str, float],
    positions: dict[str, tuple[float, float]],
    font: pygame.font.Font,
) -> None:
    """Draw a small color-coded score integer to the right of each active joint."""
    for name, score in per_joint.items():
        if name not in positions:
            continue
        nx, ny = positions[name]
        cx, cy = _norm_to_canvas(nx, ny)
        if score >= 80:
            color = (0, 220, 0)
        elif score >= 50:
            color = (255, 200, 0)
        else:
            color = (220, 60, 60)
        label = font.render(f"{score:.0f}", True, color)
        surface.blit(label, (cx + 10, cy - label.get_height() // 2))


def _manual_hit_test(
    manual_pts: dict[str, list],
    nx: float,
    ny: float,
    radius: float = 0.025,
) -> str | None:
    """Return the name of the nearest manual skeleton joint within radius, or None."""
    for name, xy in manual_pts.items():
        if math.sqrt((nx - xy[0]) ** 2 + (ny - xy[1]) ** 2) < radius:
            return name
    return None


def _draw_letter_background(surface: pygame.Surface, shape_id: str, font_large: pygame.font.Font) -> None:
    """Render a semi-transparent letter at 90% window height, centered in the canvas."""
    # Render glyph scaled to 90% of window height
    glyph = font_large.render(shape_id, True, (255, 255, 255))
    gw, gh = glyph.get_size()

    target_h = int(WINDOW_H * 0.9)
    if gh > 0:
        scale = target_h / gh
        glyph = pygame.transform.smoothscale(glyph, (max(1, int(gw * scale)), target_h))
        gw, gh = glyph.get_size()

    # Center in canvas
    x = (CANVAS_W - gw) // 2
    y = (WINDOW_H - gh) // 2

    alpha_surf = pygame.Surface((gw, gh), pygame.SRCALPHA)
    alpha_surf.fill((0, 0, 0, 0))
    alpha_surf.blit(glyph, (0, 0))
    alpha_surf.set_alpha(35)
    surface.blit(alpha_surf, (x, y))


def _draw_panel(
    surface: pygame.Surface,
    model: EditorModel,
    font_h: pygame.font.Font,
    font_b: pygame.font.Font,
    font_s: pygame.font.Font,
    buttons: dict,
    hovered_joint: str | None,
) -> None:
    """Draw the controls panel on the right side."""
    panel = pygame.Surface((PANEL_W, WINDOW_H))
    panel.fill((30, 30, 40))
    pygame.draw.line(panel, (80, 80, 100), (0, 0), (0, WINDOW_H), 2)

    y = 20
    pad = 12

    # Title
    title = font_h.render("Template Editor", True, (200, 200, 255))
    panel.blit(title, (pad, y)); y += title.get_height() + 4

    # Shape ID
    sid = font_b.render(f"Shape: {model.template.shape_id}", True, (255, 255, 255))
    panel.blit(sid, (pad, y)); y += sid.get_height() + 2

    # Display name
    dname = font_s.render(model.template.display_name, True, (180, 180, 180))
    panel.blit(dname, (pad, y)); y += dname.get_height() + 12

    # Difficulty buttons
    diff_label = font_s.render("Difficulty:", True, (160, 160, 160))
    panel.blit(diff_label, (pad, y))
    y += diff_label.get_height() + 4
    for d in (1, 2, 3):
        bx = pad + (d - 1) * 72
        color = (80, 180, 80) if model.template.difficulty == d else (60, 60, 80)
        brect = pygame.Rect(bx, y, 64, 32)
        pygame.draw.rect(panel, color, brect, border_radius=4)
        pygame.draw.rect(panel, (120, 120, 140), brect, 1, border_radius=4)
        label = font_b.render(str(d), True, (255, 255, 255))
        panel.blit(label, (brect.centerx - label.get_width() // 2,
                           brect.centery - label.get_height() // 2))
        # Store button rect in screen coordinates
        buttons[f"diff_{d}"] = pygame.Rect(PANEL_X + bx, y, 64, 32)
    y += 44

    # Action buttons
    for key, label_text, color in (
        ("score",  "Score Live",   (140, 0, 200)),
        ("tune",   "Tune Scoring", (0, 130, 150)),
        ("save",   "Save",         (0, 180, 80)),
        ("new",    "New Letter",   (160, 80, 0)),
    ):
        brect = pygame.Rect(pad, y, PANEL_W - pad * 2, 36)
        pygame.draw.rect(panel, color, brect, border_radius=5)
        pygame.draw.rect(panel, (200, 200, 200), brect, 1, border_radius=5)
        lbl = font_b.render(label_text, True, (255, 255, 255))
        panel.blit(lbl, (brect.centerx - lbl.get_width() // 2,
                         brect.centery - lbl.get_height() // 2))
        buttons[key] = pygame.Rect(PANEL_X + brect.x, brect.y, brect.w, brect.h)
        y += 44

    y += 8
    sep = font_s.render("─" * 28, True, (60, 60, 80))
    panel.blit(sep, (pad, y)); y += sep.get_height() + 6

    # Weight legend
    legend_label = font_s.render("Joint weights:", True, (160, 160, 160))
    panel.blit(legend_label, (pad, y)); y += legend_label.get_height() + 4
    for color, text in (
        (_COLOR_W_FULL,   "● 1.0  Full weight"),
        (_COLOR_W_HALF,   "● 0.5  Half weight"),
        (_COLOR_W_IGNORE, "◌ 0.0  Ignored"),
    ):
        t = font_s.render(text, True, color)
        panel.blit(t, (pad + 4, y)); y += t.get_height() + 2
    y += 8

    sep2 = font_s.render("─" * 28, True, (60, 60, 80))
    panel.blit(sep2, (pad, y)); y += sep2.get_height() + 6

    # Joint list
    jlist_label = font_s.render("Joints (right-click = cycle weight):", True, (160, 160, 160))
    panel.blit(jlist_label, (pad, y)); y += jlist_label.get_height() + 4

    lm = model.template.landmarks
    for name in TEMPLATE_LANDMARK_NAMES:
        if name not in lm:
            continue
        entry = lm[name]
        w = entry.weight
        if w >= 0.75:
            dot_color = _COLOR_W_FULL
        elif w >= 0.25:
            dot_color = _COLOR_W_HALF
        else:
            dot_color = _COLOR_W_IGNORE

        highlight = (name == hovered_joint)
        bg_color = (50, 50, 70) if highlight else None
        line_h = 20
        if bg_color:
            pygame.draw.rect(panel, bg_color, (pad, y, PANEL_W - pad * 2, line_h))

        dot = font_s.render("●", True, dot_color)
        panel.blit(dot, (pad + 2, y))
        name_surf = font_s.render(f"  {name.replace('_', ' ').title()}", True, (210, 210, 210))
        panel.blit(name_surf, (pad + 2, y))
        w_surf = font_s.render(f"{w:.1f}", True, dot_color)
        panel.blit(w_surf, (PANEL_W - pad - w_surf.get_width(), y))
        y += line_h

        if y > WINDOW_H - 10:
            break

    # Error message
    if model.error_message:
        err = font_s.render(model.error_message[:38], True, (255, 80, 80))
        panel.blit(err, (pad, WINDOW_H - 30))

    surface.blit(panel, (PANEL_X, 0))


# ---------------------------------------------------------------------------
# State handlers
# ---------------------------------------------------------------------------

def _handle_letter_select(
    screen: pygame.Surface,
    events: list,
    font_large: pygame.font.Font,
    font_body: pygame.font.Font,
) -> tuple[EditorState, EditorModel | None]:
    screen.fill((10, 10, 20))

    # Load which letters already have templates
    templates_dir = _resolve_templates_dir()
    try:
        existing_ids = set(load_alphabet(alphabet_path("latin", templates_dir)).keys())
    except Exception:
        existing_ids = set()

    # --- Title and sub-prompt ---
    title = font_large.render("Corpography — Template Editor", True, (180, 180, 255))
    screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, 120))

    sub = font_body.render(
        "Click a letter  |  or press Shift+key for uppercase / key for lowercase",
        True, (160, 160, 160),
    )
    screen.blit(sub, (WINDOW_W // 2 - sub.get_width() // 2, 175))

    # --- Letter grid ---
    CELL_W, TILE_W = 46, 40
    CELL_H, TILE_H = 50, 44
    UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
    start_x = (WINDOW_W - 26 * CELL_W) // 2
    y_upper, y_lower = 280, 346

    mx, my = pygame.mouse.get_pos()

    for row_letters, row_y, label in (
        (UPPERCASE, y_upper, "UPPERCASE"),
        (LOWERCASE, y_lower, "lowercase"),
    ):
        lbl = font_body.render(label, True, (90, 90, 100))
        screen.blit(lbl, (start_x, row_y - 22))

        for i, ch in enumerate(row_letters):
            tile = pygame.Rect(start_x + i * CELL_W, row_y, TILE_W, TILE_H)
            has = ch in existing_ids
            hovered = tile.collidepoint(mx, my)
            bg = (0, 140, 50) if has else (55, 55, 70)
            if hovered:
                bg = tuple(min(255, c + 40) for c in bg)
            text_color = (255, 255, 255) if has else (110, 110, 125)
            pygame.draw.rect(screen, bg, tile, border_radius=4)
            glyph = font_large.render(ch, True, text_color)
            screen.blit(glyph, glyph.get_rect(center=tile.center))

    hint = font_body.render("ESC  to quit", True, (100, 100, 100))
    screen.blit(hint, (WINDOW_W // 2 - hint.get_width() // 2, 440))

    # --- Events ---
    for event in events:
        if event.type == pygame.KEYDOWN:
            if pygame.K_a <= event.key <= pygame.K_z:
                shift_held = bool(event.mod & pygame.KMOD_SHIFT)
                shape_id = chr(event.key).upper() if shift_held else chr(event.key)
                model = EditorModel(shape_id=shape_id, templates_dir=templates_dir, alphabet="latin")
                return EditorState.EDITING, model

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for row_letters, row_y, _ in (
                (UPPERCASE, y_upper, None),
                (LOWERCASE, y_lower, None),
            ):
                for i, ch in enumerate(row_letters):
                    tile = pygame.Rect(start_x + i * CELL_W, row_y, TILE_W, TILE_H)
                    if tile.collidepoint(event.pos):
                        model = EditorModel(shape_id=ch, templates_dir=templates_dir, alphabet="latin")
                        return EditorState.EDITING, model

    return EditorState.LETTER_SELECT, None


def _handle_editing(
    screen: pygame.Surface,
    model: EditorModel,
    events: list,
    font_large: pygame.font.Font,
    font_h: pygame.font.Font,
    font_b: pygame.font.Font,
    font_s: pygame.font.Font,
    buttons: dict,
    hovered_joint: list,   # mutable single-element list so caller can read it
) -> EditorState:
    # --- Background ---
    screen.fill((18, 18, 28))
    # Canvas background
    pygame.draw.rect(screen, (22, 22, 35), (0, 0, CANVAS_W, WINDOW_H))

    # Letter background glyph
    _draw_letter_background(screen, model.shape_id, font_large)

    # Skeleton + joints
    _draw_skeleton_on_canvas(screen, model)

    # Panel
    buttons.clear()
    _draw_panel(screen, model, font_h, font_b, font_s, buttons, hovered_joint[0])

    # --- Events ---
    mouse_pos = pygame.mouse.get_pos()
    # Update hover (only inside canvas)
    if mouse_pos[0] < CANVAS_W:
        nx, ny = _pixel_to_norm(mouse_pos[0], mouse_pos[1])
        hovered_joint[0] = model.hit_test(nx, ny)
    else:
        hovered_joint[0] = None

    # Cursor
    if hovered_joint[0] is not None:
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
    else:
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

    for event in events:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return EditorState.LETTER_SELECT

        if event.type == pygame.MOUSEBUTTONDOWN:
            px, py = event.pos

            # Left click
            if event.button == 1:
                if px < CANVAS_W:
                    nx, ny = _pixel_to_norm(px, py)
                    name = model.hit_test(nx, ny)
                    if name:
                        model.begin_drag(name, nx, ny)
                else:
                    # Panel button click
                    for key, rect in buttons.items():
                        if rect.collidepoint(px, py):
                            if key == "score":
                                return EditorState.SCORING
                            if key == "tune":
                                return EditorState.SCORING_MANUAL
                            if key == "save":
                                model.save()
                                return EditorState.SAVE_CONFIRM
                            if key == "new":
                                model.end_drag()
                                return EditorState.LETTER_SELECT
                            if key.startswith("diff_"):
                                model.template.difficulty = int(key[-1])

            # Right click — cycle weight
            if event.button == 3 and px < CANVAS_W:
                nx, ny = _pixel_to_norm(px, py)
                name = model.hit_test(nx, ny)
                if name:
                    model.cycle_weight(name)

        if event.type == pygame.MOUSEMOTION and event.buttons[0]:
            if model.dragging:
                px, py = event.pos
                nx, ny = _pixel_to_norm(px, py)
                model.update_drag(nx, ny)

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            model.end_drag()

    return EditorState.EDITING



def _draw_dmax_control(
    panel: pygame.Surface,
    y: int,
    d_max: float,
    font_b: pygame.font.Font,
    font_s: pygame.font.Font,
) -> tuple[int, pygame.Rect, pygame.Rect]:
    """
    Draw the D_MAX +/- control row onto panel.
    Returns (new_y, minus_panel_rect, plus_panel_rect).
    Callers convert to screen rects by offsetting x by PANEL_X.
    """
    pad = 12
    label = font_s.render(f"D_MAX: {d_max:.2f}", True, (160, 160, 200))
    panel.blit(label, (pad, y))
    lw = label.get_width()
    btn_y = y
    btn_h = 22
    btn_w = 26
    gap = 6
    minus_x = pad + lw + gap
    plus_x  = minus_x + btn_w + gap
    for bx, txt in ((minus_x, "−"), (plus_x, "+")):
        r = pygame.Rect(bx, btn_y, btn_w, btn_h)
        pygame.draw.rect(panel, (70, 70, 100), r, border_radius=3)
        pygame.draw.rect(panel, (140, 140, 180), r, 1, border_radius=3)
        t = font_b.render(txt, True, (220, 220, 255))
        panel.blit(t, (r.centerx - t.get_width() // 2, r.centery - t.get_height() // 2))
    return y + btn_h + 8, pygame.Rect(minus_x, btn_y, btn_w, btn_h), pygame.Rect(plus_x, btn_y, btn_w, btn_h)


def _handle_scoring(
    screen: pygame.Surface,
    model: EditorModel,
    cap,
    detector,
    events: list,
    font_score: pygame.font.Font,
    font_b: pygame.font.Font,
    font_s: pygame.font.Font,
    timestamp_ms: int,
    d_max: float,
    capture_flash_ms: int,
) -> tuple[EditorState, float, int]:
    """
    SCORING state: live webcam feed + continuously updated score + Record button.
    Returns (new_state, new_d_max, new_capture_flash_ms).
    ESC returns to EDITING.
    """
    frame = cap.get_frame()
    if frame is not None:
        h, w = frame.shape[:2]
        surf = pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")
        surf = pygame.transform.scale(surf, (CANVAS_W, WINDOW_H))
        screen.blit(surf, (0, 0))
    else:
        screen.fill((10, 10, 10))

    result = detector.process(frame, timestamp_ms) if frame is not None else None
    landmarks = detector.get_landmarks(result)
    body_ok = detector.body_visible(landmarks)

    # Ghost template skeleton overlay
    _draw_ghost_skeleton_on_canvas(screen, model)

    # Player skeleton overlay
    if landmarks is not None:
        draw_skeleton(screen, landmarks, CANVAS_W, WINDOW_H)

    # Compute live score + per-joint breakdown
    live_score: float | None = None
    per_joint: dict[str, float] = {}
    if body_ok and landmarks is not None:
        live_score, per_joint = score_pose_detail(landmarks, model.template, d_max)

    # Per-joint score labels at player joint positions
    if per_joint and landmarks is not None:
        player_positions: dict[str, tuple[float, float]] = {
            name: (float(landmarks[idx].x), float(landmarks[idx].y))
            for idx, name in MP_INDEX_TO_NAME.items()
            if idx < len(landmarks)
        }
        _draw_per_joint_scores(screen, per_joint, player_positions, font_s)

    # --- Right panel ---
    panel = pygame.Surface((PANEL_W, WINDOW_H))
    panel.fill((30, 30, 40))
    pygame.draw.line(panel, (80, 80, 100), (0, 0), (0, WINDOW_H), 2)

    pad = 12
    y = 20

    title = font_b.render("SCORING LIVE", True, (220, 60, 60))
    panel.blit(title, (pad, y)); y += title.get_height() + 12

    # "Record from pose" button
    rec_color = (0, 160, 80) if landmarks is not None else (60, 60, 80)
    rec_rect = pygame.Rect(pad, y, PANEL_W - pad * 2, 36)
    pygame.draw.rect(panel, rec_color, rec_rect, border_radius=5)
    pygame.draw.rect(panel, (200, 200, 200), rec_rect, 1, border_radius=5)
    rec_lbl = font_b.render("Record from pose", True, (255, 255, 255))
    panel.blit(rec_lbl, (rec_rect.centerx - rec_lbl.get_width() // 2,
                          rec_rect.centery - rec_lbl.get_height() // 2))
    rec_sr = pygame.Rect(PANEL_X + rec_rect.x, rec_rect.y, rec_rect.w, rec_rect.h)
    y += rec_rect.height + 4

    # "Captured!" flash (shown for 1.5 s after capture)
    if capture_flash_ms > 0 and (timestamp_ms - capture_flash_ms) < 1500:
        flash = font_b.render("Captured!", True, (0, 255, 100))
        panel.blit(flash, (pad, y))
    y += font_b.size("X")[1] + 12

    hint = font_b.render("ESC  to stop", True, (160, 160, 160))
    panel.blit(hint, (pad, y)); y += hint.get_height() + 20

    # Large score display
    if live_score is not None:
        if live_score >= 80:
            score_color = (0, 220, 0)
        elif live_score >= 50:
            score_color = (255, 200, 0)
        else:
            score_color = (220, 60, 60)
        score_surf = font_score.render(f"{live_score:.0f}", True, score_color)
    else:
        score_color = (120, 120, 120)
        score_surf = font_score.render("—", True, score_color)

    sx = (PANEL_W - score_surf.get_width()) // 2
    panel.blit(score_surf, (sx, y)); y += score_surf.get_height() + 8

    status = "body visible" if body_ok else "body not fully visible"
    st = font_b.render(status, True, (0, 200, 0) if body_ok else (255, 220, 0))
    panel.blit(st, ((PANEL_W - st.get_width()) // 2, y)); y += st.get_height() + 16

    # D_MAX control
    y, minus_pr, plus_pr = _draw_dmax_control(panel, y, d_max, font_b, font_s)
    minus_sr = pygame.Rect(PANEL_X + minus_pr.x, minus_pr.y, minus_pr.w, minus_pr.h)
    plus_sr  = pygame.Rect(PANEL_X + plus_pr.x,  plus_pr.y,  plus_pr.w,  plus_pr.h)

    screen.blit(panel, (PANEL_X, 0))

    # Body-not-visible warning on canvas
    if not body_ok:
        warn = font_b.render("Step back — body not fully visible", True, (255, 220, 0))
        bw, bh = warn.get_size()
        pad8 = 8
        bg = pygame.Surface((bw + pad8 * 2, bh + pad8 * 2), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        x = (CANVAS_W - bg.get_width()) // 2
        screen.blit(bg, (x, 56))
        screen.blit(warn, (x + pad8, 56 + pad8))

    for event in events:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return EditorState.EDITING, d_max, capture_flash_ms
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if rec_sr.collidepoint(event.pos) and landmarks is not None:
                model.apply_landmarks_with_visibility(landmarks, VISIBILITY_THRESHOLD)
                capture_flash_ms = timestamp_ms
            elif minus_sr.collidepoint(event.pos):
                d_max = max(0.01, round(d_max - 0.01, 2))
            elif plus_sr.collidepoint(event.pos):
                d_max = min(1.00, round(d_max + 0.01, 2))

    return EditorState.SCORING, d_max, capture_flash_ms


def _draw_manual_skeleton(
    surface: pygame.Surface,
    manual_pts: dict[str, list],
    manual_dragging: str | None,
) -> None:
    """Draw the mouse-draggable test skeleton (white, dragged joint yellow)."""
    lm = manual_pts
    for a, b in EDITOR_CONNECTIONS:
        if a not in lm or b not in lm:
            continue
        pa = _norm_to_canvas(lm[a][0], lm[a][1])
        pb = _norm_to_canvas(lm[b][0], lm[b][1])
        pygame.draw.line(surface, (180, 180, 180), pa, pb, 2)
    for name, xy in lm.items():
        cx, cy = _norm_to_canvas(xy[0], xy[1])
        is_active = name == manual_dragging
        color = _COLOR_W_ACTIVE if is_active else _COLOR_W_WHITE
        r = JOINT_ACTIVE_R if is_active else JOINT_RADIUS
        pygame.draw.circle(surface, color, (cx, cy), r)
        pygame.draw.circle(surface, (50, 50, 50), (cx, cy), r, 2)


def _handle_scoring_manual(
    screen: pygame.Surface,
    model: EditorModel,
    events: list,
    manual_pts: dict[str, list],
    manual_dragging: str | None,
    font_score: pygame.font.Font,
    font_b: pygame.font.Font,
    font_s: pygame.font.Font,
    font_large: pygame.font.Font,
    d_max: float,
) -> tuple[EditorState, str | None, float]:
    """
    SCORING_MANUAL state: no camera, mouse-draggable test skeleton.
    Skeleton starts on top of template (score = 100). Drag joints to test sensitivity.
    Returns (new_state, new_manual_dragging, new_d_max).
    """
    # Canvas background
    screen.fill((18, 18, 28))
    pygame.draw.rect(screen, (22, 22, 35), (0, 0, CANVAS_W, WINDOW_H))
    _draw_letter_background(screen, model.shape_id, font_large)

    # Ghost template skeleton (cyan)
    _draw_ghost_skeleton_on_canvas(screen, model)

    # Compute score from manual skeleton positions
    player_pts_for_score = {name: (float(xy[0]), float(xy[1])) for name, xy in manual_pts.items()}
    live_score, per_joint = score_pose_from_pts(player_pts_for_score, model.template, d_max)

    # Manual skeleton (white, dragged joint yellow)
    _draw_manual_skeleton(screen, manual_pts, manual_dragging)

    # Per-joint score labels
    manual_positions = {name: (float(xy[0]), float(xy[1])) for name, xy in manual_pts.items()}
    _draw_per_joint_scores(screen, per_joint, manual_positions, font_s)

    # --- Right panel ---
    panel = pygame.Surface((PANEL_W, WINDOW_H))
    panel.fill((30, 30, 40))
    pygame.draw.line(panel, (80, 80, 100), (0, 0), (0, WINDOW_H), 2)

    pad = 12
    y = 20

    title = font_b.render("TUNE SCORING", True, (0, 200, 200))
    panel.blit(title, (pad, y)); y += title.get_height() + 8

    for line, color in (
        ("Drag joints to test.", (200, 200, 200)),
        ("ESC  to stop",        (160, 160, 160)),
    ):
        t = font_b.render(line, True, color)
        panel.blit(t, (pad, y)); y += t.get_height() + 4
    y += 20

    # Large score display
    if live_score >= 80:
        score_color = (0, 220, 0)
    elif live_score >= 50:
        score_color = (255, 200, 0)
    else:
        score_color = (220, 60, 60)
    score_surf = font_score.render(f"{live_score:.0f}", True, score_color)
    sx = (PANEL_W - score_surf.get_width()) // 2
    panel.blit(score_surf, (sx, y)); y += score_surf.get_height() + 16

    # D_MAX control
    y, minus_pr, plus_pr = _draw_dmax_control(panel, y, d_max, font_b, font_s)
    minus_sr = pygame.Rect(PANEL_X + minus_pr.x, minus_pr.y, minus_pr.w, minus_pr.h)
    plus_sr  = pygame.Rect(PANEL_X + plus_pr.x,  plus_pr.y,  plus_pr.w,  plus_pr.h)

    screen.blit(panel, (PANEL_X, 0))

    # --- Events ---
    mouse_pos = pygame.mouse.get_pos()
    if mouse_pos[0] < CANVAS_W:
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND
                                if _manual_hit_test(manual_pts, *_pixel_to_norm(*mouse_pos))
                                else pygame.SYSTEM_CURSOR_ARROW)
    else:
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

    for event in events:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return EditorState.EDITING, None, d_max

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            px, py = event.pos
            if px < CANVAS_W:
                nx, ny = _pixel_to_norm(px, py)
                hit = _manual_hit_test(manual_pts, nx, ny)
                if hit:
                    manual_dragging = hit
            else:
                if minus_sr.collidepoint(event.pos):
                    d_max = max(0.01, round(d_max - 0.01, 2))
                elif plus_sr.collidepoint(event.pos):
                    d_max = min(1.00, round(d_max + 0.01, 2))

        if event.type == pygame.MOUSEMOTION and event.buttons[0] and manual_dragging:
            px, py = event.pos
            nx, ny = _pixel_to_norm(px, py)
            manual_pts[manual_dragging][0] = nx
            manual_pts[manual_dragging][1] = ny

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            manual_dragging = None

    return EditorState.SCORING_MANUAL, manual_dragging, d_max


def _handle_save_confirm(
    screen: pygame.Surface,
    model: EditorModel,
    events: list,
    font_b: pygame.font.Font,
    font_large: pygame.font.Font,
    confirm_start_ms: int,
) -> tuple[EditorState, bool]:
    """Returns (new_state, reset_timer). reset_timer=True when first entering."""
    # Draw editing scene underneath (reuse existing screen)
    # Then overlay confirmation banner
    confirm_surf = pygame.Surface((500, 80), pygame.SRCALPHA)
    confirm_surf.fill((0, 120, 0, 200))
    path = f"templates/latin.json  [{model.shape_id}]"
    msg = font_b.render(f"Saved: {path}", True, (255, 255, 255))
    confirm_surf.blit(msg, (20, 25))
    x = (WINDOW_W - confirm_surf.get_width()) // 2
    y = (WINDOW_H - confirm_surf.get_height()) // 2
    screen.blit(confirm_surf, (x, y))

    elapsed = pygame.time.get_ticks() - confirm_start_ms
    if elapsed >= 1500:
        return EditorState.EDITING, False

    for event in events:
        if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
            return EditorState.EDITING, False

    return EditorState.SAVE_CONFIRM, False


# ---------------------------------------------------------------------------
# Templates dir resolution
# ---------------------------------------------------------------------------

def _resolve_templates_dir() -> str:
    """
    Resolve path to templates/latin/ directory.
    In dev mode: relative to project root.
    In bundled exe: relative to exe parent directory.
    """
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        # Running from src/, go up to project root
        base = os.path.join(os.path.dirname(__file__), "..")
    return os.path.normpath(os.path.join(base, "templates"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Corpography — Template Editor")
    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont(None, 700)
    font_title = pygame.font.SysFont(None, 36)
    font_h     = pygame.font.SysFont(None, 28)
    font_b     = pygame.font.SysFont(None, 24)
    font_s     = pygame.font.SysFont(None, 20)
    font_score = pygame.font.SysFont(None, 160)   # large score display

    state: EditorState = EditorState.LETTER_SELECT
    model: EditorModel | None = None
    buttons: dict = {}
    hovered_joint: list[str | None] = [None]
    confirm_start_ms: int = 0

    # Shared scoring parameters (persist across SCORING / SCORING_MANUAL transitions)
    d_max: float = DEFAULT_D_MAX
    capture_flash_ms: int = 0    # timestamp of last "Record from pose" capture
    manual_pts: dict[str, list] = {}
    manual_dragging: str | None = None

    # RECORDING / SCORING state resources (opened lazily)
    cap = None
    detector = None

    def _close_recording():
        nonlocal cap, detector
        if cap is not None:
            cap.release()
            cap = None
        if detector is not None:
            detector.close()
            detector = None

    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                _close_recording()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if state == EditorState.LETTER_SELECT:
                    _close_recording()
                    pygame.quit()
                    sys.exit()

        timestamp_ms = pygame.time.get_ticks()

        if state == EditorState.LETTER_SELECT:
            new_state, new_model = _handle_letter_select(screen, events, font_title, font_b)
            if new_model is not None:
                model = new_model
            state = new_state

        elif state == EditorState.EDITING:
            assert model is not None
            state = _handle_editing(
                screen, model, events,
                font_large, font_h, font_b, font_s,
                buttons, hovered_joint,
            )
            if state == EditorState.SCORING_MANUAL:
                # No camera needed — initialise manual skeleton at template positions
                manual_pts = {
                    name: [entry.x, entry.y]
                    for name, entry in model.template.landmarks.items()
                }
                manual_dragging = None

            if state == EditorState.SCORING:
                # Open camera and detector
                from capture import Capture
                from pose import PoseDetector
                cap = Capture(device_index=0)
                if not cap.open():
                    model.error_message = "No camera found."
                    cap = None
                    state = EditorState.EDITING
                else:
                    model_file = resource_path("assets/pose_landmarker_lite.task")
                    detector = PoseDetector(model_path=model_file)
                    if not detector.open():
                        model.error_message = "Pose model not found."
                        cap.release(); cap = None
                        detector = None
                        state = EditorState.EDITING

            elif state == EditorState.SAVE_CONFIRM:
                confirm_start_ms = pygame.time.get_ticks()

            elif state == EditorState.LETTER_SELECT:
                # "New Letter" — reset hover
                hovered_joint[0] = None
                model = None

        elif state == EditorState.SCORING:
            assert model is not None and cap is not None and detector is not None
            state, d_max, capture_flash_ms = _handle_scoring(
                screen, model, cap, detector, events,
                font_score, font_b, font_s, timestamp_ms, d_max, capture_flash_ms
            )
            if state == EditorState.EDITING:
                _close_recording()

        elif state == EditorState.SCORING_MANUAL:
            assert model is not None
            state, manual_dragging, d_max = _handle_scoring_manual(
                screen, model, events, manual_pts, manual_dragging,
                font_score, font_b, font_s, font_large, d_max
            )

        elif state == EditorState.SAVE_CONFIRM:
            assert model is not None
            state, _ = _handle_save_confirm(
                screen, model, events, font_b, font_large, confirm_start_ms
            )

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
