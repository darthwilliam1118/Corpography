"""
Camera selection screen.

A self-contained Pygame blocking screen that lets the user choose which webcam
to use.  Call run_camera_select() from both main.py and editor.py before
opening the camera for real.
"""
from __future__ import annotations

import pygame

from capture import enumerate_cameras

# ---------------------------------------------------------------------------
# Colours (match editor.py conventions)
# ---------------------------------------------------------------------------

_BG      = (10, 10, 20)
_PANEL   = (25, 25, 40)
_BORDER  = (60, 60, 80)
_SEL_BG  = (20, 50, 20)
_SEL_FG  = (0, 220, 0)
_HOV_BG  = (35, 35, 55)
_TEXT    = (220, 220, 220)
_DIM     = (120, 120, 140)
_WARN    = (255, 220, 0)
_BTN_BG  = (50, 90, 50)
_BTN_HOV = (70, 130, 70)
_BTN_DIS = (40, 40, 50)

_HEADER_H  = 110
_FOOTER_H  = 80
_ROW_H     = 64
_ROW_PAD_X = 40
_BTN_W, _BTN_H = 260, 46


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_camera_select(
    screen: pygame.Surface,
    clock: pygame.time.Clock,
    fonts: dict[str, pygame.font.Font],
    *,
    initial_index: int | None = None,
    error_message: str | None = None,
) -> int | None:
    """Show the camera pick-list screen.

    Returns the chosen device index (CameraInfo.index), or None if the user
    quits without selecting or if no physical cameras are found.

    If exactly one camera is detected the function returns immediately without
    rendering anything (silent selection).

    Parameters
    ----------
    screen:        Pygame display surface.
    clock:         Pygame clock (used to cap the UI at 30 FPS).
    fonts:         Dict with keys ``"title"``, ``"body"``, ``"small"``.
    initial_index: Pre-selected row — used when a saved camera failed to open.
    error_message: Yellow warning shown at the top of the screen.
    """
    font_title = fonts.get("title") or pygame.font.SysFont(None, 36)
    font_body  = fonts.get("body")  or pygame.font.SysFont(None, 24)
    font_small = fonts.get("small") or pygame.font.SysFont(None, 20)

    window_w, window_h = screen.get_size()

    cameras = enumerate_cameras()

    # Silent path: exactly one physical camera → return without showing UI
    if len(cameras) == 1:
        return cameras[0].index

    # Find which row to pre-select (match by device index)
    selected_row: int | None = None
    if initial_index is not None:
        for i, c in enumerate(cameras):
            if c.index == initial_index:
                selected_row = i
                break

    hovered_row: int | None = None

    def _row_rect(i: int) -> pygame.Rect:
        content_y = _HEADER_H + 20
        x = _ROW_PAD_X
        y = content_y + i * (_ROW_H + 8)
        w = window_w - 2 * _ROW_PAD_X
        return pygame.Rect(x, y, w, _ROW_H)

    def _btn_select_rect() -> pygame.Rect:
        bx = (window_w - _BTN_W) // 2
        by = window_h - _FOOTER_H + (_FOOTER_H - _BTN_H) // 2
        return pygame.Rect(bx, by, _BTN_W, _BTN_H)

    def _btn_refresh_rect() -> pygame.Rect:
        return pygame.Rect(window_w - 140, (_HEADER_H - _BTN_H) // 2, 120, _BTN_H)

    while True:
        mouse_pos = pygame.mouse.get_pos()

        # Hit-test rows for hover
        hovered_row = None
        for i in range(len(cameras)):
            if _row_rect(i).collidepoint(mouse_pos):
                hovered_row = i
                break

        btn_select  = _btn_select_rect()
        btn_refresh = _btn_refresh_rect()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if selected_row is not None and cameras:
                        return cameras[selected_row].index
                if event.key == pygame.K_ESCAPE:
                    return initial_index
                if event.key == pygame.K_DOWN and cameras:
                    if selected_row is None:
                        selected_row = 0
                    else:
                        selected_row = min(selected_row + 1, len(cameras) - 1)
                if event.key == pygame.K_UP and cameras:
                    if selected_row is None:
                        selected_row = len(cameras) - 1
                    else:
                        selected_row = max(selected_row - 1, 0)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i in range(len(cameras)):
                    if _row_rect(i).collidepoint(event.pos):
                        selected_row = i
                        break

                if btn_select.collidepoint(event.pos) and selected_row is not None and cameras:
                    return cameras[selected_row].index

                if btn_refresh.collidepoint(event.pos):
                    cameras = enumerate_cameras()
                    selected_row = None
                    hovered_row = None

        # -----------------------------------------------------------------
        # Render
        # -----------------------------------------------------------------
        screen.fill(_BG)

        # Header
        pygame.draw.rect(screen, _PANEL, (0, 0, window_w, _HEADER_H))
        pygame.draw.line(screen, _BORDER, (0, _HEADER_H), (window_w, _HEADER_H), 1)

        title_surf = font_title.render("Select Camera", True, _TEXT)
        screen.blit(title_surf, (30, (_HEADER_H // 2 - title_surf.get_height()) // 2 + 8))

        if error_message:
            warn_surf = font_body.render(error_message, True, _WARN)
            screen.blit(warn_surf, (30, _HEADER_H // 2 + 10))

        # Refresh button
        r_col = _BTN_HOV if btn_refresh.collidepoint(mouse_pos) else _BTN_BG
        pygame.draw.rect(screen, r_col, btn_refresh, border_radius=6)
        r_lbl = font_body.render("Refresh", True, _TEXT)
        screen.blit(r_lbl, r_lbl.get_rect(center=btn_refresh.center))

        # Footer
        pygame.draw.rect(screen, _PANEL, (0, window_h - _FOOTER_H, window_w, _FOOTER_H))
        pygame.draw.line(screen, _BORDER, (0, window_h - _FOOTER_H), (window_w, window_h - _FOOTER_H), 1)

        esc_surf = font_small.render("ESC — cancel  |  ↑↓ — navigate", True, _DIM)
        screen.blit(esc_surf, (30, window_h - _FOOTER_H + (_FOOTER_H - esc_surf.get_height()) // 2))

        can_select = selected_row is not None and bool(cameras)
        s_col = (_BTN_HOV if btn_select.collidepoint(mouse_pos) else _BTN_BG) if can_select else _BTN_DIS
        pygame.draw.rect(screen, s_col, btn_select, border_radius=6)
        s_lbl = font_body.render("Use This Camera", True, _TEXT if can_select else _DIM)
        screen.blit(s_lbl, s_lbl.get_rect(center=btn_select.center))

        # Camera rows
        if not cameras:
            msg = font_title.render("No cameras found. Connect a webcam and click Refresh.", True, _WARN)
            screen.blit(msg, msg.get_rect(center=(window_w // 2, (window_h - _HEADER_H - _FOOTER_H) // 2 + _HEADER_H)))
        else:
            for i, cam in enumerate(cameras):
                row = _row_rect(i)
                is_sel   = (i == selected_row)
                is_hover = (i == hovered_row)

                bg_col = _SEL_BG if is_sel else (_HOV_BG if is_hover else _PANEL)
                pygame.draw.rect(screen, bg_col, row, border_radius=6)

                # Left accent bar for selected row
                if is_sel:
                    bar = pygame.Rect(row.x, row.y + 8, 4, row.h - 16)
                    pygame.draw.rect(screen, _SEL_FG, bar, border_radius=2)

                # Camera name
                name_col = _SEL_FG if is_sel else _TEXT
                name_surf = font_body.render(cam.name, True, name_col)
                screen.blit(name_surf, (row.x + 20, row.centery - name_surf.get_height() // 2))

                # Index info (dimmed, right-aligned)
                idx_surf = font_small.render(f"index {cam.index}", True, _DIM)
                screen.blit(idx_surf, (row.right - idx_surf.get_width() - 16, row.centery - idx_surf.get_height() // 2))

                # Row border
                border_col = _SEL_FG if is_sel else _BORDER
                pygame.draw.rect(screen, border_col, row, 1 if not is_sel else 2, border_radius=6)

        pygame.display.flip()
        clock.tick(30)
