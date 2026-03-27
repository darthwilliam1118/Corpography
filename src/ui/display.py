import pygame
import mediapipe as mp

from pose import VISIBILITY_THRESHOLD, KEY_LANDMARK_INDICES, KEY_LANDMARK_NAMES

_CONNECTIONS = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

_COLOR_OK = (0, 220, 0)
_COLOR_LOW = (220, 0, 0)
_COLOR_WARN = (255, 220, 0)

_DEBUG_PANEL_W = 200


def draw_skeleton(surface: pygame.Surface, landmarks: list, win_w: int, win_h: int) -> None:
    """Draw the MediaPipe pose skeleton onto surface using Pygame drawing primitives."""
    for conn in _CONNECTIONS:
        lm_s = landmarks[conn.start]
        lm_e = landmarks[conn.end]
        color = (
            _COLOR_OK
            if lm_s.visibility >= VISIBILITY_THRESHOLD and lm_e.visibility >= VISIBILITY_THRESHOLD
            else _COLOR_LOW
        )
        pt_s = (int(lm_s.x * win_w), int(lm_s.y * win_h))
        pt_e = (int(lm_e.x * win_w), int(lm_e.y * win_h))
        pygame.draw.line(surface, color, pt_s, pt_e, 2)

    for lm in landmarks:
        color = _COLOR_OK if lm.visibility >= VISIBILITY_THRESHOLD else _COLOR_LOW
        px = int(lm.x * win_w)
        py = int(lm.y * win_h)
        pygame.draw.circle(surface, color, (px, py), 4)


def draw_body_warning(surface: pygame.Surface, font: pygame.font.Font, win_w: int, win_h: int) -> None:
    """Display a translucent 'body not fully visible' warning near the top-centre of the screen."""
    text = font.render("Step back \u2014 body not fully visible", True, _COLOR_WARN)
    text_w, text_h = text.get_size()
    pad = 8
    bg = pygame.Surface((text_w + pad * 2, text_h + pad * 2), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 160))
    x = (win_w - bg.get_width()) // 2
    y = 56
    surface.blit(bg, (x, y))
    surface.blit(text, (x + pad, y + pad))


def draw_debug_panel(
    surface: pygame.Surface,
    landmarks: list,
    font: pygame.font.Font,
    win_w: int,
    win_h: int,
) -> None:
    """Render a translucent right-side panel listing key landmark visibility scores."""
    panel = pygame.Surface((_DEBUG_PANEL_W, win_h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 160))

    line_h = 22
    x_text = 10
    y_text = 10
    for idx, name in zip(KEY_LANDMARK_INDICES, KEY_LANDMARK_NAMES):
        vis = landmarks[idx].visibility
        color = _COLOR_OK if vis >= VISIBILITY_THRESHOLD else _COLOR_LOW
        label = font.render(f"{name}: {vis * 100:.0f}%", True, color)
        panel.blit(label, (x_text, y_text))
        y_text += line_h

    surface.blit(panel, (win_w - _DEBUG_PANEL_W, 0))
