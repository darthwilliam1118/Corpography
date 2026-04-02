import sys
import pygame
from capture import Capture
from config import get_camera_index, load_config, save_config, set_camera_index
from pose import PoseDetector
from ui.camera_select import run_camera_select
from ui.display import draw_skeleton, draw_body_warning, draw_debug_panel, scale_and_crop
from utils import resource_path

WINDOW_W, WINDOW_H = 1280, 960
ERROR_DISPLAY_MS = 3000


def _show_error_screen(screen, font, message: str, duration_ms: int) -> None:
    """Display a centered error message for up to duration_ms, dismissible with ESC."""
    text = font.render(message, True, (255, 80, 80))
    rect = text.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2))
    start = pygame.time.get_ticks()

    while pygame.time.get_ticks() - start < duration_ms:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return
        screen.fill((0, 0, 0))
        screen.blit(text, rect)
        pygame.display.flip()


def main():
    debug_mode: bool = "--debug" in sys.argv

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Corpography")
    fps_font = pygame.font.SysFont(None, 28)
    error_font = pygame.font.SysFont(None, 48)
    warning_font = pygame.font.SysFont(None, 42)
    debug_font = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()

    cfg = load_config()
    camera_index = get_camera_index(cfg)
    fonts = {"title": error_font, "body": fps_font, "small": fps_font}

    cap = Capture(device_index=camera_index if camera_index is not None else 0)
    opened = camera_index is not None and cap.open()

    if not opened:
        error_msg = f"Camera {camera_index} not found." if camera_index is not None else None
        chosen = run_camera_select(screen, clock, fonts, error_message=error_msg)
        if chosen is None:
            pygame.quit()
            sys.exit(0)
        cfg = set_camera_index(cfg, chosen)
        save_config(cfg)
        cap = Capture(device_index=chosen)
        if not cap.open():
            _show_error_screen(screen, error_font, "Selected camera failed to open.", ERROR_DISPLAY_MS)
            pygame.quit()
            sys.exit(1)

    model_file = resource_path("assets/pose_landmarker_lite.task")
    detector = PoseDetector(model_path=model_file)
    if not detector.open():
        _show_error_screen(screen, error_font, "Pose model not found — check assets/.", ERROR_DISPLAY_MS)
        cap.release()
        pygame.quit()
        sys.exit(1)

    def quit_game():
        detector.close()
        cap.release()
        pygame.quit()
        sys.exit()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_game()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                quit_game()

        frame = cap.get_frame()
        if frame is not None:
            h, w = frame.shape[:2]
            surf = pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")
            surf = scale_and_crop(surf, WINDOW_W, WINDOW_H)
            screen.blit(surf, (0, 0))
        else:
            screen.fill((0, 0, 0))
            frame = None

        timestamp_ms = pygame.time.get_ticks()
        result = detector.process(frame, timestamp_ms) if frame is not None else None
        landmarks = detector.get_landmarks(result)

        if landmarks is not None:
            draw_skeleton(screen, landmarks, WINDOW_W, WINDOW_H)

        if not detector.body_visible(landmarks):
            draw_body_warning(screen, warning_font, WINDOW_W, WINDOW_H)

        if debug_mode and landmarks is not None:
            draw_debug_panel(screen, landmarks, debug_font, WINDOW_W, WINDOW_H)

        fps_text = fps_font.render(f"{clock.get_fps():.0f} FPS", True, (0, 255, 0))
        screen.blit(fps_text, (8, 8))

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
