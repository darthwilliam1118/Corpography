import sys
import pygame
from capture import Capture
from pose import PoseDetector
from ui.display import draw_skeleton, draw_body_warning, draw_debug_panel
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

    cap = Capture(device_index=0)
    if not cap.open():
        _show_error_screen(screen, error_font, "No camera found — check your webcam.", ERROR_DISPLAY_MS)
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
            surf = pygame.transform.scale(surf, (WINDOW_W, WINDOW_H))
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
