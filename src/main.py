import sys
import pygame
from capture import Capture
from utils import resource_path  # noqa: F401 — available for asset loading in later phases

WINDOW_W, WINDOW_H = 800, 600
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
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Corpography")
    fps_font = pygame.font.SysFont(None, 28)
    error_font = pygame.font.SysFont(None, 48)
    clock = pygame.time.Clock()

    cap = Capture(device_index=0)
    if not cap.open():
        _show_error_screen(screen, error_font, "No camera found — check your webcam.", ERROR_DISPLAY_MS)
        pygame.quit()
        sys.exit(1)

    def quit_game():
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

        fps_text = fps_font.render(f"{clock.get_fps():.0f} FPS", True, (0, 255, 0))
        screen.blit(fps_text, (8, 8))

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
