import sys
import pygame
from utils import resource_path  # noqa: F401 — available for asset loading in later phases


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Corpography")
    font = pygame.font.SysFont(None, 72)
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

        screen.fill((0, 0, 0))
        text = font.render("Hello World", True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=(400, 300)))
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
