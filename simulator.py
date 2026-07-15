# the pygame "game" will go here
# here is where the ball collision simulation happens
# and where the joints and head will create shapes for the body tracking
import pygame
import cv2
import numpy as np

from body_tracker import BodyTracker


class Simulator:
    WIDTH: int = 800
    HEIGHT: int = 600
    FPS: int = 60
    BG_COLOR: tuple[int, int, int] = (20, 20, 20)

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("HRTFBallSoundsSimulation")

        self.clock = pygame.time.Clock()
        self.running = True

        self.bt = BodyTracker()

    def _convert_frame_to_surface(self, cv_frame):
        cv_frame = np.swapaxes(cv_frame, 0, 1)
        surface = pygame.surfarray.make_surface(cv_frame)
        surface = pygame.transform.smoothscale(surface, (self.WIDTH, self.HEIGHT))
        return surface

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

    def _update(self, frame: np.ndarray):
        """Update physics, body tracking data, and game state."""
        # Future physics updates (e.g., self.ball.update()) will go here
        pass

    def _render(self, frame: np.ndarray):
        """draw visuals to screen"""
        frame_surface = self._convert_frame_to_surface(frame)
        self.screen.blit(frame_surface,(0,0))
        pygame.display.flip()


    def start(self):
        while self.running:
            # Event handling...
            current_frame: np.ndarray = self.bt.track_body()

            self._handle_events()
            self._update(current_frame)
            self._render(current_frame)

            # Update physics and render...
            self.clock.tick(self.FPS)

        pygame.quit()



s = Simulator()
s.start()