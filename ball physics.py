import pygame as pg
from random import randrange
import pymunk.pygame_util
pymunk.pygame_util.positive_y_is_up = False

#параметры PyGame
RES = WIDTH, HEIGHT = 900, 720
FPS = 60

pg.init()
surface = pg.display.set_mode(RES)
clock = pg.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(surface)

#настройки Pymunk
space = pymunk.Space()
space.gravity = 0, 8000

#платформа
segment_shape = pymunk.Segment(space.static_body, (2, HEIGHT), (WIDTH, HEIGHT), 26)
space.add(segment_shape)
segment_shape.elasticity = 0.8
segment_shape.friction = 1.0


#квадратики
body = pymunk.Body()
def create_square(space, pos):
    square_mass, square_size = 1, (60, 60)
    square_moment = pymunk.moment_for_box(square_mass, square_size)
    square_body = pymunk.Body(square_mass, square_moment)
    square_body.position = pos
    square_shape = pymunk.Circle(square_body, 25)
    square_shape.elasticity = 0.4
    square_shape.friction = 1.0
    square_shape.color = [randrange(256) for i in range(4)]
    space.add(square_body, square_shape)

def create_curve(space, start_pos, end_pos, num_segments=20):
    points = []
    for i in range(num_segments + 1):
        # Используем уравнение квадратичной кривой (например, параболы)
        t = i / num_segments
        x = (1 - t) * start_pos[0] + t * end_pos[0]  # Линейная интерполяция по x
        y = (1 - t) * start_pos[1] + t * end_pos[1] - 600 * (t - 0.5) ** 2  # Кривая
        points.append((x, y))
    
    for i in range(len(points) - 1):
        line_shape = pymunk.Segment(space.static_body, points[i], points[i + 1], 5)
        line_shape.friction = 0.5
        space.add(line_shape)
create_curve(space, (0, 100), (700, 500), num_segments=30)

#Отрисовка
while True:
    surface.fill(pg.Color('black'))

    for i in pg.event.get():
        if i.type == pg.QUIT:
            exit()
        # спавн кубиков
        if i.type == pg.MOUSEBUTTONDOWN:
            if i.button == 1:
                create_square(space, i.pos)
                print(i.pos)

    space.step(1 / FPS)
    space.debug_draw(draw_options)

    pg.display.flip()
    clock.tick(FPS)

print('end')
