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
def create_circle(space, pos):
    square_mass, square_size = 1, (60, 60)
    square_moment = pymunk.moment_for_box(square_mass, square_size)
    square_body = pymunk.Body(square_mass, square_moment)
    square_body.position = pos
    square_shape = pymunk.Circle(square_body, 25)
    square_shape.elasticity = 0.4
    square_shape.friction = 1.0
    square_shape.color = [randrange(256) for i in range(4)]
    space.add(square_body, square_shape)
    return square_body


        
def create_curve(space, points):
    # Проверяем, что массив точек содержит больше 1 точки
    if len(points) < 2:
        raise ValueError("Должно быть как минимум 2 точки для создания кривой.")

    # Создаем отрезки между заданными точками
    for i in range(len(points) - 1):
        line_shape = pymunk.Segment(space.static_body, points[i], points[i + 1], 5)
        line_shape.friction = 0.5
        space.add(line_shape)
        
def create_polygon(space, points):
    # Преобразуем точки в Pymunk формат
    pymunk_points = [(point[0], point[1]) for point in points]  # Flip y-координаты
    polygon_shape = pymunk.Poly(space.static_body, pymunk_points)
    polygon_shape.friction = 0.1
    space.add(polygon_shape)
    return polygon_shape
    
#create_curve(space, (0, 100), (700, 500), num_segments=30)
points = [
    (100, 300),
    (200, 400),
    (300, 350),
    (400, 500),
    (500, 300),
    (300, 350),
    (400, 300)
]
print(create_polygon(space, points))

# Создаем полигон с заданными точками
create_polygon(space, points)
#Отрисовка
circle = None

while True:
    surface.fill(pg.Color('black'))

    for i in pg.event.get():
        if i.type == pg.QUIT:
            exit()
        # спавн кубиков
        if i.type == pg.MOUSEBUTTONDOWN:
            if i.button == 1:
                circle = create_circle(space, i.pos)
                print(i.pos)

    if circle is not None:
        x, y = circle.position
        print(f"Координаты шарика изменились: x={x:.2f}, y={y:.2f}")

    space.step(1 / FPS)
    space.debug_draw(draw_options)

    pg.display.flip()
    clock.tick(FPS)

print('end')
