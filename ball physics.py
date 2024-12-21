import pymunk
import pygame
import sys

# Константы
WIDTH, HEIGHT = 800, 600
FPS = 60

# Инициализация Pygame и Pymunk
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
space = pymunk.Space()
space.gravity = (0, 800)  # гравитация вниз

# Функция для создания линии
def create_line(space, start_pos, end_pos):
    line_shape = pymunk.Segment(space.static_body, start_pos, end_pos, 5)
    line_shape.friction = 0.5
    space.add(line_shape)

# Создание линий
create_line(space, (100, 500), (700, 500))  # горизонтальная линия
create_line(space, (300, 400), (500, 400))  # второй уровень

# Создание шара
def create_ball(space, position):
    mass = 1
    radius = 20
    moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
    body = pymunk.Body(mass, moment)
    body.position = position
    shape = pymunk.Circle(body, radius)
    shape.friction = 0.5
    space.add(body, shape)
    return shape

# Создаем шар
ball = create_ball(space, (400, 300))

# Главный цикл игры
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Обновляем физику
    space.step(1 / FPS)

    # Отрисовка
    screen.fill((255, 255, 255))  # белый фон

    # Отрисовка линий
    for shape in space.shapes:
        if isinstance(shape, pymunk.Segment):
            pygame.draw.line(screen, (0, 0, 0), shape.a, shape.b, 5)

    # Отрисовка шара
    pygame.draw.circle(screen, (255, 0, 0), (int(ball.body.position.x), int(ball.body.position.y)), 20)

    pygame.display.flip()
    clock.tick(FPS)
