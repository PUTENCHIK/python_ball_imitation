import cv2
import pathlib
import numpy as np

import pygame as pg
import pymunk.pygame_util

pymunk.pygame_util.positive_y_is_up = False


def on_mouse_callback(event, x, y, *params):
    global ball_position
    if event == cv2.EVENT_LBUTTONDOWN:
        ball_position = [y, x]


def lupdate(value):
    global low
    low = value


def uupdate(value):
    global up
    up = value


def dupdate(value):
    global delta
    delta = value


def get_poligons(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([standart_pixel[0] - delta, standart_pixel[1] - delta, standart_pixel[2] - delta])
    upper = np.array([standart_pixel[0] + delta, standart_pixel[1] + delta, standart_pixel[2] + delta])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.dilate(mask, np.ones((10, 10)))
    thresh = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, result = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    big_contours = dict()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10_000:
            big_contours[area] = contour

    poligons = []
    for contour_area in sorted(big_contours)[-amount - 1:]:
        contour = big_contours[contour_area]

        eps = 1e-3 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)
        new_approx = []
        for i, point in enumerate(approx):
            new_approx += [list(point[0])]
        poligons += [new_approx]

    return poligons


def get_board_bounds(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    up_limit = 56
    upper = np.array([up_limit, up_limit, up_limit])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, np.ones((5, 5)))
    mask = cv2.dilate(mask, np.ones((10, 10)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = [point[0] for cnt in contours for point in cnt]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    bounds = {
        'max_y': int(max(ys)),
        'max_x': int(max(xs)),
        'min_y': int(min(ys)),
        'min_x': int(min(xs)),
    }

    return bounds, mask


def create_space(h: int, w: int, board_bounds: dict):
    space = pymunk.Space()
    space.gravity = 0, 8000

    outer_bound_width = 5
    inner_bound_width = 5

    print(board_bounds['max_y'], board_bounds['min_y'])
    print(h - board_bounds['max_y'], h - board_bounds['min_y'])

    segments = [
        pymunk.Poly(space.static_body, ((0, 0), (w, outer_bound_width))),
        pymunk.Poly(space.static_body, ((0, 0), (outer_bound_width, h))),
        pymunk.Poly(space.static_body, ((0, h - outer_bound_width), (w, h))),
        pymunk.Poly(space.static_body, ((w - outer_bound_width, 0), (w, h))),

        pymunk.Poly(space.static_body,
                    ((board_bounds['max_x'], h - board_bounds['max_y'] - inner_bound_width),
                     (board_bounds['min_x'], h - board_bounds['max_y']))
                    ),
        pymunk.Poly(space.static_body,
                    ((board_bounds['min_x'], h - board_bounds['max_y']),
                     (board_bounds['min_x'] + inner_bound_width, h - board_bounds['min_y']))
                    ),
        pymunk.Poly(space.static_body,
                    ((board_bounds['max_x'], h - board_bounds['min_y']),
                     (board_bounds['min_x'], h - board_bounds['min_y'] - inner_bound_width))
                    ),
        pymunk.Poly(space.static_body,
                    ((board_bounds['max_x'] - inner_bound_width, h - board_bounds['max_y']),
                     (board_bounds['max_x'], h - board_bounds['min_y']))
                    ),
    ]
    for s in segments:
        space.add(s)
        s.elasticity = 0.8
        s.friction = 1.0

    return space


def add_poligons_to_space(space: pymunk.Space, poligons: list):
    for poligon in poligons:
        n = len(poligon)
        for i in range(n // 2):
            polygon_shape = pymunk.Poly(space.static_body, [
                poligon[i], poligon[i + 1], poligon[n - i - 1], poligon[n - i - 2]
            ])
            polygon_shape.friction = 0.1
            space.add(polygon_shape)
        # polygon_shape = pymunk.Poly(space.static_body, poligon)
        # polygon_shape.friction = 0.1
        # space.add(polygon_shape)


def create_circle(space, pos, radius):
    square_mass, square_size = 0.2, (150, 150)
    square_moment = pymunk.moment_for_box(square_mass, square_size)
    square_body = pymunk.Body(square_mass, square_moment)
    square_body.position = pos
    square_shape = pymunk.Circle(square_body, radius)
    square_shape.elasticity = 0
    square_shape.friction = 0.1
    space.add(square_body, square_shape)
    return square_body


main_window = "Window"
debug_window = "Changed"
cv2.namedWindow(main_window, cv2.WINDOW_NORMAL)
# cv2.namedWindow(debug_window, cv2.WINDOW_NORMAL)

is_camera_available = False
camera = cv2.VideoCapture("rtsp://192.168.254.3:8080/h264_ulaw.sdp") if is_camera_available else None

cv2.setMouseCallback(main_window, on_mouse_callback)
ball_position = None

low, up = 44, 65
delta = 35
amount = 10
FPS = 60
circle_radius = 50

cv2.createTrackbar("Lower", main_window, low, 255, lupdate)
cv2.createTrackbar("Upper", main_window, up, 255, uupdate)
cv2.createTrackbar("Delta", main_window, delta, 255, dupdate)

standart_pixel = [174, 57, 148]

path = pathlib.Path(__file__).parent
screen = cv2.imread("screen.jpg")

height, width = screen.shape[:2]
print(f"Frame size: h,w = {height}, {width}")

poligons = get_poligons(screen)
board_bounds, _ = get_board_bounds(screen)
print(board_bounds)

space = create_space(height, width, board_bounds)
add_poligons_to_space(space, poligons)

# pg.init()
# surface = pg.display.set_mode((width, height))
clock = pg.time.Clock()
# draw_options = pymunk.pygame_util.DrawOptions(surface)

# start_pos = (800, 100)
start_pos = (1180, 350)
circle = create_circle(space, start_pos, circle_radius)

while True:
    frame = screen.copy()
    # _, frame = get_board_bounds(frame)
    # surface.fill(pg.Color('black'))

    # for i in pg.event.get():
    #     if i.type == pg.QUIT:
    #         exit()

    # print(circle.position)

    x, y = circle.position
    cv2.circle(frame,
               (int(x), int(y)),
               circle_radius // 2,
               (180, 0, 0),
               circle_radius)

    cv2.imshow(main_window, frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    #
    space.step(1 / FPS)
    # space.debug_draw(draw_options)
    # pg.display.flip()
    clock.tick(FPS)

# camera.release()
cv2.destroyAllWindows()
