import cv2
import pathlib
import numpy as np

import pygame as pg
import pymunk.pygame_util

pymunk.pygame_util.positive_y_is_up = False


def on_mouse_callback(event, x, y, *params):
    global position
    if event == cv2.EVENT_LBUTTONDOWN:
        position = [y, x]


def lupdate(value):
    global low
    low = value


def uupdate(value):
    global up
    up = value


def get_poligons(frame, figure_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    low, up = 0, 91
    lower = np.array([low, figure_color[1] - delta, low])
    upper = np.array([up, figure_color[1] + delta, up])
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

    return poligons, big_contours


def get_board_bounds(frame, bound_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower = np.array([0, 0, 0])
    # up_limit = 56
    # upper = np.array([up_limit, up_limit, up_limit])
    lower = np.array([bound_color[0][0],
                      bound_color[0][1],
                      bound_color[0][2],])
    upper = np.array([bound_color[1][0],
                      bound_color[1][1],
                      bound_color[1][2],])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, np.ones((2, 2)))
    mask = cv2.dilate(mask, np.ones((10, 10)))
    thresh = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = [point[0] for cnt in contours for point in cnt]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    bounds = {
        'max_y': int(max(ys)) if len(ys) else 0,
        'max_x': int(max(xs)) if len(xs) else 0,
        'min_y': int(min(ys)) if len(ys) else 0,
        'min_x': int(min(xs)) if len(xs) else 0,
    }

    return bounds, mask


def extract_all_niggers(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower = np.array([0, 0, 0])
    # up_limit = 56
    # upper = np.array([up_limit, up_limit, up_limit])
    # lower = np.array([extract_color[0][0],
    #                   extract_color[0][1],
    #                   extract_color[0][2],])
    # upper = np.array([extract_color[1][0],
    #                   extract_color[1][1],
    #                   extract_color[1][2],])
    lower = np.array([low, 165 - delta, low])
    upper = np.array([up, 165 + delta, up])
    
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, np.ones((2, 2)))
    mask = cv2.dilate(mask, np.ones((10, 10)))
    
    thresh = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bound_index = None
    bound_area = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > bound_area:
            bound_index = i
            bound_area = area
    
    if bound_index is not None:
        print(f"bound_index = {bound_index}")
        points = [point[0] for point in contours[bound_index]]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]

        bounds = {
            'max_y': int(max(ys)) if len(ys) else 0,
            'max_x': int(max(xs)) if len(xs) else 0,
            'min_y': int(min(ys)) if len(ys) else 0,
            'min_x': int(min(xs)) if len(xs) else 0,
        }
        contours = list(contours)
        contours.pop(bound_index)
    else:
        bounds = {
            'max_y': 0,
            'max_x': 0,
            'min_y': 0,
            'min_x': 0,
        }
    
    big_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            big_contours += [contour]    

    return bounds, big_contours, mask


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
cv2.setMouseCallback(main_window, on_mouse_callback)
position = None

is_camera_available = False
camera = cv2.VideoCapture("rtsp://192.168.254.3:8080/h264_ulaw.sdp") if is_camera_available else None

low = 104
up = 111

cv2.createTrackbar("Lower", main_window, low, 255, lupdate)
cv2.createTrackbar("Upper", main_window, up, 255, uupdate)
delta = 100
amount = 10
FPS = 60
circle_radius = 50

MODE = 1
checking_position = False

cv2.createTrackbar("Lower", main_window, low, 255, lupdate)
cv2.createTrackbar("Upper", main_window, up, 255, uupdate)

standart_pixel = [15, 12, 83]

path = pathlib.Path(__file__).parent
screen = cv2.imread("screen.jpg")
static_frame = None

width, height = None, None
bounds = None
poligons, figures = [], []

# height, width = screen.shape[:2]
# 

# poligons = get_poligons(screen)
# board_bounds, _ = get_board_bounds(screen)
# print(board_bounds)

# space = create_space(height, width, board_bounds)
# add_poligons_to_space(space, poligons)

# pg.init()
# surface = pg.display.set_mode((width, height))
# clock = pg.time.Clock()
# draw_options = pymunk.pygame_util.DrawOptions(surface)

# start_pos = (800, 100)
# start_pos = (1180, 350)
# circle = create_circle(space, start_pos, circle_radius)

while True:
    if is_camera_available:
        _, frame = camera.read()
    else:
        frame = screen.copy()
    origin = frame.copy()
    # main_color = [
    #     [0, 65, 0],
    #     [111, 265, 111]
    # ]
    # print(main_color)
    
    if MODE == 1:
        if position is not None:
            cv2.circle(frame,
                    (position[1], position[0]),
                    3,
                    (255, 128, 128),
                    3)
        bounds, figures, mask = extract_all_niggers(frame)
        cv2.rectangle(frame,
                        (bounds['min_x'], bounds['min_y']),
                        (bounds['max_x'], bounds['max_y']),
                        (0, 0, 128),
                        3)
        for figure in figures:
            cv2.drawContours(frame, [figure], 0, (255, 128, 128, 3))
        
        cv2.imshow(main_window, mask)
    elif MODE == 2:
        if static_frame is not None:
            frame = static_frame.copy()
            # if position is not None:
            #     cv2.circle(frame,
            #             (position[1], position[0]),
            #             3,
            #             (255, 128, 128),
            #             3)
            # bounds, bound_mask = get_board_bounds(frame, bound_color)
            print(bounds)
            cv2.rectangle(frame,
                          (bounds['min_x'], bounds['min_y']),
                          (bounds['max_x'], bounds['max_y']),
                          (0, 0, 128),
                          3)
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            figure_color = standart_pixel if not checking_position else hsv[position[0], position[1]]
            cv2.putText(frame,
                    f"{figure_color}",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 255, 0),
                    3)
            # poligons, contours = get_poligons(frame, figure_color)
            # for contour in contours:
            #     if contour:
            #         cv2.drawContours(frame, [contour], 0, (128, 0, 0), 3)
            
            cv2.imshow(main_window, frame)
    # _, frame = get_board_bounds(frame)
    # surface.fill(pg.Color('black'))

    # for i in pg.event.get():
    #     if i.type == pg.QUIT:
    #         exit()

    # print(circle.position)

    # x, y = circle.position
    # cv2.circle(frame,
    #            (int(x), int(y)),
    #            circle_radius // 2,
    #            (180, 0, 0),
    #            circle_radius)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
    if key == ord('m'):
        MODE = 2 if MODE == 1 else 1
        if MODE == 2:
            static_frame = origin.copy()
            width, height = static_frame.shape[:2]
            print(f"Frame size: h,w = {height}, {width}")
            bounds, figures = extract_all_niggers(static_frame, main_color)
    
    if key == ord('p'):
        checking_position = not checking_position
    
    # if key == ord('s'):
    #     cv2.imwrite("screen.jpg", origin)
    #
    # space.step(1 / FPS)
    # space.debug_draw(draw_options)
    # pg.display.flip()
    # clock.tick(FPS)

if is_camera_available:
    camera.release()
cv2.destroyAllWindows()
