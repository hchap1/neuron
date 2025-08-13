import pygame, math

_ = pygame.init()
screen = pygame.display.set_mode((800, 800))
clock = pygame.time.Clock()

cart_velocity = 0
cart_position = 400

pole_angle = math.radians(10)
pole_velocity = 0

running = True

while running:

    dt = clock.tick(60) / 1000

    pole_velocity += 5 * math.sin(pole_angle) * dt
    pole_angle += pole_velocity * dt

    if pole_angle > math.radians(90) or pole_angle < math.radians(-90):
        pole_angle = 0
        pole_velocity = 0
        cart_velocity = 0
        cart_position = 400

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    old_velocity = cart_velocity
    if keys[pygame.K_a]: cart_velocity -= dt * 2000
    if keys[pygame.K_d]: cart_velocity += dt * 2000
    acceleration = cart_velocity - old_velocity
    pole_velocity += acceleration * -0.005 * math.cos(pole_angle)
    cart_position += cart_velocity * dt * 0.5

    _ = screen.fill((255, 255, 255))

    pole_x = 100 * math.sin(pole_angle) + cart_position
    pole_y = 400 - 100 * math.cos(pole_angle)

    _ = pygame.draw.line(screen, (0, 0, 0), (cart_position, 400), (pole_x, pole_y), 5)
    _ = pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(cart_position - 100, 400, 200, 10))

    pygame.display.flip()
pygame.quit()
