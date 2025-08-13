use crate::network::reinforcement_learning::DualNetwork;
use crate::network::reinforcement_learning::Game;

use crate::game::cart_pole::*;

mod network;
mod matrix;
mod game;

use simple::Point;
use simple::Rect;
use simple::Window;

fn main() {
    let mut window = Window::new("Cart Pole", 800, 800);
    window.set_color(255, 0, 0, 255);

    let mut game = CartPole::new();
    let mut network = DualNetwork::<5, 16, 32, 2, _>::new(|x: f64| x.max(0f64));

    network.train::<CartPole, CartAction>();

    while window.next_frame() {
        let action = network.predict::<CartAction>(game.get_state());
        game.step(action);
        window.clear_to_color(255, 255, 255);

        let cart_pos = (game.position() * 800f64).round() as i32;
        let pole_offset = ((game.angle().sin()) * 100f64).round() as i32;
        let pole_height = ((game.angle().cos()) * 100f64).round() as i32;

        window.fill_rect(Rect::new(cart_pos - 100, 400, 200, 50));
        window.fill_rect(Rect::new(cart_pos + pole_offset, 400 - pole_height, 5, 5));
        window.draw_polygon(vec![
            Point::new(cart_pos + pole_offset - 5, 400 - pole_height),
            Point::new(cart_pos + pole_offset + 5, 400 - pole_height),
            Point::new(cart_pos + 5, 400),
            Point::new(cart_pos - 5, 400)
        ]);

        if window.is_key_down(simple::Key::Space) {
            game.reset();
        }

        if game.is_done() {
            game.reset();
        }

        game.set_target(window.mouse_position().0 as f64);
    }
}
