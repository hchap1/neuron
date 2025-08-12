use crate::network::parameters::NetworkParameters;
use crate::network::reinforcement_learning::Network;
use crate::network::reinforcement_learning::Game;

use crate::game::grid_navigation::*;

mod network;
mod matrix;
mod game;

fn main() {
    let parameters = NetworkParameters::default();
    let mut network = Network::<4, 16, 4, _>::new(parameters, |x: f64| x.max(0f64));

    network.train::<GridNavigation<4>, Movement>();

    let mut game = GridNavigation::new();

    while !game.is_done() {
        game.step(network.predict::<Movement>(game.get_state()));
        println!("Dist: {}", game.manhattan_distance());
    }
}
