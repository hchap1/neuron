use crate::network::reinforcement_learning::Game;
use crate::network::reinforcement_learning::Action;

pub enum CartAction {
    Left,
    Right
}

pub struct CartPole {
    cart_velocity: f64,
    cart_position: f64,
    pole_angle: f64
}
