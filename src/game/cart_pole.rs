use std::f64::consts::PI;

use rand::rngs::ThreadRng;
use rand::rng;
use rand::Rng;

use crate::matrix::Matrix;
use crate::network::reinforcement_learning::Game;
use crate::network::reinforcement_learning::Action;

#[derive(Clone, Copy, Debug)]
pub enum CartAction {
    Left,
    Right,
}

impl Action for CartAction {
    fn from_usize(val: usize) -> Self {
        match val {
            0 => Self::Left,
            1 => Self::Right,
            _ => panic!("no such action")
        }
    }

    fn to_usize(&self) -> usize {
        match self {
            Self::Left => 0,
            Self::Right => 1,
        }
    }
}

pub struct CartPole {
    cart_velocity: f64,
    cart_position: f64,
    pole_angle:    f64,
    pole_velocity: f64,
    target_x: f64,

    lifetime: f64,

    rng: ThreadRng
}

impl Game<5, CartAction> for CartPole {
    fn new() -> Self {
        let mut rng = rng();
        Self {
            cart_velocity: 0f64,
            cart_position: 0.5f64,
            pole_angle: rng.random_range(-0.35f64..=0.3564),
            pole_velocity: 0f64,
            target_x: rng.random_range(0f64..1f64),
            rng,
            lifetime: 0f64
        }
    }

    fn reset(&mut self) {
        self.cart_velocity = 0f64;
        self.cart_position = 0.5f64;
        self.pole_angle = self.rng.random_range(-0.35..=0.35);
        self.pole_velocity = 0f64;
        self.target_x = self.rng.random_range(0f64..1f64);
        self.lifetime = 0f64;
    }

    fn step(&mut self, action: CartAction) -> f64 {
        self.pole_velocity += 0.1 * (self.pole_angle.sin());
        let prev_velocity = self.cart_velocity;
        match action {
            CartAction::Left => self.cart_velocity -= 0.04,
            CartAction::Right => self.cart_velocity += 0.04,
        }
        let delta_velocity = self.cart_velocity - prev_velocity;
        self.pole_velocity -= delta_velocity * 2.5f64 * (self.pole_angle.cos());
        self.cart_position += self.cart_velocity * 0.01;
        self.pole_angle += self.pole_velocity * 0.02;
        self.lifetime += 0.05f64;
        let reward_for_balance = if self.pole_angle.to_degrees().abs() < 5f64 { 1f64 } else { -1f64 };
        let position_error = (self.cart_position - self.target_x).abs();
        let mut reward = if position_error > 10f64 { 0f64 } else { reward_for_balance + self.lifetime };
        
        if self.cart_position < self.target_x && self.pole_angle > 0f64 { reward += 0.1f64; }
        if self.cart_position > self.target_x && self.pole_angle < 0f64 { reward += 0.1f64; }
        reward
    }

    fn get_state(&self) -> Matrix<5, 1> {
        Matrix::from_flat(vec![
            self.cart_position,
            self.cart_velocity,
            self.pole_angle,
            self.pole_velocity,
            self.target_x
        ])
    }

    fn is_done(&self) -> bool {
        self.pole_angle < -PI / 2f64
        || self.pole_angle > PI / 2f64
    }
}

impl CartPole {

    pub fn set_target(&mut self, pos: f64) {
        self.target_x = pos / 800f64;
    }

    pub fn angle(&self) -> f64 {
        self.pole_angle
    }

    pub fn position(&self) -> f64 {
        self.cart_position
    }
}
