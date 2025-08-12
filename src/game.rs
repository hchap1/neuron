use rand::rng;
use rand::Rng;
use rand::rngs::ThreadRng;

use crate::matrix::Matrix;
use crate::network::reinforcement_learning::Action;
use crate::network::reinforcement_learning::Game;

#[derive(Clone, Copy)]
pub enum Movement {
    Up,
    Down,
    Right,
    Left
}

impl Action for Movement {
    fn from_usize(idx: usize) -> Self {
        match idx {
            0 => Self::Up,
            1 => Self::Down,
            2 => Self::Right,
            3 => Self::Left,
            _ => panic!("Invalid Action")
        }
    }

    fn to_usize(&self) -> usize {
        match self {
            Self::Up => 0,
            Self::Down => 1,
            Self::Right => 2,
            Self::Left => 3
        }
    }
}

pub struct GridNavigation<const I: usize> {
    player_x: f64,
    player_y: f64,
    target_x: f64,
    target_y: f64,
    rng: ThreadRng
}

impl<const I: usize> GridNavigation<I> {
    pub fn new() -> Self {
        let mut game = Self {
            player_x: 0f64,
            player_y: 0f64,
            target_x: 0f64,
            target_y: 0f64,
            rng: rng()
        };

        game.reset();
        game
    }

    pub fn manhattan_distance(&self) -> f64 {
        (self.player_x - self.target_x).abs() + (self.player_y - self.target_y).abs()
    }
}

impl<const I: usize> Game<I, Movement> for GridNavigation<I> {
    fn reset(&mut self) {
        self.player_x = self.rng.random_range(0usize..10usize) as f64;
        self.player_y = self.rng.random_range(0usize..10usize) as f64;
        self.target_x = self.rng.random_range(0usize..10usize) as f64;
        self.target_y = self.rng.random_range(0usize..10usize) as f64;
    }

    fn step(&mut self, action: Movement) -> f64 {
        let previous_distance = self.manhattan_distance();
        match action {
            Movement::Up => self.player_y += 1.0,
            Movement::Down => self.player_y -= 1.0,
            Movement::Right => self.player_x += 1.0,
            Movement::Left => self.player_x -= 1.0,
        }
        previous_distance - self.manhattan_distance()
    }

    fn get_state(&self) -> Matrix<I, 1> {
        Matrix::<I, 1>::from_flat(
            vec![self.player_x, self.player_y, self.target_x, self.target_y]
                .into_iter()
                .map(|x| x / 9f64)
                .collect()
        )
    }

    fn is_done(&self) -> bool {
        self.player_x == self.target_x && self.player_y == self.target_y
    }
}
