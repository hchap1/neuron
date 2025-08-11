
// How important is new information
pub const ALPHA: f64 = 0.9;

// How important are the long-term rewards relative to immediate reward (greedy)
pub const GAMMA: f64 = 0.95;

// How quickly we transition from guessing to applying knowledge
pub const EPSILON_DECAY: f64 = 0.9995;

fn main() {
    println!("Hello, world!");
}
