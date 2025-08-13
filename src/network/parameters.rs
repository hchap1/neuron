#[derive(Clone, Copy)]
pub struct NetworkParameters {

    // --------- CONSTANT ----------- //

    // How important is new information
    pub alpha: f64,

    // How important are the long-term rewards relative to immediate reward (greedy)
    pub gamma: f64,

    // How quickly we transition from guessing to applying knowledge
    pub epsilon_decay: f64,

    // The minimum proportion of moves that are random
    pub minimum_epsilon: f64,

    // Number of times the episode is run to train the model
    pub num_episodes: usize,

    // Maximum number of steps the model can fail before episode is terminated
    pub max_steps: usize,

    // --------- MUTABLE ----------- //

    // What portion of our guesses are entirely random
    pub epsilon: f64

}

impl Default for NetworkParameters {
    fn default() -> Self {
        Self {
            alpha: 0.01,
            gamma: 0.1,
            epsilon_decay: 0.9995,
            minimum_epsilon: 0.01,
            num_episodes: 10000,
            max_steps: 1000,
            epsilon: 1.0,
        }
    }
}
