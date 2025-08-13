use rand::{rng, rngs::ThreadRng, seq::IndexedRandom, Rng};
use crate::{matrix::Matrix, network::parameters::NetworkParameters};

pub trait Action: Copy {
    fn to_usize(&self) -> usize;
    fn from_usize(val: usize) -> Self;
}

pub trait Game<const I: usize, A: Action> {
    fn new() -> Self where Self: Sized;
    fn reset(&mut self);
    fn step(&mut self, action: A) -> f64;
    fn get_state(&self) -> Matrix<I, 1>;
    fn is_done(&self) -> bool;
}

pub struct Record<const I: usize, const O: usize, A: Action> {
    state_before_choice: Matrix<I, 1>,
    chosen_action: A,
    reward: f64,
    state_after_choice: Matrix<I, 1>,
    done: bool
}

impl<const I: usize, const O: usize, A: Action> Record<I, O, A> {
    pub fn new(
        state_before_choice: Matrix<I, 1>,
        chosen_action: A,
        reward: f64,
        state_after_choice: Matrix<I, 1>,
        done: bool
    ) -> Self { Self {
        state_before_choice,
        chosen_action,
        reward,
        state_after_choice,
        done
    }}
}

pub struct ReplayBuffer<const I: usize, const O: usize, A: Action> {
    buffer: Vec<Record<I, O, A>>,
    rng: ThreadRng
}

impl<const I: usize, const O: usize, A: Action> ReplayBuffer<I, O, A> {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            rng: rng()
        }
    }

    pub fn push(&mut self, record: Record<I, O, A>) {
        self.buffer.push(record);
    }

    pub fn sample(&mut self, batch_size: usize) -> Vec<&Record<I, O, A>> {
        self.buffer
            .choose_multiple(&mut self.rng, batch_size)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    #[allow(clippy::complexity)]
    pub fn split(records: Vec<&Record<I, O, A>>) -> (
        Vec<Matrix<I, 1>>,
        Vec<A>,
        Vec<f64>,
        Vec<Matrix<I, 1>>,
        Vec<bool>
    ) {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();
        let mut d = Vec::new();
        let mut e = Vec::new();

        for record in records {
            a.push(record.state_before_choice);
            b.push(record.chosen_action);
            c.push(record.reward);
            d.push(record.state_after_choice);
            e.push(record.done)
        }

        (a, b, c, d, e)
    }
}

pub struct Network<const I: usize, const H: usize, const O: usize, M: Fn(f64) -> f64> {
    parameters: NetworkParameters,
    activation: M,
    rng: ThreadRng,

    online_hidden_weights: Matrix<H, I>,
    online_hidden_biases: Matrix<H, 1>,
    online_output_weights: Matrix<O, H>,
    online_output_biases: Matrix<O, 1>,
    frozen_hidden_weights: Matrix<H, I>,
    frozen_hidden_biases: Matrix<H, 1>,
    frozen_output_weights: Matrix<O, H>,
    frozen_output_biases: Matrix<O, 1>,
}

impl<const I: usize, const H: usize, const O: usize, M: Fn(f64) -> f64> Network<I, H, O, M> {

    pub fn epsilon_decay(&mut self) {
        self.parameters.epsilon *= self.parameters.epsilon_decay;
        self.parameters.epsilon = self.parameters.epsilon.max(self.parameters.minimum_epsilon);
    }

    pub fn update_frozen(&mut self) {
        self.frozen_hidden_weights = self.online_hidden_weights;
        self.frozen_hidden_biases = self.online_hidden_biases;
        self.frozen_output_weights = self.online_output_weights;
        self.frozen_output_biases = self.online_output_biases;
    }

    pub fn transposed_output_weight(&self) -> Matrix<H, O> {
        self.online_output_weights.transpose()
    }

    pub fn update_hidden_layer(&mut self, grad_w1: Matrix<H, I>, grad_b1: Matrix<H, 1>, batch_size: f64) {
        self.online_hidden_weights -= grad_w1 * (self.parameters.alpha / batch_size);
        self.online_hidden_biases -= grad_b1 * (self.parameters.alpha / batch_size);
    }

    pub fn update_output_layer(&mut self, grad_w2: Matrix<O, H>, grad_b2: Matrix<O, 1>, batch_size: f64) {
        self.online_output_weights -= grad_w2 * (self.parameters.alpha / batch_size);
        self.online_output_biases -= grad_b2 * (self.parameters.alpha / batch_size);
    }

    pub fn new(
        parameters: NetworkParameters,
        activation: M,
    ) -> Self {

        let mut rng = rng();

        let online_hidden_weights = Matrix::<H, I>::he_dist(&mut rng);
        let online_hidden_biases = Matrix::<H, 1>::zeros();
        let online_output_weights = Matrix::<O, H>::he_dist(&mut rng);
        let online_output_biases = Matrix::<O, 1>::zeros();
        let frozen_hidden_weights = Matrix::<H, I>::he_dist(&mut rng);
        let frozen_hidden_biases = Matrix::<H, 1>::zeros();
        let frozen_output_weights = Matrix::<O, H>::he_dist(&mut rng);
        let frozen_output_biases = Matrix::<O, 1>::zeros();

        Self {
            parameters,
            activation,
            rng,
            online_hidden_weights,
            online_hidden_biases,
            online_output_weights,
            online_output_biases,
            frozen_hidden_weights,
            frozen_hidden_biases,
            frozen_output_weights,
            frozen_output_biases,
        }
    }

    pub fn hidden_activations(&self, inputs: Matrix<I, 1>) -> Matrix<H, 1> {
        let h_raw = (self.online_hidden_weights * inputs) + self.online_hidden_biases;
        h_raw.map(&self.activation)
    }

    pub fn hidden_z(&self, inputs: Matrix<I, 1>) -> Matrix<H, 1> {
        (self.online_hidden_weights * inputs) + self.online_hidden_biases
    }

    pub fn feedforward(&self, inputs: Matrix<I, 1>) -> Matrix<O, 1> {
        let h_raw = (self.online_hidden_weights * inputs) + self.online_hidden_biases;
        let h = h_raw.map(&self.activation);
        (self.online_output_weights * h) + self.online_output_biases
    }

    pub fn frozen_feedforward(&self, inputs: Matrix<I, 1>) -> Matrix<O, 1> {
        let h_raw = (self.frozen_hidden_weights * inputs) + self.frozen_hidden_biases;
        let h = h_raw.map(&self.activation);
        (self.frozen_output_weights * h) + self.frozen_output_biases
    }

    pub fn epsilon_greedy<A: Action>(&mut self, inputs: Matrix<I, 1>) -> A {
        let q = self.feedforward(inputs);
        if self.rng.random::<f64>() > self.parameters.epsilon {
            // Let the model choose
            A::from_usize(q.argmax().0)
        } else {
            // Pick randomly
            A::from_usize(self.rng.random_range(0_usize..O))
        }
    }

    pub fn predict<A: Action>(&mut self, inputs: Matrix<I, 1>) -> A {
        A::from_usize(self.feedforward(inputs).argmax().0)
    }

    pub fn train<G: Game<I, A>, A: Action>(&mut self) {
        let mut game = G::new();
        let mut replay_buffer: ReplayBuffer<I, O, A> = ReplayBuffer::new();
        let batch_size = 32;

        for episode in 0..self.parameters.num_episodes {
            game.reset();

            let mut num_steps = 0;

            for _ in 0..self.parameters.max_steps {
                num_steps += 1;
                let current_state = game.get_state();
                let action = self.epsilon_greedy::<A>(current_state);
                let reward = game.step(action);
                let next_state = game.get_state();
                let done = game.is_done();

                replay_buffer.push(
                    Record::new(current_state, action, reward, next_state, done)
                );

                if replay_buffer.len() >= batch_size {
                    let batch = replay_buffer.sample(batch_size);
                    let (
                        states,
                        actions,
                        rewards,
                        next_states,
                        dones
                    ) = ReplayBuffer::split(batch);

                    let best_actions: Vec<A> = next_states.iter().map(|x| self.predict(*x)).collect();
                    let next_q_target: Vec<Matrix<O, 1>> = next_states.iter().map(|x| self.frozen_feedforward(*x)).collect();
                    let q_next: Vec<f64> = best_actions
                        .iter()
                        .enumerate()
                        .map(|(i, &a)| next_q_target[i].get(a.to_usize()))
                        .collect();

                    let targets: Vec<f64> = rewards
                        .iter()
                        .zip(q_next.iter())
                        .zip(dones.iter())
                        .map(|((&r, &q), &done)| {
                            if done { r } else { r + self.parameters.gamma * q }
                        })
                        .collect();

                    let q_current: Vec<Matrix<O, 1>> = states.iter().map(|x| self.feedforward(*x)).collect();
                    let hidden_activations: Vec<Matrix<H, 1>> = states.iter().map(|x| self.hidden_activations(*x)).collect();
                    let hidden_z: Vec<Matrix<H, 1>> = states.iter().map(|x| self.hidden_z(*x)).collect();
                    let q_taken: Vec<f64> = actions
                        .iter()
                        .enumerate()
                        .map(|(i, &a)| q_current[i].get(a.to_usize()))
                        .collect();

                    // Gradient Descent
                    let mut output_errors: Vec<Matrix<O, 1>> = Vec::with_capacity(batch_size);
                    for i in 0..batch_size {
                        let mut err_vec = Matrix::<O, 1>::zeros();
                        let action_index = actions[i].to_usize();
                        err_vec.set(action_index, q_taken[i] - targets[i]);
                        output_errors.push(err_vec);
                    }

                    // Gradients for output layer
                    let mut grad_w2 = Matrix::<O, H>::zeros();
                    let mut grad_b2 = Matrix::<O, 1>::zeros();

                    for i in 0..batch_size {
                        grad_w2 += output_errors[i] * hidden_activations[i].transpose();
                        grad_b2 += output_errors[i];
                    }

                    // Gradients for hidden layer
                    let mut grad_w1 = Matrix::<H, I>::zeros();
                    let mut grad_b1 = Matrix::<H, 1>::zeros();

                    for i in 0..batch_size {
                        let mut hidden_error = self.transposed_output_weight() * output_errors[i];
                        for j in 0..H {
                            if hidden_z[i].get(j) <= 0.0 {
                                hidden_error.set(j, 0.0);
                            }
                        }

                        grad_w1 += hidden_error * states[i].transpose();
                        grad_b1 += hidden_error;
                    }

                    self.update_hidden_layer(grad_w1, grad_b1, batch_size as f64);
                    self.update_output_layer(grad_w2, grad_b2, batch_size as f64);
                }

                if done {
                    break
                }
            }

            println!("Episode: {episode}. Survived: {num_steps}");

            self.update_frozen();
            self.epsilon_decay();
        }
    }
}
