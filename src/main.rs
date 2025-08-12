use crate::game::{GridNavigation, Movement};
use crate::matrix::Matrix;
use crate::network::{parameters::NetworkParameters, reinforcement_learning::Network};
use crate::network::reinforcement_learning::{Action, Game, Record, ReplayBuffer};

mod network;
mod matrix;
mod game;

fn main() {
    let mut parameters = NetworkParameters::default();
    let mut network = Network::<4, 16, 4, _>::new(parameters, |x: f64| x.max(0f64));
    let mut game = GridNavigation::<4>::new();
    let mut replay_buffer: ReplayBuffer<4, 4, Movement> = ReplayBuffer::new();
    let batch_size = 10;

    for episode in 0..parameters.num_episodes {
        let mut episode_reward = 0f64;
        let mut num_steps = 0;
        game.reset();

        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..parameters.max_steps {
            num_steps += 1;
            let current_state = game.get_state();
            let (action, raw_output) = network.choose_action::<Movement>(current_state);
            let reward = game.step(action);
            let next_state = game.get_state();
            let done = game.is_done();

            replay_buffer.push(
                Record::new(current_state, action, reward, next_state, done)
            );

            episode_reward += reward;

            if replay_buffer.len() >= batch_size {
                let batch = replay_buffer.sample(batch_size);
                let (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones
                ) = ReplayBuffer::split(batch);

                let best_actions: Vec<Movement> = next_states.iter().map(|x| network.choose_action(*x).0).collect();
                let next_q_target: Vec<Matrix<4, 1>> = next_states.iter().map(|x| network.frozen_feedforward(*x)).collect();
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
                        if done { r } else { r + parameters.gamma * q }
                    })
                    .collect();

                let q_current: Vec<Matrix<4, 1>> = states.iter().map(|x| network.feedforward(*x)).collect();
                let hidden_activations: Vec<Matrix<16, 1>> = states.iter().map(|x| network.hidden_activations(*x)).collect();
                let hidden_z: Vec<Matrix<16, 1>> = states.iter().map(|x| network.hidden_z(*x)).collect();
                let q_taken: Vec<f64> = actions
                    .iter()
                    .enumerate()
                    .map(|(i, &a)| q_current[i].get(a.to_usize()))
                    .collect();

                // Gradient Descent
                let mut output_errors: Vec<Matrix<4, 1>> = Vec::with_capacity(batch_size);
                for i in 0..batch_size {
                    let mut err_vec = Matrix::<4, 1>::zeros();
                    let action_index = actions[i].to_usize();
                    err_vec.set(action_index, q_taken[i] - targets[i]);
                    output_errors.push(err_vec);
                }

                let mut grad_w2 = Matrix::<4, 16>::zeros();
                let mut grad_b2 = Matrix::<4, 1>::zeros();

                for i in 0..batch_size {
                    grad_w2 += output_errors[i] * hidden_activations[i].transpose();
                    grad_b2 += output_errors[i];
                }

                // Gradients for hidden layer
                let mut grad_w1 = Matrix::<16, 4>::zeros(); // Accumulate dL/dW1
                let mut grad_b1 = Matrix::<16, 1>::zeros(); // Accumulate dL/db1

                for i in 0..batch_size {
                    let mut hidden_error = network.transposed_output_weight() * output_errors[i];
                    for j in 0..16 {
                        if hidden_z[i].get(j) <= 0.0 {
                            hidden_error.set(j, 0.0);
                        }
                    }

                    grad_w1 += hidden_error * states[i].transpose(); // (16×1)*(1×I) → (16×I)
                    grad_b1 += hidden_error; // (16×1)
                }

                network.update_hidden_layer(grad_w1, grad_b1, batch_size as f64);
                network.update_output_layer(grad_w2, grad_b2, batch_size as f64);
            }

            if done {
                break
            }
        }

        network.update_frozen();
        network.epsilon_decay();

        println!("Finished episode: {episode}, {episode_reward} in {num_steps} steps.");
    }

    println!("Running testcase:");
    game.reset();

    while !game.is_done() {
        game.step(network.predict::<Movement>(game.get_state()));
        println!("Dist: {}", game.manhattan_distance());
    }
}
