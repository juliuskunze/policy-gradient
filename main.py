import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gym.spaces import Box
from matplotlib import pyplot as plt

tf.enable_eager_execution()


def main(batch_size=128):
    env = gym.make("HalfCheetah-v2")

    action_size = env.action_space.shape[0] if isinstance(env.action_space,
                                                          Box) else env.action_space.n
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(action_size)
    ])

    def policy(observation):
        return tfp.distributions.Categorical(model(np.array([observation], np.float32))[0])

    optimizer = tf.train.AdamOptimizer()

    episode_returns = []
    mean_return = 0

    for i in range(250):
        observation_batch = []
        action_batch = []
        weight_batch = []

        while len(observation_batch) < batch_size:
            observation = env.reset()
            episode_length = 0
            episode_done = False
            episode_rewards = []

            while not episode_done:
                action = policy(observation).sample().numpy()
                observation_batch.append(observation)
                episode_length += 1
                action_batch.append(action)

                observation, reward, episode_done, info = env.step(action)

                episode_rewards.append(reward)

            def rewards_to_go():
                sum = 0
                out = []

                for r in reversed(episode_rewards):
                    sum += r
                    out.append(sum)

                return list(reversed(out))

            episode_return = sum(episode_rewards)
            print(i, episode_return, mean_return)
            episode_returns.append(episode_return)
            mean_return = np.mean(episode_returns[-100:])
            weight_batch += rewards_to_go()

        with tf.GradientTape() as tape:
            log_probs = policy(observation_batch).log_prob(action_batch)

            loss = -tf.reduce_mean(log_probs * weight_batch, axis=0)

        gradients = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(gradients, model.variables),
                                  global_step=tf.train.get_or_create_global_step())

    plt.plot(episode_returns)
    plt.show()

    while True:
        observation = env.reset()

        while True:
            action = policy(observation).sample().numpy()
            observation, reward, episode_done, info = env.step(action)

            env.render()

            if episode_done:
                break


if __name__ == '__main__':
    main()
