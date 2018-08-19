import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np


class ContextualBandit:
    """
    Here we define our contextual bandits. In this example, we are using three
    four-armed bandit. What this means is that each bandit has four arms that
    can be pulled. Each bandit has different success probabilities for each arm,
    and as such requires different actions to obtain the best result.

    *pull_bandit:* generates a random number from a normal distribution with a
    mean of 0. The lower the bandit number, the more likely a positive reward
    will be returned. We want our agent to learn to always choose the bandit-arm
    that will most often give a positive reward, depending on the Bandit
    presented.
    """
    def __init__(self):
        self.state = 0
        # List out our bandits. Currently arms 4, 2,
        # and 1 (respectively) are the most optimal.
        self.bandits = np.array([
            [0.2, 0, -0.0, -5],
            [0.1, -5, 1, 0.25],
            [-5, 5, 5, 5]
        ])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def get_bandit(self):
        """
        Returns a random state for each episode.

        :return: (Integer) - The state of the bandt
        """
        self.state = np.random.randint(0, len(self.bandits))
        return self.state

    def pull_arm(self, action):
        """
        Check if your action is correct and returns a reward if true

        :param action:
        :return:
        """
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            # return a positive reward.
            return 1
        else:
            # return a negative reward.
            return -1


class Agent:
    """
    The code below established our simple neural agent. It takes as input the
    current state, and returns an action. This allows the agent to take actions
    which are conditioned on the state of the environment, a critical step
    toward being able to solve full RL problems. The agent uses a single set of
    weights, within which each value is an estimate of the value of the return
    from choosing a particular arm given a bandit. We use a policy gradient
    method to update the agent by moving the value for the selected action
    toward the received reward.
    """
    def __init__(self, lr, s_size, a_size):
        # These lines established the feed-forward part of the network.
        # The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32)

        state_in_oh = slim.one_hot_encoding(self.state_in, s_size)

        output = slim.fully_connected(
            state_in_oh,
            a_size,
            biases_initializer=None,
            activation_fn=tf.nn.sigmoid,
            weights_initializer=tf.ones_initializer()
        )
        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.argmax(self.output, 0)

        # The next six lines establish the training procedure. We feed the
        # reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)

        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)

        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])

        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

        self.update = optimizer.minimize(self.loss)


"""
We will train our agent by getting a state from the environment, take an action,
and receive a reward. Using these three things, we can know how to properly 
update our network in order to more often choose actions given states that will 
yield the highest rewards over time.
"""
# Clear the TensorFlow graph.
tf.reset_default_graph()

# Load the bandits.
c_bandit = ContextualBandit()

# Load the agent.
my_agent = Agent(
    lr=0.001,
    s_size=c_bandit.num_bandits,
    a_size=c_bandit.num_actions
)

# The weights we will evaluate to look into the network.
weights = tf.trainable_variables()[0]

# Set total number of episodes to train agent on.
total_episodes = 10000

# Set scoreboard for bandits to 0.
total_reward = np.zeros([c_bandit.num_bandits, c_bandit.num_actions])

# Set the chance of taking a random action.
e = 0.1

init = tf.global_variables_initializer()

# Need to Initialize so we can use it outside of the TensorFlow Session
ww = None

# Launch the TensorFlow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        # Get a state from the environment.
        s = c_bandit.get_bandit()

        # Choose either a random action or one from our network.
        if np.random.rand(1) < e:
            action = np.random.randint(c_bandit.num_actions)
        else:
            action = sess.run(
                my_agent.chosen_action,
                feed_dict={my_agent.state_in: [s]}
            )

        # Get our reward for taking an action given a bandit.
        reward = c_bandit.pull_arm(action)

        # Update the network.
        feed_dict = {
            my_agent.reward_holder: [reward],
            my_agent.action_holder: [action],
            my_agent.state_in: [s]
        }
        _, ww = sess.run([my_agent.update, weights], feed_dict=feed_dict)

        # Update our running tally of scores.
        total_reward[s, action] += reward
        if i % 500 == 0:
            print("Mean reward for each of the " + str(c_bandit.num_bandits)
                  + " bandits: " + str(np.mean(total_reward, axis=1)))
        i += 1

print("#################################################")
print("Number of Bandits: ", c_bandit.num_bandits)
print("Number of Actions: ", c_bandit.num_actions)
print("#################################################")

for a in range(c_bandit.num_bandits):
    print("The agent thinks action " + str(np.argmax(ww[a]) + 1)
          + " for bandit " + str(a + 1) + " is the most promising....")
    if np.argmax(ww[a]) == np.argmin(c_bandit.bandits[a]):
        print("...and it was right!")
    else:
        print("...and it was wrong!")
