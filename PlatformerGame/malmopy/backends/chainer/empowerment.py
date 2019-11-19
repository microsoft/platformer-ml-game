# --------------------------------------------------------------------------------------------------
#  Copyright (c) 2018 Microsoft Corporation
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
#  associated documentation files (the "Software"), to deal in the Software without restriction,
#  including without limitation the rights to use, copy, modify, merge, publish, distribute,
#  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or
#  substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# --------------------------------------------------------------------------------------------------

"""Module containing classes representing Chainer models for empowerment approximation."""

import gym
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer.dataset import concat_examples
from ...memories import Batch
class ClosedLoopEmpowermentNet(Chain):
    """
    Class representing the network to learn inference distribution pi^q(a_t|s^q_t)
    from observations x_t, x_f (final observation after policy terminated), expressed in
    the recurrent neural network state s^q_t = f(s^q_{t-1}, x_t, a_{t-1}, x_f).
    """

    def __init__(self, input_space, output_space, embedding_size,\
                lstm_size, lstm_dropout, zero_bias=False):
        """
        Initialise this network
        Args:
            input_space     -- here space of observations x
            output_space    -- here space of actions a
            embedding_size  -- size of the embedding layers, both for
                               observations -> embeddings and embeddings -> joint embeddings
            lstm_size       -- size of the lstm hidden layer for both the behaviour
                               and inference policies
            lstm_dropout    -- lstm dropout ratio between 0 and 1, i.e. percentage of units
                               dropped to improve generalisation
            zero_bias       -- False by default, set True for low uniformly distributed initial bias
        """

        super(ClosedLoopEmpowermentNet, self).__init__()

        # TODO: Potentially relax assumptions;
        # This is presently custom-tailored to the Gridworld environment
        assert isinstance(output_space, gym.spaces.Discrete)
        assert isinstance(input_space, gym.spaces.Box)
        assert input_space.shape is not None
        assert embedding_size > 0
        assert lstm_size > 0

        self._input_space = input_space
        self._output_space = output_space
        self._lstm_size = lstm_size

        in_channels = input_space.shape[0]
        initial_bias = None if zero_bias else I.Uniform(1e-4)

        # Define the network
        with self.init_scope():
            # Embeddings are fully connected, one-layer linear networks
            # Will be wrapped in rectified linear units later for non-linearity
            # Adding initial bias means that most rectified units will have input > 0
            # and thus be active in beginning

            # Embedding for input observations x_t -> u_t
            self.lin_embed_p = L.Linear(in_size=in_channels, out_size=embedding_size,\
                                        initial_bias=initial_bias)

            # Embedding for joint embeddings (u_t, u_f) -> v
            self.lin_embed_q = L.Linear(in_size=embedding_size*2, out_size=embedding_size,\
                                        initial_bias=initial_bias)

            # LSTM for inference policy, taking input of form [h^p_t, v, a_{t-1}]
            self.lstm_q = L.NStepLSTM(n_layers=1,\
                                      in_size=(lstm_size + embedding_size + self._output_space.n),\
                                      out_size=lstm_size, dropout=lstm_dropout)

            # Linear transformation of LSTM output to actions for inference policy
            self.lin_pi_q = L.Linear(in_size=lstm_size,
                                     out_size=self._output_space.n, initial_bias=initial_bias)

    @property
    def input_space(self):
        return self._input_space

    @property
    def output_space(self):
        return self._output_space

    def sample_from_policy(self, observation_current):
        '''
        Sample from policy model.
        Args:
            observation_current -- present observation of the environment
        '''
        # TODO: Presently just random.
        return self._output_space.sample()

    def __call__(self, x):
        '''
        Do a forward pass on the inference policy

        Args:
            x -- list (!) of actions / states up to final state,
            i.e. [a_0,x_1,a_1,x_2, ..., a_{f-1}, x_f]
        '''
        #
        # PREPROCESSING

        # Sequence of (a_{t},x_{t+1}) pairs of actions and caused observations
        # concat_examples(x) converts an "array of tuples" x into a
        # "tuple of arrays" action_seq, input_observation_seq
        # I.e. from x = [a_0,x_1,a_1,x_2, ..., a_{f-1}, x_f] to
        # [a_0,a_1,...,a_{f-1}], [x_1,x_2,...,x_f]
        # where the first is a single list, and the latter are two arrays.
        action_seq, input_observation_seq = concat_examples(x)
        input_observation_seq = Batch.to_array_and_type(input_observation_seq, self.xp.float32)
        trajectory_length = action_seq.size

        # Create array of one-hot vectors for each action in the target sequence
        # We only need this for actions a_{t-1} in the LSTM,
        # so omitting last action from one-hot sequence
        prev_action_seq_oh = self.xp.zeros((trajectory_length-1, self._output_space.n),\
                                            dtype=self.xp.float32)
        prev_action_seq_oh[self.xp.arange(trajectory_length-1), action_seq[0:-1]] = 1.0

        # Wrap both input observations and target actions into variable object
        # to keep track of computational graph.
        # This is required for the backward pass / gradient computation later on
        prev_action_seq_oh = Variable(prev_action_seq_oh)
        input_observation_seq = Variable(input_observation_seq)

        #
        # FORWARD PASS

        # Create embeddings u_t of observations x_t with last
        # element = embedding of final observation x_f
        u_seq = F.relu(self.lin_embed_p(input_observation_seq))

        # Create sequence of concatenated
        # (observation embedding at t, final observation embedding) pairs [u_t, u_f] for all t < f
        # Excluding [u_f, u_f] as this is the state that we use to infer actions chosen
        # at previous states u_t, t<f
        # For concatenation, we need two arrays of same size f-1. Thus broadcasting u_f to fit size.
        # u_f=u_seq[-1], i.e. the last element in the sequence of embeddings.
        u_f = F.broadcast_to(u_seq[-1], (trajectory_length-1, u_seq[-1].size))
        input_embedding_seq = F.concat((u_seq[0:-1], u_f), axis=1)

        # Create joint embeddings v=RW[u_t,u_f] for each concatenated observation embedding
        v_seq = F.relu(self.lin_embed_q(input_embedding_seq))

        # Output from behaviour policy LSTM at different timesteps.
        # TODO: Take actual output from policy LSTM (self.lstm_p) once that's implemented.
        # Constant for now as we assume random policy for testing.
        h_p = self.xp.full((trajectory_length-1, self._lstm_size),\
                            1.0/self._lstm_size, dtype=self.xp.float32)

        # Inference LSTM input [h^p_t, v_t, a_{t-1}]
        lstm_q_in = F.concat((h_p, v_seq, prev_action_seq_oh))

        # Get into batch shape for N-step LSTM
        # LSTM expects list with 2d arrays. 1st index = time,
        # 2nd index = batch index, 3rd index = LSTM input elements
        # Here we've only got one batch, thus first dimension = trajectory length / time,
        # second dimension = 1, and third dimension is size of LSTM input.
        lstm_q_in_list = []
        for t in range(trajectory_length-1):
            lstm_q_in_list.append(F.expand_dims(lstm_q_in[t], axis=0))

        # Feed it through n-step LSTM
        # Returns a tuple: final hidden states at t=f, updated cell state at t=f,
        # updated hidden states after each t
        # We're only interested in the third, as this is the networks' prediction for time t
        h_q = self.lstm_q(None, None, lstm_q_in_list)[2]

        # Turn list of batches of hidden state vectors (list[2darray])
        # into 2-d array of hidden state vectors
        h_q = F.stack(h_q)
        h_q = F.squeeze(h_q)

        # Use another linear transformation to bring LSTM output h to the size of our action set
        # This way, the size of our action set does not constrain the expressiveness of the LSTM
        predictions = self.lin_pi_q(h_q)

        # We return the outcome predictions and target action sequences
        # Predictions = the unnormalised log-probabilities, as our inferred probabilities of
        # actions performed at t, given observation x_t and final observation x_f
        # Target action sequences = only the actions following the observations,
        # i.e. first action in original sequence omitted
        target_action_seq = action_seq[1:]
        return (predictions, target_action_seq)
