# pip packages
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
#from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.framework import try_import_tf
import gym
#Other packages in this repo
from util.settings import *

tf1, tf, tfv = try_import_tf()

class ParametricActionsModel(TFModelV2):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, gym.spaces.Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        super(ParametricActionsModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # self.input_layer = tf.keras.layers.Input(shape=orig_space["observations"].shape, name="observations")
        # hidden_layer = tf.keras.layers.Dense(
        #     NUM_HIDDEN, 
        #     activation="relu", 
        #     name="hidden_1", 
        #     kernel_initializer=normc_initializer(1.0)
        #     )(self.input_layer)
        # output_layer = tf.keras.layers.Dense(
        #     num_outputs, 
        #     activation="softmax", 
        #     name="actions", 
        #     kernel_initializer=normc_initializer(0.01)
        #     )(hidden_layer)
        # value_layer = tf.keras.layers.Dense(
        #     1, 
        #     name="value_out", 
        #     activation=None,
        #     kernel_initializer=normc_initializer(0.01)
        #     )(hidden_layer)
        # self.action_embed_model = tf.keras.Model(inputs = self.input_layer, outputs = [output_layer, value_layer])

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()

