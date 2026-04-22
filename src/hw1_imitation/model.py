"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.hidden_dims = hidden_dims
        self.mlp = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[1], self.hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[2], self.chunk_size * self.action_dim),
            #In action chunking, we have to output all actions in a single step,
            # so we output chunk_size * action_dim values and reshape them into
            # (chunk_size, action_dim).
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        #MSELoss
        pred_chunk = self.sample_actions(state, num_steps=self.chunk_size)
        loss = nn.MSELoss()(pred_chunk, action_chunk)
        return loss


    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        pred_chunk = self.mlp(state)
        #Here pred_chunk is a tensor of shape (batch, chunk_size * action_dim).
        pred_chunk = pred_chunk.reshape(-1, self.chunk_size, self.action_dim)
        # The -1 in reshape is used to compute the batch size automatically based on the input
        # (like an equation). The old shape is [128,16], and chunk size and action dim are 8 and 2.
        # During the reshape we must have the same number of elements. So 128*16 = 2048.
        # The equation is: x * 8 * 2 = 2048, where the x is the -1. Solving for x gives us 128
        # which is the batch size.
        # I could have insert 128 manually, but with -1 is more flexible.

        #So we have to reshape it into (batch, chunk_size, action_dim) to get the predicted action chunk,
        # in the expected shape for the loss and evaluation functions.
        return pred_chunk



class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.hidden_dims = hidden_dims
        self.mlp = nn.Sequential(
            nn.Linear(self.state_dim + self.chunk_size*self.action_dim + 1, self.hidden_dims[0]), #+1 for the time (is a scalar)
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[1], self.hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[2], self.chunk_size * self.action_dim)
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:

        timestep = torch.rand(size=(128,1,1)) # one for each batch

        noise = torch.normal(0.0, 1.0, size=action_chunk.shape)
        interpolation = timestep * action_chunk + (1 - timestep) * noise

        interpolation = interpolation.reshape(-1, self.chunk_size*self.action_dim) # reshape to (batch, chunk_size*action_dim) to concatenate with state and time
        timestep = timestep.reshape(-1,1) # reshape to (batch, 1) to concatenate with state and interpolation

        # interpolation [128,16], state [128, 5], time [128,1]

        pred_action_chunk = self.mlp(torch.cat([state,interpolation,timestep],dim=1))
        pred_action_chunk = pred_action_chunk.reshape(-1, self.chunk_size, self.action_dim)

        flow_matching_loss = ((pred_action_chunk-(action_chunk-noise))**2).mean()
        return flow_matching_loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        init_noise = torch.normal(0.0, 1.0, size=(state.shape[0], self.chunk_size*self.action_dim))
        for step in range(num_steps):
            time = torch.Tensor([1/num_steps * (step+1)]) # linearly spaced time from 0 to 1
            pred_action_chunk = self.mlp(torch.cat([state,init_noise,time.reshape(state.shape[0],1)],dim=1))
            init_noise = init_noise + 1/num_steps * pred_action_chunk
        init_noise = init_noise.reshape(-1,self.chunk_size,self.action_dim)
        return init_noise



PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
