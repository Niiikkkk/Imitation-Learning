## 1. Action Chunking
Given a state, we want to predict not a single action, but a sequence of actions.
### Model Architecture
I have defined a simple MLP:
```python
self.mlp = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[1], self.hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[2], self.chunk_size * self.action_dim),
        )
```
The output of the MLP is of shape (batch_size, chunk_size * action_dim). We can reshape it to (batch_size, chunk_size, action_dim) to get the predicted actions for each time step in the chunk.

### Loss function
It is a simple MSE Loss between the predicted actions and the ground truth actions for the chunk.
### Inference
Just feed the state through the MLP and reshape the output to get the predicted actions for the chunk.

## 2. Action chunking with flow matching
Flow matching predicts a velocity field, that moves simple noise, to a known distribution. 
This is: ${FM}_{\theta}(x_t, t) = v(x_t, t)$
where $x_t$ is the sample at time t (where $x_0$ is the noise and $x_T$ is the data), and $v(x_t, t)$ is the velocity field that moves $x_t$ closer to the 
target distribution ad time $t$ ($t$ is a number between 0 and 1. 0 -> noise; 1-> target).
We can train the model by minimizing the following loss:
$$\underset{\theta}{\text{argmin}} \; \mathbb{E}_{t, x_t} \Big\| {FM}_{\theta}(x_t, t) - v(x_t, t) \Big\|^2$$
which can be rewritten as:
$$\underset{\theta}{\text{argmin}} \;  \mathbb{E}_{t, X_0, X_1} \Big\| {FM}_{\theta}(x_t, t) - (X_1 - X_0) \Big\|^2\quad\quad$$
where $X_0$ is the noise and $X_1$ is the data.
### Model Architecture
Again a simple MLP, but now the input is: State + noise  ($x_t$) and $t$.
Here noise is of shape [128,16], which is the shape of the INTERPOLATED action chunk (chunk_size=8, action_dim=2; 2*8 = 16).
The +1 is for the time scalar $t$, which is of shape [128,1]. 
The output is of shape (batch_size, chunk_size * action_dim), as before. Here the output is the action chunk closer to the target distribution, at time $t$.
```python
self.mlp = nn.Sequential(
            nn.Linear(self.state_dim + self.chunk_size*self.action_dim + 1, self.hidden_dims[0]), #+1 for the time (is a scalar)
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[1], self.hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[2], self.chunk_size * self.action_dim)
        )
```
### Loss function
As describer above.
### Inference
Here's different. We have to compute the inference iteratively, starting from noise and moving towards the target distribution.
This means going from $t=0$ to $t=1$ in small steps, and at each step, we feed the current sample through the MLP to get the next sample.
```python
init_noise = torch.normal(0.0, 1.0, size=(state.shape[0], self.chunk_size*self.action_dim))
for step in range(num_steps):
    time = torch.Tensor([1/num_steps * (step+1)]) # linearly spaced time from 0 to 1
    pred_action_chunk = self.mlp(torch.cat([state,init_noise,time.reshape(state.shape[0],1)],dim=1))
    init_noise = init_noise + 1/num_steps * pred_action_chunk
init_noise = init_noise.reshape(-1,self.chunk_size,self.action_dim)
return init_noise
```