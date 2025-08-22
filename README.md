conda install -c conda-forge box2d-py swig

Training Phase

Bootstrap: Collect 2k random transitions
Model Training: Train ensemble on collected data (200 epochs)
Planning Loop:

Generate 1000 action sequences (15 steps each)
Roll out through ensemble models
Average predictions across ensemble
Select best sequence, execute first action
Store transition, retrain models every 10 steps

Key Implementation Details

Uncertainty: Use ensemble disagreement (std dev) for exploration
Model Updates: Retrain every 10 environment steps to incorporate new data
Action Bounds: Clip sampled actions to [-1, 1] for LunarLander
Terminal Handling: Stop rollouts early if model predicts episode termination

for epoch in range(ensemble_training_epochs):
    for model in ensemble:
        batch = replay_buffer.sample()
        loss = mse_loss(model(state, action), target=[next_state, reward])
