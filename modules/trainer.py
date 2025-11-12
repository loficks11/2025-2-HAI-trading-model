import random
import torch


def train_memory(agent, gamma, batch_size, optimizer, loss_fn):
    if len(agent.memory) < batch_size:
        return

    batch = random.sample(agent.memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = [s.unsqueeze(0) for s in states] 
    next_states = [ns.unsqueeze(0) for ns in next_states]

    states = torch.cat(states, dim=0)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.cat(next_states, dim=0)
    dones = torch.FloatTensor(dones).unsqueeze(1)

    current_qs = agent.nn(states).gather(1, actions)
    next_qs = agent.nn(next_states).max(1)[0].unsqueeze(1)
    target_qs = rewards + (gamma * next_qs * (1 - dones))

    loss = loss_fn(current_qs, target_qs.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train(agent, dataset, CONFIGS, optimizer, loss_fn):
    epochs = CONFIGS.EPOCHS
    gamma = CONFIGS.GAMMA
    batch_size = CONFIGS.BATCH_SIZE
    
    loss_history = []

    for i in range(len(dataset) - 1):
        state, reward = dataset[i]
        next_state, _ = dataset[i + 1]

        action = agent(state)
        agent.remember(state, action.item(), reward.item(), next_state, 0)

    state, reward = dataset[-1]
    action = agent(state)
    agent.remember(state, action.item(), reward.item(), state, 1)

    for epoch in range(1, epochs+1):
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}")

        loss = train_memory(agent, gamma, batch_size, optimizer, loss_fn)
        if loss is not None:
            loss_history.append(loss)

    return loss_history