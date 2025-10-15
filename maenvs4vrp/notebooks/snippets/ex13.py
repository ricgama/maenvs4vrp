td['action'] = torch.tensor([[3]])
td = env.step(td)
print("reward: ", td['reward'])
print("penalty: ", td['penalty'])