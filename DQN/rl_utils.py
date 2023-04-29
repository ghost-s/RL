from tqdm import tqdm
import numpy as np
import collections
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


# 异策略训练
def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    # 保存回报
    return_list = []
    best_return = -10000
    # 分十次进行迭代
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            # 每次训练 num_episodes/10 个episode
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset(seed=0)
                state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    next_state = next_state[0].__array__() if isinstance(next_state, tuple) else next_state.__array__()
                    done = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() >= minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                        # 保存最优权重
                        if episode_return > best_return:
                            best_return = episode_return
                            agent.save_models()
                # 将每个episode的回报进行保存
                return_list.append(episode_return)
                # 每训练10个episode对进度条进行更新，回报是这10次训练的回报均值
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


