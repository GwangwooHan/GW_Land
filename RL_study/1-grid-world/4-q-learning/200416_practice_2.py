import numpy as np
import random
from environment import Env
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.step_size = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.5
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    #<s,a,r,s'> 샘플로부터 큐함수 업데이트
    def learn(self, state, action, reward, next_state):
        state, next_state = str(state), str(next_state)
        q_1 = self.q_table[state][action]
        #벨만 최적 방정식을 이용한 큐함수의 업데이트
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.step_size * (q_2 - q_1)

    # 큐함수에 의거하여 e-greedy 정책에 따라 action 반환
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 행동 반환
            state = str(state)
            q_list = self.q_table[state]
            action = arg_max(q_list)
        return action

# 큐함수의 값에 따라 최적의 행동을 반환
def arg_max(q_list):
    max_idx_list = np.argwhere(q_list == np.amax(q_list))
    max_idx_list = max_idx_list.flatten().tolist()
    return random.choice(max_idx_list)

# main loop
if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent(actions=list(range(env.n_actions)))

    for episode in range(20):
        state = env.reset()

        """
        1. 현 s에 대하여 e-greedy 정책 기반 a 받음
        2. 환경으로 부터 r, s', done 받음
        3. <s,a,r,s'>로 Q함수 업데이트        
        
        """

        while True:
            #env, state 초기화
            env.render()
            #state에 대한 action 선택
            action = agent.get_action(state)
            #env로 부터 r, s', done 받음
            next_state, reward, done = env.step(action)
            #<s,a,r,s'>로 큐함수 업데이트
            agent.learn(state, action, reward, next_state)


            state = next_state

            #모든 큐함수를 화면에 표시
            env.print_value_all(agent.q_table)

            if done:
                break























