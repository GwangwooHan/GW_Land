import numpy as np
import random
from collections import defaultdict
from environment import Env

class SARSAgent:
    def __init__(self, actions):
        self.actions = actions
        self.step_size = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        """ 살사에서 에이전트가 어떤 상태를 방문하면 그 상태의 함수 업데이트
        따라서, 어떤 상태의 큐함수를 담은 자료 구조 필요 
        dictionary 자료형을 이욯하여 큐함수 저장
        """
        # 0을 초깃값으로 가지는 큐함수 테이블 생성
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # <s,a,r,s',a'>의 샘플로 부터 큐함수를 업데이트
    def learn(self, state, action, reward, next_state, next_action):
        state, next_state = str(state), str(next_state)
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        td = reward + self.discount_factor * next_state_q - current_q
        # td: temporal difference (시간차)
        new_q = current_q + self.step_size * td
        self.q_table[state][action] = new_q

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)
        else:
            #큐함수에 따른 행동 반환
            # str함수로 string 으로 변환: dictionary의 키 값으로 상태를 string 형태로 저장하기 위함
            state = str(state)
            q_list = self.q_table[state]
            # arg_max는 별도 함수
            action = arg_max(q_list)
        return action

# 큐함수의 값에 따라 최적의 행동을 반환
def arg_max(q_list):
    max_idx_list = np.argwhere(q_list == np.amax(q_list))
    max_idx_list = max_idx_list.flatten().tolist()
    return random.choice(max_idx_list)

""" 
    1.현 state에서(s) e-greedy 정책에 따라 action (a) 선택
    2. 선택한 a로 환경에서 한 time step 진행
    3. action (a) 후 환경으로 부터 r, s' 받음
    4. 다음 state에서 (s') e-greedy 정책에 따라 action (a') 전택
    5. (s,a,r,s',a')을 통해 q-function 업데이트
"""

# Main loop
if __name__ == "__main__":
    env = Env()
    agent = SARSAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        # environment, state 초기화
        state = env.reset()
        # 현재 state에 대한 action 선택
        action = agent.get_action(state)

        while True:
            env.render()
            #a 취한 후 환경으로 부터 다음 s', r, done 여부 수령
            next_state, reward, done = env.step(action)
            # 다음 s'에서의 다음 a' 선택
            next_action = agent.get_action(next_state)
            #(s,a,r,s',a')로 큐함수 업데이트
            agent.learn(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action

            #모든 큐함수를 화면에 표시
            env.print_value_all(agent.q_table)

            if done:
                break






















