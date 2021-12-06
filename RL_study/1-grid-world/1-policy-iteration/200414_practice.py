import numpy as np
from environment import GraphicDisplay, Env

class PolicyIteration:
    def __init__(self,env):
        #환경에 대한 객체
        self.env = env
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        self.policy_table = [[[0.25,0.25,0.25,0.25]] * env.width for _ in range(env.height)]
        self.policy_table[2][2] = []
        self.discount_factor = 0.9

    # 정책 평가
    def policy_evaluation(self):
        # 다음 가치함수 초기화
        next_value_table = [[0.00]*self.env.width for _ in range(env.height)]

        # 모든 상태에 대해서 벨만 기대 방정식을 계산
        for state in self.env.get_all_states():
            value = 0.0
            # 마침상태의 가치함수 = 0
            if state == [2,2]:
                next_value_table[state[0]][state[1]] = value
                continue

            #벨만 기대 방정식
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value += (self.get_policy(state)[action]*(reward + self.discount_factor * next_value))

            next_value_table[state[0]][state[1]] = value

        self.value_table = next_value_table

    # 현재 가치 함수에 대해서 탐욕 정책 발전
    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.env.get_all_states():
            if state == [2,2]:
                continue

            value_list =[]
            #반환할 정책 초기화
            result =[0.0, 0.0, 0.0, 0.0]

            #모든 행동에 대해서 [보상 + (할인율 * 다음 상태 가치함수)] 계산
            for index, action in enumerate(self.env.possible_actions):
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value = reward + self.discount_factor * next_value
                value_list.append(value)

            #받을 보상이 최대인 행동들에 대해 탐욕 정책 발전
            max_idx_list = np.argwhere(value_list == np.amax(value_list))
            #가장 큰 값이 여러개일 수도 있으므로 list로 펴서 저장
            max_idx_list = max_idx_list.flatten().tolist()
            # max_idx_list에 담긴 값이 여러개라면 행동들을 동일한 확률에 기반하여 선택
            prob = 1 / len(max_idx_list)

            for idx in max_idx_list:
                result[idx] = prob
                next_policy[state[0]][state[1]] = result

        self.policy_table = next_policy

    #특정 상태에서 정책에 따라 무작위로 action을 반환
    def get_action(self, state):
        policy = self.get_policy(state)
        policy = np.array(policy)
        return np.random.choice(4,1, p=policy)[0] #행동의 개수, 몇개의 행동을 샘플링 할지, 세번쨰 인자에 구한 정책을 넣으면 정책에 따라 행동 정해짐

    #상태에 따른 정책 반환
    def get_policy(self, state):
        return self.policy_table[state[0]][state[1]]

    #가치함수의 값을 반환
    def get_value(self, state):
        return self.value_table[state[0]][state[1]]


if __name__ == "__main__":
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()