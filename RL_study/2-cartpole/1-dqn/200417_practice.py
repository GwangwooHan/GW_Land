import os
import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

# NN 생성 (입력: state, 출력: Q function)
class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN,self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(action_size, kernel_initializer = RandomUniform(-1e-3, 1e-3)) # kernel_initializer : 가중치 초기화

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render=True

        # state, action 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN hyper-parameter
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        # Replay memory 생성, 최대 크기: 2000 (deque 함수 사용)
        self.memory = deque(maxlen=2000)

        # 예측 모델과 타겟 모델 생성
        self.model = DQN(action_size)
        self.target_model = DQN(action_size)
        self.optimizer = Adam(lr = self.learning_rate)

        # 타겟 모델 가중치 동기화
        self.update_target_model()

    # 타겟모델의 가중치를 예측모델의 가중치로 업데이트 하는 함수
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # e-greedy policy로 action 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(state)
            return np.argmax(q_value[0])

    # 샘플 <s,a,r,s'>을 replay memory 에 리스트 형태로 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # replay 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출 (파이썬 내장 모듈 random의 sample 함수 사용)
        mini_batch = random.sample(self.memory, self.batch_size)

        # 학습을 위해 리스트형태의 <s,a,r,s'>를 별도의 numpy array로 만들어 줌
        states = np.array([sample[0][0] for sample in mini_batch]) # model로 부터 나오는 출력이 [[],[],[]] 형태이므로, [],[],[]와 형태로 변경하기 위해 [0] 붙이는것?
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        # 학습 파라미터
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            # 현 상태에 대한 예측모델의 큐함수
            # states는 numpy array이므로 케라스 모델의 입력으로 바로 넣을 수 있음
            predicts = self.model(states)
            # 모델의 예측 중 실제로 한 행동의 큐함수값만 가져오는 과정
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis = 1)

            # 다음 상태에 대한 타겟모델의 큐함수 도출
            # 다음 상태에서의 큐함수를 타겟모델로 구하는 이유: 이 값을 업데이트 목표로 사용하기 위함
            target_predicts = self.target_model(next_states)
            # 학습도중 타겟모델이 학습되는 일이 없도록 tf.stop_gradient 함수 적용 (target_predicts 예측 시 NN 업데이트 안됨)
            target_predicts = tf.stop_gradient(target_predicts)

            # 벨만 최적방정식을 이용하여 타겟 업데이트
            max_q = np.amax(target_predicts, axis=-1)
            targets = rewards + (1-dones) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts))

        # loss function을 줄이는 방향으로 예측 모델 업데이트 (주의: 타겟모델 업데이트가 아니라 예측모델 업데이트임)
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))

"""
DQN Agent 환경과 상호작용 순서
1. s 에따른 a선택 (e-greedy policy base)
2. 선택한 a로 env. 에서 한 time step 진행
3. env. 로부터 r, s' 수령
4. 샘플 (s,a,r,s') replay 메모리에 저장
5. replay 메모리에서 무작위 추출한 샘플로 학습
6. 에피소드마다 타깃모델 업데이트 
"""

# Main loop
if __name__ == "__main__":
    # CartPole-v1 환경 로딩, 최대 타임스텝 수가 500인 조건
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # DQN agent 생성
    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    score_avg = 0

    num_episode = 300
    for e in range(num_episode):
        done = False
        score = 0
        #env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # 1. s 에따른 a선택 (e-greedy policy base)
            action = agent.get_action(state)

            # 2. 선택한 a로 env. 에서 한 time step 진행
            # 3. env. 로부터 r, s' 수령
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # time step 마다 reward 0.1
            # 500 타임스텝을 채우지 못하고 에피소드가 끝나면 -1 보상
            score += reward
            reward = 0.1 if not done or score == 500 else -1

            # 4. replay memory에 sample (s,a,r,s') 저장
            agent.append_sample(state, action, reward, next_state, done)

            # 5. 메모리에 무작위 샘플링할 샘플들이 어느정도 쌓이면 타임 스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            state = next_state

            if done:
                # 6. 각 에피소드마다 타겟 모델을 예측모델의 가중치로 업데이트
                agent.update_target_model()
                # 에피소드마다 학습 결과 출력
                score_avg = 0.9*score_avg + 0.1*score if score_avg !=0 else score
                print("episode: {:3d} | score avg: {:3.2f} | memory length: {:4d} | epsilon: {:.4f}". format(e, score_avg, len(agent.memory), agent.epsilon))

                # 에피소드마다 학습결과 그래프로 저장
                scores.append(score_avg)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("average score")
                pylab.savefig("./save_graph/graph2.png")

                # 이동 평균이 400이상일 때 종료
                if score_avg>400:
                    agent.model.save_weights("./save_model/model2", save_format= "tf")
                    sys.exit()