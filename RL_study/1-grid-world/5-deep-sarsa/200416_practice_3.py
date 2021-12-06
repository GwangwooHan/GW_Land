import copy
import pylab
import random
import numpy as np
from environment import Env
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# DeepSARSA Neural network

class DeepSARSA(tf.keras.Model):
    def __init__(self, action_size):
        super(DeepSARSA, self).__init__()
        self.fc1 = Dense(30, activation = 'relu')
        self.fc2 = Dense(30, activation = 'relu')
        self.fc_out = Dense(action_size)

    # x는 Neural network의 input 값으로 쓰임
    # 뒤에 48line의 q_values = self.model(state) 있는데, 여기서 state가 input값 (x)로 들어가서 output 값 (return q)로 q_values로 출력
    def call(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q

# 그리드월드 예제에서의 DeepSARSA agent
class DeepSARSAgent:
    def __init__(self, state_size, action_size):
        # state 크기, action 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DeepSARSA hyper_parameter
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.model = DeepSARSA(self.action_size)
        self.optimizer = Adam(lr = self.learning_rate)

    # e-greedy policy로 action 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model(state)
            # 2차원 형태로 나오기 때문에, 1차원 형태로 변경하기위해 [0] 붙임 (keras의 미니배치 형태 입,출력 특징)
            return np.argmax(q_values[0])

    # <s,a,r,s',a'>의 샘플로 부터 모델 업데이트
    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 계산과정을 기록하기 위한 tape scope 선언
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_params)

            # 모델을 통한 예측
            predict = self.model(state)[0]
            # 실제로 행동이 된 큐함수와 차원이 같은 출력을 만들기 위해 one_hot_action 벡터 생성
            one_hot_action = tf.one_hot([action], self.action_size)
            predict = tf.reduce_sum(one_hot_action * predict, axis=1)

            # done = True 일 경우 에피소드가 끝나서 다음 상태가 없음
            next_q = self.model(next_state)[0][next_action]
            target = reward + (1-done) *self.discount_factor * next_q

            # MSE 오류함수 계산
            loss = tf.reduce_mean(tf.square(target-predict))

        # tape를 통한 gradient 계산 (loss로 정의된 오류함수를 model_params에 대해 미분하여 편미분 값 구함)
        grads = tape.gradient(loss, model_params)
        # 편미분 값을 구한 후 optimizer를 통해 모델 업데이트
        self.optimizer.apply_gradients(zip(grads, model_params))


# Main loop

if __name__ == "__main__":
    env = Env(render_speed = 0.01)
    state_size = 15
    action_space = [0,1,2,3,4] #상하좌우 제자리
    action_size = len(action_space)
    agent = DeepSARSAgent(state_size, action_size)

    scores, episodes = [], []

    EPISODES = 1000

    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        """
        1. s에 따른 a 선택
        2. env. 에서 한 time step 진행
        3. env. 로 부터 r, s', done 받음
        4. s' 로 부터 a' 선택 
        5. (s,a,r,s',a') 토대로 학습 진행
        1~5 반복    

        """

        while not done:
            # 현재 상태에 대한 행동 선택 (NN로 부터)
            action = agent.get_action(state)

            # time step 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            next_action = agent.get_action(next_state)

            # (s,a,r,s',a') 샘플 활용 모델 학습
            agent.train_model(state, action, reward, next_state, next_action, done)
            score += reward
            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:3d} | epsilon: {:3f}".format(e, score, agent.epsilon))

                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("average score")
                pylab.savefig("./save_graph/graph2.png")

        if e % 100 == 0:
            agent.model.save_weights('save_model/model2', save_format='tf')













