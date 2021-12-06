import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
# pickle : q table save&load
style.use("ggplot")

SIZE = 10 # 10 by 10 grid 만들 것, 20 by 20으로 만들수도 있음
HM_EPISODES = 25000 # HM: How many
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 1

start_q_table = None # or filename epsilon없이 훈련 조금 한 뒤 저장했다가 불러와서 epsilon 적용해서 훈련하는 방법 등 적용시 save&load 가능

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

# color BGR 순
d = {1:(255, 175, 0),
     2:(0, 255, 0),
     3:(0, 0, 255)}

# Class 로 Environment 지정
class Blob:

    # Random 하게 초기화
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    # 디버깅 목적
    def __str__(self):
        return f"{self.x}, {self.y}"

    # 상대위치 관련
    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    # action 관련
    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    # action의 move 관련
    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2) # -1, 0, 1 만 가능
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1, 2)  # -1, 0, 1 만 가능
        else:
            self.y += y
        if self.x < 0:
            self. x = 0
        elif self.x > SIZE - 1:
            self.x = SIZE -1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1

# q-table 초기화
if start_q_table is None:
    q_table = {} # dict 형태로 만듬

# 10 by 10 q-table, x, y 상대위치 (player to food), (player to enemy)
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE + 1, SIZE):
            for x2 in range(-SIZE + 1, SIZE):
                for y2 in range(-SIZE + 1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]

# 기존 pre-trained q-table 있으면 파일 불러오는 코드
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


# 에피소드 시작
episode_rewards = []
for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    # SHOW_EVERY 주기에 따른 show 관련
    if episode % SHOW_EVERY ==0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0

    for i in range(200):

        # State
        obs = (player-food, player-enemy) # tuple 형태로 상대좌표 출력

        # Action
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)  # 0,1,2,3 중 하나 random 하게 action

        player.action(action)

        # maybe later, enemy 및 food 도 움직이게 할 수 있음
        # enemy.move()
        # food.move()

        # Reward
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        # New_state
        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q == - ENEMY_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE *(reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q

        # Visualization 관련
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype = np.uint8) # 마지막 3은 RGB 3차원, 초기 all black env.
            env[food.y][food.x] = d[FOOD_N] #dictionary, y,x축 visualization할 때 바뀌는 것 염두
            env[player.y][player.x] = d[PLAYER_N] #dictionary
            env[enemy.y][enemy.x] = d[ENEMY_N] #dictionary

            img = Image.fromarray(env, "RGB")
            img = img.resize((300, 300))
            cv2.imshow("", np.array(img))

            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                # 끝났을 경우 약간 멈춰줌
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        episode_reward += reward

        # 한 에피소드 종료 시점
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    episode_rewards.append((episode_reward))
    epsilon *= EPS_DECAY

moving_avg =np.convolve(episode_rewards, np.ones((SHOW_EVERY, )) / SHOW_EVERY, mode = "valid")

plt.plot([i for i in range( len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY} ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable -{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)













