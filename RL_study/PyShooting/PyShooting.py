import random
from time import sleep

import pygame
from pygame.locals import *

WINDOW_WIDTH = 480
WINDOW_HEIGHT = 640

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (250, 250, 50)
RED = (250, 50, 50)

FPS = 60


class Fighter(pygame.sprite.Sprite):  # 우주선 클래스
    def __init__(self):
        super(Fighter, self).__init__()  # 상속받아옴
        self.image = pygame.image.load('./resources/fighter.png')
        self.rect = self.image.get_rect()  # rect = 그림의 위치
        self.rect.x = int(WINDOW_WIDTH / 2)  # 시작위치
        self.rect.y = WINDOW_HEIGHT - self.rect.height
        self.dx = 0  # 방향
        self.dy = 0  #

    def update(self):
        self.rect.x += self.dx
        self.rect.y += self.dy

        if self.rect.x < 0 or self.rect.x + self.rect.width > WINDOW_WIDTH:  # 플레이어가 화면 바깥으로 나가는 경우
            self.rect.x -= self.dx  # 더이상 움직이지 않음
        if self.rect.y < 0 or self.rect.y + self.rect.height > WINDOW_HEIGHT:
            self.rect.y -= self.dy

    def draw(self, screen):  # 스크린을 받아옴
        screen.blit(self.image, self.rect)  # 화면에 그려줌

    def collide(self, sprites):  # 우주선이 충돌이 나는경우
        for sprite in sprites:  # sprites: 파이썬에서 만든 기능
            if pygame.sprite.collide_rect(self, sprite):  # self(우주선)과 sprites에 포함된 sprite와 충돌나는경우
                return sprite


class Missile(pygame.sprite.Sprite):  # 미사일 클래스
    def __init__(self, xpos, ypos, speed):  # 미사일 x,y 위치, 속도
        super(Missile, self).__init__()  # 상위클래스 호출
        self.image = pygame.image.load('./resources/missile.png')
        self.rect = self.image.get_rect()
        self.rect.x = xpos  # 미사일 발사위치는 우주선의 위치임, 이를 지정하기 위해 변수설정
        self.rect.y = ypos
        self.speed = speed
        self.sound = pygame.mixer.Sound('./resources/missile.wav')  # 미사일 발사시 나는 소리

    def launch(self):
        self.sound.play()  # 미사일 발사소리 플레이

    def update(self):
        self.rect.y -= self.speed  # 미사일이 발사된 후 y좌표값
        if self.rect.y + self.rect.height < 0:  # 미사일이 화면밖으로 나간 경우
            self.kill()

    def collide(self, sprites):  # 미사일이 충돌한 경우
        for sprite in sprites:
            if pygame.sprite.collide_rect(self, sprite):
                return sprite


class Rock(pygame.sprite.Sprite):  # 암석
    def __init__(self, xpos, ypos, speed):  # 암석의 위치와 속력 받아옴
        super(Rock, self).__init__()
        rock_images = ('./resources/rock01.png', './resources/rock02.png', './resources/rock03.png', './resources/rock04.png', './resources/rock05.png',
                       './resources/rock06.png', './resources/rock07.png', './resources/rock08.png', './resources/rock09.png', './resources/rock10.png',
                       './resources/rock11.png', './resources/rock12.png', './resources/rock13.png', './resources/rock14.png', './resources/rock15.png',
                       './resources/rock16.png', './resources/rock17.png', './resources/rock18.png', './resources/rock19.png', './resources/rock20.png',
                       './resources/rock21.png', './resources/rock22.png', './resources/rock23.png', './resources/rock24.png', './resources/rock25.png',
                       './resources/rock26.png', './resources/rock27.png', './resources/rock28.png', './resources/rock29.png', './resources/rock30.png')
        self.image = pygame.image.load(random.choice(rock_images))  # Rock이 호출될 때마다 랜덤으로 구성됨
        self.rect = self.image.get_rect()
        self.rect.x = xpos
        self.rect.y = ypos
        self.speed = speed

    def update(self):
        self.rect.y += self.speed  # y값은 내려오므로 +

    def out_of_screen(self):
        if self.rect.y > WINDOW_HEIGHT:  # 암석이 화면 나가는 경우
            return True


def draw_text(text, font, surface, x, y, main_color):  # 점수 전광판
    text_obj = font.render(text, True, main_color)
    text_rect = text_obj.get_rect()
    text_rect.centerx = x
    text_rect.centery = y
    surface.blit(text_obj, text_rect)


def occur_explosion(surface, x, y):  # 폭발일어나는경우
    explosion_image = pygame.image.load('./resources/explosion.png')
    explosion_rect = explosion_image.get_rect()
    explosion_rect.x = x
    explosion_rect.y = y
    surface.blit(explosion_image, explosion_rect)

    explosion_sounds = ('./resources/explosion01.wav', './resources/explosion02.wav', './resources/explosion03.wav')  # 폭발음
    explosion_sound = pygame.mixer.Sound(random.choice(explosion_sounds))  # 폭발음 랜덤재생
    explosion_sound.play()


def game_loop():
    default_font = pygame.font.Font('./resources/NanumGothic.ttf', 28)
    background_image = pygame.image.load('./resources/background.png')
    gameover_sound = pygame.mixer.Sound('./resources/gameover.wav')
    pygame.mixer.music.load('./resources/music.wav')
    pygame.mixer.music.play(-1)  # 배경음악 몇번 반복할거냐, -1: 무한반복
    fps_clock = pygame.time.Clock()

    fighter = Fighter()
    missiles = pygame.sprite.Group()  # 미사일은 여러개이므로 sprite의 group으로 만듦
    rocks = pygame.sprite.Group()  # 암석도 여러개이므로 sprite의 group으로 만듦

    occur_prob = 40  # 암석나올확률영향 인자
    shot_count = 0  # 맞춘 암석개수
    count_missed = 0  # 놓친 암석개수

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:  # 키가 무언가가 입력이 된경우
                if event.key == pygame.K_LEFT:  # left 방향키
                    fighter.dx -= 5
                elif event.key == pygame.K_RIGHT:
                    fighter.dx += 5
                elif event.key == pygame.K_UP:
                    fighter.dy -= 5
                elif event.key == pygame.K_DOWN:
                    fighter.dy += 5
                elif event.key == pygame.K_SPACE:
                    missile = Missile(fighter.rect.centerx, fighter.rect.y, 10)  # 스페이스 누르면 미사일이 발사됨
                    missile.launch()  # 미사일발사 (소리발생)
                    missiles.add(missile)

            if event.type == pygame.KEYUP:  # 방향키 키보드에서 손을 뗀경우
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    fighter.dx = 0
                elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                    fighter.dy = 0

        screen.blit(background_image, background_image.get_rect())  # 게임화면 배경

        # 게임의 dynamic 부분: 점수가 높아질경우 암석등장 개수 증가, 암석속도 증가
        occur_of_rocks = 1 + int(shot_count / 300)
        min_rock_speed = 1 + int(shot_count / 200)
        max_rock_speed = 1 + int(shot_count / 100)

        if random.randint(1, occur_prob) == 1:  # 1부터 40(occur_prob)사이에 1이 등장할 확률
            for i in range(occur_of_rocks):  # 출현 암석마다 특성 다르게 하는 함수
                speed = random.randint(min_rock_speed, max_rock_speed)
                rock = Rock(random.randint(0, WINDOW_WIDTH - 30), 0, speed)  # x, y 위치에서 스피드만큼 암석이 떨어짐
                rocks.add(rock)

        draw_text('파괴한 운석 :{}'.format(shot_count), default_font, screen, 100, 20, YELLOW)
        draw_text('놓친 운석 :{}'.format(count_missed), default_font, screen, 400, 20, RED)

        for missile in missiles:  # 전체 암석에 대하여 충돌체크
            rock = missile.collide(rocks)
            if rock:  # 충돌한 경우
                missile.kill()
                rock.kill()
                occur_explosion(screen, rock.rect.x, rock.rect.y)
                shot_count += 1

        for rock in rocks:  # 암석 놓친경우
            if rock.out_of_screen():
                rock.kill()
                count_missed += 1

        rocks.update()
        rocks.draw(screen)
        missiles.update()
        missiles.draw(screen)
        fighter.update()
        fighter.draw(screen)
        pygame.display.flip()  # 현재 값들을 flip으로 전체반영

        if fighter.collide(rocks) or count_missed >= 3:  # game over 조건
            pygame.mixer_music.stop()  # 배경음악 off
            occur_explosion(screen, fighter.rect.x, fighter.rect.y)
            pygame.display.update()  # 업데이트
            gameover_sound.play()
            sleep(1)  # 잠깐 쉼
            done = True  # 반복문 종료

        fps_clock.tick(FPS)

    return 'game_menu'  # 게임 초기 메뉴로 이동


def game_menu():
    start_image = pygame.image.load('./resources/background.png')
    screen.blit(start_image, [0, 0])  # 화면에 그려줌 위치는 0,0
    draw_x = int(WINDOW_WIDTH / 2)
    draw_y = int(WINDOW_HEIGHT / 4)
    font_70 = pygame.font.Font('./resources/NanumGothic.ttf', 70)
    font_40 = pygame.font.Font('./resources/NanumGothic.ttf', 40)

    draw_text('지구를 지켜라!', font_70, screen, draw_x, draw_y, YELLOW)
    draw_text('엔터 키를 누르면', font_40, screen, draw_x, draw_y + 200, WHITE)
    draw_text('게임이 시작됩니다.', font_40, screen, draw_x, draw_y + 250, WHITE)

    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:  # RETURN: 엔터
                return 'play'
        if event.type == QUIT:
            return 'quit'

    return 'game_menu'


def main():  # 전체 global 함수 지정
    global screen

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('PyShooting')

    action = 'game_menu'
    while action != 'quit':
        if action == 'game_menu':
            action = game_menu()

        elif action == 'play':
            action = game_loop()

    pygame.quit()


if __name__ == "__main__":
    main()


## 변경됐는지 봐볼까?