"""QWOP 的 Gymnasium 接口：浏览器按键，不占用系统键盘。"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import time
import os
import json
from urllib.request import urlopen
import websocket
from rewards import walking_reward

import gymnasium as gym
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait

ACTIONS = ('qw', 'qo', 'qp', 'q', 'wo', 'wp', 'w', 'op', 'o', 'p', '')
DEFAULT_GAME = Path(os.environ.get('QWOP_GAME_DIR', Path(__file__).resolve().parent.parent / 'game'))


class QuietHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split('?')[0] == '/QWOP2.min.js':
            source = (Path(self.directory) / 'QWOP2.min.js').read_text(encoding='utf-8')
            marker = 'updatestate: function() {'
            if source.count(marker) != 1:
                self.send_error(500, 'Unsupported game version')
                return
            source = source.replace(marker, marker + '\nwindow.__qwop = {game:this, snow:m.core.app, core:m.core};', 1)
            data = source.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/javascript')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()

    def log_message(self, *args):
        pass


class QWOPEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, game=DEFAULT_GAME, headless=False, seconds=0.1, max_steps=300, fast=False, upright=False,
                 frame_skip=6, reward_config=None, direct_cdp=False, curriculum=False):
        super().__init__()
        if not (Path(game) / 'Athletics.html').exists():
            raise FileNotFoundError(f'缺少游戏文件：{game}，请按 README 下载参考项目')
        self.seconds, self.max_steps = seconds, max_steps
        self.fast = fast
        self.headless = headless
        self.upright = upright
        if frame_skip < 1:
            raise ValueError('frame_skip 必须为正数')
        self.frame_skip = frame_skip
        self.reward_config = json.loads(Path(reward_config).read_text(encoding='utf-8')) if reward_config else None
        self.curriculum = curriculum
        self.total_steps = 0
        self.driver = self.server = None
        self.cdp_socket = None
        self.cdp_id = 0
        self.pressed = set()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(game)))
        Thread(target=self.server.serve_forever, daemon=True).start()
        try:
            options = webdriver.ChromeOptions()
            if headless:
                options.add_argument('--headless=new')
                options.add_argument('--renderer-process-limit=2')
                options.add_argument('--disable-gpu')
            # The game and all its assets are served locally. Avoid waiting for
            # unrelated analytics/advertising frames in training runs.
            options.page_load_strategy = 'eager'
            options.add_argument('--host-resolver-rules=MAP * 0.0.0.0, EXCLUDE localhost, EXCLUDE 127.0.0.1')
            options.add_argument('--window-size=1000,800')
            options.add_argument('--disable-background-timer-throttling')
            # 离线教室可通过 QWOP_CHROMEDRIVER 指定预先下载的匹配驱动。
            driver_path = os.environ.get('QWOP_CHROMEDRIVER')
            self.driver = webdriver.Chrome(options=options, service=Service(driver_path) if driver_path else Service())
            self.driver.get(f'http://127.0.0.1:{self.server.server_port}/Athletics.html')
            WebDriverWait(self.driver, 45).until(lambda d: d.execute_script(
                'return typeof globalbodystate !== "undefined" && Object.keys(globalbodystate.torso).length > 0'))
            self.driver.find_element('tag name', 'body').click()
            self.driver.execute_script((Path(__file__).parent / 'game_probe.js').read_text(encoding='utf-8'))
            if fast:
                self.driver.execute_script('window.__qwop.snow.freeze = true;')
            if fast and direct_cdp:
                address = self.driver.capabilities['goog:chromeOptions']['debuggerAddress']
                with urlopen(f'http://{address}/json', timeout=10) as response:
                    targets = json.load(response)
                target = next(t for t in targets if t['type'] == 'page' and 'Athletics.html' in t['url'])
                self.cdp_socket = websocket.create_connection(target['webSocketDebuggerUrl'],
                    suppress_origin=True, timeout=30, http_no_proxy=['localhost', '127.0.0.1'])
            body, _ = self.snapshot()
            # 显式排序，避免对象属性遍历顺序使模型输入发生变化。
            self.features = [(part, key) for part in sorted(body) for key in sorted(body[part])]
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (len(self.features),), np.float32)
            self.action_space = gym.spaces.Discrete(len(ACTIONS))
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        return self.js('return window.__rlSnapshot()')

    def js(self, source, *args):
        """可选直连 CDP：只减少通信开销，不改变物理步数和动作。"""
        if self.cdp_socket is None:
            return self.driver.execute_script(source, *args)
        self.cdp_id += 1
        expression = '(function(){' + source + '}).apply(null,' + json.dumps(args) + ')'
        self.cdp_socket.send(json.dumps({'id': self.cdp_id, 'method': 'Runtime.evaluate',
            'params': {'expression': expression, 'returnByValue': True}}))
        while True:
            response = json.loads(self.cdp_socket.recv())
            if response.get('id') == self.cdp_id:
                break
        if 'error' in response or 'exceptionDetails' in response.get('result', {}):
            raise RuntimeError(f'游戏脚本错误：{response}')
        return response['result']['result'].get('value')

    def keys(self, keys):
        wanted = set(keys)
        for event, chars in [('keyUp', self.pressed - wanted), ('keyDown', wanted - self.pressed)]:
            for char in sorted(chars):
                self.driver.execute_cdp_cmd('Input.dispatchKeyEvent', {
                    'type': event, 'key': char, 'code': 'Space' if char == ' ' else 'Key' + char.upper(),
                    'windowsVirtualKeyCode': ord(char.upper()), 'nativeVirtualKeyCode': ord(char.upper())})
        self.pressed = wanted

    def observation(self, body):
        origin = body['torso']['position_x']
        obs = np.array([body[p][k] - (origin if k == 'position_x' else 0)
                        for p, k in self.features], dtype=np.float32)
        if not np.isfinite(obs).all():
            raise RuntimeError('游戏状态包含非有限数值')
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.fast:
            if self.reward_config:
                # 新实验不让上一回合的按键参与 reset() 内部的第一次物理更新。
                self.js('const g=window.__qwop.game; g.QDown=g.WDown=g.ODown=g.PDown=false;')
            self.js('const g=window.__qwop.game; g.reset(); g.firstClick=true; g.QDown=g.WDown=g.ODown=g.PDown=false; g.update(1/60);')
            body, state = self.snapshot()
            self.previous_x = body['torso']['position_x']
            self.reward_previous_x = self.previous_x
            self.steps = 0
            prelude = self.reward_config.get('reset_prelude', []) if self.reward_config else []
            for phase in prelude:
                action = ACTIONS.index(phase['keys'])
                for _ in range(phase['steps']):
                    _, _, ended, limited, _ = self.step(action)
                    if ended or limited:
                        raise RuntimeError('The starting-pose key sequence failed')
            if prelude:
                body, state = self.snapshot()
                self.previous_x = self.reward_previous_x = body['torso']['position_x']
                self.steps = 0
            return self.observation(body), {'distance': float(state['score'])}
        self.keys('')
        self.keys('r ')
        time.sleep(0.15)
        self.keys('')
        time.sleep(0.1)  # 游戏在松开 R 后的下一帧才执行重置。
        body, state = self.snapshot()
        if state['gameOver'] or state['gameEnded']:
            raise RuntimeError('游戏重置失败，请检查游戏是否加载完毕')
        self.previous_x = body['torso']['position_x']
        self.reward_previous_x = self.previous_x
        self.steps = 0
        return self.observation(body), {'distance': float(state['score'])}

    def step(self, action):
        if self.fast:
            body, state = self.js('''const g=window.__qwop.game, keys=arguments[0];
                g.QDown=keys.includes('q'); g.WDown=keys.includes('w');
                g.ODown=keys.includes('o'); g.PDown=keys.includes('p');
                let nonfoot=0;
                for(let i=0;i<arguments[1];i++){
                    window.__qwop.core.update(1/60);
                    nonfoot=Math.max(nonfoot,window.__rlSnapshot()[1].nonfoot_contact);
                    if(g.gameEnded || g.gameOver)break;
                }
                const result=window.__rlSnapshot(); result[1].nonfoot_contact=nonfoot;
                return result;
                ''', ACTIONS[int(action)], self.frame_skip)
            if not self.headless:
                self.render()
                time.sleep(self.seconds)  # 桌面回放让学生看清动作；无头训练保持加速。
        else:
            self.keys(ACTIONS[int(action)])
            time.sleep(self.seconds)
            body, state = self.snapshot()
        x = body['torso']['position_x']
        # 有符号位移：后退会扣分，避免来回晃动反复领取前进奖励。
        reward = float(x - self.previous_x)
        if self.upright:
            # 地面约 y=8；正常站立时躯干约 y=-2。惩罚低位跪行，不奖励原地站立。
            reward -= 0.05 * max(float(body['torso']['position_y']), 0.0)
        self.previous_x = x
        self.steps += 1
        self.total_steps += 1
        terminated = bool(state['gameOver'] or state['gameEnded'])
        truncated = not terminated and (self.steps >= self.max_steps or state['scoreTime'] >= 120)
        info = {'distance': float(state['score']), 'game_time': float(state['scoreTime']),
                'torso_angle': float(body['torso']['angle']),
                'torso_y': float(body['torso']['position_y']),
                'left_knee_y': float(state['left_knee_y']),
                'right_knee_y': float(state['right_knee_y']),
                'nonfoot_contact': int(state['nonfoot_contact']),
                'left_foot_contact': int(state['left_foot_contact']),
                'right_foot_contact': int(state['right_foot_contact']),
                'reason': 'game_end' if terminated else 'limit' if truncated else ''}
        if self.upright and terminated and info['distance'] < 100:
            reward -= 2.0
        if self.reward_config:
            schedule = self.reward_config.get('pose_curriculum')
            if schedule and self.curriculum:
                fraction = min(1.0, self.total_steps / schedule['steps'])
                state['pose_curriculum_scale'] = schedule['start_scale'] + (1 - schedule['start_scale']) * fraction
            reward, terms = walking_reward(float(x - self.reward_previous_x), info['torso_y'], state,
                self.reward_config, failed=terminated and info['distance'] < 100,
                finished=terminated and info['distance'] >= 100)
            info['reward_terms'] = terms
        self.reward_previous_x = x
        return self.observation(body), reward, terminated, truncated, info

    def render(self):
        if self.fast:
            self.js('window.__qwop.core.render();')

    def close(self):
        if self.cdp_socket:
            self.cdp_socket.close()
            self.cdp_socket = None
        if self.driver:
            self.driver.quit()
            self.driver = None
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
