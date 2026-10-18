#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
command -v google-chrome >/dev/null || { echo '请先安装 Google Chrome Linux .deb 版。'; exit 1; }
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -c "import hashlib,json,pathlib; c=json.load(open('reference.lock.json')); p=pathlib.Path('../upstream/QWOP-RL/game/QWOP2.min.js'); assert hashlib.sha256(p.read_bytes()).hexdigest()==c['game_js_sha256'], '游戏文件版本不匹配'"
python -m unittest discover -s tests -v
echo '安装完成。下一步：python course.py train --out runs/check --steps 2048'
