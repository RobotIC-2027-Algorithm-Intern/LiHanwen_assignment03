"""Encode real evaluation screenshots as a captioned, shareable MP4 (no model files)."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--title', required=True)
    p.add_argument('--note', default='', help='Optional short teaching caption')
    p.add_argument('--question', required=True)
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--font', type=Path)
    args = p.parse_args()
    font_path = args.font or next((x for x in [Path('C:/Windows/Fonts/msyh.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')] if x.exists()), None)
    if font_path is None:
        raise SystemExit('Install fonts-noto-cjk or pass --font with a Chinese font')
    fonts = {n: ImageFont.truetype(str(font_path), n) for n in (22, 26, 34)}
    paths = sorted(args.run.glob(f'episode-{args.episode}-step-*.png'),
                   key=lambda x: int(x.stem.rsplit('-', 1)[-1]))
    if not paths:
        raise SystemExit('No real screenshots found')
    for path in paths:
        with Image.open(path) as shot:
            extrema = shot.convert('RGB').getextrema()
            if all(high-low < 5 for low, high in extrema):
                raise SystemExit(f'Empty game canvas; re-record instead of distributing: {path}')
            pixels = shot.convert('RGB').resize((32, 20)).tobytes()
            if sum(min(pixels[i:i+3]) > 245 for i in range(0, len(pixels), 3)) > .95 * len(pixels)/3:
                raise SystemExit(f'Blank browser capture; re-record instead of distributing: {path}')
    config = json.loads((args.run / 'evaluation.json').read_text(encoding='utf-8'))
    rows = {int(r['step']): r for r in csv.DictReader((args.run / 'steps.csv').open(encoding='utf-8'))
            if int(r['episode']) == args.episode}
    # Repeat screenshots at 30fps; elapsed time follows the fixed-step simulation clock.
    # This is a real physics replay; it does not imply the realtime browser has been validated.
    fps = 30
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit('Refusing to overwrite existing video')
    temporary = args.out.with_suffix(args.out.suffix + '.tmp')
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x800',
        '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast',
        '-threads', '2', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        '-y', '-f', 'mp4', str(temporary)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    base = Image.new('RGB', (1280, 800), '#101827')
    static = ImageDraw.Draw(base)
    static.text((40, 20), args.title, font=fonts[34], fill='white')
    if args.note:
        static.text((40, 670), args.note, font=fonts[26], fill='#63d6d0')

    def frame(path=None, row=None):
        canvas = base.copy()
        draw = ImageDraw.Draw(canvas)
        if path:
            with Image.open(path) as shot:
                shot = shot.convert('RGB')
                scale = min(1120 / shot.width, 560 / shot.height)
                shot = shot.resize((round(shot.width*scale), round(shot.height*scale)), Image.Resampling.LANCZOS)
                # Preserve full gameplay, including the original score and timer.
                canvas.paste(shot, ((1280-shot.width)//2, 94+(560-shot.height)//2))
        else:
            draw.text((80, 270), '观察任务', font=fonts[34], fill='#63d6d0')
            draw.text((80, 340), args.question, font=fonts[26], fill='white')
        if row:
            keys = '无按键' if row['keys'] in ('none', '') else row['keys'].upper()
            draw.text((40, 716), f"距离 {float(row['distance']):.1f} 米    策略按键 {keys}", font=fonts[22], fill='white')
            top = sorted(((k[2:], float(v)) for k, v in row.items() if k.startswith('p_')),
                         key=lambda item: item[1], reverse=True)[:3]
            chances = '  '.join(f"{'无' if k == 'none' else k.upper()} {v:.0%}" for k, v in top)
            draw.text((570, 716), '前三项概率：' + chances, font=fonts[22], fill='white')
        return canvas.tobytes()

    try:
        card = frame()
        for _ in range(fps*3):
            proc.stdin.write(card)
        written = 0
        elapsed = 0.0
        for i, path in enumerate(paths):
            step = int(path.stem.rsplit('-', 1)[-1])
            data = frame(path, rows[step])
            next_step = (int(paths[i+1].stem.rsplit('-', 1)[-1]) if i+1 < len(paths)
                         else min(step + config['frame_every'], max(rows)+1))
            elapsed += (next_step-step) * config.get('frame_skip', 6) / 60
            target = round(elapsed*fps)
            for _ in range(target-written):
                proc.stdin.write(data)
            written = target
        for _ in range(fps*2):
            proc.stdin.write(data)
    finally:
        proc.stdin.close()
        code = proc.wait()
    if code:
        raise SystemExit(f'ffmpeg failed: {code}')
    temporary.replace(args.out)
    print(args.out)


if __name__ == '__main__':
    main()
