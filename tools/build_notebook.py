"""meeting_transcriber.ipynb を生成するスクリプト。

ノートブックを直接編集するとJSONの差分が読みにくいため、
セルの中身はこのファイルで管理し、以下のコマンドで .ipynb を作り直します。

    python tools/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "meeting_transcriber.ipynb"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({
        "cell_type": "code",
        "metadata": {"cellView": "form"},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n"),
    })


# ---------------------------------------------------------------------------
md(r"""
# 🎙️ 会議の文字起こし＆議事録作成ツール（Google Colab版）

Googleドライブに置いた会議の録音（数時間の長さでもOK）から、

1. **文字起こし**（時刻付きテキスト・字幕ファイル）
2. **議事録**（要約・決定事項・ToDo・タイムライン）

を自動で作り、結果をGoogleドライブに保存します。

---

## 📋 はじめに一度だけやること

1. **GPUを有効にする**：メニュー「ランタイム」→「ランタイムのタイプを変更」→「T4 GPU」を選んで保存
2. **Claude APIキーを登録する**（要約に使います）
   - [Anthropic Console](https://console.anthropic.com/) でAPIキーを作成
   - 画面左の **🔑（シークレット）** アイコン → 「新しいシークレットを追加」
   - 名前：`ANTHROPIC_API_KEY`　値：作成したキー　→「ノートブックからのアクセス」をオン

## ▶️ 使い方

1. 録音ファイル（mp3 / m4a / wav / mp4 など）をGoogleドライブに入れる
2. 下のセルを **①から順番に** ▶ボタンで実行する（③の設定欄だけ書き換えます）

> 💡 **途中で止まっても大丈夫**：処理結果は区切りごとにドライブへ保存されます。
> 接続が切れたら、①〜③を実行し直してから④・⑤を実行すると **続きから再開** します。
""")

# ---------------------------------------------------------------------------
code(r"""
#@title ① 準備（ライブラリのインストール・GPUの確認）
#@markdown ▶ボタンを押して1〜2分待ちます。
import subprocess, sys

gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
if gpu.returncode == 0:
    print("✅ GPUを確認しました:", gpu.stdout.strip())
else:
    print("⚠️ GPUが見つかりません。文字起こしがとても遅くなります。")
    print("   メニュー「ランタイム」→「ランタイムのタイプを変更」→「T4 GPU」を選んでください。")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "faster-whisper", "anthropic"], check=True)
print("✅ 準備が完了しました")
""")

# ---------------------------------------------------------------------------
code(r"""
#@title ② Googleドライブに接続
#@markdown 許可を求める画面が出たら、ご自身のGoogleアカウントで「許可」してください。
from google.colab import drive
drive.mount("/content/drive")
print("✅ ドライブに接続しました（マイドライブ = /content/drive/MyDrive）")
""")

# ---------------------------------------------------------------------------
code(r"""
#@title ③ 設定（ここだけ書き換えてください）

#@markdown ### 📁 ファイルの場所
#@markdown 録音ファイルのパス。画面左の📁でファイルを右クリック →「パスをコピー」で取得できます。<br>
#@markdown **フォルダのパス** を入れると、その中の録音ファイル一覧を表示します。
AUDIO_PATH = "/content/drive/MyDrive/会議録音/" #@param {type:"string"}
#@markdown 結果の保存先フォルダ（無ければ自動で作ります）
OUTPUT_DIR = "/content/drive/MyDrive/文字起こし結果" #@param {type:"string"}

#@markdown ### 🗣️ 文字起こしの設定
LANGUAGE = "ja" #@param ["ja", "en", "auto"]
#@markdown `large-v3`：最も正確（標準） ／ `large-v3-turbo`：速いが少し精度が落ちる
WHISPER_MODEL = "large-v3" #@param ["large-v3", "large-v3-turbo", "medium", "small"]
#@markdown 専門用語・人名・社名など（カンマ区切り）。認識精度が上がります。
GLOSSARY = "" #@param {type:"string"}

#@markdown ### 📝 議事録の設定
MEETING_TITLE = "" #@param {type:"string"}
MEETING_DATE = "" #@param {type:"string"}
PARTICIPANTS = "" #@param {type:"string"}
#@markdown 要約に使うClaudeのモデル（Opus：最高品質 ／ Sonnet：バランス ／ Haiku：最安・高速）
CLAUDE_MODEL = "claude-opus-5-5" #@param ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
DETAIL_LEVEL = "標準" #@param ["簡潔", "標準", "詳細"]
#@markdown 議事録への追加の指示（例：「営業部向けに、売上の話を重点的に」）
CUSTOM_INSTRUCTIONS = "" #@param {type:"string"}

#@markdown ### ⚙️ 詳細設定（通常は変更不要）
#@markdown 文字起こしを何分ずつ区切って保存するか
CHUNK_MINUTES = 10 #@param {type:"integer"}
#@markdown 要約のとき何分ずつに区切って整理するか
BLOCK_MINUTES = 30 #@param {type:"integer"}
#@markdown GPUのメモリ不足エラーが出たら 4 → 2 と下げてください
BATCH_SIZE = 8 #@param {type:"integer"}
#@markdown ✅ にすると、このファイルの過去の結果を消して最初からやり直します
START_OVER = False #@param {type:"boolean"}

import os, shutil
from pathlib import Path

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma",
              ".mp4", ".mov", ".mkv", ".webm", ".avi", ".3gp", ".amr"}

def _show_audio_files(folder):
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        print(f"   （{folder} の中に録音ファイルが見つかりませんでした）")
    for p in files[:50]:
        print(f"   {p}   ({p.stat().st_size / 1024 / 1024:.0f} MB)")
    if files:
        print("\n👆 使いたいファイルのパスをコピーして AUDIO_PATH に貼り付け、もう一度このセルを実行してください。")

_audio = Path(AUDIO_PATH.strip())
if not str(_audio).startswith("/content/drive"):
    print("⚠️ ドライブ内のパスではないようです。②を実行したか、パスが /content/drive/MyDrive/... で始まっているか確認してください。")
if _audio.is_dir():
    print(f"📁 フォルダが指定されました。中にある録音ファイル:")
    _show_audio_files(_audio)
    raise SystemExit("ファイルを1つ選んで AUDIO_PATH に入れてください。")
if not _audio.is_file():
    print(f"❌ ファイルが見つかりません: {_audio}")
    _parent = _audio.parent
    while not _parent.exists() and _parent != _parent.parent:
        _parent = _parent.parent
    print(f"   近くのフォルダ（{_parent}）にある録音ファイル:")
    _show_audio_files(_parent)
    raise SystemExit("AUDIO_PATH を確認してください。")

AUDIO_PATH = str(_audio)
JOB_DIR = Path(OUTPUT_DIR) / _audio.stem
if START_OVER and JOB_DIR.exists():
    shutil.rmtree(JOB_DIR)
    print("🗑️ 過去の結果を削除しました")
(JOB_DIR / "chunks").mkdir(parents=True, exist_ok=True)
(JOB_DIR / "summary_parts").mkdir(parents=True, exist_ok=True)

print("✅ 設定を確認しました")
print("   録音ファイル:", AUDIO_PATH)
print("   保存先      :", JOB_DIR)
""")

# ---------------------------------------------------------------------------
code(r"""
#@title ④ 文字起こしを実行
#@markdown 目安：T4 GPU・large-v3で、1時間の録音に約5〜10分。<br>
#@markdown 途中で止まった場合は、①〜③のあとにもう一度このセルを実行すると続きから再開します。
import json, subprocess, time
import numpy as np
from pathlib import Path

def fmt_time(sec, srt=False):
    sec = max(0.0, sec)
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    if srt:
        return f"{h:02d}:{m:02d}:{s:02d},{int((sec - int(sec)) * 1000):03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"

def media_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        raise RuntimeError(f"録音ファイルを読み込めませんでした。壊れているか、未対応の形式かもしれません。\n{out.stderr}")

def load_audio(src, start, length):
    # 指定した区間だけを、Whisper向けの形式（16kHz・モノラル）で読み込む
    out = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
         "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32)

def save_json(path, data):
    # 書き込み途中で切断されても壊れたファイルが残らないよう、一時ファイル経由で保存
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)

# --- Whisperモデルの読み込み（同じ設定なら再利用） ---
from faster_whisper import WhisperModel, BatchedInferencePipeline
import ctranslate2

_use_gpu = ctranslate2.get_cuda_device_count() > 0
if globals().get("_loaded_model_name") != WHISPER_MODEL:
    print(f"⏳ 音声認識モデル（{WHISPER_MODEL}）を読み込み中…（初回は数分かかります）")
    whisper_model = WhisperModel(WHISPER_MODEL,
                                 device="cuda" if _use_gpu else "cpu",
                                 compute_type="float16" if _use_gpu else "int8")
    whisper_pipeline = BatchedInferencePipeline(model=whisper_model)
    _loaded_model_name = WHISPER_MODEL

_lang = None if LANGUAGE == "auto" else LANGUAGE
_prompt = "以下は日本語の会議の書き起こしです。句読点を付けて記録します。" if LANGUAGE == "ja" else ""
if GLOSSARY.strip():
    _prompt += f" 用語: {GLOSSARY.strip()}"

def transcribe_audio(audio):
    if audio.size < 1600:  # 0.1秒未満は認識しない
        return []
    segments, _ = whisper_pipeline.transcribe(
        audio,
        language=_lang,
        batch_size=BATCH_SIZE,
        beam_size=5,
        vad_filter=True,                    # 無音部分を飛ばす（無音での「幻聴」防止）
        condition_on_previous_text=False,   # 同じ文の繰り返し暴走を防ぐ
        without_timestamps=False,           # 発言ごとに細かい時刻を付ける
        initial_prompt=_prompt or None,
        hotwords=GLOSSARY.strip() or None,
    )
    return [s for s in segments if s.text.strip()]

# --- 区切りごとに文字起こし（再開対応） ---
duration = media_duration(AUDIO_PATH)
chunk_dir = JOB_DIR / "chunks"
chunk_sec = max(1, CHUNK_MINUTES) * 60
done = sorted(chunk_dir.glob("chunk_*.json"))
if done:
    last = json.loads(done[-1].read_text(encoding="utf-8"))
    pos, idx = last["next_start"], len(done)
    print(f"🔁 前回の続き（{fmt_time(pos)}）から再開します")
else:
    pos, idx = 0.0, 0

print(f"🎧 録音の長さ: {fmt_time(duration)}")
t_begin, pos_begin = time.time(), pos

while pos < duration - 0.5:
    length = min(chunk_sec, duration - pos)
    is_last = pos + length >= duration - 0.5
    segs = [{"start": round(pos + s.start, 2), "end": round(pos + s.end, 2), "text": s.text.strip()}
            for s in transcribe_audio(load_audio(AUDIO_PATH, pos, length))]

    # 区切り目で言葉が途中で切れないよう、最後の発言は次の区切りで改めて認識する
    next_pos = pos + length
    if not is_last and len(segs) >= 2 and segs[-1]["start"] - pos > chunk_sec * 0.5:
        next_pos = segs[-1]["start"]
        segs = segs[:-1]

    save_json(chunk_dir / f"chunk_{idx:04d}.json",
              {"index": idx, "start": pos, "next_start": next_pos, "segments": segs})

    elapsed = time.time() - t_begin
    speed = (next_pos - pos_begin) / elapsed if elapsed > 0 else 0
    eta = (duration - next_pos) / speed if speed > 0 else 0
    print(f"   {fmt_time(next_pos)} / {fmt_time(duration)}  ({next_pos / duration * 100:5.1f}%)"
          f"  残り約{int(eta // 60)}分")
    pos, idx = next_pos, idx + 1

# --- 全区切りを結合して保存 ---
segments = []
for f in sorted(chunk_dir.glob("chunk_*.json")):
    for s in json.loads(f.read_text(encoding="utf-8"))["segments"]:
        if segments and segments[-1]["text"] == s["text"]:
            continue  # 連続する全く同じ文（認識の重複）は除く
        segments.append(s)

save_json(JOB_DIR / "transcript.json", segments)
(JOB_DIR / "文字起こし.txt").write_text(
    "\n".join(f"[{fmt_time(s['start'])}] {s['text']}" for s in segments) + "\n", encoding="utf-8")
(JOB_DIR / "字幕.srt").write_text(
    "\n".join(f"{i}\n{fmt_time(s['start'], True)} --> {fmt_time(s['end'], True)}\n{s['text']}\n"
              for i, s in enumerate(segments, 1)), encoding="utf-8")

print(f"\n✅ 文字起こし完了！（{len(segments)}行・{sum(len(s['text']) for s in segments):,}文字）")
print(f"   📄 {JOB_DIR / '文字起こし.txt'}")
print(f"   🎬 {JOB_DIR / '字幕.srt'}")
print("\n--- 冒頭のプレビュー ---")
for s in segments[:10]:
    print(f"[{fmt_time(s['start'])}] {s['text']}")
""")

# ---------------------------------------------------------------------------
code(r"""
#@title ⑤ 議事録を作成（Claudeで要約）
#@markdown 長い会議は30分ごとに整理してから、最後に全体の議事録にまとめます。<br>
#@markdown 途中で止まった場合も、もう一度実行すれば終わった部分は飛ばして続きから進みます。
import json, subprocess, hashlib
from pathlib import Path
import anthropic
from google.colab import userdata

# --- APIキーの読み込み ---
try:
    _api_key = userdata.get("ANTHROPIC_API_KEY")
except Exception:
    raise SystemExit("❌ APIキーが見つかりません。画面左の🔑（シークレット）に ANTHROPIC_API_KEY を登録し、"
                     "「ノートブックからのアクセス」をオンにしてください。")
client = anthropic.Anthropic(api_key=_api_key, max_retries=5, timeout=900)

PRICES = {  # 100万トークンあたりの料金（米ドル）：入力, 出力
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
usage_total = {"in": 0, "out": 0}

def ask_claude(system, user_text, max_tokens, effort):
    params = dict(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_text}],
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
    )
    try:
        if CLAUDE_MODEL in ("claude-opus-5-5", "claude-sonnet-5-5"):
            # 安全フィルターが誤って処理を断った場合に、別モデルで自動的にやり直す設定
            with client.beta.messages.stream(**params, betas=["server-side-fallback-2026-07-01"],
                                             extra_body={"fallbacks": "default"}) as stream:
                msg = stream.get_final_message()
        else:
            with client.messages.stream(**params) as stream:
                msg = stream.get_final_message()
    except anthropic.AuthenticationError:
        raise SystemExit("❌ APIキーが正しくありません。🔑シークレットの ANTHROPIC_API_KEY を確認してください。")
    except anthropic.PermissionDeniedError as e:
        raise SystemExit(f"❌ このAPIキーでは {CLAUDE_MODEL} を使えません。別のモデルを選ぶか、Consoleで設定を確認してください。\n{e}")
    except anthropic.RateLimitError:
        raise SystemExit("❌ 利用上限に達しました。少し時間をおいてから、もう一度このセルを実行してください（続きから再開します）。")
    except anthropic.APIStatusError as e:
        if e.status_code == 400 and "credit" in str(e).lower():
            raise SystemExit("❌ APIのクレジット残高が不足しています。Anthropic Consoleでチャージしてください。")
        raise

    usage_total["in"] += msg.usage.input_tokens
    usage_total["out"] += msg.usage.output_tokens
    if msg.stop_reason == "refusal":
        raise SystemExit("❌ Claudeがこの内容の処理を辞退しました。別のモデルを選んで再実行してください。")
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    if msg.stop_reason == "max_tokens":
        text += "\n\n（※出力が長すぎたため、ここで途切れています）"
    return text

def fmt_time(sec):
    h, rem = divmod(int(max(0, sec)), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

transcript_file = JOB_DIR / "transcript.json"
if not transcript_file.exists():
    raise SystemExit("❌ 文字起こしの結果がありません。先に④を実行してください。")
segments = json.loads(transcript_file.read_text(encoding="utf-8"))
if not segments:
    raise SystemExit("❌ 文字起こし結果が空です。録音に音声が含まれているか確認してください。")

def to_text(segs):
    return "\n".join(f"[{fmt_time(s['start'])}] {s['text']}" for s in segs)

# --- 時間ごとのブロックに分ける ---
block_sec = max(5, BLOCK_MINUTES) * 60
blocks = []
for s in segments:
    if not blocks or s["start"] - blocks[-1][0]["start"] >= block_sec:
        blocks.append([])
    blocks[-1].append(s)

meeting_info = "\n".join(x for x in [
    f"会議名: {MEETING_TITLE}" if MEETING_TITLE.strip() else "",
    f"日時: {MEETING_DATE}" if MEETING_DATE.strip() else "",
    f"参加者: {PARTICIPANTS}" if PARTICIPANTS.strip() else "",
    f"専門用語・固有名詞: {GLOSSARY}" if GLOSSARY.strip() else "",
] if x) or "（会議の情報は入力されていません）"

DETAIL = {
    "簡潔": "要点だけを短く、箇条書き中心でまとめてください。",
    "標準": "要点を押さえつつ、議論の経緯や理由も分かる程度の詳しさでまとめてください。",
    "詳細": "誰が何を言ったか・数値・理由・反対意見など、できるだけ漏らさず詳しく記録してください。",
}[DETAIL_LEVEL]

ASR_NOTE = (
    "入力は音声認識（Whisper）による自動文字起こしです。次の点に注意してください。\n"
    "- 同音異義語の誤変換や聞き間違いがあります。文脈から正しい語が明らかな場合は直して書いてください。\n"
    "- 話者の区別はありません。話の流れから話者が明らかな場合のみ記載し、推測で人物を断定しないでください。\n"
    "- 「えー」「あの」などのつなぎ言葉や雑談は省いてください。\n"
    "- 文字起こしに無い内容を作り出さないでください。聞き取りが不確かな重要語には（？）を付けてください。\n"
    "- 行頭の [時:分:秒] は録音開始からの経過時間です。"
)

BLOCK_SYSTEM = f'''あなたは優秀な議事録担当者です。長時間の会議の文字起こしの「一部区間」を読み、後で全体の議事録にまとめるための整理メモを日本語で作ります。

{ASR_NOTE}

次の形式（Markdown）で出力してください。該当がない項目は「なし」と書いてください。

### 話題の流れ
- [時:分] 話題 — 要点（時刻は区間内の該当箇所）
### 主な論点・発言
### 決定事項
### ToDo・宿題（内容／担当／期限。分からない部分は「不明」）
### 数値・固有名詞・重要キーワード
### 未解決の点・確認が必要な点

{DETAIL}'''

FINAL_SYSTEM = f'''あなたは優秀な議事録担当者です。会議全体の議事録を日本語のMarkdownで作成します。
入力として、区間ごとの整理メモ（と、可能な場合は文字起こし全文）が与えられます。

{ASR_NOTE}

次の構成で出力してください。区間をまたいで続いた話題は1つにまとめ、重複は除いてください。

# （会議名。不明なら内容から付けた仮題）議事録
- **日時**：（不明なら「記載なし」）
- **参加者**：（不明なら「記載なし」）
- **録音時間**：

## 1. 会議の要約
（会議全体を3〜7行で。何のための会議で、何が決まったか）

## 2. 議題ごとの議論内容
（議題ごとに「### 見出し」を付け、結論 → 主な意見・経緯 の順に）

## 3. 決定事項
## 4. ToDo（今後のアクション）
| No | 内容 | 担当 | 期限 |
|---|---|---|---|

## 5. 未決事項・次回への持ち越し
## 6. タイムライン（主な話題と開始時刻）
## 7. 補足（聞き取りが不確かな箇所・確認が必要な点）

{DETAIL}'''

if CUSTOM_INSTRUCTIONS.strip():
    BLOCK_SYSTEM += f"\n\n追加の指示: {CUSTOM_INSTRUCTIONS.strip()}"
    FINAL_SYSTEM += f"\n\n追加の指示: {CUSTOM_INSTRUCTIONS.strip()}"

# --- ステップ1：区間ごとの整理（保存済みの区間は飛ばす） ---
parts_dir = JOB_DIR / "summary_parts"
part_texts = []
print(f"📝 {len(blocks)}個の区間に分けて整理します（モデル: {CLAUDE_MODEL}）")
for i, blk in enumerate(blocks, 1):
    head = f"## 区間{i}（{fmt_time(blk[0]['start'])}〜{fmt_time(blk[-1]['end'])}）"
    user = (f"【会議の情報】\n{meeting_info}\n\n"
            f"【区間】全{len(blocks)}区間のうち{i}番目（{head[3:]}）\n\n"
            f"【文字起こし】\n{to_text(blk)}")
    # 文字起こし・設定・モデルが前回と同じ区間だけ、保存済みの結果を再利用する
    key = hashlib.sha1(f"{CLAUDE_MODEL}\n{BLOCK_SYSTEM}\n{user}".encode("utf-8")).hexdigest()[:10]
    part_file = parts_dir / f"part_{i:03d}_{key}.md"
    if part_file.exists():
        print(f"   区間{i}/{len(blocks)}: 保存済みのため省略")
    else:
        for old in parts_dir.glob(f"part_{i:03d}*.md"):
            old.unlink()
        print(f"   区間{i}/{len(blocks)}: 整理中…", end="", flush=True)
        body = ask_claude(BLOCK_SYSTEM, user, max_tokens=16000, effort="medium")
        part_file.write_text(f"{head}\n\n{body}\n", encoding="utf-8")
        print(" 完了")
    part_texts.append(part_file.read_text(encoding="utf-8"))

for old in parts_dir.glob("part_*.md"):  # 区間数が減った場合の古いメモを片付ける
    if int(old.name[5:8]) > len(blocks):
        old.unlink()

# --- ステップ2：全体の議事録にまとめる ---
print("📚 全体の議事録にまとめています…（数分かかることがあります）")
full_text = to_text(segments)
user = (f"【会議の情報】\n{meeting_info}\n録音時間: {fmt_time(segments[-1]['end'])}\n\n"
        f"【区間ごとの整理メモ】\n\n" + "\n\n".join(part_texts))
if len(full_text) <= 300_000:  # 長すぎなければ全文も渡して精度を上げる
    user += f"\n\n【文字起こし全文（確認用）】\n{full_text}"
minutes = ask_claude(FINAL_SYSTEM, user, max_tokens=64000, effort="high")

minutes_md = JOB_DIR / "議事録.md"
minutes_md.write_text(minutes + "\n", encoding="utf-8")
(JOB_DIR / "区間ごとの詳細メモ.md").write_text("\n\n".join(part_texts), encoding="utf-8")

# Wordファイルも作成（pandocが使える場合）
try:
    subprocess.run(["pandoc", str(minutes_md), "-o", str(JOB_DIR / "議事録.docx")],
                   check=True, capture_output=True)
    _docx = True
except Exception:
    _docx = False

_pin, _pout = PRICES[CLAUDE_MODEL]
_cost = usage_total["in"] / 1e6 * _pin + usage_total["out"] / 1e6 * _pout
print(f"\n✅ 議事録が完成しました！（今回のAPI利用料の目安: 約${_cost:.2f}）")
print(f"   📄 {minutes_md}")
if _docx:
    print(f"   📄 {JOB_DIR / '議事録.docx'}（Word形式）")
print(f"   📄 {JOB_DIR / '区間ごとの詳細メモ.md'}")

from IPython.display import Markdown, display
display(Markdown(minutes))
""")

# ---------------------------------------------------------------------------
code(r"""
#@title ⑥（おまけ）会議の内容について質問する
#@markdown ⑤を実行したあとに使えます。質問を書き換えて何度でも実行できます。
QUESTION = "〇〇さんが担当になったタスクを全部教えて" #@param {type:"string"}

_full = to_text(segments)
if len(_full) <= 300_000:
    _context = f"【文字起こし全文】\n{_full}"
else:
    _context = "【区間ごとの整理メモ】\n" + "\n\n".join(part_texts)

_answer = ask_claude(
    "あなたは会議の内容に詳しいアシスタントです。与えられた会議の記録だけを根拠に、日本語で答えてください。"
    "根拠となる発言の時刻 [時:分:秒] を添え、記録から分からないことは「記録からは分かりません」と答えてください。\n\n" + ASR_NOTE,
    f"【会議の情報】\n{meeting_info}\n\n{_context}\n\n【質問】\n{QUESTION}",
    max_tokens=16000, effort="medium")
from IPython.display import Markdown, display
display(Markdown(_answer))
""")

# ---------------------------------------------------------------------------
md(r"""
## 📂 出力されるファイル（③で指定した保存先フォルダ／録音ファイル名 の中）

| ファイル | 内容 |
|---|---|
| `議事録.md` / `議事録.docx` | 完成した議事録（要約・決定事項・ToDo・タイムライン） |
| `区間ごとの詳細メモ.md` | 30分ごとの詳しい整理メモ |
| `文字起こし.txt` | 時刻付きの文字起こし全文 |
| `字幕.srt` | 動画プレーヤーで使える字幕ファイル |
| `chunks/`, `summary_parts/`, `transcript.json` | 途中再開用のデータ（消しても問題ありません） |

## ❓ 困ったときは

- **「CUDA out of memory」と出た** → ③の `BATCH_SIZE` を 4 や 2 に下げて、④を再実行
- **途中で接続が切れた** → ①②③を実行してから④（または⑤）を再実行すると続きから再開
- **議事録の内容を変えたい** → ③の設定を変えて③→⑤を再実行（変更があった部分だけ作り直されます）
- **固有名詞の誤りが多い** → ③の `GLOSSARY` に正しい表記を入れ、`START_OVER` に✅を付けて④から再実行
""")

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

for c in cells:  # Jupyter形式に合わせて、各セルを行ごとのリストにする
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"wrote {OUT}")
