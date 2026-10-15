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
**すべての処理をColabの中（無料のGPU）で行うので、料金やAPIキーは不要です。**

---

## 📋 はじめに一度だけやること

- **GPUを有効にする**：メニュー「ランタイム」→「ランタイムのタイプを変更」→「T4 GPU」を選んで保存

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

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "faster-whisper", "transformers", "accelerate", "bitsandbytes"], check=True)
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
#@markdown 要約に使うAI（Colabの中で動きます）。`Qwen3-8B`：標準 ／ `Qwen3-4B`：速いが簡素 ／ `Swallow-8B`：日本語特化
LLM_MODEL = "Qwen3-8B" #@param ["Qwen3-8B", "Qwen3-4B", "Swallow-8B"]
DETAIL_LEVEL = "標準" #@param ["簡潔", "標準", "詳細"]
#@markdown 議事録への追加の指示（例：「営業部向けに、売上の話を重点的に」）
CUSTOM_INSTRUCTIONS = "" #@param {type:"string"}

#@markdown ### ⚙️ 詳細設定（通常は変更不要）
#@markdown 文字起こしを何分ずつ区切って保存するか
CHUNK_MINUTES = 10 #@param {type:"integer"}
#@markdown 要約のとき何文字ずつに区切って整理するか（目安：日本語の会話で約15〜20分ぶん）
BLOCK_CHARS = 6000 #@param {type:"integer"}
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
#@title ⑤ 議事録を作成（Colabの中のAIで要約・料金はかかりません）
#@markdown 初回はAIモデル（約16GB）のダウンロードと読み込みに5〜10分かかります。<br>
#@markdown 目安：1時間の会議で約10〜20分。途中で止まっても、再実行すれば終わった部分は飛ばして続きから進みます。
import json, subprocess, hashlib, gc, re, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer

LLM_IDS = {
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-4B": "Qwen/Qwen3-4B-Instruct-2507",
    "Swallow-8B": "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5",
}
_use_gpu = torch.cuda.is_available()

# 文字起こし用のモデルを片付けて、GPUのメモリを空ける
for _name in ("whisper_pipeline", "whisper_model", "_loaded_model_name"):
    globals().pop(_name, None)
gc.collect()
if _use_gpu:
    torch.cuda.empty_cache()

# --- 要約用AIモデルの読み込み（同じモデルなら再利用） ---
if globals().get("_llm_name") != LLM_MODEL:
    globals().pop("llm", None)
    gc.collect()
    _repo = LLM_IDS.get(LLM_MODEL, LLM_MODEL)
    print(f"⏳ 要約用AI（{_repo}）を読み込み中…（初回はダウンロードに数分かかります）")
    tokenizer = AutoTokenizer.from_pretrained(_repo)
    if _use_gpu:
        # 4bitに圧縮して読み込み、無料GPU（T4）のメモリに収める
        llm = AutoModelForCausalLM.from_pretrained(
            _repo, device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
    else:
        print("⚠️ GPUが無いため、とても時間がかかります（T4 GPUの利用をおすすめします）")
        llm = AutoModelForCausalLM.from_pretrained(_repo, torch_dtype=torch.float32)
    llm.eval()
    _llm_name = LLM_MODEL
    print("✅ 読み込み完了")

def generate(system, user_text, max_new_tokens, show=False):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if show else None
    with torch.inference_mode():
        out = llm.generate(**inputs, max_new_tokens=max_new_tokens, streamer=streamer,
                           do_sample=True, temperature=0.5, top_p=0.8, top_k=20,
                           repetition_penalty=1.05,
                           pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if len(new_tokens) >= max_new_tokens:
        text += "\n\n（※長くなったため、ここで途切れています）"
    return text

parts_dir = JOB_DIR / "summary_parts"

def cached_generate(tag, system, user_text, max_new_tokens):
    # 文字起こし・設定・モデルが前回と同じなら、保存済みの結果を再利用する
    key = hashlib.sha1(f"{LLM_MODEL}\n{system}\n{user_text}".encode("utf-8")).hexdigest()[:10]
    f = parts_dir / f"{tag}_{key}.md"
    if f.exists():
        return f.read_text(encoding="utf-8"), True
    text = generate(system, user_text, max_new_tokens)
    f.write_text(text, encoding="utf-8")
    return text, False

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

# --- 文字数で区間に分ける ---
blocks, _size = [], 0
for s in segments:
    if not blocks or _size + len(s["text"]) > max(1000, BLOCK_CHARS):
        blocks.append([])
        _size = 0
    blocks[-1].append(s)
    _size += len(s["text"])

meeting_info = "\n".join(x for x in [
    f"会議名: {MEETING_TITLE}" if MEETING_TITLE.strip() else "",
    f"参加者: {PARTICIPANTS}" if PARTICIPANTS.strip() else "",
    f"専門用語・固有名詞: {GLOSSARY}" if GLOSSARY.strip() else "",
] if x) or "（なし）"

DETAIL = {
    "簡潔": "要点だけを短く、箇条書きでまとめてください。",
    "標準": "要点を押さえつつ、理由や経緯も分かるようにまとめてください。",
    "詳細": "発言の内容・数値・理由・反対意見などを、できるだけ漏らさず詳しく書いてください。",
}[DETAIL_LEVEL]
EXTRA = f"\n追加の指示: {CUSTOM_INSTRUCTIONS.strip()}" if CUSTOM_INSTRUCTIONS.strip() else ""

ASR_NOTE = (
    "入力は音声の自動文字起こしなので、聞き間違いや誤変換があります。文脈から正しい語が明らかなら直してください。"
    "つなぎ言葉や雑談は省き、書かれていない内容を作らないでください。[時:分:秒] は録音開始からの時刻です。"
)

BLOCK_SYSTEM = f'''あなたは会議の議事録担当者です。会議の文字起こしの一部を読み、日本語で整理メモを作ります。
{ASR_NOTE}
次の見出しで、Markdownの箇条書きで出力してください。該当がなければ「なし」と書きます。
### 話題の流れ（[時:分] 話題：要点）
### 主な意見・発言
### 決定事項
### ToDo（内容／担当／期限）
### 未解決の点
{DETAIL}{EXTRA}'''

MERGE_SYSTEM = f'''あなたは会議の議事録担当者です。会議の連続した区間の整理メモを、1つの整理メモに統合します。
重複はまとめ、決定事項・ToDo・数値・時刻は省略せず残してください。見出しは入力と同じものを使います。{EXTRA}'''

FINAL_SYSTEM = f'''あなたは会議の議事録担当者です。会議全体の整理メモから、日本語の議事録本文を作ります。
同じ話題は1つにまとめ、メモに無い内容は書かないでください。次の見出しのMarkdownで出力してください。

## 1. 会議の要約
（3〜7行。何の会議で、何が決まったか）
## 2. 議題ごとの議論内容
（議題ごとに「### 見出し」を付け、結論 → 主な意見 の順に）
## 3. 決定事項
## 4. ToDo
| No | 内容 | 担当 | 期限 |
|---|---|---|---|
## 5. 未決事項・次回への持ち越し
## 6. タイムライン（[時:分] 主な話題）
{DETAIL}{EXTRA}'''

t_begin = time.time()

# --- ステップ1：区間ごとの整理 ---
print(f"📝 {len(blocks)}個の区間に分けて整理します")
notes = []
for i, blk in enumerate(blocks, 1):
    span = f"{fmt_time(blk[0]['start'])}〜{fmt_time(blk[-1]['end'])}"
    print(f"   区間{i}/{len(blocks)}（{span}）: ", end="", flush=True)
    t0 = time.time()
    text, cached = cached_generate(
        f"part_{i:03d}", BLOCK_SYSTEM,
        f"【会議の情報】\n{meeting_info}\n\n【文字起こし（{span}）】\n{to_text(blk)}",
        max_new_tokens=1500)
    print("保存済みのため省略" if cached else f"完了（{time.time() - t0:.0f}秒）")
    notes.append(f"## 区間{i}（{span}）\n\n{text}")
(JOB_DIR / "区間ごとの詳細メモ.md").write_text("\n\n".join(notes) + "\n", encoding="utf-8")

# --- ステップ2：メモが長すぎる場合は、隣り合う区間をまとめて短くする ---
NOTES_LIMIT = 12000  # AIに一度に渡すメモの文字数の上限
level = 0
while len("\n\n".join(notes)) > NOTES_LIMIT and len(notes) > 1:
    level += 1
    groups, cur = [], []
    for n in notes:
        if cur and len("\n\n".join(cur + [n])) > NOTES_LIMIT:
            groups.append(cur)
            cur = []
        cur.append(n)
    groups.append(cur)
    if len(groups) == len(notes):  # 1つずつしか入らない場合は2つずつまとめる
        groups = [notes[j:j + 2] for j in range(0, len(notes), 2)]
    print(f"🔗 メモが長いため、{len(notes)}件を{len(groups)}件に統合します（段階{level}）")
    merged = []
    for j, g in enumerate(groups, 1):
        text, _ = cached_generate(f"merge{level}_{j:03d}", MERGE_SYSTEM, "\n\n".join(g), max_new_tokens=2500)
        merged.append(text)
    notes = merged

# --- ステップ3：議事録の本文を作る ---
print("📚 全体の議事録をまとめています…（文章が少しずつ表示されます）\n")
body = generate(FINAL_SYSTEM, f"【会議の情報】\n{meeting_info}\n\n【整理メモ】\n" + "\n\n".join(notes),
                max_new_tokens=4000, show=True)

title = MEETING_TITLE.strip() or JOB_DIR.name
header = (f"# {title} 議事録\n\n"
          f"- **日時**：{MEETING_DATE.strip() or '記載なし'}\n"
          f"- **参加者**：{PARTICIPANTS.strip() or '記載なし'}\n"
          f"- **録音時間**：{fmt_time(segments[-1]['end'])}\n\n")
minutes = header + body + ("\n\n---\n※ この議事録は音声の自動文字起こしをもとにAIが作成したものです。"
                           "重要な内容は「区間ごとの詳細メモ.md」や「文字起こし.txt」で確認してください。\n")
minutes_md = JOB_DIR / "議事録.md"
minutes_md.write_text(minutes, encoding="utf-8")
final_notes = notes

# Wordファイルも作成（pandocが使える場合）
try:
    subprocess.run(["pandoc", str(minutes_md), "-o", str(JOB_DIR / "議事録.docx")],
                   check=True, capture_output=True)
    _docx = True
except Exception:
    _docx = False

print(f"\n\n✅ 議事録が完成しました！（所要時間 約{(time.time() - t_begin) / 60:.0f}分）")
print(f"   📄 {minutes_md}")
if _docx:
    print(f"   📄 {JOB_DIR / '議事録.docx'}（Word形式）")
print(f"   📄 {JOB_DIR / '区間ごとの詳細メモ.md'}")
""")

# ---------------------------------------------------------------------------
code(r"""
#@title ⑥（おまけ）会議の内容について質問する
#@markdown ⑤を実行したあとに使えます。質問を書き換えて何度でも実行できます。
QUESTION = "決まったことと、それぞれの担当者を教えて" #@param {type:"string"}

print(generate(
    "あなたは会議の内容に詳しいアシスタントです。与えられた会議のメモだけを根拠に、日本語で簡潔に答えてください。"
    "根拠となる時刻 [時:分] があれば添え、メモから分からないことは「記録からは分かりません」と答えてください。",
    f"【会議のメモ】\n" + "\n\n".join(final_notes) + f"\n\n【質問】\n{QUESTION}",
    max_new_tokens=1500))
""")

# ---------------------------------------------------------------------------
md(r"""
## 📂 出力されるファイル（③で指定した保存先フォルダ／録音ファイル名 の中）

| ファイル | 内容 |
|---|---|
| `議事録.md` / `議事録.docx` | 完成した議事録（要約・決定事項・ToDo・タイムライン） |
| `区間ごとの詳細メモ.md` | 区間（約15〜20分）ごとの詳しい整理メモ |
| `文字起こし.txt` | 時刻付きの文字起こし全文 |
| `字幕.srt` | 動画プレーヤーで使える字幕ファイル |
| `chunks/`, `summary_parts/`, `transcript.json` | 途中再開用のデータ（消しても問題ありません） |

## ❓ 困ったときは

- **④で「CUDA out of memory」と出た** → ③の `BATCH_SIZE` を 4 や 2 に下げて、④を再実行
- **⑤で「CUDA out of memory」と出た** → メニュー「ランタイム」→「セッションを再起動」後、①②③⑤の順に実行（④はやり直し不要）。それでもダメなら③で `Qwen3-4B` を選ぶ
- **途中で接続が切れた** → ①②③を実行してから④（または⑤）を再実行すると続きから再開
- **議事録の内容を変えたい** → ③の設定を変えて③→⑤を再実行（変更があった部分だけ作り直されます）
- **議事録の出来が今ひとつ** → ⑤をもう一度実行すると、最後のまとめが別の言い回しで作り直されます。`DETAIL_LEVEL` を「詳細」にするのも効果的です
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
