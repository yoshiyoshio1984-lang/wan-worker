"""
RunPod Serverless Worker - 動画生成 (Wan 2.2 TI2V-5B, Diffusers)
推奨GPU: RTX 4090 24GB (Serverless Flex $1.34/h) または A40 / L40S
  - 5秒・720pで 1本 4-9分 (4090)
  - 1本あたり 約 $0.05
  - 無音 (Wanは音声なし)。ライセンス Apache 2.0。
環境変数:
  HF_TOKEN - Hugging Faceトークン (任意。公開モデルのため基本不要)
  MODEL_ID - デフォルト Wan-AI/Wan2.2-TI2V-5B-Diffusers
入力 (job['input']):
  prompt: str (必須)
  negative_prompt: str (任意)
  image_base64: str (任意, I2Vの場合。Fluxで作った画像を渡すと安定する)
  width: int (デフォルト 1280)
  height: int (デフォルト 704)  # Wan 2.2 TI2Vは 1280x704 が推奨
  num_frames: int (デフォルト 121 = 5秒@24fps, 4n+1 に丸める)
  num_inference_steps: int (デフォルト 30)
  guidance_scale: float (デフォルト 5.0)
  seed: int (任意, -1でランダム)
出力:
  {"video_base64": "...mp4...", "width": 1280, "height": 704, "fps": 24, "model": "..."}
注意:
  - 動画は1ジョブが数分かかる。Serverlessの runsync ではなく
    run + ポーリング方式 (status確認) を使うこと。
  - 旧版は LTX-2.3 + LTXPipeline だったが、2026-11-24時点で
    Lightricks/LTX-2.3 は Diffusers未対応 ("support coming soon"、
    model_index.json がなく from_pretrained で読めない) のため
    デプロイするとモデル読込で失敗する。LTXを使いたい場合は
    LTX-2 (LTX2Pipeline, 2段階生成) か公式 ltx-pipelines を使うこと。
    詳しくは README.md を参照。
"""
import os
import base64
import io
import random
import tempfile

import runpod
import torch
from diffusers import WanImageToVideoPipeline, WanPipeline
from diffusers.utils import export_to_video

MODEL_ID = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")

print(f"Loading {MODEL_ID} ...", flush=True)
pipe = WanPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    token=os.environ.get("HF_TOKEN") or None,
)
pipe.enable_model_cpu_offload()
try:
    pipe.vae.enable_tiling()
except Exception:
    pass
print("Model loaded.", flush=True)

_I2V_PIPE = None


def get_i2v_pipe():
    # I2V用パイプラインは初回利用時に1回だけ読み込む (以後は再利用)。
    global _I2V_PIPE
    if _I2V_PIPE is None:
        print("Loading I2V pipeline (same checkpoint) ...", flush=True)
        _I2V_PIPE = WanImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=os.environ.get("HF_TOKEN") or None,
        )
        _I2V_PIPE.enable_model_cpu_offload()
        try:
            _I2V_PIPE.vae.enable_tiling()
        except Exception:
            pass
    return _I2V_PIPE


def handler(job):
    inp = job.get("input", {}) or {}
    prompt = inp.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required"}

    width = int(inp.get("width", 1280))
    height = int(inp.get("height", 704))
    num_frames = int(inp.get("num_frames", 121))
    # Wanは (4n+1) フレームを要求する
    num_frames = ((num_frames - 1) // 4) * 4 + 1
    steps = int(inp.get("num_inference_steps", 30))
    guidance = float(inp.get("guidance_scale", 5.0))
    fps = int(inp.get("fps", 24))
    negative_prompt = inp.get("negative_prompt", "")
    seed = inp.get("seed", -1)
    if seed is None or int(seed) < 0:
        seed = random.randint(0, 2**32 - 1)
    seed = int(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )

    # I2V: 画像が渡されたら WanImageToVideoPipeline を同じ重みで使う
    image_b64 = inp.get("image_base64")
    if image_b64:
        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        result = get_i2v_pipe()(image=img, **kwargs)
    else:
        result = pipe(**kwargs)

    frames = result.frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out_path = f.name
    export_to_video(frames, out_path, fps=fps)

    with open(out_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(out_path)

    return {
        "video_base64": video_b64,
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames": num_frames,
        "seed": seed,
        "model": MODEL_ID,
    }


runpod.serverless.start({"handler": handler})
