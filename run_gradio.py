import gradio as gr
import torch
import torchaudio
import os
import time
import random
import shutil
import sys

# Add infer/ to the Python path to enable local import
sys.path.insert(0, os.path.abspath("infer"))

from infer_utils import (
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)

@torch.inference_mode()
def inference(
    cfm_model,
    vae_model,
    cond,
    text,
    duration,
    style_prompt,
    negative_style_prompt,
    start_time,
    pred_frames,
    batch_infer_num,
    steps=32,
    cfg_strength=4.0,
    chunked=False,
):
    from einops import rearrange

    print("[inference] Sampling latents...")
    latents, _ = cfm_model.sample(
        cond=cond,
        text=text,
        duration=duration,
        style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt,
        steps=steps,
        cfg_strength=cfg_strength,
        start_time=start_time,
        latent_pred_segments=pred_frames,
        batch_infer_num=batch_infer_num
    )

    outputs = []
    for i, latent in enumerate(latents):
        print(f"[inference] Decoding latent {i}...")
        latent = latent.to(torch.float32)
        latent = latent.transpose(1, 2)

        output = decode_audio(latent, vae_model, chunked=chunked)

        output = rearrange(output, "b d n -> d (b n)")
        output = (
            output.to(torch.float32)
            .div(torch.max(torch.abs(output)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )
        outputs.append(output)

    print("[inference] Done.")
    return outputs

OUTPUT_DIR = "outputs"
DUMMY_WAV = "static/test_output.wav"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading models...")

model_repo_options = [
    "ASLP-lab/DiffRhythm-1_2",
    "ASLP-lab/DiffRhythm-base",
    "ASLP-lab/DiffRhythm-full",
    "ASLP-lab/DiffRhythm-vae",
]

selected_repo = model_repo_options[0]
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
cfm, tokenizer, muq, vae = prepare_model(max_frames=2048, device=device, repo_id=selected_repo)
print("Models loaded.")

def simulate_fast_output(label="prompt"):
    dummy_path = os.path.join(OUTPUT_DIR, f"dummy_{label}.wav")
    shutil.copy(DUMMY_WAV, dummy_path)
    return dummy_path, dummy_path, "✅ Fast mode: demo file returned."

def generate(prompt_text=None, ref_audio_path=None, duration=95, chunked=False, fast_mode=False, sample_rate=24000, repo_id="ASLP-lab/DiffRhythm-1_2", steps=32, cfg_strength=4.0, batch_size=1):
    global cfm, tokenizer, muq, vae, selected_repo
    if fast_mode:
        return simulate_fast_output("prompt" if prompt_text else "ref")

    try:
        print("[generate] Starting generation...")
        start = time.time()
        max_frames = 2048 if duration == 95 else 6144

        if repo_id != selected_repo:
            print(f"[generate] Reloading models from {repo_id}")
            cfm, tokenizer, muq, vae = prepare_model(max_frames=max_frames, device=device, repo_id=repo_id)
            selected_repo = repo_id

        lrc_prompt, start_time = get_lrc_token(max_frames, "" if not prompt_text else prompt_text, tokenizer, device)

        if ref_audio_path:
            print(f"[generate] Using reference audio: {ref_audio_path}")
            style_prompt = get_style_prompt(muq, ref_audio_path)
        else:
            print(f"[generate] Using text prompt: {prompt_text}")
            style_prompt = get_style_prompt(muq, prompt=prompt_text)

        negative_style_prompt = get_negative_style_prompt(device)
        latent_prompt, pred_frames = get_reference_latent(device, max_frames, False, None, None, vae)

        outputs = inference(
            cfm_model=cfm,
            vae_model=vae,
            cond=latent_prompt,
            text=lrc_prompt,
            duration=max_frames,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            start_time=start_time,
            pred_frames=pred_frames,
            steps=steps,
            cfg_strength=cfg_strength,
            chunked=chunked,
            batch_infer_num=batch_size,
        )

        selected = random.sample(outputs, 1)[0]
        out_path = os.path.join(OUTPUT_DIR, f"generated_{int(time.time())}.wav")
        torchaudio.save(out_path, selected, sample_rate=sample_rate)

        elapsed = time.time() - start
        print(f"[generate] Done in {elapsed:.2f} seconds.")
        return out_path, out_path, f"✅ Done in {elapsed:.1f} seconds."

    except Exception as e:
        print(f"[generate] Error: {e}")
        return None, None, f"❌ Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# 🎵 DiffRhythm Music Generator")

    with gr.Row():
        duration = gr.Slider(minimum=95, maximum=180, step=5, value=95, label="Track Duration (seconds)")
        steps = gr.Slider(minimum=8, maximum=100, step=4, value=32, label="Diffusion Steps")
        cfg_strength = gr.Slider(minimum=0.1, maximum=10.0, step=0.1, value=4.0, label="CFG Strength")
        batch_size = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="Batch Size")
        sample_rate = gr.Dropdown(["24000", "44100"], value="44100", label="Output Sample Rate (Hz)")
        model_select = gr.Dropdown(choices=model_repo_options, value=model_repo_options[0], label="Model Repo ID")
        chunked = gr.Checkbox(label="Use Chunked Mode (Low VRAM)", value=True)
        fast_mode = gr.Checkbox(label="🚀 Fast Mode (demo only)", value=False)

    with gr.Tab("Text Prompt"):
        prompt_input = gr.Textbox(label="Prompt", placeholder="e.g. 'Ambient piano with lo-fi texture'")
        gen_btn1 = gr.Button("Generate from Prompt")
        out_audio1 = gr.Audio(label="🎵 Preview")
        download1 = gr.File(label="⬇️ Download")
        status1 = gr.Textbox(label="Status")
        gen_btn1.click(
            fn=lambda prompt, duration, chunked, fast_mode, sr, repo_id, steps, cfg_strength, batch_size: generate(
                prompt_text=prompt,
                duration=duration,
                chunked=chunked,
                fast_mode=fast_mode,
                sample_rate=int(sr),
                repo_id=repo_id,
                steps=steps,
                cfg_strength=cfg_strength,
                batch_size=batch_size,
            ),
            inputs=[prompt_input, duration, chunked, fast_mode, sample_rate, model_select, steps, cfg_strength, batch_size],
            outputs=[out_audio1, download1, status1]
        )

    with gr.Tab("Reference Audio"):
        audio_input = gr.Audio(type="filepath", label="Upload Reference Audio")
        gen_btn2 = gr.Button("Generate from Audio")
        out_audio2 = gr.Audio(label="🎵 Preview")
        download2 = gr.File(label="⬇️ Download")
        status2 = gr.Textbox(label="Status")
        gen_btn2.click(
            fn=lambda ref_audio, duration, chunked, fast_mode, sr, repo_id, steps, cfg_strength, batch_size: generate(
                ref_audio_path=ref_audio,
                duration=duration,
                chunked=chunked,
                fast_mode=fast_mode,
                sample_rate=int(sr),
                repo_id=repo_id,
                steps=steps,
                cfg_strength=cfg_strength,
                batch_size=batch_size,
            ),
            inputs=[audio_input, duration, chunked, fast_mode, sample_rate, model_select, steps, cfg_strength, batch_size],
            outputs=[out_audio2, download2, status2]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

