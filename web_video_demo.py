#!/usr/bin/env python
"""
Interactive LAVIS Video Demo with Gradio.

Usage:
    python web_video_demo.py

This opens a web UI where you can:
1. Upload a video.
2. Ask questions about the video (VideoQA).
3. Generate an ordered action timeline describing what happens over time.
"""

import json
import os
import tempfile
from typing import Dict, List, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from lavis.common.utils import download_url
from lavis.models import load_model_and_preprocess

# Global model state
DEVICE = None
MODEL_VIDEO_QA = None
VIS_PROCESSORS_VIDEO_QA = None
TXT_PROCESSORS_VIDEO_QA = None
MODEL_CAPTION = None
VIS_PROCESSORS_CAPTION = None
LABEL2ANSWER = {}

MSRVTT_ANS2LABEL_URL = (
    "https://storage.googleapis.com/sfr-vision-language-research/"
    "LAVIS/datasets/msrvtt/train_ans2label.json"
)
MSRVTT_ANS2LABEL_REL_PATH = "msrvtt/annotations/qa_ans2label.json"


def resolve_writable_cache_root() -> str:
    """Pick a writable cache directory for demo artifacts."""
    candidates = []

    env_cache = os.environ.get("LAVIS_VIDEO_DEMO_CACHE")
    if env_cache:
        candidates.append(os.path.expanduser(env_cache))

    candidates.extend(
        [
            os.path.join(os.getcwd(), ".cache", "lavis_video_demo"),
            os.path.join(os.path.expanduser("~"), ".cache", "lavis_video_demo"),
            os.path.join(tempfile.gettempdir(), "lavis_video_demo"),
        ]
    )

    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            test_path = os.path.join(candidate, ".write_test")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test_path)
            return candidate
        except OSError:
            continue

    raise RuntimeError(
        "No writable cache directory found. Set LAVIS_VIDEO_DEMO_CACHE "
        "to a writable path."
    )


def load_answer_vocab(cache_root: str) -> Dict[int, str]:
    """Download/load answer mapping and build label-index to answer text map."""
    ans2label_path = os.path.join(cache_root, MSRVTT_ANS2LABEL_REL_PATH)
    os.makedirs(os.path.dirname(ans2label_path), exist_ok=True)

    if not os.path.exists(ans2label_path):
        download_url(
            url=MSRVTT_ANS2LABEL_URL,
            root=os.path.dirname(ans2label_path),
            filename=os.path.basename(ans2label_path),
        )

    with open(ans2label_path, "r", encoding="utf-8") as f:
        ans2label = json.load(f)

    # Stored format is usually {answer_text: class_index}.
    return {int(label): answer for answer, label in ans2label.items()}


def load_models() -> None:
    """Load all models once at startup."""
    global DEVICE, MODEL_VIDEO_QA, VIS_PROCESSORS_VIDEO_QA, TXT_PROCESSORS_VIDEO_QA
    global MODEL_CAPTION, VIS_PROCESSORS_CAPTION, LABEL2ANSWER

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    print("Loading VideoQA model (ALPRO, MSRVTT)...")
    MODEL_VIDEO_QA, VIS_PROCESSORS_VIDEO_QA, TXT_PROCESSORS_VIDEO_QA = (
        load_model_and_preprocess(
            name="alpro_qa",
            model_type="msrvtt",
            is_eval=True,
            device=DEVICE,
        )
    )

    print("Loading image caption model for action timeline...")
    MODEL_CAPTION, VIS_PROCESSORS_CAPTION, _ = load_model_and_preprocess(
        name="blip_caption",
        model_type="base_coco",
        is_eval=True,
        device=DEVICE,
    )

    cache_root = resolve_writable_cache_root()
    print(f"Using answer cache: {cache_root}")

    print("Loading answer vocabulary...")
    LABEL2ANSWER = load_answer_vocab(cache_root=cache_root)

    print("Video demo models loaded successfully!")


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    total_seconds = max(0, int(seconds))
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def extract_keyframes(video_path: str, num_steps: int) -> List[Tuple[Image.Image, float]]:
    """Extract evenly spaced keyframes and their timestamps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open the uploaded video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if total_frames <= 0:
        cap.release()
        raise ValueError("The video has no readable frames.")

    steps = max(1, min(int(num_steps), total_frames))
    frame_indices = np.linspace(0, total_frames - 1, num=steps, dtype=int)

    keyframes: List[Tuple[Image.Image, float]] = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        timestamp = (float(frame_idx) / fps) if fps > 0 else 0.0
        keyframes.append((frame_pil, timestamp))

    cap.release()

    if not keyframes:
        raise ValueError("Could not extract keyframes from the video.")

    return keyframes


def answer_video_question(
    video_path: str,
    question: str,
    top_k: int,
    chat_history: List[Tuple[str, str]],
):
    """Answer question about a video with ALPRO VideoQA model."""
    if chat_history is None:
        chat_history = []

    clean_question = (question or "").strip()
    if not clean_question:
        return "", chat_history

    if not video_path:
        chat_history.append((clean_question, "Please upload a video first."))
        return "", chat_history

    try:
        processed_video = VIS_PROCESSORS_VIDEO_QA["eval"](video_path).unsqueeze(0).to(DEVICE)
        processed_question = TXT_PROCESSORS_VIDEO_QA["eval"](clean_question)

        # ALPRO QA expects an "answers" tensor even during inference.
        samples = {
            "video": processed_video,
            "text_input": [processed_question],
            "answers": torch.zeros(1, dtype=torch.long, device=DEVICE),
        }

        with torch.inference_mode():
            output = MODEL_VIDEO_QA.predict(samples)
            logits = output["predictions"]
            probs = torch.softmax(logits, dim=-1)

            k = max(1, min(int(top_k), probs.shape[-1]))
            top_probs, top_indices = torch.topk(probs, k=k, dim=-1)

        best_idx = int(top_indices[0, 0].item())
        best_conf = float(top_probs[0, 0].item())
        best_answer = LABEL2ANSWER.get(best_idx, f"class_{best_idx}")

        response_lines = [f"Answer: {best_answer}", f"Confidence: {best_conf:.2%}"]

        if k > 1:
            alternatives = []
            for rank in range(1, k):
                idx = int(top_indices[0, rank].item())
                conf = float(top_probs[0, rank].item())
                alt_answer = LABEL2ANSWER.get(idx, f"class_{idx}")
                alternatives.append(f"{rank}. {alt_answer} ({conf:.2%})")
            response_lines.append("Alternatives:\n" + "\n".join(alternatives))

        chat_history.append((clean_question, "\n".join(response_lines)))
        return "", chat_history

    except Exception as e:
        chat_history.append((clean_question, f"Error answering question: {str(e)}"))
        return "", chat_history


def generate_action_timeline(video_path: str, num_steps: int) -> str:
    """Generate a chronological action list using frame-wise captioning."""
    if not video_path:
        return "Please upload a video first."

    try:
        keyframes = extract_keyframes(video_path=video_path, num_steps=num_steps)

        timeline_items: List[Tuple[float, str]] = []
        previous_caption = None

        for frame_pil, timestamp in keyframes:
            processed_frame = VIS_PROCESSORS_CAPTION["eval"](frame_pil).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                caption = MODEL_CAPTION.generate(
                    {"image": processed_frame},
                    num_captions=1,
                )[0].strip()

            # Reduce repetitive consecutive lines in the final timeline.
            if caption and caption != previous_caption:
                timeline_items.append((timestamp, caption))
                previous_caption = caption

        if not timeline_items:
            return "Could not generate timeline events for this video."

        lines = ["Action timeline (estimated):"]
        for idx, (timestamp, caption) in enumerate(timeline_items, 1):
            lines.append(f"{idx}. [{format_timestamp(timestamp)}] {caption}")

        lines.append("")
        lines.append(
            "Tip: Increase the number of timeline steps for finer detail, "
            "or ask specific questions in the Video QA tab."
        )

        return "\n".join(lines)

    except Exception as e:
        return f"Error generating action timeline: {str(e)}"


def clear_chat():
    """Clear VideoQA chat history."""
    return [], ""


def create_interface():
    """Create Gradio UI."""
    with gr.Blocks(title="LAVIS Video Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # LAVIS Video Demo

            Upload a video and use either:
            - **Video QA**: ask questions about the video.
            - **Action Timeline**: get an ordered list of what happened over time.
            """
        )

        video_input = gr.Video(
            label="Upload Video",
            sources=["upload"],
            type="filepath",
            height=360,
        )

        with gr.Tabs():
            with gr.Tab("Video QA"):
                chatbot = gr.Chatbot(
                    label="Ask Questions About the Video",
                    height=350,
                )

                with gr.Row():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What is the person doing?",
                        scale=4,
                    )
                    ask_btn = gr.Button("Ask", variant="primary", scale=1)

                top_k = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Show Top-K Predictions",
                )

                clear_btn = gr.Button("Clear Chat")

                gr.Examples(
                    examples=[
                        "What is the main activity in this video?",
                        "Where is this happening?",
                        "What is the person holding?",
                        "Is the person indoors or outdoors?",
                        "What sport is shown?",
                    ],
                    inputs=question_input,
                    label="Example Questions",
                )

            with gr.Tab("Action Timeline"):
                steps_slider = gr.Slider(
                    minimum=3,
                    maximum=12,
                    value=6,
                    step=1,
                    label="Number of Timeline Steps",
                )
                timeline_btn = gr.Button("Generate Action Timeline", variant="primary")
                timeline_output = gr.Textbox(
                    label="Ordered Actions",
                    lines=14,
                    interactive=False,
                )

        gr.Markdown(
            """
            ---
            ⚠️ **Important Disclaimer:** This AI system may produce inaccurate, biased, or misleading outputs.
            Verify important information independently and treat responses as best-effort suggestions.
            """
        )

        ask_btn.click(
            fn=answer_video_question,
            inputs=[video_input, question_input, top_k, chatbot],
            outputs=[question_input, chatbot],
        )

        question_input.submit(
            fn=answer_video_question,
            inputs=[video_input, question_input, top_k, chatbot],
            outputs=[question_input, chatbot],
        )

        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, question_input],
        )

        timeline_btn.click(
            fn=generate_action_timeline,
            inputs=[video_input, steps_slider],
            outputs=[timeline_output],
        )

    return demo


def main() -> None:
    print("=" * 60)
    print("LAVIS Video Web Demo")
    print("=" * 60)

    load_models()
    demo = create_interface()

    print("\nStarting web interface...")
    print("Open http://localhost:7861 in your browser")
    print("Press Ctrl+C to stop\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
