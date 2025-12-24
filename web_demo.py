#!/usr/bin/env python
"""
Interactive LAVIS Demo with Gradio Web Interface

Usage:
    python web_demo.py

This will open a web browser with an interactive interface where you can:
1. Upload an image
2. Get automatic captions
3. Ask questions about the image in real-time
"""

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import gradio as gr

# Global variables to store loaded models
device = None
model_caption = None
model_vqa = None
vis_processors_caption = None
vis_processors_vqa = None
txt_processors_vqa = None


def load_models():
    """Load BLIP models once at startup."""
    global device, model_caption, model_vqa
    global vis_processors_caption, vis_processors_vqa, txt_processors_vqa
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading captioning model...")
    model_caption, vis_processors_caption, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=device
    )
    
    print("Loading VQA model...")
    model_vqa, vis_processors_vqa, txt_processors_vqa = load_model_and_preprocess(
        name="blip_vqa", model_type="vqav2", is_eval=True, device=device
    )
    
    print("Models loaded successfully!")


def generate_caption(image):
    """Generate caption for uploaded image."""
    if image is None:
        return "Please upload an image first."
    
    try:
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")
        
        # Process image
        processed_image = vis_processors_caption["eval"](image).unsqueeze(0).to(device)
        
        # Generate captions
        captions = model_caption.generate({"image": processed_image}, num_captions=3)
        
        result = "Generated Captions:\n"
        for i, cap in enumerate(captions, 1):
            result += f"{i}. {cap}\n"
        
        return result
    except Exception as e:
        return f"Error generating caption: {str(e)}"


def answer_question(image, question, chat_history):
    """Answer a question about the image."""
    if image is None:
        chat_history.append((question, "Please upload an image first."))
        return "", chat_history
    
    if not question.strip():
        return "", chat_history
    
    try:
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")
        
        # Process image and question
        processed_image = vis_processors_vqa["eval"](image).unsqueeze(0).to(device)
        processed_question = txt_processors_vqa["eval"](question)
        
        # Get answer
        answer = model_vqa.predict_answers(
            samples={"image": processed_image, "text_input": processed_question},
            inference_method="generate",
        )
        
        chat_history.append((question, answer[0]))
        return "", chat_history
    
    except Exception as e:
        chat_history.append((question, f"Error: {str(e)}"))
        return "", chat_history


def clear_chat():
    """Clear chat history."""
    return [], ""


def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="LAVIS Interactive Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 
            
            Upload an image and ask questions
            
            **Features:**
            - Automatic image captioning
            - Visual Question Answering (VQA)
            - Chat-style interaction
            """
        )
        
        with gr.Row():
            # Left column: Image upload
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=400
                )
                caption_btn = gr.Button("🔍 Generate Caption", variant="primary")
                caption_output = gr.Textbox(
                    label="Image Captions",
                    lines=4,
                    interactive=False
                )
            
            # Right column: Chat interface
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Ask Questions About the Image",
                    height=350
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Type your question here... (e.g., What color is the object?)",
                        scale=4
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=1)
                
                clear_btn = gr.Button("🗑️ Clear Chat")
        
        # Example questions
        gr.Markdown("### 💡 Example Questions")
        gr.Examples(
            examples=[
                "What is in the image?",
                "What color is it?",
                "How many objects are there?",
                "Is this indoors or outdoors?",
                "What is the main subject?",
                "What is happening in this image?",
            ],
            inputs=question_input,
            label=""
        )
        
        # Ethical disclaimer
        gr.Markdown(
            """
            ---
            ⚠️ **Important Disclaimer:** This AI system may produce inaccurate, biased, or misleading outputs. 
            Do not rely on AI-generated content for critical decisions. Always verify information independently 
            and use your own judgment. AI responses should be treated as suggestions, not facts.
            """
        )
        
        # Event handlers
        caption_btn.click(
            fn=generate_caption,
            inputs=[image_input],
            outputs=[caption_output]
        )
        
        submit_btn.click(
            fn=answer_question,
            inputs=[image_input, question_input, chatbot],
            outputs=[question_input, chatbot]
        )
        
        question_input.submit(
            fn=answer_question,
            inputs=[image_input, question_input, chatbot],
            outputs=[question_input, chatbot]
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, question_input]
        )
        
        # Auto-generate caption when image is uploaded
        image_input.change(
            fn=generate_caption,
            inputs=[image_input],
            outputs=[caption_output]
        )
    
    return demo


def main():
    print("=" * 60)
    print("LAVIS Interactive Web Demo")
    print("=" * 60)
    
    # Load models
    load_models()
    
    # Create and launch interface
    demo = create_interface()
    
    print("\nStarting web interface...")
    print("Open http://localhost:7860 in your browser")
    print("Press Ctrl+C to stop\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create a public link
        inbrowser=True  # Automatically open browser
    )


if __name__ == "__main__":
    main()
