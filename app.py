import gradio as gr
from src.rag_pipeline import RAGPipeline

# 1. Initialize the pipeline globally 
# It will automatically find your vector_store/chroma folder
print("Initializing CrediTrust RAG Engine...")
rag_sys = RAGPipeline(k=5)
print("✅ Core logic successfully loaded.")

# 2. Define the execution function
def process_query(question):
    if not question.strip():
        return "Please enter a valid question.", "No sources queried."
    
    try:
        # Run the full RAG pipeline
        results = rag_sys.run(question=question)
        answer = results["answer"]
        
        # Format the retrieved sources into readable Markdown
        formatted_sources = "### Ground-Truth Context Sources Used:\n\n"
        for i, source in enumerate(results["sources"], 1):
            meta = source["metadata"]
            product = meta.get("Product", "Unknown")
            issue = meta.get("Issue", "General")
            complaint_id = meta.get("Complaint ID", "N/A")
            
            formatted_sources += f"**[{i}] Product:** {product} | **Issue:** {issue} (ID: {complaint_id})\n"
            formatted_sources += f"> {source['text']}\n\n"
            
        return answer, formatted_sources
        
    except Exception as e:
        return f"An error occurred: {str(e)}", "No sources available."

# 3. Design the layout using Gradio Blocks
with gr.Blocks(theme=gr.themes.Soft(), title="CrediTrust Analyst Portal") as demo:
    gr.Markdown("# 🔍 CrediTrust Financial Analytics Portal")
    gr.Markdown("Interact with the CFPB consumer complaint database using Retrieval-Augmented Generation.")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Operational Inquiry", 
                placeholder="What common options do customers have when disputing unauthorized credit card charges?",
                lines=3
            )
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear")
                
        with gr.Column(scale=3):
            output_answer = gr.Textbox(
                label="🤖 Analyst Response", 
                interactive=False,
                lines=6
            )
            
    with gr.Row():
        output_sources = gr.Markdown(label="Sources Verification")

    submit_btn.click(
        fn=process_query, 
        inputs=[input_text], 
        outputs=[output_answer, output_sources]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", ""), 
        inputs=[], 
        outputs=[input_text, output_answer, output_sources]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)