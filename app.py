import gradio as gr
from ui import chat
from ui import summary
from models import llm_model_dict, llm_model_dict_list, \
    embedding_model_dict_list
from models import models

block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """
# 🎉ChatGLM-Chinese-Summary WebUI🎉
Link in: [https://github.com/zhongpei/ChatGLM-Chinese-Summary](https://github.com/zhongpei/ChatGLM-Chinese-Summary) 
"""

with gr.Blocks(css=block_css) as demo:
    gr.Markdown(webui_title)
    with gr.Tab("load model"):
        llm_model = gr.Radio(llm_model_dict_list,
                             label="LLM 模型",
                             value=list(llm_model_dict.keys())[0],
                             interactive=True)
        llm_lora = gr.Textbox(label="lora path", value="")
        embedding_model = gr.Radio(embedding_model_dict_list,
                                   label="Embedding 模型",
                                   value=embedding_model_dict_list[0],
                                   interactive=True)

        result = gr.Label("模型未加载")

        load_model_button = gr.Button(
            "重新加载模型" if models.is_active() else "加载模型"
        )
        load_model_button.click(
            models.init_model,
            show_progress=True,
            inputs=[llm_model, llm_lora, embedding_model],
            outputs=result
        )

    with gr.Tab("chat"):
        chat.chat_ui(embedding_model)

    with gr.Tab("Summary"):
        summary.summary_ui()

demo.queue(concurrency_count=3).launch(
    server_name='0.0.0.0', share=False, inbrowser=False
)
