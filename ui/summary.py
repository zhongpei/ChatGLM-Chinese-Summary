import re

import gradio as gr
from typing import List
from models import models
from loguru import logger
import re

PROMPT_TEMPLATE = """\
使用中文{query_str}:
{context_str}
"""


def get_text_lines(input_txt: str) -> List[str]:
    lines = input_txt.splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    return lines


stop_chars_set = {
    '.', '!', '?', '。', '！', '？', '…', ';', '；', ':', '：',
    '”', '’', '）', '】', '》', '」', '』', '〕', '〉',
    '》', '〗', '〞', '〟', '»', '"', "'", ')', ']', '}'
}


def split_in_line(input_txt: str, limit_length: int) -> List[str]:
    new_text = ''
    contents = []
    outputs = []
    for text in input_txt:
        new_text += text
        if text in stop_chars_set:
            contents.append(new_text)
            # logger.debug(f"{new_text}")
            new_text = ''
    # logger.debug(f"{input_txt[-1]} {input_txt[-1] not in stop_chars_set} {new_text}")
    if input_txt[-1] not in stop_chars_set:
        contents.append(new_text)

    text = ""
    text_length = 0
    for idx, content in enumerate(contents):
        text += content
        text_length += len(content)
        if text_length >= limit_length:
            outputs.append(text)
            text = ""
            text_length = 0
    if text_length < limit_length:
        outputs.append(text)
    return outputs


def get_text_limit_length(input_txt: str, max_length: int = 2048) -> List[str]:
    lines = get_text_lines(input_txt)
    output: List[str] = []
    for line in lines:
        if len(line) <= max_length:
            output.append(line)
        else:
            text_lines = split_in_line(line, max_length)
            logger.debug(f"split in line: {len(text_lines)}")
            # logger.debug(f"{line} ==> {text_lines}")
            output.extend(text_lines)
    return output


def split_input_text(input_txt, strip_input_lines=0, max_length=2048):
    if strip_input_lines > 0:
        pattern = r'[\r\n]{' + str(strip_input_lines) + r',}'
        re.compile(pattern=pattern)
        logger.debug(f"strip input txt: {pattern}")
        input_txt = re.sub(pattern, '', input_txt)
    lines = get_text_limit_length(input_txt, max_length)
    logger.debug(f"split input txt: {len(lines)}")
    return "\n\n\n".join(lines)


def gen_keyword_summary(input_txt, keyword_prompt, summary_prompt, max_length=2048):
    lines = input_txt.split("\n\n\n")
    keywords_output = []
    for line in lines:
        keywords = models.llm_model.generate_answer(
            keyword_prompt,
            line,
            history=None,
            max_length=max_length,
            prompt_template=PROMPT_TEMPLATE
        )[0]
        logger.debug(f"text len: {len(line)} ==> {keywords}")
        keywords_output.extend(keywords.split())
    keywords_output = [keyword.strip() for keyword in keywords_output if keyword.strip() != ""]
    keywords_output = list(set(keywords_output))
    return f"保留关键词:{' '.join(keywords_output)},{summary_prompt}"


def gen_summary(input_txt, summary_prompt, max_length=2048):
    lines = input_txt.split("\n\n\n")
    output_summary = []
    summary = ""
    for idx, line in enumerate(lines):
        if idx == 1:
            summary = models.llm_model.generate_answer(
                summary_prompt,
                line,
                history=None,
                max_length=max_length,
                prompt_template=PROMPT_TEMPLATE
            )[0]
            logger.debug(f"text len: {len(line)} ==> {summary}")
        else:
            summary = models.llm_model.generate_answer(
                summary_prompt,
                f"{summary}{line}",
                history=None,
                max_length=max_length,
                prompt_template=PROMPT_TEMPLATE
            )[0]
            logger.debug(f"summary: {len(summary)} + text: {len(line)}  ==> {summary}")
        output_summary.append(summary)

    return "\n\n\n".join(output_summary)


def summary_ui():
    with gr.Row():
        with gr.Column(scale=1):
            line_max_length = gr.Slider(minimum=512, maximum=4096, step=1, value=1024, label="每行最大长度")
            strip_input_lines = gr.Slider(
                label="去除输入文本连续的空行(0:不除去)",
                minimum=1,
                maximum=10,
                step=1,
                value=0
            )
        with gr.Column(scale=4):
            keyword_prompt = gr.Textbox(
                lines=1,
                label="抽取关键词",
                value="抽取以下内容的人物和地点:",
                placeholder="请输入抽取关键词的Prompt"
            )
            summary_prompt = gr.Textbox(
                lines=2,
                label="生成摘要",
                value="生成以下内容的摘要:",
                placeholder="请输入生成摘要的Prompt"
            )
    keyword_summary_prompt = gr.Textbox(lines=4, label="关键词+摘要", placeholder="请输入关键词+摘要的Prompt")

    with gr.Row():
        input_text = gr.Textbox(lines=20, max_lines=60, label="输入文本", placeholder="请输入文本")
        split_text = gr.Textbox(lines=20, max_lines=60, label="分段文本", placeholder="请输入分段文本")
        summary = gr.Textbox(lines=20, max_lines=60, label="生成摘要", placeholder="请输入生成摘要的Prompt")

    with gr.Row():
        btn_split = gr.Button("分段")
        btn_keyword = gr.Button("提取关键词")
        btn_summary = gr.Button("生成摘要")

    btn_split.click(
        split_input_text,
        inputs=[input_text, strip_input_lines, line_max_length],
        outputs=[split_text]
    )

    btn_summary.click(
        gen_summary,
        inputs=[split_text, keyword_summary_prompt, line_max_length],
        outputs=[summary]
    )

    btn_keyword.click(
        gen_keyword_summary,
        inputs=[split_text, keyword_prompt, summary_prompt, line_max_length],
        outputs=[keyword_summary_prompt]
    )
