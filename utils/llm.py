from textgen import ChatGlmModel, LlamaModel
from loguru import logger


class LLM(object):
    def __init__(
            self,
            gen_model_type: str = "chatglm",
            gen_model_name_or_path: str = "THUDM/chatglm-6b-int4",
            lora_model_name_or_path: str = None,

    ):

        self.model_type = gen_model_type

        if gen_model_type == "chatglm":
            self.gen_model = ChatGlmModel(
                gen_model_type,
                gen_model_name_or_path,
                lora_name=lora_model_name_or_path,
            )
        elif gen_model_type == "llama":

            self.gen_model = LlamaModel(
                gen_model_type,
                gen_model_name_or_path,
                lora_name=lora_model_name_or_path,
            )

        else:
            raise ValueError('gen_model_type must be chatglm or llama.')
        self.history = None

    def generate_answer(self, query_str, context_str, history=None, max_length=1024, prompt_template=None):
        """Generate answer from query and context."""
        if self.model_type == "t5":
            response = self.gen_model(query_str, max_length=max_length, do_sample=True)[0]['generated_text']
            return response, history
        prompt = prompt_template.format(context_str=context_str, query_str=query_str)
        response, out_history = self.gen_model.chat(prompt, history, max_length=max_length)
        return response, out_history

    def chat(self, query_str, history=None, max_length=1024):
        if self.model_type == "t5":
            response = self.gen_model(query_str, max_length=max_length, do_sample=True)[0]['generated_text']
            logger.debug(response)
            return response, history

        response, out_history = self.gen_model.chat(query_str, history, max_length=max_length)
        return response, out_history
