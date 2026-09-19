import os
import json
import time
from typing import List, Dict
import dashscope
from dashscope import Generation, MultiModalConversation
from dashscope.api_entities.dashscope_response import Role

dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

class QwenGenerator:
    def __init__(self, api_key: str = None, model_name: str = None):
        # 优先从外部传入，否则获取环境变量
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model_name = model_name
        if not self.api_key:
            raise ValueError("DashScope API Key 缺失，请设置环境变量 DASHSCOPE_API_KEY")

    def _call_qwen(self, role: str, prompt: str) -> str:
        """调用通义千问模型"""
        messages = [
            {'role': Role.SYSTEM, 'content': role},
            {'role': Role.USER, 'content': prompt}
        ]

        def _parse_response(response) -> str:
            if response.status_code == 200:
                raw_output = response.output.choices[0].message.content

                # 如果返回结果是类似于 [{'text': '...'}] 的列表结构，进行自动解包
                if isinstance(raw_output, list) and len(raw_output) > 0:
                    if isinstance(raw_output[0], dict) and 'text' in raw_output[0]:
                        return str(raw_output[0]['text'])

                # 如果是最直接的纯文本字符串，直接返回
                if raw_output is not None:
                    return str(raw_output)

            print(f"API 调用返回错误码: {response.code} - {response.message}")
            return ""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                try:
                    response = MultiModalConversation.call(
                        model=self.model_name,
                        messages=messages,
                        api_key=self.api_key,
                        result_format='message'
                    )
                    result = _parse_response(response)
                    if result:
                        return result
                except Exception as multimodal_error:
                    print(f"MultiModalConversation 调用失败，改用 Generation: {str(multimodal_error)}")

                response = Generation.call(
                    model=self.model_name,
                    messages=messages,
                    api_key=self.api_key,
                    result_format='message'
                )
                result = _parse_response(response)
                if result:
                    return result
            except Exception as e:
                print(f"调用网络异常: {str(e)}")
            time.sleep(2 ** attempt)
        return "[]"
