import os
import json
import time
from typing import Optional, List
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

'''
高级工具定义与参数验证
'''
def example_1_advanced_tools():
    from langchain_core.tools  import tool, StructuredTool
    from pydantic import BaseModel, Field, validator
    from pydantic.functional_validators import  field_validator

    # 方式1：简单 @tool 装饰器 + 手动校验
    @tool
    def simple_calculator(expression: str) -> str:
        # 描述
        """
        执行简单的数学计算
        Args:
            expression: 数学表达式，例如 '2 + 3 * 4'
        """
        try:
            # 参数校验
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return "错误：表达式包含不允许的字符"
            result = eval(expression)
            return f"计算结果：{expression} = {result}"
        except Exception as e:
            return f"计算错误：{str(e)}"

    # 方式2：简单 @tool 装饰器 + pydantic 参数模型 + 手动校验
    class WeatherInput(BaseModel):
        """
        天气查询参数
        """
        city: str = Field(description="城市名称")
        unit: str = Field(default="celsius", description="温度单位: celsius 或 fahrenheit")

        @field_validator("unit")
        def validate_unit(cls, v):
            if v not in ["celsius", "fahrenheit"]:
                raise ValueError("unit 必须是 'celsius' 或 'fahrenheit'")
            return

    @tool(args_schema=WeatherInput)
    def get_weather(city: str, unit: str = "celsius") -> str:
        """获取指定城市的天气信息。"""
        # 模拟天气数据
        weather_data = {
            "北京": {"temp_c": 15, "condition": "晴"},
            "上海": {"temp_c": 20, "condition": "多云"},
            "深圳": {"temp_c": 25, "condition": "阴"},
        }

        data = weather_data.get(city, {"temp_c": 18, "condition": "未知"})
        temp = data["temp_c"] if unit == "celsius" else data["temp_c"] * 9 / 5 + 32
        unit_symbol = "°C" if unit == "celsius" else "°F"

        return f"{city}天气: {data['condition']}, 温度: {temp:.1f}{unit_symbol}"

    # 方式3：StructuredTool
    def translate_text(text: str, target_lang: str) -> str:
        """翻译文本（模拟）"""
        translations = {
            "en": f"[English] {text}",
            "ja": f"[日本語] {text}",
            "ko": f"[한국어] {text}",
        }
        return translations.get(target_lang, f"[{target_lang}] {text}")

    translate_tool = StructuredTool.from_function(
        func=translate_text,
        name="translate",
        description="将文本翻译成指定语言。支持: en, ja, ko",
    )

    # 测试工具
    print("\n📌 测试简单计算器:")
    print(simple_calculator.invoke({"expression": "2 + 3 * 4"}))
    print(simple_calculator.invoke({"expression": "(10 - 5) / 2"}))

    print("\n📌 测试天气查询:")
    print(get_weather.invoke({"city": "北京", "unit": "celsius"}))
    print(get_weather.invoke({"city": "上海", "unit": "fahrenheit"}))

    print("\n📌 测试翻译工具:")
    print(translate_tool.invoke({"text": "你好世界", "target_lang": "en"}))
    print(translate_tool.invoke({"text": "Hello World", "target_lang": "ja"}))

    print("\n✅ 工具信息:")
    for t in [simple_calculator, get_weather, translate_tool]:
        print(f"  - {t.name}: {t.description[:50]}...")


def main():
    """运行所有示例"""
    print("╔" + "═" * 58 + "╗")
    print("║" + " 模块 15: 工具与 Agent 进阶 ".center(56) + "║")
    print("╚" + "═" * 58 + "╝")

    # 运行示例
    example_1_advanced_tools()



if __name__ == "__main__":
    main()
