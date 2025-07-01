#!/usr/bin/env python3
"""
WorldVLA Quick Test Script
用于快速测试WorldVLA模型的简化脚本
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor
import requests
from PIL import Image
import os

def download_model_if_needed(model_name):
    """下载模型（如果需要）"""
    print(f"正在加载模型: {model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        return model, processor
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None, None

def test_action_generation():
    """测试动作生成功能"""
    print("\n=== 测试动作生成模型 ===")
    
    # 使用256分辨率的Goal模型作为示例
    model_name = "jcenaa/WorldVLA-ActionModel-LIBERO-Goal-256"
    
    model, processor = download_model_if_needed(model_name)
    if model is None:
        print("模型加载失败，请检查网络连接或模型名称")
        return
    
    # 准备测试输入
    instruction = "Pick up the bowl and place it on the plate"
    
    # 这里使用示例图像（实际使用时需要真实的机器人观察图像）
    # 创建两个虚拟图像作为历史观察
    img1 = Image.new('RGB', (256, 256), color='red')
    img2 = Image.new('RGB', (256, 256), color='blue')
    
    print(f"指令: {instruction}")
    print("正在生成动作序列...")
    
    try:
        # 准备输入
        inputs = processor(
            text=f"What action should the robot take to {instruction}?",
            images=[img1, img2],
            return_tensors="pt"
        ).to(model.device)
        
        # 生成动作
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
        
        # 解码输出
        generated_text = processor.decode(outputs[0], skip_special_tokens=False)
        print(f"生成的动作序列: {generated_text}")
        
    except Exception as e:
        print(f"动作生成失败: {e}")

def main():
    print("WorldVLA 快速测试脚本")
    print("=" * 50)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
    else:
        print("警告: 未检测到GPU，将使用CPU（速度较慢）")
    
    # 运行测试
    test_action_generation()
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()