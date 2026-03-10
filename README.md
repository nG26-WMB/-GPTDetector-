# -GPTDetector-
基于GPTDetector的短视频生成文案识别方法研究
视频转文字内容基于https://gitee.com/ventl/videototext.git
GPTDetector开源参考来自：https://github.com/zejunwang1/GPTDetector.git
GPTZero开源参考来自：https://github.com/BurhanUlTayyab/GPTZero

对于分类器（GPTDetector）：RoBERTa分类器
加载预训练模型Hello-SimpleAI/chatgpt-detector-roberta-chinese，标签映射为{0:'Human',1:'ChatGPT'}，因此AI类索引设为1。对于长文本，实现基于字符数的分块函数，在换行符、句末标点处切分，确保每块不超过模型最大长度（512 token）。预测时取各块平均概率作为最终结果。

对于GPTZero的改进为：
在gptzero_style_detect函数中，先调用translate_chinese_to_english将文本翻译成英文，然后调用修改后的GPTZero模型获得困惑度（Perplexity per line）和突发性（Burstiness）。融合策略如下：
基于小样本调试的极低困惑度规则：当平均困惑度小于30时，直接判定为AI生成，概率设为0.95，忽略突发性。此规则针对极低困惑度的文本，避免因突发性较高而误判。
正常融合：否则使用加权Sigmoid融合：
pppl=11+e0.1×(ppl−60)
pburst=11+e0.05×(burst−80)
prob=0.4×pppl+0.6×pburst
参数基于小样本调试确定：困惑度中心点60，斜率0.1；突发性中心点80，斜率0.05；权重0.4/0.6。最终概率保留四位小数。
基于小样本调试确定的高突发性规则：当平均困惑度大于30且小于60时（对于严谨的学术内容），且突发性-困惑度大于50时，设置困惑度权重 w = 0.35（即突发性权重为0.65）；否则保持默认权重 w = 0.65。

对于整个自动化流程启动为：
第一个shell：
# 激活 AI 检测环境
conda activate gptdetector-py39

# 进入监控脚本所在目录
cd C:\Users\Administrator\Desktop\GPTDetector-main

# 运行监控脚本
python monitor_and_detect.py

第二个shell：
# 激活视频转文字环境
conda activate videototext-env

# 进入后端目录
cd C:\Users\Administrator\Desktop\videototext-main\backend

# 运行 Flask 后端
python main.py

第三个shell:
# 进入前端项目根目录（无需激活 conda 环境）
cd C:\Users\Administrator\Desktop\videototext-main

# 运行前端开发服务器
npm run dev

单独启动AI概率检测内容为：
conda activate gptdetector-py39
cd C:\Users\Administrator\Desktop\GPTDetector-main
python gradio_demo.py --model Hello-SimpleAI/chatgpt-detector-roberta-chinese --port 9000
