+++
date = '2025-12-04'
draft = true
title = 'qwen3-vl调用api'
subtitle = "学习如何调用qwen3-vl的api"
description = "大创加油"
author = 'BruceZhang'
categories = ["Innovation Project"]
tags = ["qwen"]
+++

参考资料 [通义千问API参考-大模型服务平台百炼(Model Studio)-阿里云帮助中心](https://help.aliyun.com/zh/model-studio/qwen-api-reference?spm=a2c4g.11186623.0.0.2bb23748oUXxUV#a749bab1d2dqq)

## 1 消息+基本调用

```python
messages = [
    {
        "role": "user" 或 "assistant" 或 "system",
        "content": [
            {"image": "图片URL 或 file://本地绝对路径"},
            {"image": "图片URL 或 file://本地绝对路径"},
            ...若干张图片
            {"text": "你的文字指令"}
        ]
    }
]
response = dashscope.MultiModalConversation.call(
    api_key = os.getenv('DASHSCOPE_API_KEY'),
    model = 'qwen3-vl-plus',  # 也可以是'qwen3-vl-flash'
    messages = messages
)
print(response.output.choices[0].message.content[0]["text"])
```

在 messages 里面`role`选择：

* "system"，表示系统消息，用于设定大模型的角色、语气、任务目标或约束条件等。一般放在`messages`数组的第一位。

* "assistant"，表示模型对用户消息的回复。
  
  * 有属性 **partial** `*boolean*` （可选）
    
    是否开启前缀续写。相关文档：[前缀续写](https://help.aliyun.com/zh/model-studio/partial-mode)。

## 2 思考模式

```python
import os
import dashscope
from dashscope import MultiModalConversation

# 若使用新加坡地域的模型，请取消下列注释
# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
enable_thinking=True
messages = [
    {
        "role": "user",
        "content": [
            {"image": "https://img.alicdn.com/imgextra/i1/O1CN01gDEY8M1W114Hi3XcN_!!6000000002727-0-tps-1024-406.jpg"},
            {"text": "解答这道题？"}
        ]
    }
]

response = MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model="qwen3-vl-plus",
    messages=messages,
    stream=True,
    # enable_thinking 参数开启思考过程
    # qwen3-vl-plus、 qwen3-vl-flash可通过enable_thinking开启或关闭思考、对于qwen3-vl-235b-a22b-thinking等带thinking后缀的模型，enable_thinking仅支持设置为开启，对其他Qwen-VL模型均不适用
    enable_thinking=enable_thinking,
    # thinking_budget 参数设置最大推理过程 Token 数
    thinking_budget=81920,

)

# 定义完整思考过程
reasoning_content = ""
# 定义完整回复
answer_content = ""
# 判断是否结束思考过程并开始回复
is_answering = False
if enable_thinking:
    print("=" * 20 + "思考过程" + "=" * 20)

for chunk in response:
    # 如果思考过程与回复皆为空，则忽略
    message = chunk.output.choices[0].message
    reasoning_content_chunk = message.get("reasoning_content", None)
    if (chunk.output.choices[0].message.content == [] and
        reasoning_content_chunk == ""):
        pass
    else:
        # 如果当前为思考过程
        if reasoning_content_chunk != None and chunk.output.choices[0].message.content == []:
            print(chunk.output.choices[0].message.reasoning_content, end="")
            reasoning_content += chunk.output.choices[0].message.reasoning_content
        # 如果当前为回复
        elif chunk.output.choices[0].message.content != []:
            if not is_answering:
                print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                is_answering = True
            print(chunk.output.choices[0].message.content[0]["text"], end="")
            answer_content += chunk.output.choices[0].message.content[0]["text"]

# 如果您需要打印完整思考过程与完整回复，请将以下代码解除注释后运行
# print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
# print(f"{reasoning_content}")
# print("=" * 20 + "完整回复" + "=" * 20 + "\n")
# print(f"{answer_content}")
```

#### 2.1 请求体的更多参数

**请求体**（`call` 函数）还有以下参数可选

- **temperature** `*float*` （可选）
  
  采样温度，控制模型生成文本的多样性。
  
  temperature越高，生成的文本更多样，反之，生成的文本更确定。
  
  取值范围： [0, 2)
  
  qwen3-vl 非思考模式默认为0.7，思考模式默认为0.8

- **top_p** `*float*` （可选）
  
  核采样的概率阈值，控制模型生成文本的多样性。
  
  top_p越高，生成的文本更多样。反之，生成的文本更确定。
  
  取值范围：（0,1.0]。

- **enable_code_interpreter** `*boolean*` （可选）默认值为 `false`
  
  是否开启代码解释器功能。仅适用于思考模式下的 qwen3-max-preview。相关文档：[代码解释器](https://help.aliyun.com/zh/model-studio/qwen-code-interpreter?spm=a2c4g.11186623.0.0.25933748ak3IAv)
  
  可选值：
  
  - `true`：开启
  
  - `false`：不开启

- **vl_high_resolution_images** `*boolean*` （可选）默认值为`false`
  
  是否将输入图像的像素上限提升至 16384 Token 对应的像素值。相关文档：[处理高分辨率图像](https://help.aliyun.com/zh/model-studio/vision#e7e2db755f9h7)。
  
  - `vl_high_resolution_images：true`，使用固定分辨率策略，忽略 `max_pixels` 设置，超过此分辨率时会将图像总像素缩小至此上限内。
  
  - `vl_high_resolution_images`为`false`，实际分辨率由 `max_pixels` 与默认上限共同决定，取二者计算结果的最大值。超过此像素上限时会将图像缩小至此上限内。

- **seed** `*integer*` （可选）
  
  随机数种子。用于确保在相同输入和参数下生成结果可复现。若调用时传入相同的 `seed` 且其他参数不变，模型将尽可能返回相同结果。

- **stream** `*boolean*` （可选） 默认值为`false`
  
  是否流式输出回复。参数值：
  
  - false：模型生成完所有内容后一次性返回结果。
  
  - true：边生成边输出，即每生成一部分内容就立即输出一个片段（chunk）。

- **incremental_output** `*boolean*` （可选）默认为`false`（Qwen3-Max、Qwen3-VL、[Qwen3 开源版](https://help.aliyun.com/zh/model-studio/models#9d516d17965af)、[QwQ](https://help.aliyun.com/zh/model-studio/deep-thinking) 、[QVQ](https://help.aliyun.com/zh/model-studio/visual-reasoning)模型默认值为 `true`）
  
  在流式输出模式下是否开启增量输出。推荐您优先设置为`true`。
  
  参数值：
  
  - false：每次输出为当前已经生成的整个序列，最后一次输出为生成的完整结果。
    
    ```plaintext
    I
    I like
    I like apple
    I like apple.
    ```
  
  - true（推荐）：增量输出，即后续输出内容不包含已输出的内容。您需要实时地逐个读取这些片段以获得完整的结果。
    
    ```plaintext
    I
    like
    apple
    .
    ```

- **response_format** `*object*` （可选） 默认值为`{"type": "text"}`
  
  返回内容的格式。可选值：`{"type": "text"}`或`{"type": "json_object"}`。设置为`{"type": "json_object"}`时会输出标准格式的JSON字符串。相关文档：[结构化输出](https://help.aliyun.com/zh/model-studio/json-mode)。
  
  > 如果指定为`{"type": "json_object"}`，需同时在**提示词**中指引模型输出JSON格式，如：“请按照json格式输出”，否则会报错。

- **result_format** `*string*` （可选） 默认为`text`（Qwen3-Max、Qwen3-VL、[QwQ](https://help.aliyun.com/zh/model-studio/deep-thinking) 模型、Qwen3 开源模型（除了qwen3-next-80b-a3b-instruct）与 Qwen-Long 模型默认值为 message）
  
  返回数据的格式。推荐您优先设置为`message`，可以更方便地进行[多轮对话](https://help.aliyun.com/zh/model-studio/multi-round-conversation)。
  
  > qwen3-vl 和 qwen3-max 默认为 `message`

- **enable_search** `*boolean*` （可选） 默认值为`false`
  
  模型在生成文本时是否使用互联网搜索结果进行参考。取值如下：
  
  - true：启用互联网搜索，模型会将搜索结果作为文本生成过程中的参考信息，但模型会基于其内部逻辑判断是否使用互联网搜索结果。
    
    > 若开启后未联网搜索，可优化提示词，或设置`search_options`中的`forced_search`参数开启强制搜索。
  
  - false：关闭互联网搜索。

- **search_options** `*object*` （可选）
  
  联网搜索的策略。仅当`enable_search`为`true`时生效。详情参见[联网搜索](https://help.aliyun.com/zh/model-studio/web-search#cbddf5b28bug8)。

- **enable_search_extension** `*boolean*`（可选）默认值为`false`
  
  是否开启特定领域增强。参数值：
  
  - `true`
    
    开启。
  
  - `false`（默认值）
    
    不开启。

## 3 响应体结构

#### 3.1 顶层结构如下

```json
{
  "status_code": 200,
  "request_id": "...",    // 这次请求在服务端的唯一ID
  "code": "",             // 当请求失败时，这里会给错误码和错误信息
  "message": "",          // 同上
  "output": { ... },      // 真正的模型输出
  "usage": { ... }        // 本次调用的计费, token 使用情况
}
```

#### 3.2 重点关注`output`结构

`qwen3-vl`的`response`为生成器，每次迭代产生一个上面的json格式。某次迭代输出结果的`output`如下 (**答复阶段**)

```json
"output": { // 4个属性
  "text": null,
  "finish_reason": null,
  "choices": [ // 是一个数组，但一般只用choices[0]即可
    { // 2个属性
      "finish_reason": "null",
      "message": { // 2个属性
        "role": "assistant",
        "content": [ 
          { "text": " cm" }
        ]
      }
    }
  ],
  "audio": null
}
```

关注到`output.choices[0].message.content`即为输出内容，它是一个`list`，每个元素是一个字典表示片段，有文本片段`{ "text": "..." }`，可能还会有图片片段`{ "image": "..." }`，要取出内容，使用`output.choices[0].message.content[0]["text"]`即可。

对比**推理阶段**

```json
{
  "status_code": 200,
  "request_id": "de0142e7-9bbe-423b-8bb1-10f55f33674f",
  "code": "",
  "message": "",
  "output": {
    "text": null,
    "finish_reason": null,
    "choices": [
      {
        "finish_reason": "null",
        "message": {
          "role": "assistant",
          "content": [],              // 关注这里
          "reasoning_content": "高"   // 关注这里
        }
      }
    ],
    "audio": null
  },
  "usage": {
    "input_tokens": 433,
    "output_tokens": 59,
    "characters": 0,
    "input_tokens_details": {
      "image_tokens": 418,
      "text_tokens": 15
    },
    "total_tokens": 492,
    "output_tokens_details": {
      "reasoning_tokens": 59,
      "text_tokens": 0
    },
    "image_tokens": 418,
    "cached_tokens": 128
  }
}
```

下面区分纯文本模型`qwen-plus`的`output`如下

```json
"output": {
  "text": null,
  "finish_reason": null,
  "choices": [
    {
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "*突然发出尖锐的笑声，身体不自觉地抽搐* \n\n..."
      }
    }
  ]
}
```

由于是纯文本模型，`response`直接就是一个json样式，不再是生成器。并且输出的`content`就是一个字符串表示回答。调用`output.choices[0].message.content`即可
