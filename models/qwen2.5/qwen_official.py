import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    
# 将MyQwen注册到AutoModel系统
# AutoConfig.register("my_qwen", MyQwenConfig)
    # AutoModel.register(Qwen2Config, QWen2_5)
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    from transformers import AutoModel
    torch.manual_seed(42)


    # 加载官方模型的权重键名
    official_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True, device_map='auto', torch_dtype='auto')
    official_keys = set(official_model.state_dict().keys())
    # print({k:v.shape for k,v in official_model.state_dict().items()})

    # 获取您的模型权重键名
    # your_keys = set(QWen2_5().state_dict().keys())
    

    # model = QWen2_5.from_pretrained(
    #     model_name,
    #     torch_dtype="auto",
    #     device_map="auto",
    #     trust_remote_code=False
    # )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f'Loaded model on device {official_model.device} with type {official_model.dtype}')

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(official_model.device)

    generated_ids = official_model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    # print(generated_ids.shape)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f'Prompt:{prompt}\nResponse:{response}')