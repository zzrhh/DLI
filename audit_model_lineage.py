from model_lists import (
    shadow_model_names,
    base_model_names, cache_dir, token
)

import os
import json
import random
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import jsonlines
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import copy
from peft import PeftModel
from openai import OpenAI
import argparse
import numpy as np
import string
from collections import Counter
import pandas as pd
from autogluon.tabular import TabularPredictor


client = OpenAI(api_key="")


def get_lora_target_modules(model_name):
    name = model_name.lower()
    if "gpt-neo" in name:
        return ["q_proj", "v_proj", "k_proj", "out_proj"]
    elif "gpt2" in name:
        return ["c_attn", "c_proj"]
    elif "llama" in name:
        return ["q_proj", "v_proj", "k_proj", "o_proj"]
    elif "opt" in name:
        return ["q_proj", "v_proj", "k_proj", "out_proj"]
    elif any(k in name for k in ["exaone", "mistral", "qwen", "starcoder", "instella", "apriel", "smollm"]):
        return ["q_proj", "v_proj", "k_proj", "o_proj"]
    elif "phi-4" in name:
        return ["qkv_proj", "o_proj"]
    elif "bloom" in name:
        return ["query_key_value", "dense"]
    elif "phi-2" in name:
        return ["q_proj", "v_proj", "k_proj"]

    return ["q_proj", "v_proj", "k_proj", "o_proj"]

def is_complete(text):
    return text and text[-1] in ['.', '。', '！', '!', '?', '？', '”', '"', "'"]

def truncate_incomplete(text):
    m = re.search(r'^(.*?[。！？!?\.])[^。！？!?\.]*$', text)
    return m.group(1).strip() if m else text.strip()

def load_dataset(dataset_path, offset=0):
    texts = []
    if dataset_path.find('llama3') != -1:
        with jsonlines.open(dataset_path) as reader:
            for obj in reader:
                target = obj.get('target', '')
                if target:
                    texts.append(target)
    else:
        with jsonlines.open(dataset_path) as reader:
            for obj in reader:
                prompt = obj.get('prompt', '')
                target = obj.get('target', '')
                if prompt and target:
                    texts.append(prompt + ' ' + target)
    return texts[offset:]


def train_distilled_model(
    model_name,
    distill_model_name,
    dataset_path,
    token,
    output_dir,
    cache_dir,
    skip_samples=0,
    num_train_epochs=3,
    max_length=512,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    bf16=True,
):
    
    
    train_texts = load_dataset(dataset_path, offset=skip_samples)
    print(f"{train_texts[0]}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )


    # for module, param in base_model.named_parameters():
    #     print(module)

    target_modules = get_lora_target_modules(model_name)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(base_model, lora_config).train().cuda()
    print("LoRA model initialized with trainable parameters:")
    model.print_trainable_parameters()

    data = []
    for item in train_texts:
        words = item.split()
        if len(words) < 6:
            continue
        midpoint = len(words) // 2
        input_text = " ".join(words[:midpoint])
        output_text = " ".join(words[midpoint:])
        data.append({"input": input_text, "output": output_text})

    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

    train_dataset = Dataset.from_dict({
        "input": [d["input"] for d in train_data],
        "output": [d["output"] for d in train_data]
    })
    val_dataset = Dataset.from_dict({
        "input": [d["input"] for d in val_data],
        "output": [d["output"] for d in val_data]
    })


    def preprocess_function(examples):
        inputs = examples["input"]
        targets = examples["output"]
        full_texts = [inp + tgt for inp, tgt in zip(inputs, targets)]

        tokenized = tokenizer(
            full_texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"]
        labels = input_ids.clone()

        for i, (inp, tgt) in enumerate(zip(inputs, targets)):
            prompt_ids = tokenizer(inp, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]
            prompt_len = len(prompt_ids)
            labels[i, :prompt_len] = -100  # mask prompt

        tokenized["labels"] = labels
        return {k: v.numpy() for k, v in tokenized.items()}

    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)


    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, distill_model_name),
        evaluation_strategy="epoch",
        save_strategy="no",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        bf16=bf16,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=None
    )

    
    trainer.train()


    save_path = os.path.join(output_dir, distill_model_name)
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"Distilled model saved to {save_path}")

def split_paragraph(text, ratio_range=(0.5, 0.5), max_length=128):
    words = text.strip().split()
    if len(words) > max_length:
        words = words[:max_length]
    split_point = int(len(words) * random.uniform(*ratio_range))
    input_text = ' '.join(words[:split_point])
    output_text = ' '.join(words[split_point:])
    return input_text


def get_dataset(dataset_name, input_file, skip_samples):
    contexts = []

    if dataset_name == "nq-simplified":
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                contexts.append(data['context'])

    elif dataset_name == "HealthCareMagic":
        with jsonlines.open(input_file) as reader:
            for obj in reader:
                text = obj.get('text', '').strip()  
                if text: 
                    text = text.replace('<human>:', '').replace('<bot>:', '')
                    contexts.append(text)

    elif dataset_name.find('legal') != -1:

        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                answer = data['Answer']
                qa = answer
                contexts.append(qa)

    elif dataset_name.find('Sciq') != -1:

        data = pd.read_csv(input_file)

        questions = data['question']

        supports = data['support']
        supports = supports.fillna("")
        answers = data['correct_answer']

        for i in range(len(questions)):
            contexts.append(questions[i] + supports[i] + answers[i])

    elif dataset_name.find('reddit') != -1:

        data = pd.read_csv(input_file)

        questions = data['selftext']

        answers = data['falcon_summary']

        for i in range(len(questions)):
            qa = questions[i] + answers[i]
            contexts.append(qa)

    elif dataset_name.find('finance') != -1:

        data = pd.read_parquet(input_file)

        data["instruction"] = data["instruction"].fillna("N/A")
        data["output"] = data["output"].fillna("N/A")
        
        contexts = [
            f"{row['instruction']} {row['output']}"
            for _, row in data.iterrows()
        ]

    contexts = contexts[skip_samples:]

    return contexts


def generate_training_dataset_local(
    teacher_model_name: str,
    token: str,
    input_file: str,
    output_file: str,
    dataset_name: str,
    n_samples: int = 1000,
    max_tokens: int = 128,
    cache_dir: str = "./hf_cache",
    skip_samples: int = 0
):
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, token=token, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        token=token,
        cache_dir=cache_dir
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id
    )

    contexts = get_dataset(dataset_name, input_file, skip_samples)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dataset = {}
    with jsonlines.open(output_file, mode='a') as writer:
        for i in tqdm(range(n_samples), desc="🦙 Generating with LLaMA 3"):
            try:
                prompt = split_paragraph(contexts[i])
                response = generator(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    eos_token_id=tokenizer.eos_token_id
                )[0]["generated_text"]

                if not is_complete(response):
                    response = truncate_incomplete(response)

                dataset[prompt] = response
                writer.write({'prompt': prompt, 'target': response})
            except Exception as e:
                print(f"⚠️ Error on sample {i}: {e}")
                
    print(f"Dataset saved to: {output_file}")


def generate_training_dataset_openai(
    model_name: str,
    input_file: str,
    output_file: str,
    dataset_name: str,
    n_samples: int = 1000,
    max_tokens: int = 128,
    skip_samples: int = 0
):
    contexts = get_dataset(dataset_name, input_file, skip_samples)

    dataset = {}
    with jsonlines.open(output_file, mode='a') as writer:
        for i in tqdm(range(n_samples), desc="🧠 Generating with GPT-4o-mini"):
            prompt = split_paragraph(contexts[i])
            messages = [
            {"role": "system", "content": "You are a helpful AI language model. Continue the text in the same style."},
            {"role": "user", "content": f"Continue this:\n\n{prompt}"}
            ]
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    stream=False,
                    messages=messages,
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=max_tokens
                )
                output = completion.choices[0].message.content

                dataset[prompt] = output
                writer.write({'prompt': prompt, 'target': output})
            except Exception as e:
                print(f"⚠️ Error on sample {i}: {e}")


    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Dataset saved to: {output_file}")

def generate_token(teacher_response, data_store_path, model_id, mask_token_file, k, cache_dir, device):

    os.makedirs(data_store_path, exist_ok=True)

    prompts = []
    targets = []
    with open(teacher_response, "r", encoding="utf-8") as f:
        responses = json.load(f)

    for prompt, target in responses.items():
        prompts.append(prompt)
        targets.append(target)

    # prompts = prompts[620:]
    # targets = targets[620:]

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        trust_remote_code=True
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir
    )

    assert len(prompts) == len(targets)

    all_tokens = []
    device = model.device
    
    datasets = {}

    for prompt, target in tqdm(zip(prompts, targets), total=len(prompts)):

        if not target or not target.strip() or not prompt or not prompt.strip():
            print(f"Skipping empty target for prompt: {prompt}")
            continue

        target_tokens = tokenizer(
            target,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=128,
            truncation=True,
            truncation_strategy="only_first"
        ).to(device)

        target_ids = target_tokens["input_ids"]

        if target_ids.shape[1] == 0:
            print(f"Skipping empty target_ids for prompt: {prompt}")
            continue

        generated_text = prompt
        prob = []
        tokens = []
        for i in range(len(target_ids[0])):
            inputs = tokenizer(generated_text, return_tensors="pt", max_length=128, truncation=True, truncation_strategy="only_first", add_special_tokens=False).to(device)
            input_ids = inputs['input_ids']

            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits

            logits = logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token_id = target_ids[0][i]

            next_token = tokenizer.decode([next_token_id])
            generated_text += next_token

            token_prob = probs[next_token_id].item()
            prob.append(token_prob)
            tokens.append(next_token)
        
        n = len(prob)
        top_k = n // k

        if top_k < 3:
            top_k = n

        indices = np.argsort(prob)[:top_k]

        indices = np.sort(indices)

        mask_tokens = [tokens[i] for i in indices]

        all_tokens.append(mask_tokens)

    with jsonlines.open(os.path.join(data_store_path, mask_token_file), mode='w') as writer:
        for i in range(len(prompts)):
            writer.write({'prompt': prompts[i], 'token': all_tokens[i]})
    
    print(f'mask tokens saved to {mask_token_file}')
    

def generate_model_style_tokens(
    teacher_model_name: str,
    prompts: str,
    dataset_name: str,
    length: int,
    token: str,
    cache_dir: str,
    device: torch.device,
    device_idx: int,
    output_dir: str,
    k: int,
    compare_model: str,
    teacher_response_file: str,
    output_file: str,
    skip_samples: int = 0,
):

    contexts = get_dataset(dataset_name, prompts, skip_samples)

    contexts = contexts[:length]

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(teacher_response_file):
        print(f"{teacher_response_file} already exists. Skipping generation.")
    else:
        if teacher_model_name.find('llama') != -1:
            teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_name,
                token=token,
                cache_dir=cache_dir,
                torch_dtype=torch.bfloat16
            ).to(device)

            teacher_tokenizer = AutoTokenizer.from_pretrained(
                teacher_model_name,
                token=token,
                cache_dir=cache_dir
            )
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

            pipe = pipeline(
                "text-generation",
                model=teacher_model,
                tokenizer=teacher_tokenizer,
                device=device_idx,
                model_kwargs={"torch_dtype": torch.bfloat16},
                trust_remote_code=True,
            )

            responses = {}
            for prompt in tqdm(contexts, desc="Generating responses from teacher"):
                try:
                    prompt = split_paragraph(prompt)
                    response_text = pipe(prompt, max_new_tokens=128, do_sample=True)[0]['generated_text']
                    responses[prompt] = response_text
                except Exception as e:
                    print(f"Error with prompt: {prompt}... — {e}")
            
            del teacher_model
            torch.cuda.empty_cache()

        elif teacher_model_name.find('gpt') != -1:
            responses = {}
            for prompt in tqdm(contexts, desc="Generating responses from OpenAI model"):
                try:
                    prompt = split_paragraph(prompt)
                    messages = [
                        {"role": "system", "content": "You are a helpful AI language model. Continue the text in the same style."},
                        {"role": "user", "content": f"Continue this:\n\n{prompt}"}
                    ]
                    completion = client.chat.completions.create(
                        model=teacher_model_name,
                        messages=messages,
                        temperature=0.7,
                        top_p=0.9,
                        max_tokens=128,
                        stream=False
                    )
                    response = completion.choices[0].message.content
                    responses[prompt] = response
                except Exception as e:
                    print(f"Error with prompt: {prompt[:30]}... — {e}")

        with open(teacher_response_file, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=4)
        print(f"Saved teacher responses to: {teacher_response_file}")


    generate_token(
        teacher_response_file,
        output_dir,
        compare_model,
        model_style_token_file,
        k,
        cache_dir,
        device
    )

    print(f"Finished: Model-style token saved to {model_style_token_file}")



def preprocess(file):
    prompts, targets = [], []
    with open(file, "r", encoding="utf-8") as f:
        responses = json.load(f)

    for prompt, target in responses.items():

        if not target or not target.strip() or not prompt or not prompt.strip():
            continue
        if target.startswith(prompt):
            
            completion = target[len(prompt):].lstrip()
        else:
            completion = target
        prompts.append(prompt)
        targets.append(completion)

    return prompts, targets

def get_mask_tokens_and_prompts(data_store_path, mask_token_file):

    mask_tokens = []
    prompts = []

    with jsonlines.open(os.path.join(data_store_path, mask_token_file)) as reader:
        for obj in reader:
            mask_token = obj.get('token', '')
            prompt = obj.get('prompt', '')
            mask_tokens.append(mask_token)
            prompts.append(prompt)

    mask_tokens_cleaned = []

    for i in range(len(mask_tokens)):
        mask_tokens_cleaned.append([word.strip(string.punctuation) for word in mask_tokens[i]])

    return mask_tokens_cleaned, prompts

def mask_text(text, hist):
    state = []
    for i in range(len(hist)):
        text = text.replace(hist[i], '[MASK]', 1)
        split_index = text.find('[MASK]')
        first_half = text[:split_index]
        second_half = text[split_index + 6:]
        text = second_half
        if i > 0:
            prefix = prefix + hist[i-1] + first_half
        else:
            prefix = first_half
        state.append(prefix)
    return state

def store_feature(all_probs, targets, labels, feature_store_path, feature_file):
    
    if not os.path.exists(feature_store_path):
        os.makedirs(feature_store_path)
    with jsonlines.open(os.path.join(feature_store_path, feature_file), mode='w') as writer:
        for i in range(len(all_probs)):
            writer.write({
                'target': targets[i],
                'probability': all_probs[i],
                'label': labels[i] 
            })

    print(f'Feature saved to {feature_file}')

def extract_token_prob_by_sampling(prompts, answers, mask_sentences, mask_tokens, model, tokenizer, device, max_new_tokens=1, times=5):
    all_probs = []

    for index, (prompt, target) in tqdm(enumerate(zip(prompts, answers)), total=len(prompts), desc="Extracting sampled token probabilities"):
        generated_text = prompt
        sample_probs = []

        for i in range(len(mask_sentences[index])):
            full_prompt = generated_text + mask_sentences[index][i]
            token_counter = Counter()
            inputs = tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    truncation=True,
                    truncation_strategy="only_first",
                    max_length=512
                ).to(device)

            if inputs["input_ids"].shape[1] == 0:
                print(f"Skipping empty input_ids for prompt: {prompt}")
                continue

            for _ in range(times):

                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        top_k=10,
                        temperature=1.0
                    )

                new_token_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
                if len(new_token_ids) > 0:
                    generated_token = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
                    token_counter[generated_token] += 1

            target_token = mask_tokens[index][i].strip()
            prob = token_counter[target_token] / times if target_token in token_counter else 0.01
            sample_probs.append(prob)

        all_probs.append(sample_probs)

    return all_probs

def extract_token_prob_features(prompts, answers, mask_sentences, mask_tokens, model, tokenizer, device):
   
    all_probs = []

    for index, (prompt, target) in tqdm(enumerate(zip(prompts, answers)), total=len(prompts), desc="Extracting features"):
        generated_text = prompt
        prob = []

        for i in range(len(mask_sentences[index])):
            inputs = tokenizer(
                generated_text + mask_sentences[index][i],
                return_tensors="pt",
                truncation=True,
                truncation_strategy="only_first",
                max_length=512
            )

            inputs["input_ids"] = inputs["input_ids"].to(device, dtype=torch.long)

            input_ids = inputs['input_ids']

            if inputs["input_ids"].shape[1] == 0:
                print(f"Skipping empty input_ids for prompt: {prompt}")
                continue

            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits

            logits = logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

            target_tokens = tokenizer(
                mask_tokens[index][i],
                return_tensors="pt",
                add_special_tokens=False
            ).to(device)
            target_ids = target_tokens["input_ids"]

            if len(target_ids[0]) == 0:
                token_prob = 0.01
            else:
                next_token_id = target_ids[0][0]
                token_prob = probs[next_token_id].item()

            prob.append(token_prob)

        all_probs.append(prob)

    return all_probs



def encoding_features(
    distill_model_names,
    base_model_names,
    teacher_response_file,
    model_style_token_file,
    model_save_dir,
    model_style_token_dir,
    feature_dir,
    dataset_name,
    token,
    cache_dir,
    device,
    setting,
    compare_model,
):
    for model_name, base_model_name in zip(distill_model_names, base_model_names):
        print(f"Distilled model: {model_name}")
        print(f"Base model: {base_model_name}")

        model_path = os.path.join(model_save_dir, model_name)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            cache_dir=cache_dir,
            token=token,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
        ).to(device)

        model = PeftModel.from_pretrained(base_model, model_path).eval().to(device)
        student_model = model.merge_and_unload()

        non_student_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            cache_dir=cache_dir,
            token=token,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
        ).to(device)

        student_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        non_student_tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir, trust_remote_code=True)
        student_tokenizer.pad_token = student_tokenizer.eos_token
        non_student_tokenizer.pad_token = non_student_tokenizer.eos_token

        
        questions, answers = preprocess(teacher_response_file)
        mask_tokens, prompts = get_mask_tokens_and_prompts(model_style_token_dir, model_style_token_file)

        
        answers_copy = copy.deepcopy(answers)
        mask_sentences = [mask_text(ans, mask_tokens[i]) for i, ans in enumerate(answers_copy)]

        
        if setting == 'kt':
            all_probs_student = extract_token_prob_features(
                prompts=prompts,
                answers=answers,
                mask_sentences=mask_sentences,
                mask_tokens=mask_tokens,
                model=student_model,
                tokenizer=student_tokenizer,
                device=device
            )
            all_probs_non_student = extract_token_prob_features(
                prompts=prompts,
                answers=answers,
                mask_sentences=mask_sentences,
                mask_tokens=mask_tokens,
                model=non_student_model,
                tokenizer=non_student_tokenizer,
                device=device
            )

            feature_file = f"{model_name.replace('/', '_')}_{compare_model}_{dataset_name}_kt.json"

        elif setting == 'kw':
            all_probs_student = extract_token_prob_by_sampling(
                prompts=prompts,
                answers=answers,
                mask_sentences=mask_sentences,
                mask_tokens=mask_tokens,
                model=student_model,
                tokenizer=student_tokenizer,
                device=device
            )


            all_probs_non_student = extract_token_prob_by_sampling(
                prompts=prompts,
                answers=answers,
                mask_sentences=mask_sentences,
                mask_tokens=mask_tokens,
                model=non_student_model,
                tokenizer=non_student_tokenizer,
                device=device
            )
            feature_file = f"{model_name.replace('/', '_')}_{compare_model}_{dataset_name}_kw.json"
        else:
            raise ValueError("Setting must be either 'kt' or 'kw'.")

        all_probs = all_probs_student + all_probs_non_student
        labels = [1] * len(all_probs_student) + [0] * len(all_probs_non_student)
        store_feature(all_probs, answers + answers, labels, feature_dir, feature_file)

        
        del student_model
        del non_student_model
        torch.cuda.empty_cache()

def train_model(feature_store_path, feature_file, lens, bin_num=10):

    probabilities = []
    labels = []

    with jsonlines.open(os.path.join(feature_store_path, feature_file)) as reader:
        for obj in reader:
            probability = obj.get('probability', '')
            label = obj.get('label', '')
            probabilities.append(probability)
            labels.append(label)

    random.seed(2)

    label_counts = Counter(labels)

    true_pro = probabilities[:label_counts[1]]

    false_pro = probabilities[label_counts[0]:]

    true_labels = labels[:label_counts[1]]

    false_labels = labels[label_counts[0]:]

    lens = min(lens, label_counts[1])

    true_indices = random.sample(range(len(true_pro)), lens)

    probabilities = [true_pro[i] for i in true_indices] + [false_pro[i] for i in true_indices]
    labels = [true_labels[i] for i in true_indices] + [false_labels[i] for i in true_indices]

    hists = []
    bins = np.arange(0, 1.1, 1 / bin_num)
    for prob in probabilities:
        hist, _ = np.histogram(prob, bins)
        hists.append(hist)

    feature_columns = [f'feature{i + 1}' for i in range(bin_num)]

    train_data = pd.DataFrame(
        {**{feature_columns[i]: [hist[i] for hist in hists] for i in range(bin_num)},
        'class': labels}
    )

    predictor = TabularPredictor(label="class", eval_metric="accuracy").fit(train_data)

    model_directory = './Model/AutoGluon'
    os.makedirs(model_directory, exist_ok=True)
    predictor.save(model_directory)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_store_path', type=str, default='prompts')
    parser.add_argument('--auxiliary_dataset_name', type=str, default='HealthCareMagic', choices=['nq-simplified', 'HealthCareMagic', 'legal', 'finance', 'Sciq'])
    parser.add_argument('--distill_store_path', type=str, default='distill_models')
    parser.add_argument('--teacher_model_name', type=str, default='gpt4')
    parser.add_argument('--compare_model_name', type=str, default="openai-community/gpt2-large")
    parser.add_argument('--model_style_token_dir', type=str, default="tokens")
    parser.add_argument('--feature_dir', type=str, default="features")
    parser.add_argument('--dataset', type=str, default="HealthCareMagic")
    parser.add_argument('--setting', choices=['kt', 'kw'], default='kw')
    parser.add_argument('--mode', choices=['prepare', 'test'], default='prepare')
    parser.add_argument('--skip_samples', type=int, default=20000)
    parser.add_argument('--n_train', type=int, default=10000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--bin_num', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default="responses")
    parser.add_argument('--Gen_traing_data', type=bool, default=False)
    parser.add_argument('--Building', type=bool, default=False)
    parser.add_argument('--Identifying', type=bool, default=False)
    parser.add_argument('--Encoding', type=bool, default=False)
    parser.add_argument('--Learning', type=bool, default=False)
    args = parser.parse_args()
    prompt_store_path = args.prompt_store_path
    distill_store_path = args.distill_store_path
    teacher_model_name = args.teacher_model_name
    auxiliary_dataset_name = args.auxiliary_dataset_name
    compare_model_name = args.compare_model_name
    model_style_token_dir = args.model_style_token_dir
    feature_dir = args.feature_dir
    skip_samples = args.skip_samples

    if teacher_model_name == 'llama3':
        teacher_model_name = 'meta-llama/Llama-3.1-8B'
    elif teacher_model_name == 'gpt4':
        teacher_model_name = 'gpt-4o-mini'


    if auxiliary_dataset_name == 'HealthCareMagic':
        auxiliary_dataset = './prompts/HealthCareMagic-100k-en.jsonl'
    elif auxiliary_dataset_name == 'legal':
        auxiliary_dataset = './prompts/qa_legal.jsonl'
    elif auxiliary_dataset_name == 'nq-simplified':
        auxiliary_dataset = './prompts/nq-simplified.json'
        skip_samples = 0
    elif auxiliary_dataset_name == 'finance':
        auxiliary_dataset = './prompts/causal_lm_finance.parquet'
        skip_samples = 0
    elif auxiliary_dataset_name == 'Sciq':
        auxiliary_dataset = './prompts/Sciq.csv'
        skip_samples = 0


    shadow_traing_dataset = os.path.join(prompt_store_path, 'shadow_' + auxiliary_dataset_name + '_' + args.teacher_model_name + '.jsonl')
    safe_model_name = teacher_model_name.replace("/", "_")
    safe_compare_model_name = compare_model_name.replace("/", "_")

    os.makedirs(args.output_dir, exist_ok=True)
    teacher_response_file = os.path.join(args.output_dir, f"{safe_model_name}_{auxiliary_dataset_name}_gen.jsonl")
    model_style_token_file = f"{safe_model_name}_{auxiliary_dataset_name}_{safe_compare_model_name}_tokens.json"

    if args.Gen_traing_data:
        print("Generating Training Data...")
        if teacher_model_name == 'gpt-4o-mini':
            generate_training_dataset_openai(
                model_name=teacher_model_name,
                input_file=auxiliary_dataset,
                output_file=shadow_traing_dataset,
                dataset_name=auxiliary_dataset_name,
                n_samples=args.n_train,
                max_tokens=128,
                skip_samples=skip_samples
            )
        else:
            generate_training_dataset_local(
                teacher_model_name=teacher_model_name,
                token=token,
                input_file=auxiliary_dataset,
                output_file=shadow_traing_dataset,
                dataset_name=auxiliary_dataset_name,
                n_samples=args.n_train,
                max_tokens=128,
                cache_dir=cache_dir,
                skip_samples=skip_samples
            )
            
    
    if args.Building:
        print("Building Distilled Proxies...")
        for base_model in shadow_model_names:

            base_name = base_model.split("/")[-1].replace("-", "_").lower()
            shadow_student_model = f"distilled_{base_name}_{args.teacher_model_name}_{auxiliary_dataset_name}"
            print(f"Training distilled model: {shadow_student_model}")
            train_distilled_model(
            model_name=base_model,
            distill_model_name=shadow_student_model,
            dataset_path=shadow_traing_dataset,
            token=token,
            output_dir=distill_store_path,
            cache_dir=cache_dir,
        )


    if args.Identifying:
        print("Identifying Model-style tokens...")

        generate_model_style_tokens(
        teacher_model_name=teacher_model_name,
        prompts=auxiliary_dataset,
        dataset_name=auxiliary_dataset_name,
        length=args.n_test,
        token=token,
        cache_dir=cache_dir,
        device=torch.device("cuda:0"),
        device_idx=0,
        output_dir=model_style_token_dir,
        k=4,
        compare_model=compare_model_name,
        teacher_response_file=teacher_response_file,
        output_file=model_style_token_file,
        skip_samples=skip_samples + args.n_train
        )


    if args.Encoding:
        print("Encoding Behavioral Signatures...")
        print("auxiliary dataset:", auxiliary_dataset_name)
        distill_models = []

        if args.mode == 'test':

            for base_name in base_model_names:
                base_name = base_name.split("/")[-1].replace("-", "_").lower()
                distill_name = f"distilled_{base_name}_{args.teacher_model_name}_{args.dataset}"
                distill_models.append(distill_name)
        else:
            for base_name in shadow_model_names:
                base_name = base_name.split("/")[-1].replace("-", "_").lower()
                distill_name = f"distilled_{base_name}_{args.teacher_model_name}_{auxiliary_dataset_name}"
                distill_models.append(distill_name)
            base_model_names = shadow_model_names

        encoding_features(
            distill_model_names=distill_models,
            base_model_names=base_model_names,
            teacher_response_file=teacher_response_file,
            model_style_token_file=model_style_token_file,
            model_save_dir=distill_store_path,
            model_style_token_dir=model_style_token_dir,
            feature_dir=feature_dir,
            dataset_name=auxiliary_dataset_name,
            token=token,
            cache_dir=cache_dir,
            device=torch.device("cuda:0"),
            setting=args.setting,
            compare_model=safe_compare_model_name
        )


    if args.Learning:
        print("Learning Distillation Signatures...")
        for base_model in shadow_model_names:
            base_name = base_model.split("/")[-1].replace("-", "_").lower()
            shadow_student_model = f"distilled_{base_name}_{args.teacher_model_name}_{auxiliary_dataset_name}"
            if args.setting == 'kt':
                feature_file = f"{shadow_student_model.replace('/', '_')}_{safe_compare_model_name}_kt.json"
            elif args.setting == 'kw':
                feature_file = f"{shadow_student_model.replace('/', '_')}_{safe_compare_model_name}_kw.json"
            else:
                raise ValueError("Setting must be either 'kt' or 'kw'.")
            train_model(feature_dir, feature_file, args.n_test, bin_num=args.bin_num)