token = ''

cache_dir = "../../huggingface_cache"


shadow_model_names = [
        "Qwen/Qwen2.5-0.5B",
        'meta-llama/Llama-3.2-1B',
        'microsoft/phi-2',
        'google/gemma-2-2b'
]

compare_model_names = [
        "openai-community/gpt2-large",
        'bigscience/bloom-560m',
        'microsoft/phi-2',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
]


base_model_names = [
"EleutherAI/gpt-neo-1.3B", 
"microsoft/Phi-4-mini-instruct",
'facebook/opt-1.3b', 'Qwen/Qwen2-0.5B', 
'openai-community/gpt2-large',
'LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct', 'amd/Instella-3B',
'bigcode/starcoder2-3b', 'HuggingFaceTB/SmolLM2-1.7B', 'ServiceNow-AI/Apriel-5B-Base'
]


audit_model_llama_healthcare_kt = []

audit_model_gpt4_healthcare_kt = []

audit_model_gpt4_healthcare_kw = []

audit_model_llama_healthcare_kw = []

audit_model_gpt4_legal_kt = []

audit_model_llama_legal_kt = []

audit_model_gpt4_legal_kw = []

audit_model_llama_legal_kw = []



audit_model_paths_llama = {
    'HealthCareMagic_kt': audit_model_llama_healthcare_kt,
    'HealthCareMagic_kw': audit_model_llama_healthcare_kw,
    'legal_kt': audit_model_llama_legal_kt,
    'legal_kw': audit_model_llama_legal_kw
}

audit_model_paths_gpt4 = {
    'HealthCareMagic_kt': audit_model_gpt4_healthcare_kt,
    'HealthCareMagic_kw': audit_model_gpt4_healthcare_kw,
    'legal_kt': audit_model_gpt4_legal_kt,
    'legal_kw': audit_model_gpt4_legal_kw
}

audit_model_paths = {
    'llama3': audit_model_paths_llama,
    'gpt4': audit_model_paths_gpt4
}


