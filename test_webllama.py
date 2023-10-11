from condense_rope_patch import replace_llama_with_condense

import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import requests
import pysbd
import json
import time
from collections import defaultdict


try:
    my_google_api_key = os.environ['GOOGLE_API_KEY']
    my_google_cx = os.environ['GOOGLE_CX']
except:
    raise ValueError("Google API key not found, please set GOOGLE_API_KEY and GOOGLE_CX environment variables")


key_args = {"main_model_name": "meta-llama/Llama-2-70b-hf",
            "main_adapter_name": "xhan77/web-llama2chat-70b-adapter", # or "xhan77/web-llama2-70b-adapter"
            "main_context_condense_ratio": 4.0,
            "query_model_name": "meta-llama/Llama-2-13b-hf",
            "query_adapter_name": "xhan77/web-query-llama2-13b-adapter",
            "query_context_condense_ratio": 1.0,
            "max_context_seqlen": 14000,
            "max_new_tokens": 2000,
            "gen_temperature": 0.7,
            "gen_top_p": 0.95,
            "gen_rep_penalty": 1.0, 
            "per_device_max_memory": 81000, # change based on the devices
            "dtype": torch.bfloat16,
            }


en_segmenter = pysbd.Segmenter(language="en", clean=False)

def trim_text(text, max_num_char=50000):
    text_list = en_segmenter.segment(text)
    new_text = ""
    cur_num_char = 0
    for _t in text_list:
        if cur_num_char + len(_t) > max_num_char:
            break
        cur_num_char += len(_t)
        new_text = new_text + _t
    return new_text.strip()

def google_search(query):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": my_google_api_key, 
        "cx": my_google_cx, 
        "q": query, 
    }
    result = requests.get(url, params=params).json()
    return [item['link'] for item in result['items']]

class WebReader_4oai(): # Han: using WebPilot (a plugin for ChatGPT) for webllama
    def __init__(self, tmp_userinfo='webllama'):
        self.learnable = False
        self.tmp_userinfo = tmp_userinfo
        self.already_seen_cache = set()
        self.query2links = defaultdict(list)
        
    def clear_cache(self):
        self.already_seen_cache = set()
        self.query2links = defaultdict(list)

    def run(self, query, gold_webpage=None):
        optional_input = 'official_google_api'
        preprocessor = lambda x: x

        reading_link = None
        if gold_webpage is not None:
            reading_link = gold_webpage
        elif optional_input == 'official_google_api':
            if len(self.query2links[preprocessor(query)]) == 0:
                for result in google_search(preprocessor(query)): 
                    self.query2links[preprocessor(query)].append(result)
            while len(self.query2links[preprocessor(query)]) > 0:
                link = self.query2links[preprocessor(query)].pop(0)
                key_for_cache = tuple([preprocessor(query), link])
                if key_for_cache in self.already_seen_cache:
                    continue
                else:
                    reading_link = link
                    self.already_seen_cache.add(key_for_cache)
                    break
            if reading_link is None:
                print("########\n\nNo more links for this query (remember to clear cache if you want to start over)\n\n########")
                return "" # Han: (TODO) better way to handle edge cases
        else:
            raise ValueError("check whether the reader strategy is valid")

        url = "https://webreader.webpilotai.com/api/visit-web"
        params = {
            "link": reading_link, "user_has_request": False
        }
        rt_request = requests.post(url, json=params, headers={"WebPilot-Friend-UID": self.tmp_userinfo})
        rt_dict = rt_request.json() # fallback to .text if failing
        try:
            _ = rt_dict['content']
        except:
            raise ValueError(f"WebPilot Exception: {rt_dict}")
            # rt_dict['content'] = ''
        
        return json.dumps({"content": trim_text(rt_dict['content'])})
    
web_reader_4oai = WebReader_4oai()

func_call_prep = {"func": web_reader_4oai.run, 
                  "func_name": "google_search_and_read", 
                  "func_description": "Read into top google search results",
                  "func_parameters": {"type": "object",
                                      "properties": {"query": {"type": "string", "description": "The query the user wants to search"},
                                                    },
                                      "required": ["query"]
                                     }
                 }
func_prep_list = [func_call_prep]
func_name2func = {_e["func_name"]: _e["func"] for _e in func_prep_list}

################################

model_name = key_args["main_model_name"]
adapters_name = key_args["main_adapter_name"]

replace_llama_with_condense(ratio=key_args["main_context_condense_ratio"])
try:
    from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()
except:
    print("Flash attention disabled")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=key_args["dtype"],
    device_map="auto",
    max_memory= {i: f'{key_args["per_device_max_memory"]}MB' for i in range(torch.cuda.device_count())},
)
model = PeftModel.from_pretrained(model, adapters_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=False) # Han: making sure eos is not added during preprocessing
model.eval()

####

query_model_name = key_args["query_model_name"]
query_adapters_name = key_args["query_adapter_name"]

replace_llama_with_condense(ratio=key_args["query_context_condense_ratio"])

query_model = AutoModelForCausalLM.from_pretrained(
    query_model_name,
    torch_dtype=key_args["dtype"],
    device_map="auto",
    max_memory= {i: f'{key_args["per_device_max_memory"]}MB' for i in range(torch.cuda.device_count())},
)
query_model = PeftModel.from_pretrained(query_model, query_adapters_name)
query_model.eval()

####

def webllama_decode(prompt, context):
    # Han: assuming batch_size=1
    if prompt is None:
        formatted_prompt = (
            f"\n[|Human|]"
        )
    else:
        formatted_prompt = (
            f"\n[|Human|] {prompt.strip()}\n[|AI|]"
        )
    prompt_ids = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0").input_ids
    prompt_len = len(prompt_ids[0])
    max_context_seqlen = key_args["max_context_seqlen"] - prompt_len
    max_new_tokens = key_args["max_new_tokens"]

    trunc_context_ids = tokenizer(context.strip(), return_tensors="pt", add_special_tokens=False).to("cuda:0").input_ids[:, :max_context_seqlen - 1]
    trunc_context = tokenizer.batch_decode(trunc_context_ids, skip_special_tokens=True)[0]
    # print('Truncated context:', trunc_context)
    # print('\n')

    input_text = f"{trunc_context.strip()}{formatted_prompt}"
    input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=True).to("cuda:0").input_ids
    
    print('Input text:', input_text)

    outputs = model.generate(inputs=input_ids, max_new_tokens=max_new_tokens, temperature=key_args["gen_temperature"], top_p=key_args["gen_top_p"], repetition_penalty=key_args["gen_rep_penalty"], do_sample=True)
    # return tokenizer.decode(outputs[0], skip_special_tokens=False)
    # current only for bs=1
    return tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True).strip()

####

def webllama_query_decode(prompt): # optimize hyperparam later
    formatted_prompt = (
        f"Prompt: {prompt.strip()}\n\nQuery:"
    )
    input_ids = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True).to("cuda:0").input_ids
    outputs = query_model.generate(inputs=input_ids, max_new_tokens=1000, top_p=0.9, do_sample=True)
    # return tokenizer.decode(outputs[0], skip_special_tokens=False)
    # current only for bs=1
    return tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True).strip()

################################

def webllama_run_conversation_with_func(query, repeat_func_calls=5): # Han: change here for the number of links to read into
    time_list = []
    base_time = time.time()
    function_name = "google_search_and_read"
    function_query = webllama_query_decode(query)
    time_list.append(('generate search query', round(time.time() - base_time, 3)))
    print("*Func call query*:", function_query)
    
    function_response_list = []
    func_call_result_len = 0
    base_time = time.time()
    for i in range(repeat_func_calls):
        function_response = func_name2func[function_name](query=function_query)
        func_call_result_len += len(function_response)
        if func_call_result_len < 100000: # Han: change here for max char len
            function_response_list.append(function_response)
    func_call_result = '\n\n|||\n\n'.join([json.loads(rl)['content'] for rl in function_response_list])
    # print("*Func call result*:", func_call_result)
    time_list.append(('search and parse webpages', round(time.time() - base_time, 3)))

    base_time = time.time()
    second_response = dict()
    second_response['choices'] = list()
    second_response['choices'].append({'message': dict()})
    second_response['choices'][0]['message']['content'] = webllama_decode(query, func_call_result)
    time_list.append(('generate final answer', round(time.time() - base_time, 3)))
    return {'final_return': second_response["choices"][0]["message"]["content"],
            'pre_func_message': {"name": "google_search_and_read", "arguments": json.dumps({'query': function_query})},
            'func_call_result': func_call_result, 'time_list': time_list}

def webllama_generate_data_with_func(webpage):
    time_list = []
    base_time = time.time()
    function_name = "google_search_and_read" # dummy
    
    function_response_list = []
    function_response = func_name2func[function_name](query=None, gold_webpage=webpage)
    function_response_list.append(function_response)
    func_call_result = '\n\n|||\n\n'.join([json.loads(rl)['content'] for rl in function_response_list])
    # print("*Func call result*:", func_call_result)
    time_list.append(('parse webpages', round(time.time() - base_time, 3)))

    base_time = time.time()
    second_response = dict()
    second_response['choices'] = list()
    second_response['choices'].append({'message': dict()})
    second_response['choices'][0]['message']['content'] = webllama_decode(prompt=None, context=func_call_result)
    time_list.append(('generate final answer', round(time.time() - base_time, 3)))
    return {'final_return': second_response["choices"][0]["message"]["content"],
            'pre_func_message': {"webpage": webpage},
            'func_call_result': func_call_result, 'time_list': time_list}

####

def answer(prompt):
    web_reader_4oai.clear_cache() # optional
    output = webllama_run_conversation_with_func(prompt)
    print(f"\n================================\nstage 1: {output['pre_func_message']['arguments']} ||| overall time: {output['time_list']}\n================================\n")
    print(f"\n================================\n*WebLlama final return*:\n================================\n")
    print(output['final_return'])

def generate(webpage):
    web_reader_4oai.clear_cache() # optional
    output = webllama_generate_data_with_func(webpage)
    print(f"\n================================\noverall time: {output['time_list']}\n================================\n")
    print(f"\n================================\n*WebLlama final return*:\n================================\n")
    print(output['final_return'])

with torch.no_grad():
    # Han: if dtype error persists, can try torch.autocast("cuda")
    # answer("Give me some information about Xiaochuang Han.")
    # generate("https://huggingface.co/datasets/allenai/dolma")
    print("################")
    print("Welcome to WebLlama!")
    print("To enter the question answering mode: answer(<prompt>), for example, answer(\"Write a short bio for Xiaochuang Han\")")
    print("To enter the webpage reading mode: generate(<url>), for example, generate(\"https://huggingface.co/datasets/allenai/dolma\")")
    print("Have fun!")
    print("################")
    breakpoint()
