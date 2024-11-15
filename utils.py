from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
import torch
import os
import spaces
import torch
from transformers import BitsAndBytesConfig

def setGPU():
    torch.cuda.empty_cache()
    #os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if torch.cuda.is_available():
        print(f"Numero di GPU disponibili: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        current_device = torch.cuda.current_device()
        print(f"GPU in uso: {current_device}, {torch.cuda.get_device_name(current_device)}")
    else:
        print("CUDA non disponibile. Utilizzando CPU.")

        
def setLLM():
    # Define the quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, 
    )
        
    # Define the generate_kwargs
    generate_kwargs = {
        "do_sample": True,
        "min_length": 50,  # Lunghezza minima della risposta generata
        "no_repeat_ngram_size": 5,  # Evita la ripetizione di n-grammi
        "temperature": 0.1,
        "top_p": 0.95,      
        "top_k": 10,
    }
    
    # Define the prompt template
    prompt_template = PromptTemplate("<s> [INST] {query_str} [/INST] ")
    
    # Load the HuggingFaceLLM with specified configurations
    llm = HuggingFaceLLM(
        model_name="swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA",
        tokenizer_name="swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA",
        query_wrapper_prompt=prompt_template,
        context_window=3900,
        max_new_tokens=512,
        generate_kwargs=generate_kwargs,
        model_kwargs={"quantization_config": quantization_config},
        # tokenizer_kwargs={"token": hf_token},
        device_map="auto", # Automatically allocate the model to GPU if available
    )
    
    return llm

def setPromptTemplate():
    text_qa_template_str = (
        "Sei un chatbot in grado di rispondere solo alle domande su bandi regionali e avvisi della regione Puglia. Le informazioni di contesto recuperate da diverse sorgenti sono qua sotto.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Usa le informazioni di contesto sopra fornite e non la tua conoscenza pregressa per rispondere, l'unica regione che conosci è la regione Puglia. Se le informazioni di contesto non sono utili rispondi usando la tua conoscenza pregressa."
        "rispondi alla seguente query usando le informazioni dei bandi della regione Puglia \n"
        "Query: {query_str}\n"
        "Risposta: "
    )

    refine_template_str = (
        "La domanda orginale è la seguente: {query_str}\n Abbiamo fornito la"
        " seguente risposta: {existing_answer}\nAbbiamo l'opportunità di aggiornare"
        " la risposta (solo se necessario) con il seguente contesto in più"
        " .\n------------\n{context_msg}\n------------\nUsando il nuovo"
        " contesto, aggiorna o ripeti la risposta.\n"
    )


    text_qa_template = PromptTemplate(text_qa_template_str)
    refine_template = PromptTemplate(refine_template_str)

    return text_qa_template, refine_template
