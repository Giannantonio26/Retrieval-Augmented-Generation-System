import gradio as gr
from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.core.memory import ChatMemoryBuffer
import re
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
# Retrievers
from llama_index.core.retrievers import (
    VectorIndexRetriever,
)
from llama_index.core.chat_engine import ContextChatEngine 
from llama_index.core.memory import ChatMemoryBuffer
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
import time
from utils import *
import spaces
import threading
import sys
import torch
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)


head = """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">

    <script>
        // JavaScript function to toggle text visibility
        function toggleText(id){
            console.log(id)
            if(id=="span1"){
              nodo_id = "nodo1"
            }else if(id=="span2"){
              nodo_id = "nodo2"
            }else if(id=="span3"){
              nodo_id = "nodo3"
            }else{
              nodo_id = "nodo4"
            }
            var text = document.getElementById(nodo_id);
            if (text.style.display === "none") {
                text.style.display = "block";
            } else {
                text.style.display = "none";
            }
        }
    </script>
"""

css = """
#chatbot {
  margin-top: 1%;
  width: 75%;
  position:relative;
  height:70%;
 }

#textBox{
  width: 75%;
  position:relative;
}

.wrapper.svelte-nab2ao p{
  font-size: 14px;
}

#btnClear{
  width: 75%;
}

#buttonChat{
  width:50%;
  position: relative;
}


#colonnaElementi{
  position: absolute;
  left: 77%;
  top: 10%;
  bottom: 10%; /* Adjust this value as necessary */
  width: 10%;
  height: auto; /* Let the height be determined by the top and bottom properties */
  max-height: 80%; /* Ensure it does not exceed 80% of the parent container's height */
  overflow-y: auto; /* Allow scrolling if content overflows vertically */
  overflow-x: hidden; /* Hide horizontal overflow */
  word-wrap: break-word; /* Ensure words break to fit within the width */
  box-sizing: border-box; /* Include padding and border in the element's total width and height */
}

#responseMode{
    width: 5%;
}

.message.user.svelte-gutj6d.message-bubble-border{
  padding: 5px;
}
.message.bot.svelte-gutj6d.message-bubble-border{
  padding: 5px;
}
.icon {
  cursor: pointer;
}
/* Style for the hidden text */
.hidden-text {
  display: none;
}

.wrap svelte-1sk0pyu{
  width: 12%
}


"""
user_message=""
current_chat_mode=""
current_response_mode="tree_summarize"
current_collection="BANDI_SISTEMA_PUGLIA"
file_path=""
num_responses=0
current_chat_mode="STANDARD"
retriever=None
token_count_bandi=0
token_count_bandi_sistema_puglia=0
chat_engine_bandi=None
chat_engine_bandi_sistema_puglia=None
memory_bandi=None
memory_bandi_sistema_puglia=None
stream_response=None
divDocumenti=None
llm = None

def main():
    global llm
    setGPU()
    llm = setLLM()
    Settings.llm = llm
    Settings.embed_model = "local:google-bert/bert-base-multilingual-cased"
    embed_model = Settings.embed_model
    text_qa_template, refine_template = setPromptTemplate()

    def select_initial_collection():
        global current_collection
        global retriever
        global index
        
        pc = Pinecone(api_key="7e412663-a2dc-44a6-ab57-25dd0bdce226")
        # connect to index
        pinecone_index = pc.Index("indexbandisistemapuglia")

        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            add_sparse_vector=True,
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

        retriever = VectorIndexRetriever(index=index, similarity_top_k=3, vector_store_query_mode="hybrid", embed_model=embed_model, alpha=0.5)

        current_collection = "BANDI_SISTEMA_PUGLIA"
        return "collezione settata"
    
    select_initial_collection()


    def select_collection(evt: gr.SelectData):
        global current_collection
        global retriever
        global chat_engine_bandi
        global chat_engine_bandi_sistema_puglia
        global token_count_bandi
        global token_count_bandi_sistema_puglia
        global memory_bandi
        global memory_bandi_sistema_puglia
        selected_collection = evt.value

        if(selected_collection != current_collection):
            if(selected_collection == "BANDI_SISTEMA_PUGLIA"):  
                chat_engine_bandi.reset()
                chat_engine_bandi_sistema_puglia.reset()
                memory_bandi_sistema_puglia.reset()
                memory_bandi.reset()
                token_count_bandi = 0
                token_count_bandi_sistema_puglia = 0
                pc = Pinecone(api_key="7e412663-a2dc-44a6-ab57-25dd0bdce226")
                # connect to index
                pinecone_index = pc.Index("indexbandisistemapuglia")

                vector_store = PineconeVectorStore(
                    pinecone_index=pinecone_index,
                    add_sparse_vector=True,
                )

                storage_context = StorageContext.from_defaults(vector_store=vector_store)

                # load your index from stored vectors
                index = VectorStoreIndex.from_vector_store(
                    vector_store, storage_context=storage_context
                )

                retriever = VectorIndexRetriever(index=index, similarity_top_k=3, vector_store_query_mode="hybrid", embed_model=embed_model, alpha=0.5)
            else:
                chat_engine_bandi.reset()
                chat_engine_bandi_sistema_puglia.reset()
                memory_bandi_sistema_puglia.reset()
                memory_bandi.reset()
                token_count_bandi = 0
                token_count_bandi_sistema_puglia = 0
                pc = Pinecone(api_key="7e412663-a2dc-44a6-ab57-25dd0bdce226")
                # connect to index
                pinecone_index = pc.Index("indexbandi")

                vector_store = PineconeVectorStore(
                    pinecone_index=pinecone_index,
                    add_sparse_vector=True,
                )

                storage_context = StorageContext.from_defaults(vector_store=vector_store)

                # load your index from stored vectors
                index = VectorStoreIndex.from_vector_store(
                    vector_store, storage_context=storage_context
                )

                retriever = VectorIndexRetriever(index=index, similarity_top_k=3, vector_store_query_mode="hybrid", embed_model=embed_model, alpha=0.4)     
            
            current_collection = selected_collection
        
        return "<div class='alert alert-success' role='alert'> Collezione "+selected_collection+" selezionata </div>"
    
    def select_response_mode(evt: gr.SelectData):
        global current_response_mode
        current_response_mode = evt.value
        return "<div class='alert alert-success' role='alert'>"+current_response_mode+" selezionato </div>"

    def select_chat_mode():
        global current_chat_mode
        global memory_bandi
        global memory_bandi_sistema_puglia
        global chat_engine_bandi
        global chat_engine_bandi_sistema_puglia
        global token_count_bandi
        global token_count_bandi_sistema_puglia
        memory_bandi_sistema_puglia.reset()
        memory_bandi.reset()
        chat_engine_bandi.reset()
        chat_engine_bandi_sistema_puglia.reset()
        token_count_bandi = 0
        token_count_bandi_sistema_puglia = 0
        current_chat_mode = "CHAT"

        return "<div class='alert alert-success' role='alert'>Hai selezionato la modalità "+current_chat_mode+" </div>"

    def select_standard_mode():
        global current_chat_mode
        current_chat_mode = "STANDARD"
        return "<div class='alert alert-success' role='alert'>Hai selezionato la modalità "+current_chat_mode+" </div>"

    def set_chat_engine():
        global chat_engine_bandi
        global chat_engine_bandi_sistema_puglia
        global memory_bandi
        global memory_bandi_sistema_puglia
        global token_count_bandi_sistema_puglia
        global token_count_bandi
        memory_bandi = ChatMemoryBuffer.from_defaults(token_limit=5000)
        memory_bandi_sistema_puglia = ChatMemoryBuffer.from_defaults(token_limit=3000)

        pc = Pinecone(api_key="7e412663-a2dc-44a6-ab57-25dd0bdce226")
        pinecone_index = pc.Index("indexbandi")
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            add_sparse_vector=True,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

        retriever_bandi = VectorIndexRetriever(index=index, similarity_top_k=3, vector_store_query_mode="hybrid", embed_model=embed_model, alpha=0.4) 
        chat_engine_bandi = ContextChatEngine(retriever=retriever_bandi, 
                                                        context_template="Sei un chatbot in grado di rispondere alle domande su bandi regionali e avvisi della regione Puglia. Hai accesso ai bandi della regione Puglia. Qui sotto le informazioni di contesto recuperate. \n"
                "---------------------\n"
                "Informazioni di contesto: "+"{context_str}\n"
                "---------------------\n"
                "Usa le informazioni di contesto sopra fornite e non la tua conoscenza pregressa per rispondere, l'unica regione che conosci è la regione Puglia. "
                "rispondi sempre alla seguente query sul bando regionale della Puglia usando le informazioni di contesto."
                "\n", llm=llm, memory=memory_bandi, prefix_messages=[])
        
        pinecone_index = pc.Index("indexbandisistemapuglia")
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            add_sparse_vector=True,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )     
        retriever_bandi_sistema_puglia = VectorIndexRetriever(index=index, similarity_top_k=3, vector_store_query_mode="hybrid", embed_model=embed_model, alpha=0.5)

        chat_engine_bandi_sistema_puglia = ContextChatEngine(retriever=retriever_bandi_sistema_puglia, 
                                                        context_template="Sei un chatbot in grado di rispondere alle domande su bandi regionali e avvisi della regione Puglia. Hai accesso ai bandi della regione Puglia. Qui sotto le informazioni di contesto recuperate. \n"
                "---------------------\n"
                "Informazioni di contesto: "+"{context_str}\n"
                "---------------------\n"
                "Usa le informazioni di contesto sopra fornite e non la tua conoscenza pregressa per rispondere, l'unica regione che conosci è la regione Puglia. "
                "rispondi sempre alla seguente query sul bando regionale della Puglia usando le informazioni di contesto."
                "\n", llm=llm, memory=memory_bandi_sistema_puglia, prefix_messages=[])

    set_chat_engine()


    def html_escape(text):
        html_entities = {
            'à': '&agrave;',
            'è': '&egrave;',
            'é': '&eacute;',
            'ì': '&igrave;',
            'ò': '&ograve;',
            'ù': '&ugrave;',
            'À': '&Agrave;',
            'È': '&Egrave;',
            'É': '&Eacute;',
            'Ì': '&Igrave;',
            'Ò': '&Ograve;',
            'Ù': '&Ugrave;',
            'ç': '&ccedil;',
            'Ç': '&Ccedil;',
            'ä': '&auml;',
            'ö': '&ouml;',
            'ü': '&uuml;',
            'Ä': '&Auml;',
            'Ö': '&Ouml;',
            'Ü': '&Uuml;',
            'ß': '&szlig;',
            'ñ': '&ntilde;',
            'Ñ': '&Ntilde;',
            'œ': '&oelig;',
            'Œ': '&OElig;',
            'æ': '&aelig;',
            'Æ': '&AElig;',
            'ø': '&oslash;',
            'Ø': '&Oslash;',
            'å': '&aring;',
            'Å': '&Aring;',
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        }
        return ''.join(html_entities.get(c, c) for c in text)

    def reset():
        global chat_engine_bandi
        global chat_engine_bandi_sistema_puglia
        global memory_bandi
        global memory_bandi_sistema_puglia
        global token_count_bandi
        global token_count_bandi_sistema_puglia
        chat_engine_bandi.reset()
        chat_engine_bandi_sistema_puglia.reset()
        memory_bandi_sistema_puglia.reset()
        memory_bandi.reset()
        token_count_bandi = 0
        token_count_bandi_sistema_puglia = 0
        return ""


    with gr.Blocks(css=css, head=head) as demo:
        with gr.Row():
            output = gr.HTML()
        with gr.Row(elem_id="buttonChat"):
            gr.Button("STANDARD", size="sm").click(fn=select_standard_mode, outputs=output)
            gr.Button("CHAT",size="sm").click(fn=select_chat_mode, outputs=output)
            gr.Dropdown(
            ["BANDI_SISTEMA_PUGLIA","BANDI"], min_width= 185, label="Collezione di documenti", info="", container=False, interactive=True, value="BANDI_SISTEMA_PUGLIA", elem_id="dropdown"
            ).select(fn=select_collection, outputs=output)
        
        chatbot = gr.Chatbot(elem_id="chatbot", container=False)

        with gr.Column(elem_id="colonnaElementi"):
            
            gr.Radio(["compact","tree_summarize"], label="Response mode", info="Influenzerà il modo in cui il chatbot risponde", interactive=True,container=False, value="tree_summarize",elem_id="responseMode").select(fn=select_response_mode, outputs=output),

            divDocumenti = gr.HTML("<div id='divDocumenti'></div>")

        msg = gr.Textbox(elem_id="textBox", container=False)
        clear = gr.ClearButton([msg, chatbot], elem_id="btnClear")
        clear.click(fn=reset, outputs=divDocumenti)
        
        def user(userMessage, history):
            global user_message 
            user_message = userMessage
            if history is None:
                history = []

            return "", history + [[user_message, None]]
        
        @spaces.GPU(duration=150)
        def bot(history):
            lenght = len(history)
            userMessage = history[lenght-1][0]
            global chat_engine_bandi
            global chat_engine_bandi_sistema_puglia
            global memory_bandi
            global memory_bandi_sistema_puglia
            global current_response_mode
            global current_collection
            global retriever
            global file_path
            global current_chat_mode
            global token_count_bandi
            global token_count_bandi_sistema_puglia
            
            if(current_chat_mode=="CHAT"):
                print("MODALITA CHAT")
                if(current_collection=="BANDI"):
                    if(token_count_bandi >= 1000):
                        print("RESET!!!")
                        token_count_bandi = 0
                        memory_bandi.reset()
                        chat_engine_bandi.reset()                
                    print(chat_engine_bandi.chat_history)
                    print(memory_bandi)
                    stream_response = None
                    print(userMessage)
                    stream_response = chat_engine_bandi.stream_chat(userMessage)
                    print("risposta con chat engine")
                    responseHTML = ""
                    for i, node in enumerate(stream_response.source_nodes):
                        responseHTML += "<p><b>"+node.metadata['nome_bando']+"</b><a href='"+node.metadata['file_path']+"' download> <svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='currentColor' class='bi bi-download' viewBox='0 0 16 16'><path d='M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5'/><path d='M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708z'/> </svg></a><br>Nodo <span id='span"+str(i+1)+"' class='icon' onclick='toggleText(this.id)'>🔍</span>   <!-- Text to show/hide --><p class='hidden-text' id='nodo"+str(i+1)+"'>"+node.text+"</p>"

                    history[-1][1] = ""
                    for character in stream_response.response_gen: 
                        tokens = character.split(" ") 
                        num_tokens = len(tokens)
                        token_count_bandi = token_count_bandi + num_tokens 
                        print(token_count_bandi)
                        history[-1][1] += html_escape(str(character))
                        time.sleep(0.05)
                        yield history, responseHTML

                else:
                    if(token_count_bandi_sistema_puglia >= 1000):
                        print("RESET!!!")
                        token_count_bandi_sistema_puglia = 0
                        memory_bandi_sistema_puglia.reset()
                        chat_engine_bandi_sistema_puglia.reset()                
                    print(chat_engine_bandi_sistema_puglia.chat_history)
                    print(memory_bandi_sistema_puglia)
                    stream_response = None
                    print(userMessage)
                    stream_response = chat_engine_bandi_sistema_puglia.stream_chat(userMessage)
                    print("risposta con chat engine")
                    responseHTML = "" 
                    for i, node in enumerate(stream_response.source_nodes):
                        responseHTML += "<p><b>"+node.metadata['nome_bando']+"</b><a href='"+node.metadata['file_path']+"' download> <svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='currentColor' class='bi bi-download' viewBox='0 0 16 16'><path d='M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5'/><path d='M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708z'/> </svg></a><br>Nodo <span id='span"+str(i+1)+"' class='icon' onclick='toggleText(this.id)'>🔍</span>   <!-- Text to show/hide --><p class='hidden-text' id='nodo"+str(i+1)+"'>"+node.text+"</p>"
                    
                    history[-1][1] = ""
                    for character in stream_response.response_gen: 
                        tokens = character.split(" ") 
                        num_tokens = len(tokens)
                        token_count_bandi_sistema_puglia = token_count_bandi_sistema_puglia + num_tokens 
                        print(token_count_bandi_sistema_puglia)
                        history[-1][1] += html_escape(str(character))
                        time.sleep(0.05)
                        yield history,responseHTML   

            else:
                print("MODALITA STANDARD")
                nome_bando = ""
                userMessage = userMessage.lower()
                if("diploma professionale" in userMessage):
                        nome_bando += "Scheda Avviso Pubblico Diploma Professionale 2022.pdf,"
                if("red" in userMessage):
                        nome_bando += "Scheda RED 2020.pdf,"
                if("ifts" in userMessage):
                        nome_bando += "Scheda Avviso Pubblico IFTS_2023.pdf,"
                if(("impianti" in userMessage) or ("idrogeno" in userMessage)):
                        nome_bando += "Scheda Avviso PNRR - Impianti idrogeno rinnovabile.pdf,"
                if("laureati" in userMessage):
                        nome_bando += "Scheda Pass Laureati 2023.pdf,"
                if("nidi" in userMessage):
                        nome_bando += "Scheda NIDI - Nuove iniziative d'impresa_ Strumento di ingegneria finanziaria.pdf,"
                if("microprestito" in userMessage):
                        nome_bando += "Scheda MicroPrestito della Regione Puglia - edizione 2021.pdf,"
                if("gol" in userMessage):
                        nome_bando += "Scheda Garanzia di occupabilità dei lavoratori - GOL.pdf,"
                if("edifici pubblici" in userMessage):
                        nome_bando += "Scheda Efficientamento Energetico Edifici Pubblici.pdf,"
                if("innoaid" in userMessage):
                        nome_bando += "Scheda di sintesi Avviso _INNOAID - RIAPERTURA_.pdf,"
                if("tecnonidi" in userMessage):
                        nome_bando += "Scheda Avviso Tecnonidi - Aiuti alle piccole imprese innovative.pdf,"
                if(("bando of" in userMessage) or ("avviso of" in userMessage)):
                        nome_bando += "Scheda Avviso Pubblico OF a_f_ 2023_2024.pdf,"
                if("giardin" in userMessage):
                        nome_bando += "Scheda Avviso Pubblico _Giardiniere d'arte per giardini e parchi storici_.pdf,"
                if("punti cardinali" in userMessage):
                        nome_bando += "Scheda Avviso _Punti Cardinali_ punti di orientamento per la formazione e il lavoro.pdf,"
                if("multimisura POC" in userMessage):
                        nome_bando += "Avviso Multimisura POC.pdf,"
                if("garanzia giovani" in userMessage):
                        nome_bando += "Avviso Multimisura - Garanzia Giovani II Fase.pdf,"
                if("apprendistato professionalizzante" in userMessage):
                        nome_bando += "Apprendistato Professionalizzante.pdf,"                  
            
                if(nome_bando!=""):                       
                        # Rimuovi l'ultima virgola
                        if nome_bando.endswith(","):
                            nome_bando = nome_bando[:-1]                  
                        # Crea una lista di bandi separati dalla virgola
                        lista_bandi = nome_bando.split(",")               
                        # Crea una lista di oggetti MetadataFilter
                        filter_list = []
                        for bando in lista_bandi:
                            filter_list.append(MetadataFilter(key="nome_bando", value=bando))
                        
                        #crea una lista di MetadataFilter
                        filters = MetadataFilters(
                            filters=filter_list,
                            condition=FilterCondition.OR,
                        )

                        retriever = VectorIndexRetriever(index=index, similarity_top_k=3, vector_store_query_mode="hybrid", embed_model=embed_model, alpha=0.5, filters=filters)
                else:
                        retriever = VectorIndexRetriever(index=index, similarity_top_k=3, vector_store_query_mode="hybrid", embed_model=embed_model, alpha=0.5)
                       
                if(str(current_response_mode)=="tree_summarize"):
                    # define response synthesizer
                    response_synthesizer = get_response_synthesizer(streaming=True,response_mode="tree_summarize",text_qa_template=text_qa_template)
                    query_engine = None
                    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)
                    stream_response = None
                    print(userMessage)
                    stream_response = query_engine.query(userMessage)
                    print("risposta con query engine")
                    
                    responseHTML = ""
                    for i, node in enumerate(stream_response.source_nodes):
                        responseHTML += "<p><b>"+node.metadata['nome_bando']+"</b><a href='"+node.metadata['file_path']+"' download> <svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='currentColor' class='bi bi-download' viewBox='0 0 16 16'><path d='M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5'/><path d='M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708z'/> </svg></a><br>Nodo <span id='span"+str(i+1)+"' class='icon' onclick='toggleText(this.id)'>🔍</span>   <!-- Text to show/hide --><p class='hidden-text' id='nodo"+str(i+1)+"'>"+node.text+"</p>"
        
                    history[-1][1] = ""
                    # Misura il tempo di inizio
                    start_time = time.time()
                    for character in stream_response.response_gen:      
                        history[-1][1] += html_escape(str(character))
                        time.sleep(0.05)
                        yield history, responseHTML
                    # Misura il tempo di fine
                    end_time = time.time()
                    # Calcola il tempo di esecuzione
                    execution_time = end_time - start_time
                    print(f"Tempo di esecuzione: {execution_time} secondi")
                else:
                    # define response synthesizer
                    response_synthesizer = get_response_synthesizer(streaming=True,response_mode="compact",text_qa_template=text_qa_template, refine_template=refine_template)
                    query_engine = None
                    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)
                    stream_response = None
                    print(userMessage)
                    stream_response = query_engine.query(userMessage)
                    print("risposta con query engine")
                    responseHTML = ""
                    for i, node in enumerate(stream_response.source_nodes):
                        responseHTML += "<p><b>"+node.metadata['nome_bando']+"</b><a href="+node.metadata['file_path']+" download> <svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='currentColor' class='bi bi-download' viewBox='0 0 16 16'><path d='M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5'/><path d='M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708z'/> </svg></a><br>Nodo <span id='span"+str(i+1)+"' class='icon' onclick='toggleText(this.id)'>🔍</span>   <!-- Text to show/hide --><p class='hidden-text' id='nodo"+str(i+1)+"'>"+node.text+"</p>"
        
                    history[-1][1] = ""
                    for character in stream_response.response_gen:      
                        history[-1][1] += html_escape(str(character))
                        time.sleep(0.05)
                        yield history, responseHTML


            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()
            
        
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, [chatbot, divDocumenti]
        )
        
        demo.queue()
        demo.launch(debug=True, share=True)


if __name__ == "__main__":
    main()
