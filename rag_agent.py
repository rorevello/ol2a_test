import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from persistent_index import FaissManager, QwenVectoriser
import ir_datasets
from sentence_transformers import SentenceTransformer


class RAGAgent:
    def __init__(self, dataset, index_path="qwen.index"):
        print("Initializing RAG Agent...")

        # 1. Configurando o Buscador (Retrieval)
        self.m = FaissManager()
        # É necessário definir o vectoriser antes de fazer buscas!
        self.m.vectoriser = QwenVectoriser(model="Qwen/Qwen3-Embedding-0.6B")
        self.m.load_index(index_path)
        print("FAISS loaded successfully!")

        # 2. Configurando o Modelo de Geração (LLM)
        print("Loading the local generation model (Qwen)...")
        gen_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            gen_model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )

        self.dataset = dataset
        self.docs_map = {doc.doc_id: doc for doc in self.dataset.docs_iter()}

        # --- CONSTRUÇÃO DO MAPEAMENTO LEVE (Apenas Título e Resumo) ---
        print("Mapping titles and abstracts")
        docs_in_qrels = []
        for qrel in self.dataset.qrels_iter():
            docs_in_qrels.append(qrel.doc_id)
        docs_in_qrels = set(docs_in_qrels)

        self.docs_for_indexing = []
        self.doc_id_index = {}
        for doc in self.dataset.docs_iter():
            doc_id = doc[0]
            if doc_id in docs_in_qrels and doc_id not in self.doc_id_index.values():
                title = doc[1]
                abstract = doc[4]
                
                # Junta o título, o resumo e os primeiros 2500 caracteres do texto completo para dar contexto real
                content = f"Title: {title}\nAbstract: {abstract}\n"
                
                full_text_parts = []
                for f in doc[5]:
                    full_text_parts.append(f"{f.title}\n{f.text}")
                full_text = "\n".join(full_text_parts)
                
                # Limited to 2500 characters of the full text to avoid VRAM overflow on 6GB cards
                content += f"Article Text:\n{full_text[:2500]}\n"
                
                self.docs_for_indexing.append(content)
                self.doc_id_index[len(self.docs_for_indexing) - 1] = doc_id
        print(f"Mapping completed! {len(self.docs_for_indexing)} documents mapped.")



    def retrieve(self, query, k=5):
        print(f"\n[Retrieval] Searching for the {k} most relevant documents for: '{query}'...")

        # Faz a busca vetorial no FAISS
        # text_search nos dá as distâncias (scores) e os índices correspondentes
        distances, indices = self.m.text_search(
            query,
            prompt='Find scientific documents that answer the question',
            k=k
        )

        retrieved_docs = []
        # O FAISS retorna uma lista de listas (uma para cada query). Como fazemos só 1 query, pegamos o índice 0.
        for idx in indices[0]:
            if idx == -1:
                continue

            retrieved_docs.append(idx)

        return retrieved_docs

    def context_fusion(self, retrieved_indices):
        print("\n[Context Fusion] Funding texts of the documents recovered...")

        fused_text = []

        for i, idx in enumerate(retrieved_indices):
            doc_id = self.doc_id_index[idx]
            doc_content = self.docs_for_indexing[idx]

            # Formatamos cada documento com um cabeçalho para o LLM saber separar as fontes
            formatted_doc = f"--- DOCUMENTO {i + 1} (ID: {doc_id}) ---\n{doc_content.strip()}\n"
            fused_text.append(formatted_doc)

        # Unimos todos os documentos com uma linha divisória clara
        context = "\n".join(fused_text)
        return context

    def generate_response(self, query, context):
        print("\n[Generation] Generating response based on context...")

        system_instruction = (
            "You are a scientific assistant expert in COVID-19. "
            "Answer the user's question based strictly on the provided context. "
            "If the context does not contain the answer, say that you do not know. "
            "Always cite the source documents used (e.g., [DOCUMENT 1])."
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        # Organiza no formato de chat que o Qwen espera
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]

        # Converte as mensagens para o formato especial do modelo (chat template)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Converte o texto em tensores e envia para a placa de vídeo
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")

        # Gera os tokens de resposta
        generated_ids = self.llm.generate(
            **model_inputs,
            max_new_tokens=512,  # Limite máximo da resposta
            temperature=0.1,  # Temperatura baixa deixa a resposta mais factual/precisa
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Remove os tokens da question para ficar apenas com a resposta gerada
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decodifica a resposta gerada de volta para texto legível
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def ask(self, user_question):
        indices = self.retrieve(user_question, k=5)

        context = self.context_fusion(indices)

        response = self.generate_response(user_question, context)

        return response
