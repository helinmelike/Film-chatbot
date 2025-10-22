import os
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy

load_dotenv()

st.set_page_config(
    page_title="Film Öneri Asistanı",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
    }
    .assistant-message {
        background-color: #F3E5F5;
        border-left: 5px solid #9C27B0;
    }
    .movie-card {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #FF9800;
    }
    .tech-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background-color: #4CAF50;
        color: white;
        border-radius: 15px;
        margin: 0.25rem;
        font-size: 0.85rem;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class LangChainMovieRAG:
    """LangChain tabanlı Film Öneri RAG Sistemi"""
    
    def __init__(self, api_key):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=api_key
        )
        self.vectorstore = None
        self.chain = None
        self.chat_history = []
        
    def create_vectorstore(self, df):
        """DataFrame'den FAISS vektör DB oluştur"""
        texts = []
        metadatas = []
        embeddings_list = []
        
        for idx, row in df.iterrows():
            rating = row.get('rating', 'N/A')
            text = f"Film: {row['title']}\nTür: {row['genre']}\nDuygu: {row['emotion']}\nRating: {rating}/10\nYorum: {row['review'][:300]}"
            texts.append(text)
            metadatas.append({
                'title': row['title'],
                'genre': str(row['genre']),
                'emotion': row['emotion'],
                'rating': float(rating) if rating != 'N/A' else 0.0,
                'review': row['review'][:200]
            })
            embeddings_list.append(row['embedding'])
        
        embeddings_array = np.array(embeddings_list).astype('float32')
        dimension = len(embeddings_list[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        documents = [
            Document(page_content=text, metadata=meta) 
            for text, meta in zip(texts, metadatas)
        ]
        
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {i: str(i) for i in range(len(documents))}
        
        self.vectorstore = LangChainFAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        
        return self.vectorstore
    
    def load_vectorstore(self, path="langchain_faiss_db"):
        self.vectorstore = LangChainFAISS.load_local(
            path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
    
    def create_chain(self):
        if not self.vectorstore:
            raise ValueError("Önce vectorstore oluşturun!")
        
        prompt_template = """Sen profesyonel bir film öneri asistanısın. AYNEN aşağıdaki formatı kullan!

ÖNEMLİ KURALLAR:
1. SADECE verilen "Bulunan filmler" listesindeki filmleri öner
2. Rating sorusu: "X filminin ratingu Y/10" (tek satır)
3. Film önerisi formatı - AYNEN BÖYLE YAZ:

Size [kategori] filmleri öneriyorum:

**Film Adı 1** (⭐ X.X/10)
Tek cümle açıklama.

**Film Adı 2** (⭐ X.X/10)
Tek cümle açıklama.

**Film Adı 3** (⭐ X.X/10)
Tek cümle açıklama.

4. Her film için:
   - Önce **Film Adı** (kalın)
   - Sonra (⭐ rating/10)
   - Yeni satırda tek cümle açıklama
   - Sonra boş satır
5. Sadece 3-5 film öner, fazla detay verme

Önceki sohbet geçmişi:
{chat_history}

Soru: {input}

Bulunan filmler:
{context}

Cevabın (AYNEN YUKARIDAKI FORMAT):"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Document chain oluştur
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Retrieval chain oluştur
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 30,
                "lambda_mult": 0.5
            }
        )
        
        self.chain = create_retrieval_chain(retriever, document_chain)
        
        return self.chain
    
    def query(self, question):
        if not self.chain:
            self.create_chain()
        
        # Chat history'yi mesaj formatına çevir
        chat_history_messages = []
        for msg in self.chat_history[-6:]:  # Son 6 mesajı al
            if msg['role'] == 'user':
                chat_history_messages.append(HumanMessage(content=msg['content']))
            else:
                chat_history_messages.append(AIMessage(content=msg['content']))
        
        result = self.chain.invoke({
            "input": question,
            "chat_history": chat_history_messages
        })
        
        # Tekrar eden filmleri temizle
        seen_titles = set()
        unique_docs = []
        
        for doc in result.get('context', []):
            title = doc.metadata.get('title', '')
            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)
        
        # Chat history'ye ekle
        self.chat_history.append({'role': 'user', 'content': question})
        self.chat_history.append({'role': 'assistant', 'content': result['answer']})
        
        return {
            'answer': result['answer'],
            'source_docs': unique_docs
        }

@st.cache_resource
def initialize_rag():
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        st.error("OpenAI API Key bulunamadı!")
        st.stop()
    
    rag = LangChainMovieRAG(api_key)
    
    try:
        if os.path.exists('langchain_faiss_db'):
            rag.load_vectorstore('langchain_faiss_db')
        else:
            st.warning("FAISS veritabanı bulunamadı!")
            st.stop()
        
        rag.create_chain()
        return rag
    except Exception as e:
        st.error(f"Sistem başlatılamadı: {e}")
        st.stop()

def main():
    st.markdown('<h1 class="main-header">AI Film Öneri Asistanı</h1>', unsafe_allow_html=True)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'recommended_movies' not in st.session_state:
        st.session_state.recommended_movies = set()
    if 'conversation_count' not in st.session_state:
        st.session_state.conversation_count = 0
    if 'rag' not in st.session_state:
        with st.spinner("Sistem başlatılıyor..."):
            st.session_state.rag = initialize_rag()
            st.success("Sistem hazır! 45,000+ film veritabanı yüklendi!")
    
    with st.sidebar:
        st.header("Sistem Bilgileri")
        
        st.markdown("""
        <div style='background-color: #E8F5E9; padding: 1rem; border-radius: 10px;'>
        <h4>Teknolojiler</h4>
        <span class='tech-badge'>GPT-4o-mini</span>
        <span class='tech-badge'>FAISS</span>
        <span class='tech-badge'>LangChain</span>
        <span class='tech-badge'>MMR</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("İstatistikler")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='stat-box'>
                <h2>{st.session_state.conversation_count}</h2>
                <p>Sohbet</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='stat-box'>
                <h2>{len(st.session_state.recommended_movies)}</h2>
                <p>Film Önerildi</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Örnek Sorular")
        
        example_questions = [
            "Komik bir aksiyon filmi öner",
            "Ağlatıcı bir drama izlemek istiyorum",
            "Macera dolu bir film öner",
            "Romantik komedi var mı?",
            "Gerilim filmi öner",
            "Aile ile izlenecek film"
        ]
        
        for question in example_questions:
            if st.button(f"{question}", key=f"ex_{question}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()
        
        st.markdown("---")
        if st.button("Sohbeti Temizle", use_container_width=True, type="primary"):
            st.session_state.chat_history = []
            st.session_state.recommended_movies = set()
            st.session_state.conversation_count = 0
            st.session_state.rag.chat_history = []
            st.rerun()
    
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class='chat-message user-message'>
                <strong>Siz:</strong><br>{message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='chat-message assistant-message'>
                <strong>AI Asistanı:</strong><br>{message['content']}
            </div>
            """, unsafe_allow_html=True)
    
    if 'pending_question' in st.session_state:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question
    else:
        user_input = st.chat_input("Film tercihinizi yazın... (Örn: Komik aksiyon filmi)")
    
    if user_input:
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        with st.spinner("Düşünüyorum..."):
            try:
                result = st.session_state.rag.query(user_input)
                
                movies = []
                for doc in result['source_docs'][:6]:
                    rating_val = doc.metadata.get('rating', 0.0)
                    movie = {
                        'title': doc.metadata.get('title', 'Bilinmiyor'),
                        'genre': doc.metadata.get('genre', ''),
                        'emotion': doc.metadata.get('emotion', ''),
                        'rating': rating_val if rating_val and rating_val > 0 else None
                    }
                    movies.append(movie)
                    st.session_state.recommended_movies.add(movie['title'])
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': result['answer'],
                    'movies': movies
                })
                
                st.session_state.conversation_count += 1
                st.rerun()
                
            except Exception as e:
                st.error(f"Hata: {e}")

if __name__ == "__main__":
    main()