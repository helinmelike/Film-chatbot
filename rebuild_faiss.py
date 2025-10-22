import os
import pickle
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy

load_dotenv()

print("🔄 FAISS Veritabanını Rating ile Yeniden Oluşturuluyor...")
print("=" * 80)

# 1. Embeddings'leri yükle
print("\n📂 movie_embeddings_full.pkl yükleniyor...")
with open('movie_embeddings_full.pkl', 'rb') as f:
    data = pickle.load(f)
    df = data['df']

total_movies = len(df)
print(f"✅ {total_movies} film yüklendi")
print(f"📊 Kolonlar: {df.columns.tolist()}")

# Rating kolonunu kontrol et
if 'rating' in df.columns:
    rating_col = 'rating'
elif 'rrating' in df.columns:
    rating_col = 'rrating'
elif 'Ratings' in df.columns:
    rating_col = 'Ratings'
else:
    print("❌ Rating kolonu bulunamadı!")
    exit(1)

print(f"📊 Rating kolonu: {rating_col}")
print(f"📊 Örnek rating: {df[rating_col].iloc[0]}")

# 2. OpenAI Embeddings
api_key = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)

# 3. Yeni metinler oluştur - BATCH PROCESSING (Bellek dostu)
print("\n🔨 Yeni dokümanlar oluşturuluyor (rating bilgisiyle)...")
print("💡 Bellek optimizasyonu aktif - Batch processing kullanılıyor...")

texts = []
metadatas = []
embeddings_list = []

# İLERLEME TAKIBI
batch_size = 50000
processed = 0

for idx in range(total_movies):
    try:
        # Tek satır al (bellekten tasarruf)
        row = df.iloc[idx]
        
        title = row.get('title', 'Bilinmiyor')
        genre = row.get('genre', '')
        emotion = row.get('emotion', '')
        review = row.get('review', '')
        rating = row.get(rating_col, 0.0)
        embedding = row.get('embedding', None)
        
        if embedding is None:
            continue
        
        # ÖNEMLİ: Metin içinde rating'i açıkça belirt!
        text = f"""Film: {title}
Tür: {genre}
Duygu: {emotion}
Rating: {float(rating):.1f}/10
Yorum: {str(review)[:300]}"""
        
        texts.append(text)
        metadatas.append({
            'title': title,
            'genre': str(genre),
            'emotion': emotion,
            'rating': float(rating),
            'review': str(review)[:200]
        })
        embeddings_list.append(embedding)
        
        processed += 1
        
        # İlerleme göster
        if processed % batch_size == 0:
            print(f"  ⏳ {processed:,}/{total_movies:,} işlendi ({processed/total_movies*100:.1f}%)")
            
    except Exception as e:
        print(f"⚠️ Satır {idx} atlandı: {e}")
        continue

print(f"\n✅ Toplam {processed:,} film işlendi")

# 4. FAISS oluştur
print("\n🚀 FAISS index oluşturuluyor...")
embeddings_array = np.array(embeddings_list).astype('float32')
dimension = len(embeddings_list[0])
index = faiss.IndexFlatL2(dimension)

# Batch halinde ekle (bellek tasarrufu)
batch_size = 100000
for i in range(0, len(embeddings_array), batch_size):
    batch = embeddings_array[i:i+batch_size]
    index.add(batch)
    print(f"  ✅ {min(i+batch_size, len(embeddings_array)):,}/{len(embeddings_array):,} eklendi")

# 5. Documents oluştur - Batch processing
print("\n📝 LangChain Documents oluşturuluyor...")
documents = []
for i in range(0, len(texts), 100000):
    batch_docs = [
        Document(page_content=text, metadata=meta) 
        for text, meta in zip(texts[i:i+100000], metadatas[i:i+100000])
    ]
    documents.extend(batch_docs)
    print(f"  ✅ {len(documents):,}/{len(texts):,} döküman oluşturuldu")

# 6. Docstore oluştur
print("\n💾 Docstore oluşturuluyor...")
docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
index_to_docstore_id = {i: str(i) for i in range(len(documents))}

# 7. LangChain FAISS
print("\n🔗 LangChain FAISS oluşturuluyor...")
vectorstore = LangChainFAISS(
    embedding_function=embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
)

# 8. KAYDET
print("\n💾 FAISS veritabanı kaydediliyor...")
vectorstore.save_local("langchain_faiss_db")

print("\n" + "=" * 80)
print("✅ BAŞARILI! FAISS veritabanı rating bilgisiyle yeniden oluşturuldu!")
print("=" * 80)
print(f"\n📊 İstatistikler:")
print(f"  • Toplam Film: {len(documents):,}")
print(f"  • Embedding Boyutu: {dimension}")
print(f"  • Rating Kolonu: {rating_col}")
print("\n📋 Test için örnek doküman:")
print(documents[0].page_content[:250])
print(f"\n📊 Metadata örneği:")
print(documents[0].metadata)
print("\n🚀 Şimdi app.py'yi çalıştırabilirsiniz: streamlit run app.py")