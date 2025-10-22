**AI Film Öneri Sistemi - RAG Tabanlı ChatbotProje Hakkında**

web site link:https://film-chatbot-j5s9ya38ry5fbbnjukw97v.streamlit.app/

Bu proje, Retrieval-Augmented Generation (RAG) teknolojisi kullanarak kullanıcıların film tercihlerine göre kişiselleştirilmiş öneriler sunan yapay zeka destekli bir chatbot uygulamasıdır. Sistem, 900,000'den fazla film verisi üzerinde çalışarak doğal dil işleme ve semantik arama teknikleriyle en uygun filmleri önermektedir.Projenin AmacıBu projenin temel amaçları şunlardır:

Kullanıcıların doğal dilde belirttikleri tercihlerine göre yüksek kaliteli film önerileri sunmak

Tür, duygu durumu ve rating bilgilerine dayalı akıllı filtreleme ve sıralama yapmak

Konuşma bağlamını koruyarak interaktif ve tutarlı bir kullanıcı deneyimi sağlamak

Modern RAG mimarisi ile yüksek ilgililik skoruna sahip ve çeşitli sonuçlar üretmek

Vektör veritabanı ve dil modeli teknolojilerinin entegrasyonunu göstermek



**Proje Yapısı ve Dosya OrganizasyonuFilm-Oneri-RAG-Chatbot/**

│

├── filmkod.ipynb                  # Ana geliştirme notebook'u

│   ├── Cell 1: Kütüphane import ve API key kontrolü

│   ├── Cell 2: Embedding'lerin yüklenmesi

│   ├── Cell 3: LangChainMovieRAG class tanımı

│   ├── Cell 4: RAG sistemi başlatma

│   ├── Cell 5: Chatbot fonksiyonu

│   └── Cell 6: İnteraktif test interface

│

├── app.py                         # Streamlit web uygulaması

│   ├── LangChainMovieRAG class

│   ├── initialize\_rag() fonksiyonu

│   ├── Streamlit UI components

│   └── Session state yönetimi

│

├── film.csv                       # Ham veri seti (900K+ kayıt)

│   ├── Columns: movie\_name, genres, Reviews, Ratings, emotion, Description

│   └── Size: ~2.5 GB

│

├── movie\_embeddings\_full.pkl      # Önceden hesaplanmış embeddings

│   ├── Structure: {'df': DataFrame with 'embedding' column}

│   ├── Size: ~5.2 GB

│   └── Creation time: ~2-3 saat

│

├── langchain\_faiss\_db/            # FAISS vektör veritabanı

│   ├── index.faiss               # Binary FAISS index

│   ├── index.pkl                 # Index metadata

│   └── docstore.pkl              # Document store

│

├── .env                          # Çevre değişkenleri (gitignore'da)

│   └── OPENAI\_API\_KEY=sk-...

│

├── requirements.txt              # Python bağımlılıkları

└── README.md                     # Proje dokümantasyonu



**Veri Seti**

**Veri Kaynağı ve Boyut**

Projede kullanılan veri seti IMDb film veritabanından derlenmiş olup, 900,000'den fazla film kaydı içermektedir. Veri CSV formatında saklanmakta ve her kayıt çoklu özellikler barındırmaktadır.

**Veri Seti Özellikleri**

ÖzellikAçıklamaVeri Tipimovie\_nameFilmin orijinal başlığıStringgenresFilm türleri (Action, Comedy, Drama, Thriller vb.)List\[String]ReviewsKullanıcı yorumları ve eleştiri metinleriTextRatings1-10 arası puanlamaFloatemotionFilmin genel duygu tonu (joy, sadness, anger, fear, anticipation, optimism)StringDescriptionFilm özeti ve kısa açıklamaText

Veri Ön İşleme Pipeline



**Veri Temizleme**



Eksik değerlerin (null/NaN) ortalama veya medyan değerlerle doldurulması

Duplicate kayıtların tespit edilip kaldırılması

Encoding hatalarının düzeltilmesi





**Metin Birleştirme**



Tüm film özellikleri (başlık, tür, duygu, rating, yorum) tek bir metin alanında birleştirildi

Her kayıt için text\_for\_embedding alanı oluşturuldu

Metin formatı standardize edildi





**Embedding Üretimi**



OpenAI text-embedding-3-small modeli kullanıldı

Her film için 1536 boyutlu vektör temsili üretildi

Toplam embedding süresi: yaklaşık 2-3 saat (900K kayıt için)





**Vektör İndeksleme**



FAISS kütüphanesi ile verimli similarity search için indeksleme yapıldı

IndexFlatL2 (L2 mesafe metriği) kullanıldı

İndeks ve embedding'ler pickle formatında kaydedildi



Kullanılan Teknolojiler ve Yöntemler

Teknik Altyapı ve Mimari

1\. Embedding Modeli

Model: OpenAI text-embedding-3-small

Teknik Özellikler:



Boyut: 1536 dimension

Maksimum token: 8191 token

Maliyet: $0.00002 per 1K token

Performans: state-of-the-art semantik benzerlik performansı



Kullanım Amacı:

Film metinlerini yüksek boyutlu vektör uzayında temsil ederek semantik benzerlik hesaplamaları yapmak. Bu sayede kullanıcı sorgusu ile filmler arasında anlam bazlı eşleştirme gerçekleştirilmektedir.

2\. Vektör Veritabanı - FAISS

Teknoloji: Facebook AI Similarity Search (FAISS)

Seçilen İndeks Yapısı:



IndexFlatL2: Brute-force L2 mesafe hesaplama

Arama karmaşıklığı: O(n)

Doğruluk: %100 (approximate değil, exact search)



Performans Özellikleri:



900K vektör üzerinde arama süresi: ~0.3-0.5 saniye

Bellek kullanımı: ~5.5 GB (900K × 1536 × 4 byte)

İndeks yükleme süresi: ~3-5 saniye



Alternatif İndeks Tipleri ile Karşılaştırma:



IndexIVFFlat: Daha hızlı ama %95 doğruluk

IndexHNSW: Çok hızlı ama daha fazla bellek



Projede doğruluk öncelikli olduğu için IndexFlatL2 tercih edilmiştir.

3\. Large Language Model (LLM)

Model: GPT-4o-mini

Konfigürasyon Parametreleri:



Temperature: 0.7 (dengeli yaratıcılık-tutarlılık)

Max tokens: 1000

Top-p: 0.9



Model Seçim Gerekçesi:



GPT-4o-mini, GPT-4'ün hafif versiyonu olup maliyet-performans dengesinde üstün

Türkçe dil desteği mükemmel

Bağlamsal anlama kapasitesi yüksek

API response time: 1-2 saniye



4\. RAG Framework - LangChain

Framework Bileşenleri:

a) ConversationalRetrievalChain:



Retriever ve LLM'i entegre eder

Sohbet geçmişini yönetir

Prompt engineering için template desteği



b) Retrieval Stratejisi - MMR (Maximum Marginal Relevance):

MMR algoritması iki kriteri optimize eder:



Relevance: Sorguyla en yüksek benzerlik

Diversity: Seçilen dokümanlar arası minimum benzerlik

