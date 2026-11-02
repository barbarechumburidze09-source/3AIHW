import os
import psycopg

from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_core.messages import SystemMessage, HumanMessage


# ==========================================
# 1. გარემოს ცვლადების ჩატვირთვა
# ==========================================

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY ვერ მოიძებნა .env ფაილში")

if not POSTGRES_PASSWORD:
    raise ValueError("POSTGRES_PASSWORD ვერ მოიძებნა .env ფაილში")


# ==========================================
# 2. PostgreSQL მონაცემთა ბაზასთან კავშირი
# ==========================================

DB_CONFIG = {
    "dbname": "hotel_db",
    "user": "postgres",
    "password": POSTGRES_PASSWORD,
    "host": "localhost",
    "port": 5432,
}


# ==========================================
# 3. სასტუმროს ინფორმაცია
# ==========================================

hotel_text = """
სასტუმრო ალპური სახლი მდებარეობს ყაზბეგში.

ნომრების ფასები:
სტანდარტული ნომერი გათვლილია 2 ადამიანზე და ღირს 120 ლარი ერთ ღამეში.
საოჯახო ნომერი გათვლილია 4 ადამიანზე და ღირს 200 ლარი ერთ ღამეში.
ლუქს ნომერი გათვლილია 2 ადამიანზე და ღირს 250 ლარი ერთ ღამეში.

ყველა ნომრის ფასში შედის საუზმე.
დამატებითი საწოლის ღირებულებაა 40 ლარი.

სასტუმროში არის უფასო Wi-Fi, უფასო პარკინგი და საერთო დასასვენებელი სივრცე.

სასტუმროს რესტორანში მზადდება ქართული კერძები.
ვახშამი ნომრის ფასში არ შედის.

შინაური ცხოველების მიყვანა შესაძლებელია მხოლოდ ადმინისტრაციასთან წინასწარი შეთანხმებით.

შესვლის დროა 15:00 საათი.
გასვლის დროა 11:00 საათი.
ადრეული შესვლის შემთხვევაში სტუმარმა უნდა დაუკავშირდეს ადმინისტრაციას.

უფასო გაუქმება შესაძლებელია ჩამოსვლამდე მინიმუმ 48 საათით ადრე.
თუ ჯავშანი გაუქმდა ჩამოსვლამდე 48 საათზე ნაკლები დროით ადრე,
პირველი ღამის ღირებულება არ ბრუნდება.

სასტუმრო სტუმრებს სთავაზობს ლაშქრობებს და ადგილობრივი გიდის მომსახურებას.
შესაძლებელია გერგეტის სამების მონახულება.
"""


# ==========================================
# 4. ტექსტის ჩანკებად დაყოფა
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

chunks = splitter.split_text(hotel_text)

print(f"სულ შეიქმნა {len(chunks)} ჩანკი.")


# ==========================================
# 5. Embedding მოდელის შექმნა
# ==========================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

print("მიმდინარეობს ჩანკების ვექტორიზაცია...")

chunk_embeddings = embeddings.embed_documents(chunks)

print("ვექტორიზაცია დასრულდა!")


# ==========================================
# 6. მონაცემების PostgreSQL-ში შენახვა
# ==========================================

print("მიმდინარეობს მონაცემების PostgreSQL-ში შენახვა...")

with psycopg.connect(**DB_CONFIG) as conn:

    register_vector(conn)

    with conn.cursor() as cur:

        cur.execute("DELETE FROM hotel_documents")

        for chunk, embedding in zip(chunks, chunk_embeddings):

            cur.execute(
                """
                INSERT INTO hotel_documents (content, embedding)
                VALUES (%s, %s)
                """,
                (chunk, Vector(embedding))
            )

print("მონაცემები წარმატებით შეინახა PostgreSQL-ში!")


# ==========================================
# 7. PostgreSQL-იდან შესაბამისი ჩანკების მოძებნა
# ==========================================

def retrieve_chunks(question):

    question_embedding = embeddings.embed_query(question)

    question_embedding = Vector(question_embedding)

    with psycopg.connect(**DB_CONFIG) as conn:

        register_vector(conn)

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT content,
                       embedding <=> %s AS distance
                FROM hotel_documents
                ORDER BY embedding <=> %s
                LIMIT 3
                """,
                (question_embedding, question_embedding)
            )

            results = cur.fetchall()

    return results


# ==========================================
# 8. Gemini-ს საშუალებით პასუხის გენერირება
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)


def ask_question(question):

    results = retrieve_chunks(question)

    context = "\n\n".join(
        [row[0] for row in results]
    )

    messages = [
        SystemMessage(
            content=f"""
შენ ხარ სასტუმრო ალპური სახლის ასისტენტი.

უპასუხე მომხმარებელს მხოლოდ ქვემოთ მოცემული კონტექსტის
საფუძველზე.

თუ პასუხი კონტექსტში არ არის, უთხარი მომხმარებელს,
რომ ეს ინფორმაცია არ გაქვს.

უპასუხე ქართულ ენაზე.

კონტექსტი:
{context}
"""
        ),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages)

    return "".join(
    item["text"]
    for item in response.content
    if isinstance(item, dict) and item.get("type") == "text"
)
if __name__ == "__main__":
    print("სასტუმრო ალპური სახლის ასისტენტი")
    print("კითხვების დასასრულებლად აკრიფე 'exit'.")

    while True:
        question = input("\nშენი კითხვა: ")

        if question.lower() == "exit":
            print("ნახვამდის!")
            break

        try:
            answer = ask_question(question)
            print("\nპასუხი:", answer)
        except Exception as error:
            print("შეცდომა:", error)

