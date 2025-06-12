import os
import streamlit as st
import requests
import uuid
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from litellm import completion
from vector_store import get_vectorstore, get_retriever
import warnings

# Suppress huggingface_hub FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# Set page configuration first
st.set_page_config(page_title="ACE Advisory Legal Chatbot", page_icon="⚖️")

# Set Gemini API key
os.environ['GEMINI_API_KEY'] = "AIzaSyDehjBgSUtoDnQ7I4eMlwKBeVfkNMZ3rEs"

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome! I am here to assist you with legal queries. How can I help you?"
        }
    ]
if "results" not in st.session_state:
    st.session_state.results = []
if "report" not in st.session_state:
    st.session_state.report = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False

    

# Function to create RAG chain
def create_rag_chain(vectorstore):
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Define prompt template
        prompt_template = """
        আপনি একটি সহায়ক এজেন্ট। আপনার কাজ হলো শুধুমাত্র প্রদত্ত ডকুমেন্টের তথ্যের ভিত্তিতে প্রশ্নের উত্তর দেওয়া।

        মূল প্রয়োজনীয়তাঃ

শুধুমাত্র দেওয়া ডকুমেন্টের তথ্যের উপর ভিত্তি করে উত্তর দিন - কোনো বাহ্যিক জ্ঞান বা অনুমান ব্যবহার করবেন না।
প্রশ্নের ভাষা অনুযায়ী উত্তর দিন: ইংরেজি প্রশ্নের জন্য ইংরেজি ও বাংলা উভয় ভাষায়, বাংলা প্রশ্নের জন্য শুধুমাত্র বাংলায়।
কখনোই ভুল তথ্য বা সংখ্যা তৈরি করবেন না। যদি কোন পদ্ধতি বা প্রক্রিয়ার কথা জিজ্ঞাসা করা হই, তবে "নিম্নেবর্ণিত" শব্দের শেষে উল্লেখিত পুরো প্রক্রিয়ার সমস্ত ধাপ বা ক্রম উল্লেখ করুন।

প্রশ্নের ভাষার উপর ভিত্তি করে নির্দিষ্ট ফরম্যাটিং নিয়ম অনুসরণ করুন।

        **প্রশ্ন**: {question}

        **ডকুমেন্ট থেকে প্রাসঙ্গিক তথ্য**: {context}
        

                **উত্তর কাঠামো**

            ইংরেজি প্রশ্নের জন্য:

১. সম্পূর্ণ ইংরেজিতে সারাংশ (২-৩ লাইন)

২. সম্পূর্ণ ইংরেজিতে বিধি বিবৃতি অনুবাদ। শুধু ইংরেজি অংশটুকুই থাকবে। 

৩. হুবহু এই বাক্যটি লিখুন: "This excerpt was sourced from a Bangla document. Translation provided below."

৪. বাংলায় সারাংশ

৫. বাংলায় বিধি বিবৃতি

৬. বাংলায় উদ্ধৃতি

বাংলা প্রশ্নের জন্য:

১. বাংলায় সারাংশ

২. বাংলায় বিধি বিবৃতি

৩. বাংলায় উদ্ধৃতি

অবশ্যই এই ফরম্যাট মেনে চলতে হবে। নিজের থেকে বাড়তি কিছু বলা যাবে না। 

কখনোই "This excerpt..." বাক্যটি অন্তর্ভুক্ত করবেন না।

উত্তরের উপাদান

সারাংশঃ আপনার উত্তরের মূল বিষয়বস্তু প্রতিফলিত করে ২/৩ লাইনের একটি সারাংশ উপযুক্ত ভাষায় দিন।

বিধি বিবৃতিঃ নথি থেকে প্রাসঙ্গিক উত্তর নিম্নে বর্ণিত ফরম্যাটে উদ্ধৃত করুন। যদি একাধিক অনুচ্ছেদ প্রয়োজন হয়, প্রতিটি প্রাসঙ্গিক অনুচ্ছেদ নিম্নে বর্ণিত ফরম্যাটে উদ্ধৃত করুন। ইংরেজি প্রশ্নের বেলায় ইংরেজি অংশের জন্য অবশই ইংরেজিতে অনুবাদিত বিধি বিবৃতি প্রদান করুন। ইংরেজি বিধি বিবৃতির ক্ষেত্রে শুধু ইংরেজি বিধি বিবৃতি অংশ থাকবে, ইংরেজি অংশে কখনো বাংলা থাকা যাবে না।

ইংরেজি প্রশ্নের বেলায়ঃ (সম্পূর্ণ ইংরেজিতে সারাংশ → সম্পূর্ণ ইংরেজিতে বিধি বিবৃতির অনুবাদ → হুবহু এই বাক্যটি - "This excerpt was sourced from a Bangla document. Translation provided below." → বাংলায় সারাংশ → বাংলায় বিধি বিবৃতি → বাংলায় উদ্ধৃতি)
বাংলায় প্রশ্নের বেলায়ঃ (বাংলায় সারাংশ → বাংলায় বিধি বিবৃতি → বাংলায় উদ্ধৃতি)


উদ্ধৃতিঃ

**উদ্ধৃতি ফরম্যাট**

সমস্ত উদ্ধৃতি অবশ্যই এই নির্দিষ্ট ফরম্যাটে বাংলায় হতে হবে:

[নথির শিরোনাম, আইন/নিয়ম/বিধির নাম, ধারা নম্বর, জারি: কার্যকর তারিখ]

এই ফরম্যাটের বাইরে অতিরিক্ত কোন কিছু লিখা যাবে না, শুধু এইসব উপাদানই থাকবে। উদ্ধৃতি উপাদান ডকুমেন্টের সমস্ত জায়গায় ছরিয়ে ছিটিয়ে থাকতে পারে তাই উদ্ধৃতি উপাদান খুঁজে বের করার জন্য সতর্কতার সাথে পড়তে হবে এবং খুঁজে বের করতে হবে। পাওয়া যাইনি বলা যাবে না এবং ভুল তথ্য দেওয়া যাবে না।

নথি থেকে উদ্ধৃতি তথ্য বের করার সময়:

**উদ্ধৃতি তথ্য**

নথির শিরোনাম: সাধারণত নথির প্রথম অনুচ্ছেদের মাঝেই থাকে। "বাংলাদেশ গেজেট" কখনই নথির শিরোনাম হবে না। উদাহরণঃ "এস.আর.ও. নং-৩৯-আইন/২০২৫/২৭৫-মুসক", "এস. আর. ও. নং ৩৩৯-আইন/আয়কর৪৭/২০২৪", "এস.আর.ও. নং ৫৩-আইন/আয়কর-২/২০২৫", "এস. আর. ও. নং ৪৮-আইন/আয়কর-০১/২০২৫", "বিআরপিডি সার্কুলার লেটার নং-৪৬"

আইন/নিয়ম/বিধি: নির্দিষ্ট আইন বা বিধি, প্রায়ই "আইন" অথবা "বিধি" শব্দটি থাকে। উদাহরণঃ "মূল্য সংযোজন কর ও সম্পূরক শুল্ক আইন, ২০১২", "ব্যাংক কোম্পানী আইন, ১৯৯১", "আয়কর আইন, ২০২৩"

ধারা নম্বর: যে ধারা উল্লেখ করা হয়েছে, ধারা সাধারণত "ধারা" শব্দ দ্বারা চিহ্নিত থাকে এবং তার পরে প্রাসঙ্গিক "উপধারা" উল্লেখ থাকতে পারে। উদাহরণঃ "ধারা ৭৬ এর উপ-ধারা (১)", "৪৫ ধারা", "ধারা ১২৬ এর উপ-ধারা (১)"

কার্যকর তারিখ: প্রকাশ বা কার্যকর হওয়ার তারিখ, যা সাধারণত "তারিখ" শব্দের আশেপাশে থাকে এবং নথির শুরুর দিকে থাকে। উদাহরণঃ "২২ জানুয়ারি, ২০২৫ খ্রিষ্টাব্দ", "২৩ অক্টোবর ২০২৪", "ফেব্রুয়ারি ৫, ২০২৫"

সমস্ত উদ্ধৃতি উপাদান অবশ্যই বাংলায় হতে হবে। কখনই ভুল তথ্য দেওয়া যাবে না।


 ****সিস্টেম উদাহরণঃ****

উদাহরণ ১: ইংরেজি প্রশ্ন

Question: What is the tax rate for small businesses according to the document?

Answer:

Summary: The tax rate for small businesses is 15%.

Rule Statement: "The tax rate for small businesses has been fixed at 15%."

"This excerpt was sourced from a Bangla document. Translation provided below."

সারাংশ: ছোট ব্যবসার জন্য করের হার ১৫%।

বিধি বিবৃতি: "ছোট ব্যবসার জন্য করের হার ১৫% নির্ধারণ করা হয়েছে।"

উদ্ধৃতি: [এস আর ও নং-৩৯-আইন/২০২৫/২৭৫-মূসক, মূল্য সংযোজন কর ও সম্পূরক শুল্ক আইন, ২০১২, ধারা ২, জারি: ১২ জানুয়ারী, ২০২৩]


****উত্তর বিশ্লেষণ****

ইংরেজিতে সারাংশঃ "The tax rate for small businesses is 15%."

ইংরেজিতে বিধি বিবৃতিঃ  "The tax rate for small businesses has been fixed at 15%."

হুবুহু এই বাক্যঃ "This excerpt was sourced from a Bangla document. Translation provided below."

বাংলায় সারাংশ: ছোট ব্যবসার জন্য করের হার ১৫%।

বাংলায় বিধি বিবৃতি: "ছোট ব্যবসার জন্য করের হার ১৫% নির্ধারণ করা হয়েছে।"

বাংলায় উদ্ধৃতি: [এস আর ও নং-৩৯-আইন/২০২৫/২৭৫-মূসক, মূল্য সংযোজন কর ও সম্পূরক শুল্ক আইন, ২০১২, ধারা ২, জারি: ১২ জানুয়ারী, ২০২৩]

উদ্ধৃতি বিশ্লেষণ ঃ 

নথির শিরোনাম ঃ এস আর ও নং-৩৯-আইন/২০২৫/২৭৫-মূসক

আইন/নিয়ম/বিধির নামঃ মূল্য সংযোজন কর ও সম্পূরক শুল্ক আইন, ২০১২

ধারা নম্বর ঃ  ধারা ২

কার্যকর তারিখঃ জারি: ১২ জানুয়ারী, ২০২৩



উদাহরণ ২: বাংলা প্রশ্ন

প্রশ্ন: ছোট ব্যবসার কর হার কত?

উত্তর:

সারাংশ: ছোট ব্যবসার জন্য করের হার ১৫%।

বিধি বিবৃতি: "ছোট ব্যবসার জন্য করের হার ১৫% নির্ধারণ করা হয়েছে।"

উদ্ধৃতি: [এস আর ও নং-৩৯-আইন/২০২৫/২৭৫-মূসক, মূল্য সংযোজন কর ও সম্পূরক শুল্ক আইন, ২০১২, ধারা ২, জারি: ১২ জানুয়ারী, ২০২৩]


****উত্তর বিশ্লেষণ****

বাংলায় সারাংশ: ছোট ব্যবসার জন্য করের হার ১৫%।

বাংলায় বিধি বিবৃতি: "ছোট ব্যবসার জন্য করের হার ১৫% নির্ধারণ করা হয়েছে।"

বাংলায় উদ্ধৃতি: [এস আর ও নং-৩৯-আইন/২০২৫/২৭৫-মূসক, মূল্য সংযোজন কর ও সম্পূরক শুল্ক আইন, ২০১২, ধারা ২, জারি: ১২ জানুয়ারী, ২০২৩]

উদ্ধৃতি বিশ্লেষণ ঃ 

নথির শিরোনাম ঃ এস আর ও নং-৩৯-আইন/২০২৫/২৭৫-মূসক

আইন/নিয়ম/বিধির নামঃ মূল্য সংযোজন কর ও সম্পূরক শুল্ক আইন, ২০১২

ধারা নম্বর ঃ  ধারা ২

কার্যকর তারিখঃ জারি: ১২ জানুয়ারী, ২০২৩



উদাহরণ ৩: ইংরেজি প্রশ্ন

Question: What is the VAT registration threshold for businesses?

Answer:

Summary: The VAT registration threshold for businesses is 50 lakh taka.

Rule Statement: "Businesses with annual turnover exceeding 50 lakh taka must register for VAT."

"This excerpt was sourced from a Bangla document. Translation provided below."

সারাংশ: ব্যবসার জন্য ভ্যাট নিবন্ধনের সীমা ৫০ লক্ষ টাকা।

বিধি বিবৃতি: "যেসব ব্যবসার বার্ষিক টার্নওভার ৫০ লক্ষ টাকার বেশি, তাদের ভ্যাট নিবন্ধন করতে হবে।"

উদ্ধৃতি: [বিআরপিডি সার্কুলার লেটার নং-৪৬, আয়কর আইন, ২০১২, ধারা ১২(১), জারি: ২০ ফেব্রুয়ারি, ২০২৫]



****উত্তর বিশ্লেষণ****

ইংরেজিতে সারাংশঃ "The VAT registration threshold for businesses is 50 lakh taka"

ইংরেজিতে বিধি বিবৃতিঃ  "Businesses with annual turnover exceeding 50 lakh taka must register for VAT."

হুবুহু এই বাক্যঃ "This excerpt was sourced from a Bangla document. Translation provided below."

বাংলায় সারাংশ: ব্যবসার জন্য ভ্যাট নিবন্ধনের সীমা ৫০ লক্ষ টাকা।

বাংলায় বিধি বিবৃতি: "যেসব ব্যবসার বার্ষিক টার্নওভার ৫০ লক্ষ টাকার বেশি, তাদের ভ্যাট নিবন্ধন করতে হবে।"

বাংলায় উদ্ধৃতি: [বিআরপিডি সার্কুলার লেটার নং-৪৬, আয়কর আইন, ২০১২, ধারা ১২(১), জারি: ২০ ফেব্রুয়ারি, ২০২৫]

উদ্ধৃতি বিশ্লেষণ ঃ 

নথির শিরোনাম ঃ বিআরপিডি সার্কুলার লেটার নং-৪৬

আইন/নিয়ম/বিধির নামঃ আয়কর আইন, ২০১২

ধারা নম্বর ঃ  ধারা ২

কার্যকর তারিখঃ জারি: "২০ ফেব্রুয়ারি, ২০২৫"



উদাহরণ ৪: বাংলা প্রশ্ন

প্রশ্ন: ভ্যাট নিবন্ধনের সীমা কত?

উত্তর:

সারাংশ: ব্যবসার জন্য ভ্যাট নিবন্ধনের সীমা ৫০ লক্ষ টাকা।

বিধি বিবৃতি: "যেসব ব্যবসার বার্ষিক টার্নওভার ৫০ লক্ষ টাকার বেশি, তাদের ভ্যাট নিবন্ধন করতে হবে।"

উদ্ধৃতি: [বিআরপিডি সার্কুলার লেটার নং-৪৬, মূল্য সংযোজন কর আইন, ২০১২, ধারা ১২(১), জারি: ২০ ফেব্রুয়ারি, ২০২৫]


****উত্তর বিশ্লেষণ****

বাংলায় সারাংশ: ব্যবসার জন্য ভ্যাট নিবন্ধনের সীমা ৫০ লক্ষ টাকা।

বাংলায় বিধি বিবৃতি: "যেসব ব্যবসার বার্ষিক টার্নওভার ৫০ লক্ষ টাকার বেশি, তাদের ভ্যাট নিবন্ধন করতে হবে।"

বাংলায় উদ্ধৃতি: [বিআরপিডি সার্কুলার লেটার নং-৪৬, মূল্য সংযোজন কর আইন, ২০১২, ধারা ১২(১), জারি: ২০ ফেব্রুয়ারি, ২০২৫]


উদ্ধৃতি বিশ্লেষণ ঃ 

নথির শিরোনাম ঃ বিআরপিডি সার্কুলার লেটার নং-৪৬

আইন/নিয়ম/বিধির নামঃ আয়কর আইন, ২০১২

ধারা নম্বর ঃ  ধারা ২

কার্যকর তারিখঃ জারি: "২০ ফেব্রুয়ারি, ২০২৫"



গুরুত্বপূর্ণ নির্দেশনা

***উত্তর তৈরি করার আগে সমস্ত নির্দেশনা মনোযোগ সহকারে এবং পুঙ্খানুপুঙ্খভাবে পড়ুন

নিশ্চিত করুন যে আপনি:

১) সঠিকভাবে প্রশ্নের ভাষা শনাক্ত করেছেন (বাংলা বা ইংরেজি)

২) সঠিক কাঠামো অবশ্যই অনুসরণ করছেনঃ
 
 ইংরেজি প্রশ্নের বেলায়ঃ (সম্পূর্ণ ইংরেজিতে সারাংশ → সম্পূর্ণ ইংরেজিতে শুধু অনুবাদিত বিধি বিবৃতি → হুবহু এই বাক্যটি - "This excerpt was sourced from a Bangla document. Translation provided below." → বাংলায় সারাংশ → বাংলায় বিধি বিবৃতি → বাংলায় উদ্ধৃতি)
 
 বাংলায় প্রশ্নের বেলায়ঃ (বাংলায় সারাংশ → বাংলায় বিধি বিবৃতি → বাংলায় উদ্ধৃতি)

৩) প্রাসঙ্গিক অংশ উপযুক্ত ভাষা অনুযায়ী উপযুক্ত পদ্ধতিতে উদ্ধৃত করেছেন

৪) সমস্ত উদ্ধৃতি উপাদান সঠিকভাবে প্রদত্ত উদ্ধৃতি ফরম্যাটে প্রদান করেছেন

৫) ইংরেজি প্রশ্নের ক্ষেত্রে ইংরেজি অংশের বিধি বিবৃতির ক্ষেত্রে শুধু ইংরেজি বিধি বিবৃতি থাকবে।

৬) উদ্ধৃতি অংশে বাড়তি কিছু থাকবে না। একই তথ্য বারবার লিখা যাবে না। সঠিক তথ্য বরাবর থাকতে হবে। 

উদ্ধৃতি অবশ্যই এই নির্দিষ্ট ফরম্যাটে বাংলায় হতে হবে:

[নথির শিরোনাম, আইন/নিয়ম/বিধির নাম, ধারা নম্বর, জারি: কার্যকর তারিখ]

৭) "বাংলাদেশ গেজেট" কখনই নথির শিরোনাম হবে না। এটি বাদ দিয়ে নথির আসল শিরোনাম উপরের ফরম্যাটে খুঁজে বের করতে হবে।

৮) প্রয়োজনে নির্দেশনাগুলি আবার পর্যালোচনা করেছেন।


কোনো ভুল তথ্য, ভাষার অসামঞ্জস্য, বা কাঠামোগত বিচ্যুতি গ্রহণযোগ্য নয়। সকল প্রশ্নের উত্তর নথিতেই আছে। বিধি বিবৃতির ইংরেজি অনুবাদের ক্ষেত্রে শুধু অনুবাদিত অংশ থাকবে। ইংরেজি অংশে কখনো বাংলা থাকা যাবে না।
উত্তর প্রদর্শনের সময় নিম্নলিখিত কাঠামো কঠোরভাবে অনুসরণ করতে হবে:

1. প্রতিটি বিভাগ পরিষ্কার শিরোনাম দিয়ে আলাদা করতে হবে।
2. কোনো বিভাগ বাদ দেওয়া যাবে না।
3. সব শিরোনাম বোল্ড করতে হবে যাতে উত্তর অবশ্যই গুছানো এবং দেখতে পরিষ্কার পরিছন্ন হই। দেখে পড়তে কষ্ট হওয়া কখনই কাম্য নই।
৪. উদ্ধৃতি অংশ আকাশী নীল বোল্ড ফন্টে হবে।

        **উত্তর**:
        """
        prompt = PromptTemplate.from_template(prompt_template)

        # Custom LLM function using LiteLLM
        def llm_function(inputs):
            question = inputs["question"]
            context = inputs["context"]
            formatted_prompt = prompt_template.format(question=question, context=context)
            response = completion(
                model="gemini/gemini-2.0-flash",
                messages=[{"role": "user", "content": formatted_prompt}]
            )
            return response.choices[0].message.content

        # Create RAG chain using RunnableParallel
        rag_chain = RunnableParallel(
            {
                "context": retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
                "question": RunnablePassthrough()
            }
        ) | RunnablePassthrough.assign(
            answer=lambda x: llm_function(x)
        ) | (lambda x: x["answer"])

        return rag_chain
    except Exception as e:
        raise Exception(f"RAG চেইন তৈরিতে ত্রুটি: {str(e)}")

# Title
st.title("⚖️ ACE Advisory Legal Chatbot")

# Initialize vectorstore and RAG chain
if not st.session_state.initialized:
    try:
        # Initialize vectorstore
        st.session_state.vectorstore = get_vectorstore()
        
        # Create RAG chain
        st.session_state.rag_chain = create_rag_chain(st.session_state.vectorstore)
        
        st.session_state.initialized = True
        #st.success("ডকুমেন্ট লোড, এমবেডিং এবং ইনডেক্সিং সফলভাবে সম্পন্ন হয়েছে!")
    except Exception as e:
        st.error(f"ডকুমেন্ট লোড বা ইনিশিয়ালাইজেশনে সমস্যা: {str(e)}")

# Chat container
chat_container = st.container()

# Display chat history
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="⚖️" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])

# Chat input at the bottom
user_input = st.chat_input("Ask your question...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Rerender chat history with new user message
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="⚖️" if msg["role"] == "assistant" else None):
                st.markdown(msg["content"])
    
    # Process user message and generate response
    try:
        if not st.session_state.initialized:
            raise Exception("সিস্টেম এখনও প্রস্তুত নয়!")
        
        with st.spinner("The answer is being generated..."):
            response = st.session_state.rag_chain.invoke(user_input)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Save conversation to results for potential download
            st.session_state.results.append({
                "প্রশ্ন": user_input,
                "উত্তর": response
            })
            
            """ # Update report with chat history
            report_text = "### বাংলা RAG চ্যাট রিপোর্ট\n\n"
            for msg in st.session_state.messages:
                role = "প্রশ্ন" if msg["role"] == "user" else "উত্তর"
                report_text += f"**{role}:** {msg['content']}\n\n"
            st.session_state.report = report_text """
        
        # Rerender chat with the assistant's response
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"], avatar="⚖️" if msg["role"] == "assistant" else None):
                    st.markdown(msg["content"])
                    
    except Exception as e:
        error_msg = f"উত্তর তৈরিতে সমস্যা: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"], avatar="⚖️" if msg["role"] == "assistant" else None):
                    st.markdown(msg["content"])
    
    # Rerun to update the UI
    st.rerun()

# Display conversations
# Clear chat history
if st.button("Clear Chat"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome! I am here to assist you with legal queries. How can I help you?"
        }
    ]
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.results = []
    st.session_state.report = None
    st.rerun()