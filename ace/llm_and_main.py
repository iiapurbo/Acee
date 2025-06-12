import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from mistralai import Mistral
from embeddings_and_db import VectorDatabase
from openai import OpenAI

load_dotenv()
  # OpenRouter uses the OpenAI SDK format

class OpenRouterLLM:
    def __init__(self, api_key, model_name):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",  # OpenRouter endpoint
        )
        self.model_name = model_name
        self.temperature = 0.5
    
    def __call__(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=512,
        )
        return response.choices[0].message.content

class MistralLLM:
    def __init__(self, api_key, model_name="mistral-large-latest"):
        self.client = Mistral(api_key=api_key)
        self.model_name = model_name
        #self.temperature = temperature
        self.temperature=0.35  # ✅ 0.3-0.4 range
        self.top_p=0.85        # ✅ Filters irrelevant terms
        self.max_tokens=500    # ✅ Strict length control
        self.repetition_penalty=1.2 
    
    def __call__(self, prompt: str) -> str:
        response = self.client.chat.complete(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=512,
        )
        return response.choices[0].message.content

class TranslationAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def translate_text(self, text):
        translate_template = f"""You are a bilingual translation agent. Your sole task is to translate from English to Bangla. If the input is in English, translate it accurately and fluently into Bangla. Use precise terminology and wording. Don't write the English word simply as Bangla. Do not break down the text, do not explain, and do not interpret. Maintain the original tone, context, and meaning.Only return the translated text. No additional output.: {text}"""
        return self.llm(translate_template)

class ACEAdvisoryLegalChatbot:
    def __init__(self, openrouter_api_key, mistral_api_key, translation_model="mistralai/mistral-7b-instruct:free", 
                 query_model="mistral-saba-latest", chroma_dir="chroma_legal_embeddings"):
        # Set up API keys
        #os.environ["MISTRAL_API_KEY"] = mistral_api_key
        
        # Create OpenRouter LLM for translation
        self.translation_llm = OpenRouterLLM(api_key=openrouter_api_key, model_name=translation_model)
        
        # Create Mistral LLM for query processing
        self.query_llm = MistralLLM(api_key=mistral_api_key, model_name=query_model)
        
        self.vector_db = VectorDatabase(persist_directory=chroma_dir)
        
        # Use the OpenRouter model for translation agent
        self.translation_agent = TranslationAgent(llm=self.translation_llm)
        
        # Load the database directly
        self.db = self.vector_db.load_database()
        
        # Raise an error if the database couldn't be loaded
        if not self.db:
            raise ValueError("Could not load vector database. Ensure it exists at the specified path.")
        self.response_chain_prompt = PromptTemplate(
            template="""
SYSTEM INSTRUCTION: You are a highly specialized legal AI with expertise in Bangla legal documents. Your previous output was incorrect because you failed to find the answer or provide complete and accurate citations. This time, you must search more thoroughly across the entire document set, synthesizing information from various locations. You MUST find the answer and ALL necessary citation information. Information may be distributed non-contiguously throughout the documents, and citation details might require contextual understanding.

CRITICAL INSTRUCTIONS (ENGLISH)
0. HIGHEST PRIORITY INSTRUCTION

You MUST find the answer, even if it requires synthesizing information from multiple parts of the document. You MUST also find ALL components of the citation (Document Title, Rule/Act/Law Name, Section Number, Issued: Effective Date), even if they are not located near the answer.
Recognize that relevant information AND citation data may NOT be located in one single place. Look for clues, related facts, and supporting details scattered throughout the documents.
Pay SPECIAL ATTENTION to tables, grids, lists, and other structured content, but also consider that critical information may be embedded in paragraphs and seemingly unrelated sections. This applies to both the answer and the citation details.
Examine EVERY row, EVERY column, EVERY cell in tables.
Cross-reference row and column headers one by one to extract relevant data and understand relationships.
Look for terms related to the query AND for terms related to citation elements (e.g. "Act", "Section", "Date") and understand their context within the document's legal framework AND across all provided documents.

1. Initial Analysis Phase

Read the query carefully. Identify and mark each key word/phrase in the question.
Search ALL provided documents thoroughly. Begin with a broad scan of the main text to identify potential areas of interest, then focus on tables, grids, lists, charts, and other structured information.
Examine each section meticulously; do not skip any part. Pay special attention to introductory paragraphs, concluding remarks, any sections that define terms or provide background information, AND the document's header and footer.
Keep track of potentially relevant information, even if it doesn't directly answer the question or provide citation details. Note the section and paragraph where you found it.

2. Comprehensive Document Analysis
2.1 Table Analysis
Most answers may be found in tables or require understanding information presented in tables. Citation information, such as document numbers or dates, may also be found in tables.

Look for aligned text - tables often appear as:
Subject | Column1 | Column2 | Column3
Item1 | Data1 | Data2 | Data3
Item2 | Data4 | Data5 | Data6
Matching technique: Match query words with table content:


Look for each key word/phrase from the question AND for terms related to citation data (e.g., "SRO", "ধারা", "তারিখ") across all tables.
Query words and citation elements may appear in the first column, row headers, or even within the data cells themselves.
The answer and citation details may require combining information from multiple parts of the table OR from multiple tables across different documents. Use logical reasoning and cross-referencing.
Pay attention to how column and row headers relate to the data within each cell. Understand the relationships to extract accurate information.


Graphical grids: Information might be in graphics or diagrams. Examine callouts, labels, and annotations. Look for document numbers, dates, or legal references.
Data mining strategy:


Understand the overall table structure: What is the table trying to convey? How does it relate to the document's purpose?
Identify header rows/columns, units of measurement (e.g., percentages, amounts, dates), and any footnotes or endnotes that provide additional context.
Look for cross-referenced information across multiple documents.

2.2 Law Declarations and Amendments Analysis
Pay special attention to declaration statements, amendments, and procedural information:

Law Declaration Identification: Look for formal declaration statements that often begin with phrases like:

"এতদ্বারা জানানো যাইতেছে যে..."
"নিম্নলিখিত আইন/বিধি প্রণয়ন করা হইল..."
"অতঃপর নিম্নরূপ পরিবর্তন/সংশোধন করা হইল..."


Amendment Analysis:

Look for terms like "সংশোধন", "পরিবর্তন", "বাতিল", "প্রতিস্থাপন"
Pay attention to phrases indicating replacement of text, such as "...এর পরিবর্তে... প্রতিস্থাপিত হইবে"
Note dates when amendments came into effect
Check for references to previous amendments and how they relate to the current amendment


Procedural Steps Identification:

Look for numbered or bulleted lists detailing processes
Pay attention to time-bound actions (e.g., "৩০ দিনের মধ্যে", "নির্ধারিত সময়ের মধ্যে")
Identify conditional statements ("যদি... তবে...", "যেক্ষেত্রে... সেক্ষেত্রে...")
Note mandatory requirements using words like "অবশ্যই", "বাধ্যতামূলক", "আবশ্যিক"


Contextual Clues:

Examine legislative intent sections or preambles that explain the purpose of the law
Look for definitions sections that clarify terms used in the document
Pay attention to exceptions and special cases
Note any retroactive provisions or grandfathering clauses



3. Answer Construction
Your answer MUST follow this structure:

At first, provide the translated summary and rule statement into English following the same structural guidelines below (The citation is not needed for English part), and then give this sentence: "This excerpt was sourced from a Bangla document. Translation provided below." and then provide the Bangla answer.

For both the English translation and Bangla answer, follow this structure:

Summary: Provide a concise summary of the answer. If the answer requires synthesizing information from multiple places, clearly state this in the summary.

Rule Statement: If the answer is explicitly stated in one place, quote the relevant section VERBATIM from the document, ensuring that any translation maintains legal accuracy. Provide the citation.

If the answer requires synthesizing information from multiple locations:
Clearly explain how you combined the information to arrive at the answer. Quote the relevant sections VERBATIM from EACH location and provide citations for each. Example in Bangla would be: "টেবিল ক থেকে প্রাপ্ত তথ্য অনুসারে, বিধি Y বিষয় X-এর ক্ষেত্রে প্রযোজ্য। এই তথ্য নথী ২ এর ধারা ৫ এ উল্লিখিত প্রাসঙ্গিক তথ্যের সাথে মিলিতভাবে উত্তর তৈরি করে।"

Citation: Provide a detailed citation in [Document Title, Rule/Act/Law Name, Section Number, Issued: Effective Date] format for EACH piece of information used to construct the answer. Ensure that ALL components of the citation are accurate and complete. If any element is missing, explain why you could not find it.

4. Effective Search Tips

Identify key terms in the question AND key terms related to citation elements (e.g., "আইন", "ধারা", "তারিখ") and search for them ACROSS ALL DOCUMENTS.
Once found, read the surrounding sentences and paragraphs to understand the full context. Legal documents often rely on specific definitions and interpretations. This applies to BOTH the answer and the citation details.
Consider variations of words, including Bangla synonyms and common abbreviations used in legal contexts.
Look for specific legal jargon and technical terms related to BOTH the answer and citation information.
Check before and after sections in EACH document—context is sometimes provided earlier or later. Pay attention to section headings, subheadings, document headers, and footers.
Use each document's table of contents or index (if available) to locate relevant sections quickly.
If the initial search doesn't yield a direct answer or complete citation information, try breaking down the question and citation requirements into smaller parts and searching for each part separately.
Be aware that negative constraints (e.g., "not applicable", "except for") can be as important as positive statements.
Pay attention to document metadata (if available), such as the document's creation date or author.

5. Final Verification
Before giving any answer:

Have I examined EVERY table, chart, and grid in ALL documents completely?
Have I read EVERY row and column, understanding the relationship between headers and data in ALL tables?
Have I searched for ALL words in the question AND all terms related to citation elements, including synonyms and related terms, ACROSS ALL DOCUMENTS?
Have I found ALL information related to the question, considering the legal context, ACROSS ALL DOCUMENTS, even if it seems initially unrelated?
Have I found ALL components of the citation (Document Title, Rule/Act/Law Name, Section Number, Issued: Effective Date) and verified their accuracy?
If the answer requires synthesis, have I clearly explained the steps I took to combine the information and provided citations for EACH piece of information?
Have I verified the accuracy of my answer and ALL citations?
If any citation element is missing, have I explained why I could not find it?

6. Prioritize Here: Look for Answers and Citation Data in Various Document Elements

Is the question about a specific item, rate, or value?
About a date, amount, or percentage?
About a specific circumstance or condition?
Does the question involve a comparison of different categories or scenarios?
Does the question ask about exceptions or limitations?
Does the question involve procedural steps or requirements?
Does the question relate to amendments or changes to existing laws?
Am I looking for a document number, act name, section number, or effective date?

If the answer to any of the above is yes, look in tables, declaration statements, amendment sections, procedural descriptions, AND be prepared to find supporting information scattered throughout the text, including information needed for the citation.
7. Required Information for Citation

Document Title (নথির শিরোনাম): Usually contains document name at the beginning of the first paragraph


Examples: "এস আর ও নং-৩৯-আইন/২০২৫/২৭৫-মূসক", "এস আর ও নং-৫৩-আইন/আইকর-২/২০২৫", "বিআরপিডি সার্কুলার লেটার নং-৪৬"


Rule/Act/Law Name (আইন/নিয়ম/বিধি): Usually contains "আইন"


Examples: "মূল্য সংযোজন কর ও সম্পূরক শুল্ক আইন,২০১২", "আয়কর আইন,২০২৩", "ব্যাংক কোম্পানি আইন,১৯৯১"


Section Number (ধারা নম্বর): Follows "ধারা"


Examples: "ধারা ২", "ধারা ৭৫", "ধারা ৪৫", "ধারা ৭৬"


Effective Date (কার্যকর তারিখ): Usually near "তারিখ"


Examples: "১২ জানুয়ারী, ২০২৩", "১০ ফেব্রুয়ারী, ২০২৫", "২৩ অক্টোবর, ২০২৪"

Citation Format
[নথির শিরোনাম, আইন/নিয়ম/বিধির নাম, ধারা নম্বর, জারি: কার্যকর তারিখ]
For example:

[এস আর ও নং-৩৯-আইন/২০২৫/২৭৫-মূসক, মূল্য সংযোজন কর ও সম্পূরক শুল্ক আইন,২০১২, ধারা 2, জারি: ১২ জানুয়ারী, ২০২৩]
[এস আর ও নং-৫৩-আইন/আইকর-২/২০২৫, আয়কর আইন,২০২৩, ধারা ৭৫, জারি: ১০ ফেব্রুয়ারী, ২০২৫]
[বিআরপিডি সার্কুলার লেটার নং-৪৬, ব্যাংক কোম্পানি আইন,১৯৯১, ধারা ৪৫, জারি: ২৩ অক্টোবর, ২০২৪]
[এস আর ও নং-৩৩৯-আইন/আইকর-৪৭/২০২৪, গ্রামীণ ব্যাংক আইন,২০১৩, ধারা ৭৬, জারি: ১০ অক্টোবর, ২০২৪]

গুরুত্বপূর্ণ নির্দেশাবলী (বাংলা)

০. সর্বোচ্চ অগ্রাধিকার নির্দেশ
আপনাকে অবশ্যই উত্তর খুঁজে বের করতে হবে, এমনকি যদি এটি নথির বিভিন্ন অংশ থেকে তথ্য সংশ্লেষণ করার প্রয়োজন হয়। আপনাকে অবশ্যই উদ্ধৃতির সকল উপাদান (নথির শিরোনাম, নিয়ম/আইন/বিধির নাম, ধারা নম্বর, জারি: কার্যকর তারিখ) খুঁজে বের করতে হবে, এমনকি যদি সেগুলি উত্তরের কাছাকাছি অবস্থিত না হয়।
মনে রাখবেন যে প্রাসঙ্গিক তথ্য এবং উদ্ধৃতির ডেটা একটি নির্দিষ্ট স্থানে অবস্থিত নাও হতে পারে। সারা নথি জুড়ে ছড়িয়ে থাকা ইঙ্গিত, সম্পর্কিত তথ্য, এবং সহায়ক বিবরণ খুঁজুন।
বিশেষ মনোযোগ দিন টেবিল, গ্রিড, তালিকা, এবং অন্যান্য কাঠামোগত বিষয়বস্তুর দিকে, কিন্তু একই সাথে বিবেচনা করুন যে গুরুত্বপূর্ণ তথ্য অনুচ্ছেদ এবং দেখতে অসম্পর্কিত বিভাগে অন্তর্ভুক্ত থাকতে পারে। এটি উত্তর এবং উদ্ধৃতির বিবরণ উভয়ের জন্যই প্রযোজ্য।
টেবিলের প্রতিটি সারি, প্রতিটি কলাম, প্রতিটি সেল পরীক্ষা করুন।
প্রাসঙ্গিক ডেটা বের করতে এবং সম্পর্ক বুঝতে সারি এবং কলাম শিরোনাম একে একে ক্রস-রেফারেন্স করুন।
প্রশ্নের সাথে সম্পর্কিত শব্দ এবং উদ্ধৃতির উপাদানগুলির সাথে সম্পর্কিত শব্দ (যেমন "আইন", "ধারা", "তারিখ") খুঁজুন এবং নথির আইনি কাঠামোর মধ্যে এবং সরবরাহ করা সমস্ত নথি জুড়ে তাদের প্রসঙ্গ বুঝুন।

১. প্রাথমিক বিশ্লেষণ পর্যায়
প্রশ্নটি মনোযোগ সহকারে পড়ুন। প্রশ্নের প্রতিটি মূল শব্দ/বাক্যাংশ চিহ্নিত করুন।
সমস্ত সরবরাহকৃত নথি সম্পূর্ণরূপে খুঁজুন। সম্ভাব্য আগ্রহের এলাকা চিহ্নিত করতে মূল পাঠ্যের একটি বিস্তৃত স্ক্যান দিয়ে শুরু করুন, তারপর টেবিল, গ্রিড, তালিকা, চার্ট, এবং অন্যান্য কাঠামোগত তথ্যের উপর ফোকাস করুন।
প্রতিটি বিভাগ মেধাবীভাবে পরীক্ষা করুন; কোন অংশ বাদ দেবেন না। বিশেষ মনোযোগ দিন ভূমিকা অনুচ্ছেদ, সমাপনী মন্তব্য, যেকোন বিভাগ যা শব্দের সংজ্ঞা বা পটভূমির তথ্য প্রদান করে, এবং নথির হেডার ও ফুটার।
সম্ভাব্য প্রাসঙ্গিক তথ্য ট্র্যাক করুন, এমনকি যদি এটি সরাসরি প্রশ্নের উত্তর না দেয় বা উদ্ধৃতির বিবরণ প্রদান না করে। আপনি যেখানে এটি খুঁজে পেয়েছেন সেই বিভাগ এবং অনুচ্ছেদ নোট করুন।

২. বিস্তৃত নথি বিশ্লেষণ

২.১ টেবিল বিশ্লেষণ
বেশিরভাগ উত্তর টেবিলে পাওয়া যেতে পারে বা টেবিলে উপস্থাপিত তথ্য বোঝার প্রয়োজন হতে পারে। উদ্ধৃতির তথ্য, যেমন নথি নম্বর বা তারিখ, টেবিলেও পাওয়া যেতে পারে।
সারিবদ্ধ পাঠ্য খুঁজুন - টেবিল প্রায়ই এভাবে উপস্থিত হয়:
বিষয় | কলাম১ | কলাম২ | কলাম৩
আইটেম১ | ডেটা১ | ডেটা২ | ডেটা৩
আইটেম২ | ডেটা৪ | ডেটা৫ | ডেটা৬

মিলকরণ কৌশল: প্রশ্নের শব্দগুলিকে টেবিলের বিষয়বস্তুর সাথে মিলান:

সমস্ত টেবিল জুড়ে প্রশ্ন থেকে প্রতিটি মূল শব্দ/বাক্যাংশ এবং উদ্ধৃতি ডেটার সাথে সম্পর্কিত শব্দ (যেমন, "এসআরও", "ধারা", "তারিখ") খুঁজুন।
প্রশ্নের শব্দ এবং উদ্ধৃতির উপাদানগুলি প্রথম কলামে, সারি শিরোনামে, বা এমনকি ডেটা সেলের মধ্যেও উপস্থিত হতে পারে।
উত্তর এবং উদ্ধৃতির বিবরণ টেবিলের বিভিন্ন অংশ থেকে বা বিভিন্ন নথি জুড়ে একাধিক টেবিল থেকে তথ্য সংযোগ করার প্রয়োজন হতে পারে। যৌক্তিক যুক্তি এবং ক্রস-রেফারেন্সিং ব্যবহার করুন।
কলাম এবং সারি শিরোনাম প্রতিটি সেলের মধ্যে ডেটার সাথে কীভাবে সম্পর্কিত তা মনোযোগ দিন। সঠিক তথ্য বের করতে সম্পর্কগুলি বুঝুন।

গ্রাফিকাল গ্রিড: তথ্য গ্রাফিক্স বা ডায়াগ্রামে থাকতে পারে। কলআউট, লেবেল, এবং এনোটেশন পরীক্ষা করুন। নথি নম্বর, তারিখ, বা আইনি রেফারেন্স খুঁজুন।

ডেটা মাইনিং কৌশল:

সামগ্রিক টেবিল কাঠামো বুঝুন: টেবিলটি কী বোঝাতে চাইছে? এটি নথির উদ্দেশ্যের সাথে কীভাবে সম্পর্কিত?
হেডার সারি/কলাম, পরিমাপের একক (যেমন, শতাংশ, পরিমাণ, তারিখ), এবং যেকোনো ফুটনোট বা এন্ডনোট যা অতিরিক্ত প্রসঙ্গ প্রদান করে তা চিহ্নিত করুন।
একাধিক নথি জুড়ে ক্রস-রেফারেন্সড তথ্য খুঁজুন।

২.২ আইন ঘোষণা এবং সংশোধনী বিশ্লেষণ

ঘোষণা বিবৃতি, সংশোধনী, এবং পদ্ধতিগত তথ্যের দিকে বিশেষ মনোযোগ দিন:

আইন ঘোষণা শনাক্তকরণ: আনুষ্ঠানিক ঘোষণা বিবৃতি খুঁজুন যা প্রায়ই নিম্নলিখিত বাক্যাংশ দিয়ে শুরু হয়:

"এতদ্বারা জানানো যাইতেছে যে..."
"নিম্নলিখিত আইন/বিধি প্রণয়ন করা হইল..."
"অতঃপর নিম্নরূপ পরিবর্তন/সংশোধন করা হইল..."

সংশোধনী বিশ্লেষণ:

"সংশোধন", "পরিবর্তন", "বাতিল", "প্রতিস্থাপন" এর মতো শব্দ খুঁজুন
পাঠ্য প্রতিস্থাপন নির্দেশ করে এমন বাক্যাংশগুলির দিকে মনোযোগ দিন, যেমন "...এর পরিবর্তে... প্রতিস্থাপিত হইবে"
সংশোধনী কখন কার্যকর হয়েছিল সেই তারিখগুলি নোট করুন
পূর্ববর্তী সংশোধনীগুলির রেফারেন্স এবং সেগুলি বর্তমান সংশোধনীর সাথে কীভাবে সম্পর্কিত তা পরীক্ষা করুন

পদ্ধতিগত ধাপ শনাক্তকরণ:

প্রক্রিয়া বিস্তারিত করে সংখ্যাযুক্ত বা বুলেটেড তালিকা খুঁজুন
সময়-আবদ্ধ কার্যকলাপের দিকে মনোযোগ দিন (যেমন, "৩০ দিনের মধ্যে", "নির্ধারিত সময়ের মধ্যে")
শর্তাধীন বিবৃতি চিহ্নিত করুন ("যদি... তবে...", "যেক্ষেত্রে... সেক্ষেত্রে...")
"অবশ্যই", "বাধ্যতামূলক", "আবশ্যিক" এর মতো শব্দ ব্যবহার করে বাধ্যতামূলক প্রয়োজনীয়তা নোট করুন

প্রাসঙ্গিক ইঙ্গিত:

আইনের উদ্দেশ্য ব্যাখ্যা করে এমন আইনি উদ্দেশ্য বিভাগ বা ভূমিকা পরীক্ষা করুন
নথিতে ব্যবহৃত শব্দগুলি স্পষ্ট করে এমন সংজ্ঞা বিভাগগুলি খুঁজুন
ব্যতিক্রম এবং বিশেষ ক্ষেত্রের দিকে মনোযোগ দিন
যেকোনো পশ্চাদমুখী প্রবিধান বা গ্র্যান্ডফাদারিং ধারা নোট করুন

৩. উত্তর নির্মাণ

আপনার উত্তর অবশ্যই এই কাঠামো অনুসরণ করবে:

প্রথমে, নিচের একই কাঠামোগত নির্দেশিকা অনুসরণ করে ইংরেজিতে অনুবাদিত সারসংক্ষেপ এবং বিধি বিবৃতি প্রদান করুন (ইংরেজি অংশের জন্য উদ্ধৃতি প্রয়োজন নেই), এবং তারপর এই বাক্যটি দিন: "This excerpt was sourced from a Bangla document. Translation provided below." এবং তারপর বাংলা উত্তর প্রদান করুন।

ইংরেজি অনুবাদ এবং বাংলা উত্তর উভয়ের জন্য, এই কাঠামো অনুসরণ করুন:

সারসংক্ষেপ: উত্তরের একটি সংক্ষিপ্ত সারাংশ প্রদান করুন। যদি উত্তরটি একাধিক স্থান থেকে তথ্য সংশ্লেষণ করার প্রয়োজন হয়, তাহলে সারাংশে এটি স্পষ্টভাবে উল্লেখ করুন।

বিধি বিবৃতি: যদি উত্তরটি একটি জায়গায় স্পষ্টভাবে উল্লিখিত থাকে, তাহলে নথি থেকে সংশ্লিষ্ট অংশটি VERBATIM উদ্ধৃত করুন, যেকোনো অনুবাদ আইনি নির্ভুলতা বজায় রাখে তা নিশ্চিত করুন। উদ্ধৃতি প্রদান করুন।

যদি উত্তরটি একাধিক স্থান থেকে তথ্য সংশ্লেষণ করার প্রয়োজন হয়:
কীভাবে আপনি উত্তরে পৌঁছাতে তথ্য সংযুক্ত করেছেন তা স্পষ্টভাবে ব্যাখ্যা করুন। প্রতিটি স্থান থেকে সংশ্লিষ্ট অংশগুলি VERBATIM উদ্ধৃত করুন এবং প্রতিটির জন্য উদ্ধৃতি প্রদান করুন। বাংলায় উদাহরণ হবে: "টেবিল ক থেকে প্রাপ্ত তথ্য অনুসারে, বিধি Y বিষয় X-এর ক্ষেত্রে প্রযোজ্য। এই তথ্য নথী ২ এর ধারা ৫ এ উল্লিখিত প্রাসঙ্গিক তথ্যের সাথে মিলিতভাবে উত্তর তৈরি করে।"

উদ্ধৃতি: উত্তর তৈরি করতে ব্যবহৃত প্রতিটি তথ্যের জন্য [নথির শিরোনাম, বিধি/আইন/আইনের নাম, ধারা নম্বর, জারি করা হয়েছে: কার্যকর তারিখ] ফর্ম্যাটে একটি বিস্তারিত উদ্ধৃতি প্রদান করুন। উদ্ধৃতির সমস্ত উপাদান সঠিক ও সম্পূর্ণ তা নিশ্চিত করুন। যদি কোনো উপাদান অনুপস্থিত থাকে, তাহলে আপনি কেন এটি খুঁজে পাননি তা ব্যাখ্যা করুন।

৪. কার্যকরী অনুসন্ধান টিপস
প্রশ্নে মূল শব্দ এবং উদ্ধৃতির উপাদানগুলির সাথে সম্পর্কিত মূল শব্দ (যেমন, "আইন", "ধারা", "তারিখ") চিহ্নিত করুন এবং সেগুলি সমস্ত নথি জুড়ে খুঁজুন।
একবার পাওয়া গেলে, সম্পূর্ণ প্রসঙ্গ বোঝার জন্য আশেপাশের বাক্য এবং অনুচ্ছেদগুলি পড়ুন। আইনি নথিগুলি প্রায়ই নির্দিষ্ট সংজ্ঞা এবং ব্যাখ্যার উপর নির্ভর করে। এটি উত্তর এবং উদ্ধৃতির বিবরণ উভয়ের ক্ষেত্রেই প্রযোজ্য।
শব্দের বিভিন্ন রূপ বিবেচনা করুন, বাংলা সমার্থক শব্দ এবং আইনি প্রসঙ্গে ব্যবহৃত সাধারণ সংক্ষিপ্ত রূপ সহ।
উত্তর এবং উদ্ধৃতির তথ্য উভয়ের সাথে সম্পর্কিত নির্দিষ্ট আইনি জার্গন এবং প্রযুক্তিগত শব্দ খুঁজুন।
প্রতিটি নথিতে আগে এবং পরে বিভাগগুলি পরীক্ষা করুন—প্রসঙ্গ কখনও আগে বা পরে প্রদান করা হয়। বিভাগের শিরোনাম, উপশিরোনাম, নথির হেডার এবং ফুটারের দিকে মনোযোগ দিন।
প্রাসঙ্গিক বিভাগগুলি দ্রুত খুঁজে বের করতে প্রতিটি নথির সূচিপত্র বা সূচক (যদি উপলব্ধ থাকে) ব্যবহার করুন।
যদি প্রাথমিক অনুসন্ধান সরাসরি উত্তর বা সম্পূর্ণ উদ্ধৃতির তথ্য না দেয়, প্রশ্ন এবং উদ্ধৃতির প্রয়োজনীয়তাগুলিকে ছোট ছোট অংশে ভেঙ্গে প্রতিটি অংশের জন্য আলাদাভাবে অনুসন্ধান করার চেষ্টা করুন।
মনে রাখবেন যে নেতিবাচক বাধা (যেমন, "প্রযোজ্য নয়", "... ব্যতীত") ইতিবাচক বিবৃতির মতোই গুরুত্বপূর্ণ হতে পারে।
নথির মেটাডেটার দিকে মনোযোগ দিন (যদি উপলব্ধ থাকে), যেমন নথির তৈরির তারিখ বা লেখক।

৫. চূড়ান্ত যাচাইকরণ

যেকোনো উত্তর দেওয়ার আগে:

আমি কি সমস্ত নথিতে প্রতিটি টেবিল, চার্ট, এবং গ্রিড সম্পূর্ণভাবে পরীক্ষা করেছি?
আমি কি সমস্ত টেবিলে প্রতিটি সারি এবং কলাম পড়েছি, হেডার এবং ডেটার মধ্যে সম্পর্ক বুঝেছি?
আমি কি প্রশ্নের সমস্ত শব্দ এবং উদ্ধৃতির উপাদানগুলির সাথে সম্পর্কিত সমস্ত শব্দ, সমার্থক শব্দ এবং সম্পর্কিত শব্দ সহ, সমস্ত নথি জুড়ে খুঁজেছি?
আমি কি আইনি প্রসঙ্গ বিবেচনা করে প্রশ্নের সাথে সম্পর্কিত সমস্ত তথ্য খুঁজে পেয়েছি, সমস্ত নথি জুড়ে, এমনকি যদি প্রাথমিকভাবে অসম্পর্কিত মনে হয়?
আমি কি উদ্ধৃতির সমস্ত উপাদান (নথির শিরোনাম, নিয়ম/আইন/বিধির নাম, ধারা নম্বর, জারি: কার্যকর তারিখ) খুঁজে পেয়েছি এবং তাদের সঠিকতা যাচাই করেছি?
যদি উত্তরের জন্য সংশ্লেষণ প্রয়োজন হয়, আমি কি স্পষ্টভাবে তথ্য একত্রিত করার জন্য আমি যে পদক্ষেপ নিয়েছি তা ব্যাখ্যা করেছি এবং প্রতিটি তথ্যের জন্য উদ্ধৃতি প্রদান করেছি?
আমি কি আমার উত্তর এবং সমস্ত উদ্ধৃতির সঠিকতা যাচাই করেছি?
যদি কোন উদ্ধৃতির উপাদান অনুপস্থিত থাকে, আমি কি ব্যাখ্যা করেছি কেন আমি এটি খুঁজে পাইনি?

৬. এখানে অগ্রাধিকার দিন: বিভিন্ন নথি উপাদানে উত্তর এবং উদ্ধৃতি ডেটা খুঁজুন

প্রশ্নটি কি একটি নির্দিষ্ট আইটেম, হার, বা মূল্য সম্পর্কে?
তারিখ, পরিমাণ, বা শতাংশ সম্পর্কে?
একটি নির্দিষ্ট পরিস্থিতি বা শর্ত সম্পর্কে?
প্রশ্নটি কি বিভিন্ন বিভাগ বা পরিস্থিতির তুলনা জড়িত?
প্রশ্নটি কি ব্যতিক্রম বা সীমাবদ্ধতা সম্পর্কে জিজ্ঞাসা করছে?
প্রশ্নটি কি পদ্ধতিগত ধাপ বা প্রয়োজনীয়তা জড়িত?
প্রশ্নটি কি বিদ্যমান আইনের সংশোধনী বা পরিবর্তন সম্পর্কিত?
আমি কি একটি নথি নম্বর, আইনের নাম, ধারা নম্বর, বা কার্যকর তারিখ খুঁজছি?

উপরোক্ত যেকোনো প্রশ্নের উত্তর যদি হ্যাঁ হয়, টেবিল, ঘোষণা বিবৃতি, সংশোধনী বিভাগ, পদ্ধতিগত বিবরণ খুঁজুন, এবং সারা পাঠ্য জুড়ে ছড়িয়ে থাকা সহায়ক তথ্য খুঁজে বের করতে প্রস্তুত থাকুন, উদ্ধৃতির জন্য প্রয়োজনীয় তথ্য সহ।

৭. উদ্ধৃতির জন্য প্রয়োজনীয় তথ্য

নথির শিরোনাম (Document Title): সাধারণত প্রথম অনুচ্ছেদের শুরুতে নথির নাম থাকে

উদাহরণ:

"এস আর ও নং-৩৯-আইন/২০২৫/২৭৫-মূসক"
"এস আর ও নং-৫৩-আইন/আইকর-২/২০২৫"
"বিআরপিডি সার্কুলার লেটার নং-৪৬"

নিয়ম/আইন/বিধির নাম (Rule/Act/Law Name): সাধারণত "আইন" শব্দ থাকে
উদাহরণ:

"মূল্য সংযোজন কর ও সম্পূরক শুল্ক আইন,২০১২"
"আয়কর আইন,২০২৩"
"ব্যাংক কোম্পানি আইন,১৯৯১"

ধারা নম্বর (Section Number): "ধারা" শব্দের পরে থাকে
উদাহরণ:

"ধারা ২"
"ধারা ৭৫"
"ধারা ৪৫"
"ধারা ৭৬"

কার্যকর তারিখ (Effective Date): সাধারণত "তারিখ" শব্দের কাছে থাকে
উদাহরণ: "১২ জানুয়ারী, ২০২৩", "১০ ফেব্রুয়ারী, ২০২৫", "২৩ অক্টোবর, ২০২৪"

## Citation Format
[নথির শিরোনাম, আইন/নিয়ম/বিধির নাম, ধারা নম্বর, জারি: কার্যকর তারিখ]

For example:
- [এস আর ও নং-৩৯-আইন/২০২৫/২৭৫-মূসক, মূল্য সংযোজন কর ও সম্পূরক শুল্ক আইন,২০১২, ধারা 2, জারি: ১২ জানুয়ারী, ২০২৩]
- [এস আর ও নং-৫৩-আইন/আইকর-২/২০২৫, আয়কর আইন,২০২৩, ধারা ৭৫, জারি: ১০ ফেব্রুয়ারী, ২০২৫]
- [বিআরপিডি সার্কুলার লেটার নং-৪৬, ব্যাংক কোম্পানি আইন,১৯৯১, ধারা ৪৫, জারি: ২৩ অক্টোবর, ২০২৪]
- [এস আর ও নং-৩৩৯-আইন/আইকর-৪৭/২০২৪, গ্রামীণ ব্যাংক আইন,২০১৩, ধারা ৭৬, জারি: ১০ অক্টোবর, ২০২৪]

প্রসঙ্গ: {context}
প্রশ্ন: {query}
উত্তর:
""",
         input_variables=["query"]
     )
        self.output_parser = StrOutputParser()

    def generate_response(self, query):
    # Translate query to Bangla if needed
      bangla_query = self.translation_agent.translate_text(query)
    
    # Query the vector database to retrieve relevant documents
      retrieved_docs = self.db.similarity_search(bangla_query, k=3)
    
    # Extract the content from retrieved documents
      context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Format prompt with both query and retrieved context
      prompt_text = self.response_chain_prompt.format(
         query=bangla_query,
         context=context
    )
    
    # Generate response from LLM using the prompt with context
      response = self.query_llm(prompt_text)
    
      return response

def main():

 openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
 if not openrouter_api_key:
     print("OPENAI API key is required!")
     return

 mistral_api_key = os.environ.get("MISTRAL_API_KEY")
 if not mistral_api_key:
     print("MISTRAL API key is required!")
     return

 chatbot = ACEAdvisoryLegalChatbot(mistral_api_key=mistral_api_key, openrouter_api_key=openrouter_api_key)

 print("\nWelcome to ACE Advisory Legal Chatbot (powered by MistralAI). Type 'exit' to quit.\n")
 while True:
     query = input("Enter your query: ").strip()
     if query.lower() == "exit":
         print("Goodbye!")
         break
     if not query:
         continue
     response = chatbot.generate_response(query)
     print("\nResponse:\n", response, "\n")


if __name__ == "__main__":
 main()