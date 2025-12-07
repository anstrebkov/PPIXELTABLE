import os
import uvicorn
import pixeltable as pxt
from pixeltable.functions import gemini, huggingface
from pixeltable.iterators import DocumentSplitter
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
from dotenv import load_dotenv
from mangum import Mangum  # Adapter for Netlify/Lambda

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Pixeltable Multimodal Demo")
handler = Mangum(app)  # Entry point for Netlify Functions

# Configuration
PIXELTABLE_DIR = Path("/tmp/pixeltable_home") # Use /tmp for serverless/read-only envs
os.environ['PIXELTABLE_HOME'] = str(PIXELTABLE_DIR)

# --- Pixeltable Initialization & Schema ---

def init_pixeltable():
    """Initialize tables and views. Using /tmp for serverless compatibility."""
    pxt.init()
    
    # 1. Document RAG System
    try:
        docs = pxt.get_table('docs')
    except:
        docs = pxt.create_table('docs', {'doc': pxt.Document})

    try:
        chunks = pxt.get_table('doc_chunks')
    except:
        chunks = pxt.create_view(
            'doc_chunks',
            docs,
            iterator=DocumentSplitter.create(
                document=docs.doc,
                separators='sentence'
            )
        )
        # Use a lightweight embedding model to save RAM/Time
        embed_model = huggingface.sentence_transformer.using(model_id='sentence-transformers/all-MiniLM-L6-v2')
        chunks.add_embedding_index('text', string_embed=embed_model)

    # 2. Image Analysis System
    try:
        images = pxt.get_table('images')
    except:
        images = pxt.create_table('images', {'input_image': pxt.Image})
        
        # GEMINI 2.0 FLASH INTEGRATION
        # Using generate_content for vision/multimodal analysis
        images.add_computed_column(
            vision_description=gemini.generate_content(
                contents=images.input_image, # Gemini 2.0 is natively multimodal
                model='gemini-2.0-flash'
            )
        )

    return docs, chunks, images

# Global references (lazy loaded in endpoints to avoid startup timeouts)
TABLES = {}

def get_tables():
    if not TABLES:
        d, c, i = init_pixeltable()
        TABLES['docs'] = d
        TABLES['chunks'] = c
        TABLES['images'] = i
    return TABLES

# --- HTML Template (Embedded for simplicity) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pixeltable + Gemini 2.0 Flash</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #f3f4f6; }
        .gradient-text {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .btn-primary {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
    </style>
</head>
<body class="p-8">
    <div class="max-w-4xl mx-auto bg-white rounded-xl shadow-lg p-8">
        <h1 class="text-4xl font-bold mb-2 gradient-text">Pixeltable Multimodal</h1>
        <p class="text-gray-500 mb-8">Powered by Gemini 2.0 Flash</p>

        <!-- Tabs -->
        <div class="mb-6 border-b">
            <nav class="-mb-px flex space-x-8">
                <button onclick="switchTab('docs')" class="tab-btn border-b-2 border-indigo-500 py-4 px-1 text-sm font-medium text-indigo-600" id="tab-docs">Document RAG</button>
                <button onclick="switchTab('images')" class="tab-btn border-b-2 border-transparent py-4 px-1 text-sm font-medium text-gray-500 hover:text-gray-700" id="tab-images">Image Analysis</button>
            </nav>
        </div>

        <!-- Document Section -->
        <div id="content-docs" class="tab-content">
            <div class="grid grid-cols-1 gap-6">
                <div class="bg-gray-50 p-6 rounded-lg">
                    <h3 class="font-bold mb-4">1. Upload Document</h3>
                    <form action="/upload_doc" method="post" enctype="multipart/form-data" class="flex gap-4">
                        <input type="file" name="file" accept=".pdf,.txt" class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-violet-50 file:text-violet-700 hover:file:bg-violet-100"/>
                        <button type="submit" class="btn-primary px-4 py-2 rounded">Upload</button>
                    </form>
                </div>

                <div class="bg-gray-50 p-6 rounded-lg">
                    <h3 class="font-bold mb-4">2. Ask Question (Gemini 2.0)</h3>
                    <form action="/ask" method="post" class="flex gap-4">
                        <input type="text" name="question" placeholder="Ask about your documents..." class="flex-1 p-2 border rounded"/>
                        <button type="submit" class="btn-primary px-4 py-2 rounded">Ask</button>
                    </form>
                    {% if answer %}
                    <div class="mt-4 p-4 bg-white border-l-4 border-indigo-500 shadow-sm">
                        <p class="font-semibold text-gray-700">Answer:</p>
                        <p class="mt-1">{{ answer }}</p>
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>

        <!-- Image Section -->
        <div id="content-images" class="tab-content hidden">
            <div class="bg-gray-50 p-6 rounded-lg">
                <h3 class="font-bold mb-4">Upload Image for Analysis</h3>
                <form action="/upload_image" method="post" enctype="multipart/form-data" class="flex gap-4 items-center">
                    <input type="file" name="file" accept="image/*" class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-violet-50 file:text-violet-700 hover:file:bg-violet-100"/>
                    <button type="submit" class="btn-primary px-4 py-2 rounded">Analyze</button>
                </form>
            </div>

            {% if image_desc %}
            <div class="mt-6 p-6 bg-white rounded-lg shadow border border-gray-100">
                <h3 class="font-bold text-lg mb-2 text-indigo-600">Gemini Vision Analysis:</h3>
                <p class="text-gray-800 leading-relaxed">{{ image_desc }}</p>
            </div>
            {% endif %}
        </div>
    </div>

    <script>
        function switchTab(tab) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            document.getElementById('content-' + tab).classList.remove('hidden');
            
            document.querySelectorAll('.tab-btn').forEach(el => {
                el.classList.remove('border-indigo-500', 'text-indigo-600');
                el.classList.add('border-transparent', 'text-gray-500');
            });
            document.getElementById('tab-' + tab).classList.add('border-indigo-500', 'text-indigo-600');
            document.getElementById('tab-' + tab).classList.remove('border-transparent', 'text-gray-500');
            
            // Persist tab selection (simple local storage or just keep view)
            localStorage.setItem('activeTab', tab);
        }
        
        // Restore tab if reloaded
        const active = localStorage.getItem('activeTab');
        if(active) switchTab(active);
    </script>
</body>
</html>
"""

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_TEMPLATE

@app.post("/upload_doc", response_class=HTMLResponse)
async def upload_doc(file: UploadFile = File(...)):
    tables = get_tables()
    
    # Save temp file
    temp_path = Path(f"/tmp/{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Insert into Pixeltable
    try:
        tables['docs'].insert([{'doc': str(temp_path)}])
        msg = f"Processed {file.filename}"
    except Exception as e:
        msg = f"Error: {str(e)}"
        
    return HTML_TEMPLATE.replace("Pixeltable Multimodal", f"Status: {msg}")

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(question: str = Form(...)):
    tables = get_tables()
    chunks = tables['chunks']
    
    # 1. Retrieve relevant context
    sim = chunks.text.similarity(question)
    context_rows = chunks.order_by(sim, asc=False).limit(3).select(chunks.text).collect()
    context_text = "\n".join([row['text'] for row in context_rows])
    
    # 2. Generate Answer using Gemini 2.0 Flash
    # Note: We call Gemini directly via Pxt function or just generic Python if Pxt table insertion isn't needed for the QA log.
    # To keep it "Pixeltable style", we usually insert into a QA table, but for speed in this demo we can use the Pxt expression directly or simple logic.
    
    prompt = f"""
    Context: {context_text}
    
    Question: {question}
    
    Answer concisely based on the context.
    """
    
    # Using Pixeltable's function execution engine
    # We can create a temporary table or just use the function directly if supported, 
    # but Pixeltable functions are designed for columns.
    # Let's use a QA table approach for correctness.
    
    try:
        qa_table = pxt.get_table('qa_log')
    except:
        qa_table = pxt.create_table('qa_log', {'question': pxt.String, 'context': pxt.String})
        qa_table.add_computed_column(
            answer=gemini.generate_content(
                contents="Question: " + qa_table.question + "\nContext: " + qa_table.context,
                model='gemini-2.0-flash'
            )
        )
        
    qa_table.insert([{'question': question, 'context': context_text}])
    result = qa_table.select(qa_table.answer).tail(1)
    answer_text = result[0]['answer'] if result else "No answer generated."

    # Return template with answer injected
    from jinja2 import Template
    t = Template(HTML_TEMPLATE)
    return t.render(answer=answer_text)

@app.post("/upload_image", response_class=HTMLResponse)
async def upload_image(file: UploadFile = File(...)):
    tables = get_tables()
    images = tables['images']
    
    temp_path = Path(f"/tmp/{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    images.insert([{'input_image': str(temp_path)}])
    
    # Get analysis
    result = images.select(images.vision_description).tail(1)
    desc = result[0]['vision_description'] if result else "Analysis failed."
    
    from jinja2 import Template
    t = Template(HTML_TEMPLATE)
    return t.render(image_desc=desc)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
