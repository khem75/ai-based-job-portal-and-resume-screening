import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import sqlite3
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "secret_key_for_session"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS jobs 
                 (id INTEGER PRIMARY KEY, title TEXT, description TEXT)''')
    
    # We need to ensure the table has all columns. 
    # If you get an error, delete database.db and restart.
    c.execute('''CREATE TABLE IF NOT EXISTS applicants 
                 (id INTEGER PRIMARY KEY, job_id INTEGER, name TEXT, 
                  filename TEXT, match_score REAL, missing_keywords TEXT, 
                  ats_status TEXT, resume_quality TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def check_resume_quality(text):
    # Simple check: Does it have standard resume sections?
    text_lower = text.lower()
    required_sections = ["education", "experience", "skills", "projects"]
    found_sections = [sec for sec in required_sections if sec in text_lower]
    
    if len(found_sections) >= 3:
        return "High Quality (Standard Format)"
    elif len(found_sections) >= 1:
        return "Medium Quality (Missing Sections)"
    else:
        return "Low Quality / Invalid Format"

def get_missing_keywords(job_desc, resume_text):
    cv = CountVectorizer(stop_words='english')
    try:
        cv.fit([job_desc])
        job_keywords = set(cv.get_feature_names_out())
        cv.fit([resume_text])
        resume_keywords = set(cv.get_feature_names_out())
        missing = job_keywords - resume_keywords
        return ", ".join(list(missing)[:5])
    except:
        return "N/A"

def calculate_match_score(job_description, resume_text):
    if not resume_text: return 0.0
    text_list = [job_description, resume_text]
    cv = TfidfVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(text_list)
    matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
    return round(matchPercentage, 2)

# --- Routes ---

@app.route('/')
def index():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM jobs")
    jobs = c.fetchall()
    conn.close()
    return render_template('index.html', jobs=jobs)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/post_job', methods=['GET', 'POST'])
def post_job():
    if request.method == 'POST':
        title = request.form['title']
        desc = request.form['description']
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("INSERT INTO jobs (title, description) VALUES (?, ?)", (title, desc))
        conn.commit()
        conn.close()
        flash('Job Posted Successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('post_job.html')

@app.route('/apply/<int:job_id>', methods=['GET', 'POST'])
def apply(job_id):
    if request.method == 'POST':
        name = request.form['name']
        if 'resume' not in request.files: return redirect(request.url)
        file = request.files['resume']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            resume_text = extract_text_from_pdf(file_path)
            job_desc = sqlite3.connect('database.db').cursor().execute("SELECT description FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
            
            # 1. AI Score
            score = calculate_match_score(job_desc, resume_text)
            
            # 2. Missing Keywords
            missing = get_missing_keywords(job_desc, resume_text)
            
            # 3. Resume Quality Check
            quality = check_resume_quality(resume_text)
            
            # 4. ATS Status Decision
            if score >= 50 and "Low" not in quality:
                ats_status = "Shortlisted"
            else:
                ats_status = "Rejected"

            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("INSERT INTO applicants (job_id, name, filename, match_score, missing_keywords, ats_status, resume_quality) VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (job_id, name, filename, score, missing, ats_status, quality))
            conn.commit()
            conn.close()
            
            flash(f'Application Sent! AI Score: {score}% ({ats_status})', 'info')
            return redirect(url_for('index'))
            
    return render_template('apply.html', job_id=job_id)

@app.route('/dashboard/<int:job_id>')
def dashboard(job_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
    job = c.fetchone()
    c.execute("SELECT * FROM applicants WHERE job_id=? ORDER BY match_score DESC", (job_id,))
    applicants = c.fetchall()
    conn.close()
    return render_template('dashboard.html', job=job, applicants=applicants)

if __name__ == '__main__':
    if not os.path.exists('uploads'): os.makedirs('uploads')
    app.run(debug=True)