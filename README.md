# Multi-Grade Classroom Agent (RAG)

A hackathon MVP for rural schools where **one teacher teaches Grades 1–5 in the same room**.
Enter a topic, and the agent:

1. **Retrieves curriculum** chunks for each selected grade from `data/curriculum/` (ChromaDB + local embeddings).
2. Writes **grade-specific, syllabus-aligned objectives** (`with_structured_output(LessonPlanOutline)`).
3. Generates, **in parallel for every grade**, a low-resource activity, a 3–4 task independent worksheet and a 3-question quiz with answers.
4. Builds a **teacher rotation timetable** that shows who the teacher is with and what every other grade does, checked by a Python validator.
5. Exports a **printable PDF packet** with the timetable, worksheets and answer keys.

Output languages: English, Kannada, Hindi.

> ⚠️ The 55 files in `data/curriculum/` (one unit per grade 1-5 for Maths, EVS, Science, English, Kannada, Hindi, Social Studies, Art and Craft, Music, Physical Education and Computer Science) are **DEMO material written for this prototype, not official syllabus documents**. Replace them with your state or board curriculum. The UI and PDF label demo sources and say so when no relevant source was retrieved.

## Tech stack

* **Backend:** Python 3.12, Flask 3, python-dotenv, Pydantic 2, Groq API through LangChain (`langchain-core`, `langchain-groq`), `langchain-text-splitters`, ChromaDB (`langchain-chroma`), HuggingFace Sentence Transformers (`langchain-huggingface`), **pypdf** (PDF reading), ReportLab (PDF output), JSON file storage.
* **Frontend:** semantic HTML5 rendered with Jinja2, hand-written CSS3 (custom properties, no Tailwind or Bootstrap) and vanilla ES6 JavaScript. There are no frameworks and no build step. Google Fonts supplies Noto Sans Kannada and Devanagari so Indic text renders correctly.
* **Tests:** pytest, offline: AI calls are replaced by fakes, and embeddings are faked in the ChromaDB tests.

## Pages

| Page | URL | What it does |
|---|---|---|
| Dashboard | `/` | Quick actions, real counts (lessons, curriculum files, classes, pending approvals), recent lessons and classes |
| Lesson Planner | `/planner` | Lesson form, Load Sample Lesson, accessibility settings. Old `/?subject=...` links redirect here. |
| Lesson results | `/lesson/<id>` | Grade tabs, answers on demand, accessible preview, inline editing, rotation timetable, RAG sources, PDF and print |
| Adaptive Classroom Assistant | `/assistant/` | Class setup, break schedule, attention pattern, timetable proposals, editing, feedback |
| Curriculum Library | `/curriculum` | Upload (drag and drop), file list with grade, type and Demo badges, per-file errors, delete, rebuild vector store |

## Project structure (frontend)

```
templates/
  base.html                      HTML5 shell: sidebar navigation, notifications, loading overlay, confirm dialog
  dashboard.html  not_found.html
  components/   icons.html  navigation.html  notifications.html
  lesson/       planner.html  results.html  _accessibility.html  _grade_tabs.html  _rotation.html  _sources.html
  assistant/    dashboard.html  timetable.html  _setup.html  _feedback.html  _macros.html
  curriculum/   library.html
static/css/     style.css (tokens + components)  layout.css  dashboard.css  lesson.css  assistant.css  curriculum.css  responsive.css
static/js/      app.js (shared)  lesson.js  assistant.js  curriculum.js
```

## New: Dyslexia-Friendly Worksheets

On the lesson form, open **♿ Worksheet Accessibility Settings**. You can turn on **Dyslexia-Friendly Mode**, the recommended preset, or pick individual adaptations: larger font, extra spacing, simple step-by-step instructions, short sentences, one question per section, reduced clutter, more answer space, visual hints, keyword highlights, and fewer questions.

* Groq rewrites each grade's worksheet from the same objectives, tasks and retrieved curriculum (`chains/accessible_chain.py`, run in parallel for all grades). The prompt forbids lowering the challenge. It requires a worked example first and an alternative way to answer each question. "Reduce questions" is off in the preset.
* Python then checks the output. Keywords are kept only if they really appear in the text. The question count is enforced, and visual hints are removed unless enabled.
* **Preview:** each grade tab shows an *Accessible worksheet - teacher preview* in Verdana, with large spacing and highlighted keywords.
* **PDF:** the main packet adds an accessible page after each standard worksheet, plus an answer key. *Accessible worksheets only* downloads just those pages. English uses Verdana/Arial, falling back to DejaVu Sans and then Helvetica. Kannada and Hindi use the Indic font.
* You can add or change accessible worksheets for an existing lesson from the results page.

## New: AI Adaptive Classroom Assistant (`/assistant`)

This is a separate dashboard for a class's **weekly subject timetable**. It complements, and does not replace, the multi-grade teacher rotation inside a lesson.

1. **Setup:** class info, breaks, subjects with editable cognitive load (High/Medium/Low), periods per week, max per day, teacher, teacher unavailability, fixed activities, and a clickable **attention pattern** per hour. Attention levels are your observations, not measurements.
2. **Timetable:** **Generate** builds a *proposal*.
   * A deterministic Python optimiser (`adaptive/scheduler.py`) scores candidate timetables. It uses greedy construction, simulated-annealing swaps and several seeded restarts, then ranks the candidates. It puts high-load subjects in attentive periods, avoids back-to-back demanding periods without a break, spreads subjects across the week, respects teacher availability (including other classes' approved timetables), and keeps locked periods and fixed activities.
   * Groq gives per-subject advice (a small tie-breaking bonus only) and a plain-language summary. Any summary sentence that contradicts the grid is dropped.
   * `adaptive/validator.py` decides validity. A proposal with errors **cannot be approved**, and the conflicts come with suggested fixes.
   * **Nothing changes until you approve.** You can swap, set or lock any period. Every change is re-validated and the score change is shown. The *Why did AI recommend this timetable?* panel is generated from the actual placements and your data.
   * **Export PDF** and **Print** are on the Timetable tab. **Plan lesson for this period** opens the lesson planner pre-filled with the subject, period length, grade and the period's attention level.
3. **Feedback & learning:** record understanding, engagement and concentration per period. After at least 2 entries, a period's attention becomes 60 % your level and 40 % observed. Subjects that struggle at a time of day are steered away from it. Recommendations appear on the Timetable tab, and you generate and approve an updated proposal yourself.

Data is stored as JSON in `data/adaptive/` (git-ignored). Optional `.env` settings: `ADAPTIVE_USE_AI=0` runs the optimiser only, and `ADAPTIVE_RESTARTS=10` sets the number of search restarts.

## Editing generated content

On any lesson page, click **✏️ Edit content**, change any outlined text, then **Save changes**. Edits are saved to the lesson, so the PDF uses them. Grades, times and curriculum sources cannot be edited. The rotation timetable is re-validated after each save.

## Setup (Windows PowerShell)

```powershell
cd C:\Users\Yuktha\Downloads\multigrade-agent\multigrade-agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

Put your Groq key in `.env` (copy it from `.env.example` if `.env` is missing):

```
GROQ_API_KEY=gsk_...your key...
GROQ_MODEL=openai/gpt-oss-120b
```

Get a free key at https://console.groq.com/keys. `.env` is git-ignored; never commit it.

## Run

```powershell
python scripts/ingest.py      # optional: build the vector store now (first run downloads the ~90 MB embedding model)
python app.py                 # open http://127.0.0.1:5000
```

Click **⚡ Load Sample Lesson** to generate Fractions for Grades 1–5, 45 minutes, in English.

Run the tests (no API key needed; 580+ tests covering the rotation, accessibility, adaptive scheduler, curriculum ingestion, routes and pages):

```powershell
python -m pytest -q
```

## Updating the curriculum (RAG)

* **In the UI:** open **Curriculum Library** and upload a `.pdf` or `.txt` (max 10 MB). It is validated, moved into `data/curriculum/` and indexed right away. Choose a grade, or include `grade3` in the file name. You can delete files there too.
* **On disk:** copy files into `data/curriculum/`. New or changed files are indexed automatically before the next lesson. The index tracks SHA-256 hashes in `vectorstore/manifest.json`.
* **CLI:**
  * `python scripts/ingest.py`: incremental sync
  * `python scripts/ingest.py --rebuild`: wipe and rebuild
  * `python scripts/ingest.py --status`: show what is indexed
  * `python scripts/ingest.py --query "fractions" --grades 3 4`: test retrieval
* **Rebuild vector store** button in the UI: same as `--rebuild`.

Subject metadata comes from the first word of the file name: `maths_`, `evs_`, `science_`, `english_`, `kannada_`, `hindi_`, `social_`, `art_`, `music_`, `pe_` / `physical_`, `computer_`. Otherwise a `Subject:` line in the text is used, and anything else is tagged "General".
Grade metadata comes from the file name (`grade3`, `class 3`, `std3`) or the first lines of text. If neither names a grade, the file applies to all grades.

PDFs are read page by page with **pypdf** (`rag.read_pdf_pages`). Damaged files, password-protected files (PDFs with only an owner password still work), files with no pages, and scanned PDFs with no selectable text are rejected with a plain-language message. That message appears on upload and next to the file in the library. A broken re-upload never removes the passages of the previous version.

## Fonts for Kannada and Hindi PDFs

The web page renders Kannada and Hindi with Google Noto fonts loaded from a CDN. For the **ReportLab PDF**, the generator looks for a Unicode TTF font in this order:

1. `fonts/NotoSansKannada-Regular.ttf` (and `-Bold.ttf`) for Kannada, or `fonts/NotoSansDevanagari-Regular.ttf` (and `-Bold.ttf`) for Hindi. Download them free from https://fonts.google.com/noto. Use the static `.ttf` files, not variable fonts.
2. Windows system fonts: `Nirmala.ttc` / `Nirmala.ttf`, `Tunga` (Kannada), `Mangal` (Hindi).
3. Common Linux paths for Noto fonts.

If no font is found, the PDF shows a warning and Indic text may print as boxes.
**Known limitation:** ReportLab does not fully shape Indic scripts, so some conjuncts and vowel signs may look split. For pixel-perfect Kannada or Hindi, use **🖨 Print / Save as PDF** on the results page. The browser shapes the text correctly, and the print stylesheet prints all grades and opens every answer key.

## How it works

| Step | File | Technique |
|---|---|---|
| Retrieval | `rag.py` | `RecursiveCharacterTextSplitter` → `HuggingFaceEmbeddings` (all-MiniLM-L6-v2, CPU) → `Chroma` (cosine, persisted in `vectorstore/`). Each grade is searched with a metadata filter `grade ∈ {g, 0}`. Chunks below `MIN_RELEVANCE` are dropped. |
| Objectives | `chains/intake_chain.py` | `ChatPromptTemplate \| ChatGroq.with_structured_output(LessonPlanOutline)`. Retries if a grade is missing. Citations to labels that were never retrieved are removed. |
| Grade content | `chains/content_chain.py` | One LCEL chain → `GradeContent`, run for all grades with `.batch(..., max_concurrency)`. Failed grades are retried once. |
| Timetable | `chains/timetable_chain.py` | Deterministic skeleton: optional opening and closing, a rotation that starts with the youngest grade, extra minutes spread across the earliest slots, adjacent grades paired when slots would be under 5 min, and a second round when slots would exceed 25 min. The LLM only writes the task text. |
| Validation | `validators.py` | Checks exact coverage of 0 to duration, no gaps or overlaps, one non-empty task per grade per block, and modes that match the teacher focus. Also validates the lesson form and uploads. |
| Orchestration | `chains/pipeline.py` | Runs the steps and saves the result to `output/lessons/<id>.json`. |
| PDF | `pdf_generator.py` | ReportLab packet: cover and sources, landscape timetable, one worksheet and quiz page per grade, then the teacher activities and answer keys. |
| Web | `app.py`, `assistant_routes.py`, `templates/`, `static/` | Flask routes, Jinja2 templates, custom CSS3 and vanilla JavaScript. |

## Troubleshooting

* **"GROQ_API_KEY is not set"**: edit `.env`, then restart `python app.py`.
* **Rate limit (429)**: on the Groq free tier, set `MAX_CONCURRENCY=1` in `.env` and/or `TIMETABLE_LLM=0`.
* **First lesson is slow**: the embedding model downloads once and the index builds on first use.
* **Model unavailable**: change `GROQ_MODEL` in `.env` to a model listed in your Groq console.
