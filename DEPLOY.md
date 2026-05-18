# Deploying to Render

Step-by-step guide to deploy the Email Spam Detection Dashboard to
[Render](https://render.com) for free.

---

## Prerequisites

- A [GitHub](https://github.com) account
- A [Render](https://render.com) account (free)
- The `spam.csv` dataset file

---

## Step 1 — Push to GitHub

```bash
# From inside the email_spam_detection/ folder

git init
git add .
git commit -m "Initial commit — Email Spam Detection System"

# Create a new repo on github.com (do NOT initialise with README)
# Then run:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

> **Note:** `spam.csv` is in `.gitignore` — do not commit it.
> You will upload it separately in Step 3.

---

## Step 2 — Create a Render Web Service

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **New +** → **Web Service**
3. Click **Connect** next to your GitHub repository
4. Render auto-detects `render.yaml` — confirm the settings:

| Setting | Value |
|---|---|
| **Name** | `email-spam-detection` |
| **Runtime** | Python 3 |
| **Build Command** | `bash build.sh` |
| **Start Command** | `gunicorn "dashboard.app:app" --bind 0.0.0.0:$PORT --workers 2 --timeout 120` |
| **Plan** | Free |

5. Click **Create Web Service**

Render will run `build.sh` which installs all dependencies and downloads
NLTK data automatically. The first deploy takes about 3–5 minutes.

---

## Step 3 — Upload the Dataset

Render's free tier has no persistent disk, so you have two options:

### Option A — Upload via Render Shell (quickest)

1. In your Render service dashboard, click **Shell**
2. Run:
```bash
mkdir -p data
cat > data/spam.csv << 'EOF'
# paste the contents of spam.csv here, or use the method below
EOF
```

Or use `curl` if your CSV is hosted somewhere:
```bash
curl -o data/spam.csv "https://your-hosted-file-url/spam.csv"
```

### Option B — Commit a small sample to GitHub

If you want the dataset in the repo (not recommended for large files):
1. Remove `data/spam.csv` from `.gitignore`
2. `git add data/spam.csv && git commit -m "add dataset" && git push`
3. Render will pick it up automatically on next deploy

### Option C — Render Disk (paid, most reliable)

Uncomment the `disk:` block in `render.yaml`:
```yaml
disk:
  name: spam-data
  mountPath: /opt/render/project/src/data
  sizeGB: 1
```
Then upload `spam.csv` via the Render dashboard → Disks.

---

## Step 4 — Train the Pipeline

Once deployed and the dataset is in place:

1. Open your Render URL (e.g. `https://email-spam-detection.onrender.com`)
2. Click **Training Console** in the sidebar
3. Click **▶ Start Training**
4. Watch the live log — training takes 1–3 minutes on Render's free tier
5. All sections populate automatically when training completes

---

## Step 5 — Use the Dashboard

| Section | What you can do |
|---|---|
| **Overview** | Dataset stats, class distribution, algorithm comparison |
| **Classify Email** | Paste email text → get SPAM/HAM verdict instantly |
| **Model Performance** | Ranked metrics table, ROC curves, confusion matrices |
| **Data Analysis** | EDA plots (text length, top words, class balance) |
| **Training Console** | Retrain at any time, watch live progress |

---

## Environment Variables (optional)

Set these in Render → Environment:

| Variable | Value | Purpose |
|---|---|---|
| `FLASK_ENV` | `production` | Disables debug mode |
| `NLTK_DATA` | `/opt/render/project/src/nltk_data` | NLTK data path |
| `PYTHON_VERSION` | `3.11.0` | Pin Python version |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Dataset not found` | Upload `spam.csv` to the `data/` folder (Step 3) |
| `NLTK resource not found` | Re-run build: Render dashboard → Manual Deploy |
| `Build timeout` | Upgrade to Starter plan for longer build times |
| App sleeps after inactivity | Expected on free tier; upgrade for always-on |
| `gunicorn: command not found` | Make sure `gunicorn>=21.2.0` is in `requirements.txt` |

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Place spam.csv in data/

# Run locally
cd dashboard
python app.py
# Open http://127.0.0.1:5000
```
