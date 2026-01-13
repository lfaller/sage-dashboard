# Supabase Setup Guide

This guide walks you through setting up Supabase for the SAGE Dashboard.

## Step 1: Create Supabase Project

Since you already have a Supabase account:

1. Go to https://supabase.com/dashboard
2. Click **New Project**
3. Fill in:
   - **Name:** `sage-dashboard`
   - **Password:** Generate a strong password (save it!)
   - **Region:** Choose the closest to you
4. Click **Create new project**
5. Wait for provisioning (2-3 minutes)

## Step 2: Get Your API Credentials

1. In your project, go to **Settings** → **API**
2. Copy these values:
   - **Project URL** (looks like `https://xxxxx.supabase.co`)
   - **anon public key** (starts with `eyJ...`)
3. Keep these safe - you'll need them next

## Step 3: Create Database Schema

1. In Supabase, go to **SQL Editor**
2. Click **New query**
3. Copy the entire contents of `sql/schema.sql`
4. Paste into the SQL editor
5. Click **Run**

This creates the tables and indexes needed for the dashboard.

## Step 4: Configure Local Secrets

1. Create the secrets file:
   ```bash
   mkdir -p .streamlit
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

2. Edit `.streamlit/secrets.toml`:
   ```toml
   [connections.supabase]
   SUPABASE_URL = "https://xxxxx.supabase.co"
   SUPABASE_KEY = "your-anon-public-key"
   ```

3. Replace with your actual values from Step 2

4. **IMPORTANT:** Never commit this file! It's already in `.gitignore`

## Step 5: Verify Connection

Test that the connection works:

```bash
poetry run python -c "from sage.database import get_supabase_client; print('✓ Connected!')"
```

If you see `✓ Connected!`, you're ready to load data!

## Troubleshooting

### "ModuleNotFoundError: No module named 'supabase'"
Make sure you have Supabase installed:
```bash
poetry install
```

### "KeyError: 'connections'"
Make sure `.streamlit/secrets.toml` exists and has the correct format.

### "CORS Error" when running Streamlit app
This is expected for now. We'll fix it when we integrate with the UI.

## Next Steps

Once Supabase is set up:

1. Load sample data (see `../scripts/` when ready)
2. Update the overview page to use real data
3. Build additional pages with database queries

---

**See:** [ROADMAP.md](../ROADMAP.md) for the full development plan
