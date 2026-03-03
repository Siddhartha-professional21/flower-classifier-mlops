# ============================================================
# Dockerfile - Containerize the Flower Classifier API
# ============================================================
# What is Docker?
#   Think of Docker like a LUNCHBOX 🍱
#   - Your app needs specific food (libraries, python, model)
#   - Docker packs EVERYTHING into one sealed box
#   - Anyone can open that box on ANY computer and it works!
#
# What this Dockerfile does step by step:
#   1. Gets a clean Python computer (base image)
#   2. Sets up a working folder inside the container
#   3. Copies requirements.txt and installs all libraries
#   4. Copies all your project files into the container
#   5. Opens port 8000 so outside world can access the API
#   6. Runs the FastAPI server when container starts
# ============================================================


# ============================================================
# STEP 1: Choose the base image (starting point)
# ============================================================
# FROM = "start with this pre-built computer"
# python:3.10-slim = Python 3.10 installed on a SLIM Linux OS
# "slim" means minimal size - no unnecessary extras
# This is like choosing a clean empty laptop with Python ready

FROM python:3.10-slim


# ============================================================
# STEP 2: Set the working directory inside the container
# ============================================================
# WORKDIR = "go to this folder" (creates it if doesn't exist)
# All future commands will run from /app folder
# Think of it like: cd /app

WORKDIR /app


# ============================================================
# STEP 3: Copy requirements.txt FIRST (smart caching trick!)
# ============================================================
# COPY source destination
# We copy ONLY requirements.txt first (not everything)
# Why? Docker caches each step - if requirements.txt didn't
# change, it SKIPS reinstalling libraries next time = FASTER!

COPY requirements.txt .


# ============================================================
# STEP 4: Install all Python libraries
# ============================================================
# RUN = "execute this command during build"
# --no-cache-dir = don't save download cache (saves disk space)
# This installs tensorflow, fastapi, uvicorn, pillow etc.
# This step takes the longest (downloading libraries)

RUN pip install --no-cache-dir -r requirements.txt


# ============================================================
# STEP 5: Copy ALL project files into the container
# ============================================================
# COPY . .  means "copy everything from current folder
#            to the container's /app folder"
# This copies: app.py, model/ folder, dataset/ etc.
# Note: We do this AFTER pip install for faster rebuilds

COPY . .


# ============================================================
# STEP 6: Tell Docker which port our app uses
# ============================================================
# EXPOSE = "this container will use port 8000"
# This is just documentation/label - doesn't actually open it
# The actual port mapping happens when running the container

EXPOSE 8000


# ============================================================
# STEP 7: The command to run when container starts
# ============================================================
# CMD = "run this command when the container launches"
# This starts the FastAPI server using uvicorn
# --host 0.0.0.0 = accept connections from outside container
# --port 8000    = run on port 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]