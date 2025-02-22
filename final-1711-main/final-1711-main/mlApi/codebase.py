# from PIL import Image
# import torch
# from transformers import CLIPProcessor, CLIPModel
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression

# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# complaints = [
#     "dirty platform",
#     "broken equipment",
#     "overcrowded platform",
#     "unsafe conditions",
#     "missing amenities",
#     "inadequate lighting",
#     "broken signage",
#     "empty platform",
#     "poor seating",
#     "lack of information"
# ]

# complaint_descriptions = [
#     "dirty platform",
#     "broken equipment",
#     "overcrowded platform",
#     "unsafe conditions",
#     "missing amenities",
#     "inadequate lighting",
#     "broken signage",
#     "empty platform",
#     "poor seating",
#     "lack of information"
# ]

# complaint_categories = [
#     "Cleanliness",
#     "Maintenance",
#     "Crowd Control",
#     "Safety",
#     "Facilities",
#     "Lighting",
#     "Signage",
#     "Timeliness",
#     "Comfort",
#     "Information"
# ]

# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(complaint_descriptions)
# classifier = LogisticRegression()
# classifier.fit(X, complaint_categories)


# def generate_complaint_description(image):
#     inputs = processor(text=complaints, images=image, return_tensors="pt", padding=True).to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits_per_image = outputs.logits_per_image
#     probs = logits_per_image.softmax(dim=1)
#     description = complaints[probs.argmax().item()]
#     return description


# def classify_complaint_description(description):
#     X_desc = vectorizer.transform([description])
#     category = classifier.predict(X_desc)[0]
#     return category


from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pytesseract
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from pymongo import MongoClient
import io

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["complaints_db"]
complaints_collection = db["complaints"]

# For Windows, specify the path to tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP Model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Complaint categories
complaints = [
    "dirty platform",
    "broken equipment",
    "overcrowded platform",
    "unsafe conditions",
    "missing amenities",
    "inadequate lighting",
    "broken signage",
    "empty platform",
    "poor seating",
    "lack of information"
]

complaint_descriptions = complaints

complaint_categories = [
    "Cleanliness",
    "Maintenance",
    "Crowd Control",
    "Safety",
    "Facilities",
    "Lighting",
    "Signage",
    "Timeliness",
    "Comfort",
    "Information"
]

# Train Text Classifier
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(complaint_descriptions)
classifier = LogisticRegression()
classifier.fit(X, complaint_categories)

# Function to generate complaint description from image
def generate_complaint_description(image):
    inputs = processor(text=complaints, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    description = complaints[probs.argmax().item()]
    return description

# Function to classify complaint description into categories
def classify_complaint_description(description):
    X_desc = vectorizer.transform([description])
    category = classifier.predict(X_desc)[0]
    return category

# OCR Preprocessing
def simple_preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return binary

# Perform OCR
def perform_ocr(image):
    processed_image = simple_preprocess(image)
    text = pytesseract.image_to_string(processed_image)
    return text

# FastAPI Application
app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running!"}

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Generate complaint description
    description = generate_complaint_description(image)

    # Classify the complaint
    category = classify_complaint_description(description)

    # Store in MongoDB
    complaint_data = {"description": description, "category": category}
    complaints_collection.insert_one(complaint_data)

    return {"description": description, "category": category}
