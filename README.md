
# 📦 Image and Text Search Engine with VGG16, Elasticsearch, FastAPI and HTML/CSS/JS

This project is a high-performance **image and text search engine** leveraging **deep learning** and modern search technology. It uses the **VGG16** model for feature extraction from images and **Elasticsearch** for fast indexing and retrieval of data. A seamless backend API is provided with **FastAPI** and a user-friendly frontend interface with **HTML/CSS/JS** is used for efficient data ingestion.

---


## 📜 Table of Contents

- [📂 Project Architecture](#-project-architecture)
- [🚀 How It Works](#-how-it-works)
  - [VGG16 Feature Extraction](#1-vgg16-feature-extraction)
  - [Elasticsearch Integration](#2-elasticsearch-integration)
  - [FastAPI Backend](#3-fastapi-backend)
  - [HTML/CSS/JS Web Interface](#4-streamlit-web-interface)
- [📸 Features](#-features)
- [🛠️ Technology Stack](#%EF%B8%8F-technology-stack)
- [🔧 How to Run the Project](#-how-to-run-the-project)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#1-clone-the-repository)
  - [Navigate to the Infra Folder](#2-navigate-to-the-infra-folder)
  - [Manually Run the Frontend and Backend](#3-manually-run-the-frontend-and-backend)
  - [Access the Web Interface and API](#5-access-the-web-interface-and-api)
- [📂 Folder Structure](#-folder-structure)
  - [Backend](#backend)
  - [Frontend](#frontend)
  - [Data](#data)
- [📊 Results](#-results)
  - [Global Interface Description](#global-interface-description)
  - [🔍 Search Features](#-search-features)
    - [Search by Text](#search-by-text)
    - [Search by Image](#search-by-image)

---

## 📂 **Project Architecture**
![Architecture](images/Architecture.png)

## 🚀 **How It Works**

### 1. **VGG16 Feature Extraction**  
- Uses the **VGG16 deep learning model** to extract high-level **feature vectors** from uploaded images.  
- These vectors represent images in a meaningful way, capturing important features like texture, color, and shape.

### 2. **Elasticsearch Integration**  
- Indexes both **image features and textual data** using **Elasticsearch**, ensuring **fast and accurate** search results with similarity matching. 

### 3. **FastAPI Backend**  
- Provides a **RESTful API** using **FastAPI** for handling search requests.  
- Users can query by **text** or **images** and receive relevant results from Elasticsearch.

### 4. **HTML/CSS/JS Web Interface**  
- Built with **HTML**, **CSS**, and **JavaScript** to provide an intuitive and responsive web interface.
- Enables users to **upload images**, enter **search queries**, and interactively explore the retrieved results.



---

## 📸 **Features**

- **Image Similarity Search**: Upload an image to retrieve visually similar results.
- **Text-based Search**: Search using keywords or descriptive phrases to find matching images.
- **Deep Learning with VGG16**: Utilizes VGG16 to extract powerful feature vectors, enhancing search accuracy.
- **FastAPI Backend**: Provides high-performance and reliable API endpoints for processing search requests.
- **Interactive UI with HTML/CSS/JS**: Provides a modern, responsive, and user-friendly web interface for easy interaction.


---

## 🛠️ **Technology Stack**

| Component         | Technology    |
|-------------------|---------------|
| **Feature Extraction** | VGG16 (Keras) |
| **Search Engine**     | Elasticsearch |
| **Backend API**       | FastAPI       |
| **Frontend Interface**| HTML CSS JS   |


---


## 🔧 **How to Run the Project**

### Prerequisites  
- Ensure all dependencies listed in the `requirements.txt` files are properly configured.

### 1. **Clone the Repository**

```bash
git clone https://github.com/rim373/ImgTextSearch.git
cd ImageText-Search-ElasticVGG
```

### 2. **Run the Frontend and Backend** 
#### 2.1 Run the Streamlit Frontend
```bash
cd Frontend
streamlit run app.py
```
Access the web interface at http://localhost:5000.

#### 2.2 Run the FastAPI Backend
```bash
cd Backend
uvicorn route:app --reload
```
#### 2.3 Run the Elasticsearch
```bash
cd elasticsearch-8.*.*\bin
bin\elasticsearch.bat
```

### 3. **Access the Web Interface and API**  
- **Frontend:** Open [http://localhost:5000](http://localhost:5000).  
- **Backend API (FastAPI):** Visit the Swagger docs at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 📂 **Folder Structure**

**Backend**

The Backend directory contains the core components of the backend service, responsible for data ingestion, feature extraction, and API routing.


```
├── app.py        # Configuration settings for the backend service
├── ingest_data_elastic.py   # Script to ingest data into Elasticsearch
├── feature_extractor.py      # Module for extracting features from the ingested data
├── Requirements.txt                  # The requirements 
        
```
**Frontend**

The Frontend directory for user interaction.

```
├── index.html                    # Main entry point for the frontend application
```
**Infra**

The data directory contains the project’s datasets, including images, their corresponding tags, and related JSON files used for metadata storage and indexing.

```

├── data 
│   ├── photo_metadata.json   # a Flickr dataset contain image link and their metadata
│   ├── images               # Download a dataset contain images/tags
|   ├── tags                 # Download a dataset contain images/tags   
```
---


## 📊 Results
### Global Interface Description
The interface of the image search engine is designed to be user-friendly and intuitive. It features a clean layout where users can easily access both the text and image search functionalities. Users can quickly navigate through the results, making the search experience efficient and enjoyable. The interface also allows for easy uploads of images for content-based searches, enhancing the overall usability of the application.

![Result](images/interface.png)

### 🔍 Search Features

#### Search by Text
- Users can enter keywords or phrases to find relevant images.
- The search is performed against both indexed image metadata and textual descriptions.
- Results display relevant images alongside their corresponding metadata.

![Text](images/text.png)
#### Search by Image
- Users can upload an image to perform a content-based search.
- The engine utilizes VGG16 to extract features from the uploaded image.
- Results display similar images based on visual content similarity.

![Image](images/image.png)

