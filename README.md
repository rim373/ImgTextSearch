# ElasticImageSearch

A full-text search engine for images using Elasticsearch, Logstash, Kibana, and a web frontend built with Express.js, Tailwind CSS, and vanilla JS.

---

## 🏗 Project Pipeline Overview

The project implements a full-text image search pipeline:

1. **Data Storage & Indexing**
   - Images and metadata are stored and indexed in **Elasticsearch**.
   - Elasticsearch provides fast and scalable search capabilities.

2. **Data Ingestion**
   - **Logstash** is used to parse and feed image metadata into Elasticsearch.
   - Supports batch and real-time ingestion.

3. **Visualization & Monitoring**
   - **Kibana** is used to explore and visualize image indices.
   - You can monitor indexing status, view search results, and create dashboards.

4. **Backend Server**
   - Built with **Express.js**.
   - Provides APIs for:
     - Searching images
     - Serving image metadata
     - Handling frontend requests

5. **Frontend**
   - Built with **Vanilla JS**, **HTML**, and **Tailwind CSS**.
   - Features:
     - Search bar for querying images
     - Grid display of search results
     - Responsive layout

---

## ⚙️ Setup Instructions

### Prerequisites
- Node.js ≥ 18
- Elasticsearch ≥ 8.x
- Logstash ≥ 8.x
- Kibana ≥ 8.x
- npm or yarn

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ElasticImageSearch.git
cd ElasticImageSearch
