# ElasticImageSearch

ElasticImageSearch is a full-text image search engine that indexes image metadata and provides text search + visualization.  
Tech stack: **Elasticsearch**, **Logstash**, **Kibana**, **Express.js** (backend), and **Vanilla JS + HTML + Tailwind CSS** (frontend).

---

## Table of contents

- [Project overview](#project-overview)  
- [Pipeline diagram](#pipeline-diagram)  
- [Prerequisites](#prerequisites)  
- [Install & run (all components)](#install--run-all-components)  
  - [1. Elasticsearch](#1-elasticsearch)  
  - [2. Kibana](#2-kibana)  
  - [3. Logstash (ingestion)](#3-logstash-ingestion)  
  - [4. Backend — Express.js](#4-backend--expressjs)  
  - [5. Frontend — Vanilla JS + Tailwind](#5-frontend--vanilla-js--tailwind)  
- [Index mapping (recommended)](#index-mapping-recommended)  
- [Sample Logstash config (example)](#sample-logstash-config-example)  
- [API examples (curl)](#api-examples-curl)  
- [Project structure](#project-structure)  
- [Screenshots (placeholders)](#screenshots-placeholders)  
- [Troubleshooting & tips](#troubleshooting--tips)  
- [License](#license)

---

## Project overview

1. **Data storage & indexing:** Elasticsearch stores image metadata (title, description, tags, path/url, date, etc.) for high-speed text search.  
2. **Ingestion:** Logstash reads metadata (JSON/CSV) and writes documents into Elasticsearch.  
3. **Visualization:** Kibana queries the same indices for dashboards / monitoring.  
4. **Backend API:** Express.js exposes search endpoints used by the UI.  
5. **Frontend:** Simple, responsive UI (Vanilla JS + Tailwind) that calls the backend.

---

## Pipeline diagram

Mermaid (GitHub may render this automatically):

```mermaid
flowchart TD
  A[Image metadata (JSON files)] --> B[Logstash]
  B --> C[Elasticsearch]
  C --> D[Kibana]
  C --> E[Express.js API]
  E --> F[Frontend (Vanilla JS + Tailwind)]
