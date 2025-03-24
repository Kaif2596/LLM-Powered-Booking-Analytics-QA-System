
# LLM-Powered Booking Analytics & QA System



## 📋  Description

This project develops an LLM-powered system to analyze hotel booking data and provide insights through analytics and natural language question-answering (QA). Built as part of an assignment, it processes a dataset of hotel bookings, generates key metrics like revenue trends and cancellation rates, and uses retrieval-augmented generation (RAG) to answer user queries. The system is deployed via a REST API using FastAPI, with additional features like real-time data updates and query history tracking. The goal is to create an efficient, scalable tool for hotel booking analysis and interaction.

## 🚀 Key Features

- **Data Preprocessing**: Cleans and structures hotel booking data.
- **Analytics**: Generates insights such as revenue trends, cancellation rates, geographical distribution, and lead time distribution.
- **RAG QA System**: Answers natural language questions using FAISS and an open-source LLM (Mistral).
- **REST API**: Exposes analytics and QA functionalities via FastAPI endpoints.
- **Bonus Features**: Includes real-time data updates, query history, and a health check endpoint.
## 🛠️ Installation and Setup


### Prerequisites
- Python 3.8+
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Kaif2596/LLM-Powered-Booking-Analytics-QA-System.git
   cd <LLM-Powered-Booking-Analytics-QA-System>

2. Ensure the dataset (cleaned_hotel_bookings.csv) is in the root directory.

### Running the Application

1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
2. The API will be available at http://127.0.0.1:8000.

### API Endpoints
- POST /analytics
- Description: Returns analytics reports.
- Request Body: None


### Sample Test Queries & Expected Answers

![image alt](https://github.com/Kaif2596/LLM-Powered-Booking-Analytics-QA-System/blob/main/Screenshot%20(25).png)


## 🧠 How It Works

- **Data Preprocessing**: Handled using pandas for missing values and date formatting.
- **Analytics**: Computed with pandas and visualized with Matplotlib/Seaborn.
- **RAG QA System**: Uses FAISS for vector search and Mistral (via Hugging Face) for QA.
- **REST API**: Built with FastAPI for fast and asynchronous endpoints.
- **Bonus Features**: SQLite for real-time updates, query history tracking, and health checks.
## 📊 Demo

![image alt](https://github.com/Kaif2596/LLM-Powered-Booking-Analytics-QA-System/blob/main/revenue_trends.png)

![image alt](https://github.com/Kaif2596/LLM-Powered-Booking-Analytics-QA-System/blob/main/lead_time_distribution.png)

![image alt](https://github.com/Kaif2596/LLM-Powered-Booking-Analytics-QA-System/blob/main/geo_distribution.png)

## 📈 Challenges:

*   **Dataset Size:**  Large datasets may slow down RAG; mitigated by optimizing FAISS retrieval.
*   **LLM Accuracy:** Fine-tuning might be needed for precise answers; currently relies on pre-trained Mistral.
*   **Performance:** API response time optimized by limiting retrieval to top 5 matches.
