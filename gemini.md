# Financial Prediction Backend Project

This document outlines the purpose, architecture, and design principles of the Financial Prediction backend project.

---

## 1. Project Overview

This project is the backend system for a financial application. Its primary purpose is to fetch financial market data, process it, train a predictive machine learning model, and expose the results via a web API.

The core functionalities are:

1.  **Data Ingestion:** Fetches time-series financial data (Open, High, Low, Close, Volume) from the Alpha Vantage API.
2.  **Data Processing:** Cleans the raw data and transforms it into sequences suitable for training a machine learning model.
3.  **Model Training:** (Future Step) Uses the processed data to train a predictive model, which is then saved for later use.
4.  **API Endpoints:** Provides a simple REST API for other applications (like a mobile or web app) to interact with the backend, allowing them to trigger model training and retrieve predictions.

This project was initiated as a practical exercise to solidify Python programming skills, with a specific focus on applying both Object-Oriented and Functional Programming principles in a real-world scenario.

---

## 2. Project Architecture

The project is structured into distinct directories, each with a clear responsibility. This separation of concerns makes the codebase modular, scalable, and easy to maintain.

```
financial_prediction/
├── api/                # Handles all API-related logic (FastAPI).
│   └── main.py
├── data/               # Manages data sourcing and processing.
│   ├── connector.py
│   ├── extractor.py
│   └── processor.py
├── services/           # Orchestrates the core business logic.
│   └── data_pipeline_service.py
├── models/             # (Future) For model training logic and saved model files.
├── tests/              # Contains all tests for the application.
└── .env                # Stores secret keys and environment variables.
```

### Data Flow

The typical data flow for a training request is as follows:

1.  **API Layer (`api/main.py`):**
    *   A `POST` request is received at the `/train` endpoint.
    *   It validates the incoming request data (e.g., stock ticker).

2.  **Service Layer (`services/data_pipeline_service.py`):**
    *   The API calls the `run_data_pipeline` function, which acts as the central coordinator.

3.  **Data Layer (`data/`):
    *   **`connector.py`:** The service uses the `MarketAPIConnector` class to establish a connection with the Alpha Vantage API.
    *   **`extractor.py`:** The `get_raw_data` function is called to fetch the data and convert it from a raw JSON response into a structured pandas DataFrame.
    *   **`processor.py`:** The `clean_data` and `create_sequences` functions are called to clean and transform the DataFrame into feature (X) and target (y) arrays for the model.

4.  **Model Layer (`models/`):
    *   (Future) The features and targets are passed to a model trainer, which trains and saves a predictive model.

5.  **API Layer (Response):
    *   The API layer receives the results from the service and sends a JSON response back to the client, confirming the operation and providing a summary of the processed data.

---

## 3. The Synergy of OOP and FP

This project intentionally blends principles from both Object-Oriented Programming (OOP) and Functional Programming (FP). This hybrid approach leverages the strengths of each paradigm to create a robust and maintainable system.

### The Role of Object-Oriented Programming (OOP)

OOP is used to structure the application and manage its components. It helps define the "nouns" of the system.

*   **Encapsulation:** The `MarketAPIConnector` class is a prime example. It encapsulates all the logic and state (like the API key and session) required to communicate with the external API. The rest of the application doesn't need to know the internal details of how requests are made; it just uses the connector's public methods. This makes the code easier to manage and allows for swapping out the connector with a different one without changing the rest of the application.

*   **Abstraction & Data Structures:** FastAPI's `app` object is an instance of a class that abstracts away the complexities of the underlying web server. Similarly, Pydantic's `BaseModel` is used to create the `TickerRequest` class, which provides a clear, object-oriented structure for the data our API expects.

### The Role of Functional Programming (FP)

FP is used to handle the data processing and transformation logic. It helps define the "verbs" of the system in a predictable way.

*   **Stateless Functions:** The functions in `processor.py` (`clean_data`, `create_sequences`) are pure or near-pure functions. They take data as input and produce new, transformed data as output without causing side effects (i.e., they don't modify the original data). This makes them highly predictable, easy to reason about, and simple to test in isolation.

*   **Function Composition:** The entire data pipeline is a great example of function composition. The `run_data_pipeline` service is essentially a pipeline that chains together a series of functions: `get_raw_data() -> clean_data() -> create_sequences()`. This creates a clear and linear flow of data, making the logic easy to follow and debug.

### Why This Hybrid Approach is Important

By using OOP to structure the major components and FP to handle the data flow, we get the best of both worlds:

*   The **OOP structure** provides a solid, scalable foundation for the application.
*   The **FP approach** for data processing ensures that the core logic is predictable, testable, and less prone to bugs.

This combination results in a backend system that is both well-organized and easy to evolve over time.
