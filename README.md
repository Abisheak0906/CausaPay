````markdown
# CausaPay

**Causal Recovery Intelligence for Payment Operations**

CausaPay is an AI-assisted decision intelligence platform for payment recovery. Instead of treating every recovery event the same way, CausaPay uses causal inference and outcome data to estimate which intervention is most likely to create **incremental recovery**.

The system is designed to answer a more useful question than:

> Which customer is likely to pay?

It asks:

> **Which action is most likely to cause this recovery outcome?**

CausaPay combines causal estimation, event-level decisioning, policy comparison, model diagnostics, outcome feedback, and an interactive frontend into a deployable end-to-end application.

## Live Demo

### Frontend

**Live Application:** https://causa-pay-olive.vercel.app/

### Backend

**API Health Check:** https://causapay.onrender.com/api/health

The backend health endpoint confirms application status, model availability, database connectivity, and fallback configuration.

> Note: The root backend URL does not expose a `/` route. API endpoints are available under `/api/...`.

---

## The Problem

Traditional recovery systems often optimize for correlation.

For example:

- Customers who receive Action A may recover more often.
- Customers who receive Action B may recover less often.
- A predictive model may therefore conclude that Action A is better.

But this can be misleading.

Different customers may receive different interventions because of their risk level, payment history, outstanding amount, or other characteristics. Simply observing which group recovered more does not tell us whether the intervention itself caused the difference.

CausaPay approaches the problem from a causal perspective.

For each recovery scenario, the platform estimates the expected incremental impact of an intervention and supports decisions based on:

- customer and event characteristics
- treatment assignment
- observed outcomes
- causal effect estimation
- policy comparison
- model diagnostics
- outcome feedback

The goal is to move from:

```text
Correlation → Prediction
````

towards:

```text
Intervention → Causal Effect → Decision → Outcome → Feedback
```

---

# Key Features

## 1. Causal Decision Engine

CausaPay implements a causal framework for estimating the incremental impact of recovery interventions.

The decision system is designed around treatment-effect estimation rather than simple outcome prediction.

This helps answer questions such as:

* Should this event receive an intervention?
* Which recovery action is expected to create the highest incremental impact?
* What would likely happen under alternative treatment strategies?
* How does the recommended policy compare with baseline policies?

---

## 2. Cross-Fitted AIPW Estimation

The project uses a causal estimation approach based on **Augmented Inverse Probability Weighting (AIPW)** with cross-fitting.

Conceptually, the estimator combines:

* an outcome model
* a treatment or propensity model
* inverse probability weighting
* augmentation for robustness

The objective is to estimate treatment effects while accounting for differences in the characteristics of events receiving different interventions.

A simplified intuition is:

```text
Observed Outcome
        +
Correction using Treatment Propensity
        +
Outcome Model Adjustment
        ↓
Estimated Incremental Treatment Effect
```

Cross-fitting helps reduce overfitting-related bias by separating model training and evaluation across folds.

---

## 3. Event-Level Recovery Decisions

The API can evaluate individual events and return decision-oriented information.

The application supports workflows around:

* event retrieval
* event-level decisions
* next-action recommendations
* outcome recording
* audit information

This makes the causal model usable as part of an operational workflow rather than only as an offline analysis.

---

## 4. Policy Comparison

CausaPay compares decision strategies and recovery policies.

The platform can be used to evaluate the relative impact of different approaches, helping answer questions such as:

* How does the causal policy compare with a baseline strategy?
* What is the expected recovery impact?
* How many interventions can potentially be avoided?
* Where does targeted treatment outperform blanket intervention?

The goal is not simply to maximize the number of interventions.

It is to identify interventions that are expected to create meaningful incremental value.

---

## 5. Model Diagnostics

The backend exposes diagnostic information to help inspect the behavior of the causal system.

Diagnostics can support analysis of:

* model performance
* treatment distribution
* causal estimation behavior
* evaluation metrics
* policy impact

This makes the system more transparent than a simple black-box recommendation engine.

---

## 6. Evidence and Decision Explanation

The frontend includes an evidence-oriented interface that communicates the reasoning and signals behind the decision framework.

The application presents information related to:

* causal framework
* recovery impact
* treatment recommendations
* model evidence
* decision flow

The purpose is to make recommendations easier to inspect and communicate.

---

## 7. Outcome Feedback Loop

Recovery outcomes can be recorded back into the system.

Supported workflow endpoints allow outcomes to be associated with events, creating a foundation for:

```text
Decision
   ↓
Intervention
   ↓
Observed Outcome
   ↓
Evaluation
   ↓
Future Model Improvement
```

This creates a feedback-oriented architecture instead of a one-time static prediction system.

---

## 8. Dataset Upload and Evaluation

CausaPay supports dataset-oriented workflows.

The backend includes functionality for:

* dataset upload
* schema inspection
* evaluation
* diagnostics
* model and policy analysis

This allows the system to be used beyond a single hardcoded demonstration scenario.

---

# System Architecture

```text
                         ┌──────────────────────┐
                         │      User / Demo     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────┐
                    │       React + Vite UI       │
                    │                             │
                    │  • Dashboard                │
                    │  • Event Analysis           │
                    │  • Evidence Panel           │
                    │  • Policy Comparison        │
                    │  • Diagnostics              │
                    └──────────────┬──────────────┘
                                   │
                                   │ HTTPS / REST API
                                   ▼
                    ┌─────────────────────────────┐
                    │        FastAPI Backend      │
                    │                             │
                    │  /api/decision              │
                    │  /api/events                │
                    │  /api/summary               │
                    │  /api/diagnostics           │
                    │  /api/evaluate              │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
             ┌────────────┐ ┌────────────┐ ┌────────────┐
             │ Causal ML  │ │  Database  │ │   Redis /  │
             │   Engine   │ │            │ │  Fallback  │
             └────────────┘ └────────────┘ └────────────┘
                    │
                    ▼
          ┌──────────────────────────┐
          │ Cross-Fitted AIPW Engine │
          │                          │
          │ • Outcome Models         │
          │ • Propensity Models      │
          │ • Treatment Effects      │
          │ • Policy Evaluation      │
          └──────────────────────────┘
```

---

# Technology Stack

## Frontend

* React
* TypeScript
* Vite
* CSS
* Lucide Icons

## Backend

* Python
* FastAPI
* Uvicorn
* Pydantic
* SQLAlchemy

## Machine Learning and Data

* scikit-learn
* NumPy
* Pandas
* SciPy
* Cross-fitted causal estimation
* Augmented Inverse Probability Weighting

## Infrastructure

* Vercel — Frontend deployment
* Render — Backend deployment
* SQLite / database-backed local deployment support
* Redis with graceful fallback behavior

---

# API Endpoints

## Health

### `GET /api/health`

Returns the current backend status.

Example:

```json
{
  "status": "ok",
  "model_loaded": true,
  "database": {
    "status": "connected"
  },
  "redis": {
    "enabled": false,
    "connected": false,
    "fallback_mode": true
  },
  "n8n": {
    "enabled": false,
    "configured": false,
    "webhook_url_configured": false,
    "mode": "local"
  }
}
```

---

## Summary

### `GET /api/summary`

Returns high-level information and summary metrics for the system.

---

## Decision

### `POST /api/decision`

Runs the causal decision workflow for a recovery scenario.

---

## Policy Comparison

### `GET /api/policy-comparison`

Provides information for comparing recovery policies or intervention strategies.

---

## Model Diagnostics

### `GET /api/model-diagnostics`

Returns diagnostics related to the causal model.

---

## Events

### `GET /api/events`

Retrieves available events.

### `GET /api/events/{event_id}`

Retrieves details for a specific event.

### `GET /api/events/{event_id}/audit`

Returns audit-oriented information for an event.

---

## Event Outcomes

### `POST /api/events/{event_id}/outcome`

Records an observed outcome for an event.

### `POST /api/workflows/outcome`

Supports workflow-based outcome handling.

---

## Next Action

### `POST /api/events/{event_id}/next-action`

Generates or records the next recommended action for an event.

---

## Diagnostics

### `GET /api/diagnostics`

Returns additional system or evaluation diagnostics.

---

## Dataset Upload

### `POST /api/upload-dataset`

Uploads data for processing and evaluation.

---

## Evaluation

### `GET /api/evaluate/schema`

Returns the expected evaluation schema.

### `POST /api/evaluate`

Runs an evaluation workflow.

### `POST /evaluation/run`

Runs the evaluation endpoint available outside the `/api` prefix.

---

# Local Development

## Prerequisites

Install:

* Python
* Node.js
* npm

Clone the repository:

```bash
git clone https://github.com/Abisheak0906/CausaPay.git
cd CausaPay
```

---

# Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI application:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The backend should be available at:

```text
http://127.0.0.1:8000
```

Test the health endpoint:

```text
http://127.0.0.1:8000/api/health
```

---

# Frontend Setup

Open another terminal and navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

Vite will provide a local development URL, typically:

```text
http://localhost:5173
```

---

# Production Build

To verify that the frontend builds successfully:

```bash
cd frontend
npm install
npm run build
```

The production build is generated by Vite.

---

# Deployment

## Frontend Deployment

The frontend is deployed on Vercel.

Production URL:

```text
https://causa-pay-olive.vercel.app
```

Deployment workflow:

```text
GitHub
   ↓
Push to main
   ↓
Vercel Build
   ↓
Production Deployment
```

---

## Backend Deployment

The FastAPI backend is deployed on Render.

Production URL:

```text
https://causapay.onrender.com
```

Health check:

```text
https://causapay.onrender.com/api/health
```

The production server runs using:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Deployment workflow:

```text
GitHub
   ↓
Render Build
   ↓
Install Python Dependencies
   ↓
Start Uvicorn
   ↓
Load Causal Model
   ↓
Connect Database
   ↓
Expose REST API
```

---

# Deployment Notes

The backend intentionally does not define a root route:

```text
GET /
```

Therefore, visiting:

```text
https://causapay.onrender.com/
```

may return:

```json
{
  "detail": "Not Found"
}
```

This does **not** indicate that the deployment has failed.

Use the health endpoint instead:

```text
https://causapay.onrender.com/api/health
```

A successful response confirms that:

* the FastAPI application is running
* the causal model has loaded
* the database is connected
* the service is reachable

---

# Causal Decision Framework

A simplified representation of the CausaPay workflow is:

```text
                    Historical Recovery Data
                              │
                              ▼
                  ┌──────────────────────┐
                  │ Feature Preparation  │
                  └──────────┬───────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
        ┌──────────────────┐     ┌──────────────────┐
        │ Outcome Model    │     │ Propensity Model │
        │ P(Y | X, T)      │     │ P(T | X)         │
        └────────┬─────────┘     └────────┬─────────┘
                 │                        │
                 └───────────┬────────────┘
                             ▼
                 ┌──────────────────────┐
                 │ Cross-Fitted AIPW    │
                 │ Treatment Estimator  │
                 └──────────┬───────────┘
                            │
                            ▼
                 Estimated Incremental
                    Recovery Impact
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Recovery Decision    │
                 │ / Next Best Action   │
                 └──────────┬───────────┘
                            │
                            ▼
                     Observed Outcome
                            │
                            ▼
                       Feedback Loop
```

---

# Why Causal Inference?

A predictive model can estimate:

```text
P(Customer Recovers)
```

But operational decisions often require estimating:

```text
What happens if we intervene?
```

These are not necessarily the same problem.

A customer may have a high probability of recovering without intervention. Sending them an expensive recovery action may produce little incremental value.

Another customer may have a lower baseline recovery probability but respond strongly to a specific intervention.

CausaPay is designed around estimating this **incremental impact**.

The decision objective is therefore closer to:

```text
Expected Value of Intervention
=
Expected Outcome With Action
-
Expected Outcome Without Action
```

This distinction is central to the project's architecture.

---

# Example Decision Flow

```text
Payment Recovery Event
        │
        ▼
Extract Event Features
        │
        ▼
Estimate Baseline Outcome
        │
        ▼
Estimate Treatment Propensity
        │
        ▼
Estimate Incremental Causal Effect
        │
        ▼
Compare Possible Actions
        │
        ▼
Select Recommended Action
        │
        ▼
Record Outcome
```

---

# Project Structure

```text
CausaPay/
│
├── backend/
│   ├── api/
│   │   └── main.py
│   │
│   ├── requirements.txt
│   └── ...
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── App.tsx
│   │   ├── App.css
│   │   └── ...
│   │
│   ├── package.json
│   └── ...
│
├── README.md
└── ...
```

---

# Design Principles

CausaPay is built around the following principles:

### Incremental Impact Over Raw Prediction

The objective is not simply to predict who will recover.

The objective is to estimate where an intervention is expected to make a difference.

### Decision-Oriented ML

Model outputs should support an operational decision rather than exist only as an accuracy metric.

### Transparency

Diagnostics, policy comparison, evidence, and event-level information are included to make the decision process more inspectable.

### Feedback

Observed outcomes can feed back into the evaluation workflow, creating a foundation for continuous improvement.

### Deployability

The project is designed to run as a complete system:

```text
Frontend
    +
Backend API
    +
Causal Model
    +
Database
    =
Deployable Decision Intelligence Application
```

---

# Current Status

CausaPay is currently deployed end-to-end.

| Component    | Status                        | Platform |
| ------------ | ----------------------------- | -------- |
| Frontend     | Live                          | Vercel   |
| Backend API  | Live                          | Render   |
| Causal Model | Loaded                        | Render   |
| Database     | Connected                     | Backend  |
| Redis        | Optional / Fallback Supported | Backend  |
| Health Check | Operational                   | Render   |

---

# Future Improvements

Potential next steps for CausaPay include:

* real-time payment event ingestion
* experiment tracking
* treatment-effect confidence intervals
* richer counterfactual explanations
* intervention cost optimization
* multi-action treatment policies
* model retraining pipelines
* drift detection
* automated evaluation workflows
* authentication and role-based access
* production-grade database configuration
* asynchronous background jobs
* workflow orchestration integration
* monitoring and observability dashboards

---

# Demo Checklist

To verify the complete deployment:

### 1. Open the frontend

```text
https://causa-pay-olive.vercel.app
```

### 2. Verify backend health

```text
https://causapay.onrender.com/api/health
```

### 3. Confirm

* the frontend loads
* backend API requests succeed
* event data is available
* causal decision workflows respond
* diagnostics are accessible
* the application functions end-to-end

---

# Repository

GitHub:

```text
https://github.com/Abisheak0906/CausaPay
```

---

# Disclaimer

CausaPay is a project demonstrating the application of causal inference and decision intelligence to payment recovery workflows.

The outputs of the system should be evaluated appropriately before use in high-stakes production financial decisions. Real-world deployment should include proper validation, monitoring, experimentation, governance, security, and compliance controls.

---

# License

This project is intended for educational, research, hackathon, and demonstration purposes unless otherwise specified by the repository license.

---

## CausaPay

**From predicting recovery to estimating the impact of recovery actions.**

```text
Observe
   ↓
Understand
   ↓
Estimate Causal Impact
   ↓
Choose an Intervention
   ↓
Measure the Outcome
   ↓
Learn
```

**CausaPay — Causal Recovery Intelligence for Better Payment Decisions.**

```
```
