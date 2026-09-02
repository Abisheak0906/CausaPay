# CausaPay: AI-Powered Causal Payment Recovery Engine

CausaPay is an AI-powered payment recovery intelligence platform that helps businesses determine the optimal intervention for failed payments.

Instead of applying the same recovery strategy to every failed payment, CausaPay uses causal inference to estimate the incremental impact of different recovery actions. The system evaluates whether an intervention such as retrying a payment or sending a WhatsApp reminder is expected to generate additional revenue beyond what would have happened naturally.

The objective is not simply to maximize recovery attempts. CausaPay aims to maximize incremental recovery while minimizing unnecessary intervention costs and avoiding actions that would have resulted in self-cure.

## Problem Statement

Payment failures are a significant challenge for subscription and recurring payment businesses. Traditional recovery systems often follow static rules such as retrying every failed payment after a fixed interval or sending reminders to every customer.

This approach creates several problems:

- Some customers would successfully complete payment without any intervention.
- Repeated retries can increase payment processing costs.
- Communication campaigns can create unnecessary customer friction.
- Different customers respond differently to the same intervention.
- Traditional predictive models estimate the probability of recovery but do not estimate whether an intervention actually caused the recovery.

CausaPay addresses this problem using causal inference and decision intelligence.

## Solution

For every failed payment, CausaPay evaluates the expected incremental value of available recovery actions.

The system compares potential interventions against the estimated counterfactual outcome: what would likely have happened if no intervention had been performed.

Each action is evaluated using its expected incremental net value.

**ENIV = Expected Incremental Recovery Value − Intervention Cost**

The system recommends an action only when it produces positive expected incremental value.

Possible decisions include:

- `RETRY`
- `WHATSAPP`
- `NONE`
- `ABSTAIN`

The system also applies operational safeguards such as WhatsApp consent validation, cooldown restrictions, intervention limits, and uncertainty-based abstention.

## System Flow

```text
                         ┌──────────────────────┐
                         │   Payment Failure    │
                         │      Occurs          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Input Data Layer   │
                         │                      │
                         │ • Manual Input       │
                         │ • CSV Upload         │
                         │ • Payment Events     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Feature Processing        │
                    │                               │
                    │ • Payment History             │
                    │ • Failure Context             │
                    │ • Engagement Signals          │
                    │ • Customer Attributes         │
                    │ • Consent Information         │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Causal AI Engine         │
                    │                               │
                    │ Estimate Counterfactuals      │
                    │                               │
                    │ What happens with:            │
                    │                               │
                    │ • No Intervention             │
                    │ • Retry                       │
                    │ • WhatsApp                    │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Incrementality Estimation   │
                    │                               │
                    │ Calculate Incremental Effect  │
                    │ of Each Intervention          │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      ENIV Calculation         │
                    │                               │
                    │ Expected Incremental Value    │
                    │              −                │
                    │ Intervention Cost             │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Decision Engine          │
                    │                               │
                    │ Select Highest Positive ENIV  │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Safety and Policy Layer     │
                    │                               │
                    │ • Consent Validation          │
                    │ • Cooldown Checks             │
                    │ • Intervention Limits         │
                    │ • Uncertainty Abstention      │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Final Decision     │
                         │                      │
                         │  RETRY               │
                         │  WHATSAPP            │
                         │  NONE                │
                         │  ABSTAIN             │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Execution and Tracking    │
                    │                               │
                    │ • Recovery Workflow           │
                    │ • State Tracking              │
                    │ • Audit Logs                  │
                    │ • Decision History            │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       Analytics Dashboard     │
                    │                               │
                    │ • Incremental Recovery        │
                    │ • Gross Recovery              │
                    │ • Self-Cure Estimation        │
                    │ • ENIV                        │
                    │ • Intervention Cost           │
                    │ • Action Distribution         │
                    └───────────────────────────────┘
```

## Key Features

### Causal Decision Making

CausaPay estimates the causal impact of each recovery intervention instead of relying only on correlation or recovery probability.

### Counterfactual Estimation

The platform estimates what would likely happen under different treatment scenarios, including no intervention, payment retry, and customer communication.

### Incrementality Optimization

The system prioritizes actions that generate additional recovery rather than crediting interventions for payments that would have recovered naturally.

### Expected Incremental Net Value

Each intervention is evaluated based on its expected financial value after accounting for intervention costs.

### Uncertainty-Based Abstention

When the model does not have sufficient confidence in the expected benefit of an intervention, the system can abstain instead of taking unnecessary action.

### Operational Safeguards

The decision layer supports safeguards such as:

- WhatsApp opt-in validation
- Intervention cooldown periods
- Maximum intervention limits
- Blocked action handling
- Fallback actions
- Decision auditability

### CSV Dataset Upload

Users can upload their own payment failure dataset through the dashboard.

The uploaded data is processed through the CausaPay decision engine, and the dashboard dynamically updates based on the uploaded dataset.

The system supports fields such as:

```text
event_id
customer_id
payment_id
amount
plan_tier
payment_method
failure_context
decline_signal_bucket
engagement_score
historical_failure_count
whatsapp_opted_in
email_verified
days_since_last_failure
historical_payment_count
day_of_month
tenure_days
```

If `event_id` or `payment_id` is missing, the system generates identifiers during processing.

### Interactive Dashboard

The frontend provides multiple views for analyzing payment recovery performance, including:

- Executive Overview
- Policy Intelligence
- Payment Explorer
- Recovery Attribution
- Diagnostics
- Manual Payment Analysis
- CSV Dataset Upload
- Demo Dataset Switching
- Decision and Audit Information

## Technology Stack

### Frontend

- React
- TypeScript
- Vite

### Backend

- Python
- FastAPI
- SQLAlchemy

### Machine Learning and Causal Inference

- Causal modeling
- Augmented Inverse Probability Weighting
- Counterfactual estimation
- Incrementality estimation
- Uncertainty-aware decision making

### Data and Infrastructure

- SQLite
- REST APIs
- CSV ingestion and validation

## Project Structure

```text
CausaPay/
│
├── backend/
│   ├── api/
│   │   └── main.py
│   │
│   ├── causal_engine/
│   │   └── aipw_estimator.py
│   │
│   ├── evaluation/
│   │   └── engine.py
│   │
│   ├── policies/
│   │   └── incrementality.py
│   │
│   ├── simulator/
│   │   ├── generator.py
│   │   └── bias.py
│   │
│   └── requirements.txt
│
├── frontend/
│   └── src/
│       ├── components/
│       ├── lib/
│       ├── App.tsx
│       └── types.ts
│
├── README.md
└── .gitignore
```

## Running the Project

### Backend

Navigate to the backend directory:

```powershell
cd backend
```

Activate the Python environment if required:

```powershell
.\.venv\Scripts\Activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the FastAPI server:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

### Frontend

Open another terminal and navigate to the frontend directory:

```powershell
cd frontend
```

Install dependencies:

```powershell
npm install
```

Start the development server:

```powershell
npm run dev
```

Open the local address displayed by Vite in your browser.

## API Endpoints

### Health Check

```text
GET /api/health
```

Checks whether the backend service is running.

### Payment Decision

```text
POST /api/decision
```

Accepts an individual failed payment event and returns the recommended recovery action along with causal decision information.

### Dataset Upload

```text
POST /api/upload-dataset
```

Accepts a CSV dataset, validates the input, processes payment events, persists decision results, and returns updated dataset-level analytics.

### Payment Events

```text
GET /api/events
```

Retrieves processed payment events.

### Decision Audit

```text
GET /api/events/{event_id}/audit
```

Retrieves the decision history and audit information for a specific payment event.

## Decision Logic

```text
                    Failed Payment
                          │
                          ▼
               Estimate No-Action Outcome
                          │
                          ▼
             Estimate Intervention Outcomes
                    │           │
                    ▼           ▼
                  RETRY      WHATSAPP
                    │           │
                    └─────┬─────┘
                          │
                          ▼
              Calculate Incremental Effect
                          │
                          ▼
                    Subtract Cost
                          │
                          ▼
                     Calculate ENIV
                          │
                          ▼
               Is ENIV Positive and Safe?
                     │            │
                   YES            NO
                     │            │
                     ▼            ▼
              Apply Best       NONE /
               Intervention    ABSTAIN
```

## Example Outcomes

### Retry

The system recommends `RETRY` when retrying the payment produces the highest positive expected incremental net value.

### WhatsApp

The system recommends `WHATSAPP` when customer communication is expected to create additional recovery and the customer has provided the required opt-in consent.

### None

The system recommends `NONE` when interventions do not provide sufficient incremental value.

### Abstention

The system abstains when model uncertainty is high or when the available information does not support a sufficiently confident intervention.

## Core Principle

The central idea behind CausaPay is:

> Do not ask which customers are most likely to pay. Ask which intervention is most likely to cause additional payment recovery.

This distinction helps reduce unnecessary retries, avoid spending resources on customers who would self-cure, and focus interventions where they can create measurable incremental value.

## Future Improvements

Potential future extensions include:

- Integration with live payment gateways
- Real-time event streaming
- Production-scale databases
- Advanced treatment effect models
- Dynamic intervention timing
- Additional recovery channels
- Automated model retraining
- A/B testing integration
- Cloud deployment
- Role-based authentication
- Multi-tenant support

## License

This project is currently developed as a prototype and hackathon project.
