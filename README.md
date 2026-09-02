# Razorpay AI Revenue Recovery: Causal Inference Prototype

## The Problem
Merchants often optimize payment recovery based on **Gross Recovery**—the revenue collected *after* a recovery intervention (like sending a WhatsApp reminder). However, this is heavily confounded. High-value, engaged customers might pay anyway (self-cure). If an intervention is disproportionately applied to these customers, it appears artificially successful. 

We need to optimize for **True Incremental Recovery**—the revenue collected *because* of the intervention.

### Why this is not a smart dunning system
This system does not primarily decide "How do we retry failed payments better?". It asks: *"Did the recovery intervention itself create the payment, and where should future recovery effort be allocated based on that incremental effect?"*

## Architecture & Synthetic Environment
Because true incremental revenue is fundamentally unobservable in real life (the Fundamental Problem of Causal Inference), we built a **Synthetic Causal Environment**.

1. **Simulator**: Generates customers with hidden potential outcomes ($Y_{none}$, $Y_{retry}$, $Y_{whatsapp}$).
2. **Selection Bias**: Historically assigns interventions based on observables (e.g., Enterprise customers get more WhatsApp), creating confounded data.
3. **Causal Engine**: Uses a Doubly Robust / T-Learner approach with Random Forests to estimate Conditional Average Treatment Effects (CATE).
4. **Policies**:
   - **Baseline (Gross Recovery)**: Chooses the intervention with the highest historical gross recovery rate for that segment.
   - **Incrementality-Aware (Our Policy)**: Uses the Causal Engine to choose the action that maximizes Expected Net Incremental Value (ENIV).
   - **Oracle**: Has access to the hidden ground truth and makes perfect decisions (theoretical upper bound).

## How to Run

### Backend
```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt # (fastapi uvicorn pandas numpy scikit-learn pydantic)
python api/main.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Results & Limitations
The causal engine uncovers instances where a cheaper intervention (Retry) has a higher *incremental* effect than a more expensive one (WhatsApp), even if WhatsApp has higher *gross* recovery. By penalizing costs against only the *incremental* probability, the Incrementality-Aware policy outperforms the Baseline policy in True Incremental Revenue. 

**Limitations:** The dataset is synthetic. Real-world causal inference suffers from unobserved confounders, which this simulation neatly avoids by design. The Oracle policy proves that even our machine learning model leaves money on the table compared to perfect knowledge.
