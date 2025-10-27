  EquiHire – LLM-Driven Adaptive Interview Platform  


Problem Statement
Traditional hiring uses generic assessments that ignore real skills and can introduce bias.  
**EquiHire**solves this by providing *personalized, fair, and transparent* AI-based interviews.

## Vision
- Adaptive interview flow tailored to each candidate  
- Skill graph mapping and evidence-based scoring  
- Real-time fairness and bias monitoring  

##  System Architecture
**Front-End:** Streamlit – candidate & recruiter UI  
**Back-End:** Python + Pandas for data & logic  
**LLM Module:** Groq API (LLaMA 3.1) for adaptive question generation  
**Fairness Module:** AIR (Adverse Impact Ratio) visualization using Streamlit charts  



## ⚙️ Features
- Personalized question generation  
- Dynamic skill graph visualization  
- Transparent scoring (0–100)  
- Fairness dashboard for bias detection  

 Tech Stack
Python, Streamlit, Groq LLM, Pandas, Matplotlib, NetworkX

 Future Enhancements
- Video interviews with sentiment analysis  
- Large-scale data integration  
- Company-level fairness analytics  

 Run Locally
```bash
git clone https://github.com/<your-username>/EquiHire.git
cd EquiHire
pip install -r requirements.txt
streamlit run app.py
