# Vehicle_Detection_webapp
Computer Vision project. Vehicle detection in images. Complete ML pipeline with model tuning and web interface.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Installation & Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Vehicle_Detection_webapp.git
```

#### 2. Backend Setup
```bash
cd src/api
python -m venv venv
```

- For Windows:
```bash
venv\Scripts\activate
```
- For MacOS/Linux:
```bash
source venv/bin/activate
```
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Frontend Setup
```bash
cd ../frontend
pip install -r requirements.txt
```
#### 4. Run Backend
```bash
cd src/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Backend will be available at: http://localhost:8000

#### 5. Run Frontend
```bash
cd src/frontend
streamlit run frontend.py
```
