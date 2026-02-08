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

#### 2. Сreating a virtual environment
```bash
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
#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
#### 4. Download and Setup Models
Models are too large for GitHub. You need to download them manually:
- Download models from: https://cloud.mail.ru/public/NKd4/zFmfaEUeB
- Extract downloaded files into src/models/ folder
- Verify structure should look like:
```text
src/models/
├── yolov8nano.pt
└── yolov8m.pt
```

#### 5. Run Backend
```bash
cd src/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Backend will be available at: http://localhost:8000

#### 6. Run Frontend
```bash
cd src/frontend
streamlit run frontend.py
```
