# Human Computer Interaction API (Backend)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/hci-back.git
   cd hci-back
   ```

2. Create and activate a Virtual Environment
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install requirements.txt
   ```
4. Create a `.env` file and add the OpenAI API key variable
   ```
   OPENAI_API_KEY= "<YOUR OPENAI API KEY>"
   ```

## Usage

Start the development server:

```bash
fastapi dev main.py
```

The API will be available at `http://127.0.0.1:8000` by default.

There's also an interactive API available at `http://127.0.0.1:8000/docs`
