<!DOCTYPE html>
<html lang="en">
<body>
    <br>
    <br>
    python3.11 -m venv agent-env
    <br>
    <br>
    source agent-env/bin/activate
    <br>
    <br>
    pip install --upgrade pip setuptools wheel ipywidgets notebook
    <br>
    <br>
    pip install -r requirements.txt
    <br>
    <br>
    pm2 start agent-env/bin/python --name "agent-app" -- -m uvicorn app:app --host 127.0.0.1 --port 8002
</html>