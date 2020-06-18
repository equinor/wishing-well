    FROM python:3
    EXPOSE 5000
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY hello.py .
    CMD ["python3","./hello.py"]