    FROM python:3.7

    WORKDIR /usr/src/app
    
    COPY requirements.txt ./
    RUN pip install --no-cache-dir -r requirements.txt

    COPY . .
    
    #Install our custom openai gym
    RUN pip install --no-cache-dir -e ./env/gym-ww

    CMD [ "python","-u","./main.py" ]	

    #The "-u" is there to run it "unbuffered". That means that the printed output from python wil be shown in our host terminal, not the invisible container terminal