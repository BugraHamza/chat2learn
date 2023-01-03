FROM python:3.10

# Create app directory
WORKDIR /app

# Install app dependencies
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Bundle app source
COPY . .

RUN python -m spacy download en

EXPOSE 9090
CMD [ "flask", "run","--host","0.0.0.0","--port","9090"]