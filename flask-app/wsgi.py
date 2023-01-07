from app import app

if __name__ == "__main__":
    app.run()

# gunicorn --bind 0.0.0.0:9090 wsgi:app -w 2 --threads 4