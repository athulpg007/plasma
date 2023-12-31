FROM python:3.11.7-slim-bullseye
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . ./
WORKDIR /app_v1 # Set working directory here if changing versions
EXPOSE 8050
# For local development use "python app.py"
CMD ["python", "app.py"] 
# For production deployment, use gunicorn
# CMD [ "gunicorn", "-b 0.0.0.0:8050", "app:server"] 

