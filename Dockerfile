
# Use a Python base image
FROM python:3.11.2
    
# Set the working directory
WORKDIR /app
    
# Copy only necessary files
COPY model-settings.json /app/model-settings.json
COPY custom_runtime.py /app/custom_runtime.py
COPY model/requirements.txt /app/model/requirements.txt
    
# Install required Python packages
RUN pip install --no-cache-dir -r /app/model/requirements.txt 
RUN pip install --no-cache-dir mlserver-mlflow mlserver
EXPOSE 8080
    
CMD ["mlserver","start","/app"]
        