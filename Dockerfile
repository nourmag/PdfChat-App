# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Run Streamlit when the container launches
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
