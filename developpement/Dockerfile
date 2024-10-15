# Use an official Python runtime as the parent image
FROM pytorch/pytorch

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Make port 8888 available for jupyter (notebooks)
EXPOSE 8888

# Make port 6006 available for tensorboard
EXPOSE 6006

# Define environment variable for Flask to run in production mode
ENV FLASK_ENV=production
