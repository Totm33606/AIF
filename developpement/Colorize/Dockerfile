# Use an official Python runtime as the parent image
FROM pytorch/pytorch

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r base_requirements.txt

CMD ["python", "code/train.py", "--epochs", "1"]