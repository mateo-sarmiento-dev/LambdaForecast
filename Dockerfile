# Use the official Python 3.11 image
#FROM  public.ecr.aws/lambda/python:3.8
FROM public.ecr.aws/lambda/python:3.11


# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}
COPY main.py ${LAMBDA_TASK_ROOT}
COPY appmain.py ${LAMBDA_TASK_ROOT}
# Install the specified packages
RUN pip install -r requirements.txt

# Copy all files in ./src
#COPY app/ . ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler.
CMD [ "main.handler" ]