# Pull python base image
FROM python:3.11
LABEL maintainer="Alexander Dolich"

# set a user
RUN adduser s2auser
USER s2auser
WORKDIR /home/s2auser

# set the path
ENV PATH="/home/s2auser/.local/bin:${PATH}"

# Update pip
RUN pip install --upgrade pip

# Install stgrid2area
RUN pip install stgrid2area

# Install jupyter
RUN pip install jupyter

# open port 8888
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser"]