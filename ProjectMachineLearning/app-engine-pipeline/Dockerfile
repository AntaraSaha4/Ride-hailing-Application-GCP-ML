FROM gcr.io/google-appengine/python

# Install correct python version
RUN apt-get update && apt-get install libbz2-dev -y
RUN wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz \ 
    && tar -xf Python-3.8.10.tgz \
    && cd Python-3.8.10 && ./configure --enable-optimizations && make -j 4 && make install

# Create a virtualenv for dependencies. This isolates these packages from
# system-level packages.
RUN pip3 install --upgrade pip
RUN pip3 install virtualenv
RUN python3.8 -m virtualenv /env 

# RUN echo "python --version"
# Setting these environment variables are the same as running
# source /env/bin/activate.
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH

# Copy the application's requirements.txt and run pip to install all
# dependencies into the virtualenv.
ADD requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt


# Add the application source code.
ADD . /app

# Run a WSGI server to serve the application. gunicorn must be declared as
# a dependency in requirements.txt.
CMD gunicorn -b :$PORT main:app
