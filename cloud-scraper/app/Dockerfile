FROM --platform=linux/amd64 python:3.10 as build

# install google chrome
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN apt-get -y update
RUN apt-get install -y google-chrome-stable

# install chromedriver
RUN apt-get install -yqq unzip
RUN wget -O /tmp/chromedriver.zip https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/119.0.6045.105/linux64/chromedriver-linux64.zip
RUN unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/




# set display port to avoid crash
ENV DISPLAY=:99

# Set the working directory
WORKDIR /app

# Copy your Lambda function code and dependencies into the container
COPY redfin_scraper.py .
COPY requirements.txt .

# upgrade pip
RUN pip install --upgrade pip

RUN pip3 install -r requirements.txt --target .

ENV PATH=$PATH:/usr/local/bin/:usr/bin/


CMD ["python", "redfin_scraper.py"]