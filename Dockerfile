FROM nvidia/cuda:11.0-base
CMD nvidia-smi

ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get update -y
RUN apt-get install -y python3-pip python3.10 build-essential libgl1-mesa-glx ffmpeg libsm6 libxext6


EXPOSE 5000
ENV FLASK_APP 'app.py'
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY . .

RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

ENTRYPOINT [ "flask"]
CMD [ "run", "--host", "0.0.0.0"]
