FROM scratch

ADD alpine-minirootfs-3.11.12-x86_64.tar.gz /
RUN /bin/sh

EXPOSE 5001
RUN apk add --update python3 python3-dev py3-pip build-base


COPY . /app

WORKDIR /app

RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "python3" ]

CMD ["server.py"]