FROM mambaorg/micromamba:0.15.3
USER root
#RUN apt-get update && apt-get install -y redis-server
RUN apt-get update && DEBIAN_FRONTEND=“noninteractive” apt-get install -y --no-install-recommends \
#       nginx \
       gcc \
       g++ \
       ca-certificates \
       apache2-utils \
       certbot \
       python3-certbot-nginx \
       sudo \
       cifs-utils \
       openjdk-11-jdk \
       && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get -y install cron
RUN mkdir /opt/cloud_deploy
RUN chmod -R 777 /opt/cloud_deploy
WORKDIR /opt/cloud_deploy
USER micromamba
COPY environment.yml environment.yml
RUN micromamba install -y -n base -f environment.yml && \
   micromamba clean --all --yes
COPY run.sh run.sh
COPY project_contents project_contents
USER root
RUN cd /opt/cloud_deploy/project_contents/ && \
   tar -xvzf public_datasets.tar.gz && \
   rm public_datasets.tar.gz && \
   chown -R micromamba:micromamba /opt/cloud_deploy/project_contents
USER micromamba
#COPY nginx.conf /etc/nginx/nginx.conf
USER root
RUN chmod a+x run.sh
CMD ["./run.sh"]