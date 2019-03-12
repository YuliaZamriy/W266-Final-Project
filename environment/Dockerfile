### update 3/5/2019: added requests to the installation pack

FROM tensorflow/tensorflow:latest-py3-jupyter

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  git \
  tmux \
  python3-bs4

RUN pip install --upgrade pip
RUN pip install -U nltk pandas requests feedparser gspread oauth2client df2gspread

RUN python -m nltk.downloader -d /usr/share/nltk_data all
