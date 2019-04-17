### update 3/5/2019: added requests to the installation pack
### update 3/19/2019: added spicy and sklearn to the installation pack
### update 3/20/2019: switched from tf latest to 1.13 because tf 2.0 has significant changes
### update 3/24/2019: removed git, tmux, feedparser
### update 4/7/2019: added gensim

# FROM tensorflow/tensorflow:latest-py3-jupyter

FROM tensorflow/tensorflow:1.13.0rc2-py3-jupyter

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  python3-bs4

RUN pip install --upgrade pip gensim
RUN pip install -U nltk pandas requests gspread oauth2client df2gspread scipy scikit-learn

#RUN python -m nltk.downloader -d /usr/share/nltk_data all


