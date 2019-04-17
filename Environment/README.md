# Setting up the environment for W266 final project

## Solution 1. Get tensorflow docker image

Instructions are [here.](https://www.tensorflow.org/install/docker)

Step 1. Pull the image:
```
docker pull tensorflow/tensorflow:latest-py3-jupyter
```

Step 2a. Start `bash` container:

Note: replace path with your own:
```
docker run -it --rm -v <local/path/to/working/directory>:/tf/notebooks tensorflow/tensorflow:latest-py3-jupyter bash
```

Step 2b. Start `Jupyter Notebook` container:

Note: replace path with your own:
docker run -it --rm -v <local/path/to/working/directory>:/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-py3-jupyter

Step 3. Open Jupyter Notebook

Navigate to http://localhost:8888 and enter the token from the command line.

## Solution 2. Extend this image with additional packages

Step 1. Copy `Dockerfile` to your machine

Step 2. Build the image

Note: run this command from the same directory where `Dockerfile` is
```
docker build -t tf-nltk .
```

Run this command to re-build the image (if there are changes to it)
```
docker build --no-cache -t tf-nltk .
```

Step 3a. Start `bash` container:

Note: replace path with your own:
```
docker run -it --rm -v <local/path/to/working/directory>:/tf/notebooks tf-nltk:latest bash
```

Step 3b. Start `Jupyter Notebook` container:

Note: replace path with your own:
```
docker run -it --rm -v <local/path/to/working/directory>:/tf/notebooks -p 8888:8888 tf-nltk:latest
```

Step 4. Open Jupyter Notebook

Navigate to http://localhost:8888 and enter the token from the command line.

# Adding symbolic links to helper files

Helper functions are located in `Classification` folder. But there is no need to copy them elsewhere, just create symbolic links to them:
1. Within docker container (bash or notebook)
2. Navigate to the working directory (wherever you need to run your code)
3. Run the following commands:

```
ln -s /tf/notebooks/2019-spring-main/common w266_common
ln -s /tf/notebooks/final-project/Classification/helpers/ helpers
```


# Working with Google Sheets

Step 1. Connect Google Sheets and Jupyter Notebooks

Very good instructions are [here](https://socraticowl.com/post/integrate-google-sheets-and-jupyter-notebooks/)

Step 2. Adjust paramters in `./environment/gs_connect.py`: 

- the location of the credentials json file
- spreadsheet key (if different) from the Google Sheets URL

Step 3. Enable Google Sheets API here:

    + https://console.developers.google.com/apis/api/sheets.googleapis.com/overview?project=zzzzzzzzzzzzzzz
    + The `zzzzzzzzzzzzzzz` needs to be adjusted for your project id. It can be obtained from client_id.json ("client_id":"zzzzzzzzzzzzzzz-xxxxxxxxxxxxxxxxxxxxxxxxx.apps.googleusercontent.com")
    + Alternatively, you can run `gs_connect.py` and it'll give an error with the instructions

Step 4. To call functions from `gs_connect.py` either:
- copy this file to the working directory of the jupyter notebook where you need it
- create a symbolic link to the file in the directory of the jupyter notebook where you need it (that way you don't need to re-copy the file if there are any changes to it):
```
ln -s <path to>/gs_connect.py gs_connect.py
```

# Working on Google Cloud Instance

- ssh into instance:

`gcloud compute ssh --ssh-flag="-L 8896:127.0.0.1:8896" --ssh-flag="-L 8006:127.0.0.1:8006" w266`

- create directories on the instance:

`mkdir -p ~/final-project/data/raw/hein-daily`
`mkdir -p ~/final-project/data/QA`
`mkdir -p ~/final-project/Classification`
`mkdir -p ~/final-project/Classification/helpers`
`mkdir -p ~/final-project/Classification/w266_common`

- copy files into instance:

*Speeches*:

`gcloud compute scp ./final-project/data/raw/hein-daily/speeches* w266:~/final-project/data/raw/hein-daily`

*Congresspeople Description*:

`gcloud compute scp ./final-project/data/QA/full_descr.txt w266:~/final-project/data/QA`

*Working notebooks and scripts*:

`gcloud compute scp ./final-project/Classification/helpers/*.py w266:~/final-project/Classification/helpers`

`gcloud compute scp ./2019-spring-main/common/*.py  w266:~/final-project/Classification/w266_common`

`gcloud compute scp ./final-project/Classification/Baseline-all-test.ipynb w266:~/final-project/Classification/Baseline-all-gce.ipynb`

### using tmux

- open new terminal on the local machine (don't open tmux)
- `gcloud compute ssh` into the instance
- run:
    + `tmux new-session -s notebook`
- new terminal will open. run:
    + `jupyter notebook`
- open browser:
    + `localhost:8896`
- detach if desired:
    + `Ctrl+b d`
- re-attach:
    + `tmux attach -t notebook`
- kill session:
    + `tmux kill-session -t notebook`

