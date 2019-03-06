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
