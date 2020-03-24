

# Setup 
- make sure darknet framework is properly installed from https://sc01-trt.thales-systems.ca/gitlab/BRiTE/darknet.git. (follow instructions in readme file)
# Dependencies 
1-Anaconda

- go to setup-script folder , right click and open in terminal 
- run the command : sh anaconda-install.sh
- wait for the installation to finish.
- open a new terminal and run the command ```conda config --set auto_activate_base false```

2- xampp (mysql server)
If you want to run the project locally you need sql server, you can use xampp. intall xampp run the xamp-install.sh by running the command : sh xampp-install.sh

For kubernetes usages, refer to the [wiki](https://wiki-trt.thales-systems.ca/bin/view/Trt%20Quebec/Engineering/Infrastructure/VLANs/Collaborative_VLAN/Kubernetes/Usage/).
# Content
1- Anchor mapping
The goal is to map the bus route, detect all anchors and  create a database of these anchors.

2- Bus Localization
Using the proposed probabilistic approach ( in the Above publications) the bus locations is infered from the anchors detection pattern
![work flow](https://github.com/hayouni15/Cognitive-Bus/blob/master/md_images/Presentationd.jpg)

3- Speed estimation
A deep neural network is trained to estimate the vehicle speed using only the camera. The proposed method is explained in the following figure where an optical flow matrix is derived from two consecutive frames and then converted to an HSV frame and finally an RGB image , the RGB image is fed to the trained neural network to give accurate estimates about the vehicle speed.
![Speed estimation](https://github.com/hayouni15/Cognitive-Bus/blob/master/md_images/Presentation2.jpg)
This approach is very reliable and results in an estimation error less than 2 mph


4- Behavior Cloning
A neural network is trained to estimate the appropriate steering angle

5- Collision Early warning
A new collision early warning approach is proposed to detect any potential collision risks and deliver warnings to the driver.
# Versioning

Version is based on Git commit hash and tag using [setuptools-scm](https://pypi.org/project/setuptools-scm/)

**Please note that unique job identifier/name will be generated using, in part, the git commit hash.** It means that you currently cannot submit two concurrent jobs using the same version of the code.

# Build 

The following will:
1. Build a .whl (wheel) file containing the application code and copy it in the dist/ folder;
2. Build a Docker image containing the application code and all its dependencies (using deployment/Dockerfile);
3. Push the Docker image to the [collaborative docker registry](http://collaborative-docker-registry.collaborative.local/)
4. Create a kubernetes job file from the (using deployment/mapping-raw-data-kubernetes-job.tpl template) and copy it in the jobs/ folder;

To build, simply run:
```
# Login using your TGI credentials
docker login ${collaborative_docker_registry}

# The job filename once the build is completed
./build.sh <template-file-path>
```

# Launch Job

The job can be launch using:
```
kubectl apply -f ./jobs/<JOB_NAME>.yml
```

# Monitor Job

You can monitor your job using different tools

1. [Dashboard](https://dashboard.k8s.collaborative.local/#!/job?namespace=brite)
2. [kubectl commands](https://wiki-trt.thales-systems.ca/bin/view/Trt%20Quebec/Engineering/Infrastructure/VLANs/Collaborative_VLAN/Kubernetes/Usage/#HJob27slogs)
3. [#k8s-collaborative](https://thales-quebec.slack.com/messages/CLALVMM6U) Slack channel

# Outputs

TBD

# Stop/delete job

Once job is completed (or failed to complete) and you don't to access its logs anymore, you have to manually delete it using the following command.

```
kubectl delete -f ./jobs/<JOB_NAME>.yml
```

