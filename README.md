# Set up the environment
Conda is a package management, the required packages for this project is as follows:
```
$ conda create -n nlp-prompt-attack-env
$ conda install python
$ conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
$ conda install -c conda-forge pytorch-lightning
$ conda install -c anaconda scikit-learn
$ conda install pandas
$ conda install -c conda-forge tensorboard
```
Some useful commands:
```
$ conda search <package-name>
$ conda list 
$ conda list | grep <package-name>
```