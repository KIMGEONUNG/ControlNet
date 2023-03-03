if [[ $(hostname | grep mark12) ]]; then
    # IN DOCKER CONTAINER MARK12
    if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
      export CUDA_VISIBLE_DEVICES=4
    fi
    export condapath=/opt/conda/etc/profile.d/conda.sh
else
    # IN LOCAL SYSTEM
    export CUDA_VISIBLE_DEVICES=0
    export condapath=$HOME/anaconda3/etc/profile.d/conda.sh
fi
