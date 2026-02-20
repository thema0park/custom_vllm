docker run -it \
	--gpus all \
	--net host \
	--entrypoint /bin/bash \
	-v /home/jhs/.vimrc:/root/.vimrc \
	-v /home:/home \
	jhs/cuda:12.0.0-base-ubuntu22.04
