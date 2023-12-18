NAME?=hw
GPUS?=0
DATA_DIR?=/hdd/aidar/dcgan_data
NOTEBOOKS?=/hdd/aidar/notebooks/dcgan
SAVE_DIR?=/hdd/aidar/dcgan_save_dir

.PHONY: build run

build:
	docker build -t $(NAME) .

run:
	docker run --rm -it --runtime=nvidia \
	-e NVIDIA_VISIBLE_DEVICES=$(GPUS) \
	--ipc=host \
	--net=host \
	-v $(PWD):/workspace \
	-v $(DATA_DIR):/workspace/data \
	-v $(NOTEBOOKS):/workspace/notebooks \
	-v $(SAVE_DIR):/workspace/saved \
	--name=$(NAME) \
	$(NAME) \
	bash
