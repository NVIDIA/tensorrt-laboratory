BASE_IMAGE ?= nvcr.io/nvidia/tensorrt:19.02-py3
IMAGE_NAME ?= tensorrt-laboratory
RELEASE_IMAGE ?= ryanolson/tensorrt-laboratory

.PHONY: build tag push release clean distclean

default: clean build

build: 
	@git submodule update --init third_party/pybind11
	@echo FROM ${BASE_IMAGE} > .Dockerfile
	@cat Dockerfile >> .Dockerfile
	docker build -t ${IMAGE_NAME} -f .Dockerfile . 

tag: build
	docker tag ${IMAGE_NAME} ${RELEASE_IMAGE}

push: tag
	docker push ${RELEASE_IMAGE}

release: push

clean:
	@rm -f .Dockerfile 2> /dev/null ||:
	@docker rm -v `docker ps -a -q -f "status=exited"` 2> /dev/null ||:
	@docker rmi `docker images -q -f "dangling=true"` 2> /dev/null ||:

distclean: clean
	@docker rmi ${IMAGE_NAME} 2> /dev/null ||:
	@docker rmi ${RELEASE_IMAGE} 2> /dev/null ||:
