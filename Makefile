# nvcr.io/nvidia/tensorrt:18.06-py2 is not yet available
BASE_IMAGE ?= ryanolson/trt4
IMAGE_NAME ?= yais
RELEASE_IMAGE ?= ryanolson/yais

.PHONY: build tag push release clean distclean

default: clean build

build: 
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
