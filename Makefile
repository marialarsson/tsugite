#! make

build:
	docker build -t tsugite .

up:
	docker-compose up -d

browse:
	firefox http://localhost:8083 &

