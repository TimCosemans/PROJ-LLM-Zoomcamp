source .env

docker-compose up -d 
docker cp elasticsearch:/usr/share/elasticsearch/config/certs/http_ca.crt .
