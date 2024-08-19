FROM node:20.11.1-slim AS builder

RUN mkdir /ui
WORKDIR /ui
COPY . /ui

RUN npm install && npm run build

FROM nginx:1.25
COPY --from=builder /ui/build /usr/share/nginx/html