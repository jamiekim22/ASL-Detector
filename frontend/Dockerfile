# Build stage
FROM node:18-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Production stage: serve static files with Nginx
FROM nginx:alpine
WORKDIR /usr/share/nginx/html
COPY --from=build /app/index.html ./
COPY --from=build /app/js ./js
COPY --from=build /app/styles ./styles
COPY --from=build /app/assets ./assets
COPY --from=build /app/components ./components
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]