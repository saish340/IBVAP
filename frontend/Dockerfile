# ------------------------------------------------------------------
# IBVAP frontend — production image (node build -> nginx serve).
#
# Single-origin mode (default, used by docker-compose / VPS):
#   browser -> nginx (:80) -> /api/*, /ws/* proxied to backend:8000
#   so the React app can keep using relative "/api" + same-host WS.
#
# Split mode (Vercel frontend + Render/Railway backend):
#   build with  --build-arg VITE_API_URL=https://<backend> 
#   then the app calls the backend directly (CORS must allow it).
# ------------------------------------------------------------------
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Optional at build time. Empty = same-origin via nginx proxy.
ARG VITE_API_URL=""
ARG VITE_WS_URL=""
ENV VITE_API_URL=$VITE_API_URL \
    VITE_WS_URL=$VITE_WS_URL

RUN npm run build

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

