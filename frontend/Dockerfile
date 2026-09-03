# ------------------------------------------------------------------
# Placeholder frontend image — runs the Vite dev server so the
# container is reachable at http://localhost:5173.
# TODO: replace with a multi-stage build (node build -> nginx serve)
# once the dashboard is production-ready.
# ------------------------------------------------------------------
FROM node:20-alpine

WORKDIR /app

ENV VITE_API_TARGET=http://backend:8000

COPY package.json ./
RUN npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
