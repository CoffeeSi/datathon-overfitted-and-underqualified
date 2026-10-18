FROM node:24-bookworm-slim AS build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . ./
RUN npm run build

FROM node:24-bookworm-slim

ENV NODE_ENV=production
WORKDIR /app

COPY --from=build --chown=node:node /app/package.json /app/package-lock.json ./
COPY --from=build --chown=node:node /app/next.config.ts ./next.config.ts
COPY --from=build --chown=node:node /app/node_modules ./node_modules
COPY --from=build --chown=node:node /app/.next ./.next
COPY --from=build --chown=node:node /app/public ./public

USER node
EXPOSE 3000
CMD ["npm", "run", "start", "--", "--hostname", "0.0.0.0"]
