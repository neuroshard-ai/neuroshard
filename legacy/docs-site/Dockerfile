# NeuroShard Documentation Site
# Multi-stage build for optimized production image

# Stage 1: Build
FROM node:20-alpine AS builder

# Install git for vitepress lastUpdated feature
RUN apk add --no-cache git

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm install

# Copy source files
COPY . .

# Build the documentation
RUN npm run docs:build

# Stage 2: Production
FROM nginx:alpine AS production

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built docs from builder
COPY --from=builder /app/.vitepress/dist /usr/share/nginx/html

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

