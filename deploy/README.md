# Static deployment

Production: http://212.193.11.210

The CRA build runs in the independent `ml-playground` Nginx container on port 80.
The existing API on port 3000 and its PostgreSQL container are not modified.

Server files:
- `/opt/ml-playground/releases/<revision>/`: versioned build and Nginx configuration.
- `/opt/ml-playground/current`: symlink to the active release.

Build locally with `npm run build`. Transfer `build/` and `deploy/nginx.conf` into
 a new release directory. Point `current` at that release and recreate only the
`ml-playground` container with the read-only release directory mounted at
`/usr/share/nginx/html`, the configuration at `/etc/nginx/conf.d/default.conf`,
and `--restart unless-stopped -p 80:80`. Never run Compose commands in the
existing backend directory for this deployment.

The running container mounts the resolved release path, so switching the symlink
alone does not change the live site. Recreate this container to deploy or roll back.
No server credentials belong in this repository.
