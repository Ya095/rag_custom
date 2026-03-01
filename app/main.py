import uvicorn

from config import config
from initializer import create_app


app = create_app()

if __name__ == '__main__':
    app = 'main:app' if config.app_config.reload else app
    uvicorn.run(
        app=app,
        host=config.app_config.host,
        port=config.app_config.port,
        forwarded_allow_ips='*',
        reload=config.app_config.reload,
    )
