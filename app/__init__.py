import os
import logging

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

from app.config import config


def create_app(config_name=None):
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')

    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
        static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static'),
    )
    app.config.from_object(config.get(config_name, config['default']))

    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_prefix=1)

    logging.basicConfig(level=logging.INFO)

    from app.extensions import init_chroma
    init_chroma(app)

    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(get_remote_address, app=app, default_limits=["30 per minute"])
    app.limiter = limiter

    from app.chat.routes import chat_bp
    app.register_blueprint(chat_bp)

    from app.errors import register_error_handlers
    register_error_handlers(app)

    return app
