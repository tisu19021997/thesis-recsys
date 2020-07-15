import os


class BaseConfig(object):
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    TESTING = True


class ProductionConfig(BaseConfig):
    DEBUG = True
    TESTING = True


config = {
    "default": "main.config.BaseConfig",
    "development": "main.config.DevelopmentConfig",
    "production": "main.config.ProductionConfig",
}


def configure_app(app):
    config_name = os.getenv('FLASK_ENV')
    app.config.from_object(config[config_name])
    app.config.from_pyfile('application.cfg', silent=True)
