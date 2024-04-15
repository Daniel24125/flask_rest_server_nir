from flask_restful import Resource


class Measurement(Resource):
    def get(self):
        return {'method': 'GET',
                'path': '/measurement'}