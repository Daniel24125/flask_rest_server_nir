from flask_restful import Resource


class Home(Resource):
    def get(self):
        return {'method': 'GET',
                'path': '/'}

    def post(self):
        return {'method': 'POST',
                'path': '/'}