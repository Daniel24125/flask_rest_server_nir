from flask_restful import Resource


class Test(Resource):
    def get(self):
        return {'method': 'GET',
                'path': '/test'}