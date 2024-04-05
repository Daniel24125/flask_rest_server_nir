from flask import Flask
from flask_restful import Api
from routes.index import Home
from routes.test.index import Test
import inspect

app = Flask(__name__)
api = Api(app)

routes = [
    Home,
    Test
]

def route_registration(): 
    for r in routes: 
        route_path = inspect.getfile(r)
        route_path = inspect.getfile(r).replace("\\", "/")
        routes_index = route_path.find("routes")+6
        index_location = route_path.find("index.py")
        route = route_path[routes_index:index_location]
        api.add_resource(r, route)


if __name__ == '__main__':
    route_registration()
    app.run(debug=True)