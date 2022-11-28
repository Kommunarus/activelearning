from flask import Flask, jsonify, request
# from flask_cors import CORS
from flask_restful import Api, Resource, reqparse
from scripts.classification.train_plants import for_api
# import os
import uuid


app = Flask(__name__)
# CORS(app)
api = Api(app)

def make_lab(method, device, path_to_dataset, num_samples, train, check, num_train_for_model0, n_epoch):
    out = for_api(method, device, path_to_dataset, rawnum_samples=num_samples, train=(True if train == 'y' else False),
                  check=(True if check == 'y' else False), num_train_for_model0=num_train_for_model0, n_epoch=n_epoch)
    return jsonify(out)


class active_learning(Resource):
    @staticmethod
    def get():
        check = reqparse.request.args['check']
        train = reqparse.request.args['train']
        num_train_for_model0 = reqparse.request.args['num_train_for_model0']
        if len(num_train_for_model0) > 0:
            num_train_for_model0 = int(num_train_for_model0)
        else:
            num_train_for_model0 = 0
        n_epoch = reqparse.request.args['n_epoch']
        if len(n_epoch) > 0:
            n_epoch = int(n_epoch)
        else:
            n_epoch = 0
        device = reqparse.request.args['device']
        method = reqparse.request.args['method']
        num_samples_in_epoch = reqparse.request.args['num_samples_in_epoch']
        path_to_dataset = reqparse.request.args['path_to_dataset']
        # print(num_samples)
        # print(device, method)
        return make_lab(method, device, path_to_dataset, num_samples_in_epoch, train, check, num_train_for_model0,
                        n_epoch)


api.add_resource(active_learning, '/active_learning')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
    # app.run(host='127.0.0.1', debug=True, ssl_context=('./ssl/mkskom.crt', './ssl/mkskom.key'))