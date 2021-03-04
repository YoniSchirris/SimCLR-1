import requests
import numpy as np
import math
import json
from PIL import Image
import io


class SlideScoreResult:
    def __init__(self, slide_dict=None):
        if slide_dict is None:
            self.image_id = 0
            self.image_name = ''
            self.user = None
            self.tma_row = None
            self.tma_col = None
            self.tma_sample_id = None
            self.question = None
            self.answer = None
            return

        self.image_id = int(slide_dict['imageID'])
        self.image_name = slide_dict['imageName']
        self.user = slide_dict['user']
        self.tma_row = int(slide_dict['tmaRow']) if 'tmaRow' in slide_dict else 0
        self.tma_col = int(slide_dict['tmaCol']) if 'tmaCol' in slide_dict else 0
        self.tma_sample_id = slide_dict['tmaSampleID'] if 'tmaSampleID' in slide_dict else ""
        self.question = slide_dict['question']
        self.answer = slide_dict['answer']

        if self.answer[:2] == '[{':
            annos = json.loads(self.answer)
            if len(annos) > 0:
                if hasattr(annos[0], 'type'):
                    self.annotations = annos
                else:
                    self.points = annos

    def to_row(self):
        ret = str(self.image_id) + "\t" + self.image_name + "\t" + self.user + "\t"
        if self.tma_row is not None:
            ret = ret + str(self.tma_row) + "\t" + str(self.tma_col) + "\t" + self.tma_sample_id + "\t"
        ret = ret + self.question + "\t" + self.answer
        return ret


class APIClient(object):
    print_debug = False

    def __init__(self, server, api_token, disable_cert_checking=False):
        if server[-1] == "/":
            server = server[:-1]
        self.end_point = "{0}/Api/".format(server)
        self.api_token = api_token
        self.disable_cert_checking = disable_cert_checking
        self.base_url, self.cookie = None, None

    def perform_request(self, request, data, method="POST"):
        headers = {'Accept': 'application/json', 'Authorization': 'Bearer {auth}'.format(auth=self.api_token)}
        url = "{0}{1}".format(self.end_point, request)
        verify = True
        if self.disable_cert_checking:
            verify = False

        if method == "POST":
            response = requests.post(url, verify=verify, headers=headers, data=data)
        else:
            response = requests.get(url, verify=verify, headers=headers, data=data)
        if response.status_code != 200:
            response.raise_for_status()

        return response

    def get_images(self, studyid):
        response = self.perform_request("Images", {"studyid": studyid})
        rjson = response.json()
        return rjson

    def get_results(self, studyid):
        response = self.perform_request("Scores", {"studyid": studyid})
        rjson = response.json()
        return [SlideScoreResult(r) for r in rjson]

    def upload_results(self, studyid, results):
        sres = "\n" + "\n".join([r for r in results])
        response = self.perform_request("UploadResults", {
            "studyid": studyid,
            "results": sres
        })
        rjson = response.json()
        if not rjson['success']:
            raise SlideScoreErrorException(rjson['log'])
        return True

    def upload_asap(self, imageid, user, questions_map, annotation_name, asap_annotation):
        response = self.perform_request("UploadASAPAnnotations", {
            "imageid": imageid,
            "questionsMap": '\n'.join(key + ";" + val for key, val in questions_map.items()),
            "user": user,
            "annotationName": annotation_name,
            "asapAnnotation": asap_annotation})
        rjson = response.json()
        if not rjson['success']:
            raise SlideScoreErrorException(rjson['log'])
        return True

    def get_image_metadata(self, imageid):
        response = self.perform_request("GetImageMetadata", {
            "imageId": imageid}, "GET")
        rjson = response.json()
        if not rjson['success']:
            raise SlideScoreErrorException(rjson['log'])
        return rjson['metadata']

    def export_asap(self, imageid, user, question):
        response = self.perform_request("ExportASAPAnnotations", {
            "imageid": imageid,
            "user": user,
            "question": question})
        rawresp = response.text
        if rawresp[0] == '<':
            return rawresp
        rjson = response.json()
        if not rjson['success']:
            raise SlideScoreErrorException(rjson['log'])

    def get_image_server_url(self, imageid):
        response = self.perform_request("GetTileServer?imageId=" + str(imageid), None, method="GET")
        rjson = response.json()
        return (
            self.end_point.replace("/Api/", "/i/" + str(imageid) + "/" + rjson['urlPart'] + "/_files"), rjson['cookiePart'])

    def set_cookie(self, image_id):
        (self.base_url, self.cookie) = self.get_image_server_url(image_id)

    def get_tile(self, level, x, y):
        response = requests.get(self.base_url + "/{}/{}_{}.jpeg".format(str(level), str(x), str(y)), stream=True, cookies=dict(t=self.cookie))
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            raise SlideScoreErrorException()


class SlideScoreErrorException(Exception):
    pass
