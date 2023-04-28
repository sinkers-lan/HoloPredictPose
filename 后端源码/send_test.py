# -*- coding:utf-8 -*-
import time
from typing import Union, List

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import asyncio

import uvicorn
import json

app = FastAPI()


class ConnectionManager:
    def __init__(self):
        self.ptr0 = 0
        self.ptr1 = 0
        with open("out0.json", 'r', encoding="utf-8") as jsonfile_0:
            self.json0 = json.load(jsonfile_0)
        with open("out1.json", 'r', encoding="utf-8") as jsonfile_1:
            self.json1 = json.load(jsonfile_1)

    def start(self):
        self.ptr0 = 0
        self.ptr1 = 0
        # 你的socket初始化
        print(f"start:{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}")

    def end(self):
        self.ptr0 = 0
        self.ptr1 = 0
        print(f"end{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}")

    def send_message(self, avail: int):
        # self.ptr0 = 0
        # self.ptr1 = 0
        # 取缓冲区
        var0 = self.json0[self.ptr0]
        var1 = self.json1[self.ptr1]
        self.ptr0 += 1
        self.ptr1 += 1
        return {
            "org": var0,
            "pred": var1,
            "avail": avail
        }


manager = ConnectionManager()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/cv")
async def get_cv():
    return manager.send_message(1)


@app.get("/start")
async def get_start():
    manager.start()
    pass


@app.get("/end")
async def get_end():
    manager.end()
    pass


# 接收文件
# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile = File(...)):
#     s = await file.read()
#     # print(s)
#     # print(type(s))
#     with open("new_photo.jpg", 'wb') as new_file:
#         new_file.write(s)
#     return {"filename": file.filename}

my_json = {
    "org": [[]],
    "pred": [[]],
    # "avail": True
}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 异步接收消息
    # json_data = None

    # async def read_from_socket(websocket: WebSocket):
    #     nonlocal json_data
    #     try:
    #         async for i in websocket.iter_json():
    #             json_data = i
    #     except WebSocketDisconnect:
    #         print("socket close")
    #
    # asyncio.create_task(read_from_socket(websocket))

    with open("out0.json", 'r', encoding="utf-8") as jsonfile_0:
        data_0 = json.load(jsonfile_0)
    with open("out1.json", 'r', encoding="utf-8") as jsonfile_1:
        data_1 = json.load(jsonfile_1)
        # print(type(data_1))
    try:
        i = 0
        while True:
            my_json['org'] = data_0[i]
            my_json['pred'] = data_1[i]
            # print(data_1)
            # await websocket.send_json(data_1)
            await websocket.send_text(json.dumps(my_json))
            i += 1
            await asyncio.sleep(0.1)
    except BaseException:
        print(f"socket close{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}")


if __name__ == '__main__':
    uvicorn.run(app="send_test:app", host="0.0.0.0", port=8001, reload=True, ws="websockets", ws_ping_interval=5, ws_ping_timeout=5, log_level="trace")
# , log_level="trace"
