import uvicorn

if __name__ == '__main__':

    print("5555555")
    uvicorn.run(app="wyl:app", host="0.0.0.0", port=8001, reload=True, ws="websockets", log_level="trace")