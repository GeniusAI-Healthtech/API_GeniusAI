import json
from loguru import logger

from fastapi import FastAPI, File, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from controllers.detection_controller import router as detection_router
from controllers.healthcheck_controller import router as healthcheck_router
from controllers.gpt_controller import router as gpt_router


###################### FastAPI Setup #############################

# title
app = FastAPI(
    title="Object Detection FastAPI Template",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="2024.5.01",
)

# This function is needed if you want to allow client requests
# from specific domains (specified in the origins argument)
# to access resources from the FastAPI server,
# and the client and server are hosted on different domains.
origins = ["http://localhost", "http://localhost:8008", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def save_openapi_json():
    """This function is used to save the OpenAPI documentation
    data of the FastAPI application to a JSON file.
    The purpose of saving the OpenAPI documentation data is to have
    a permanent and offline record of the API specification,
    which can be used for documentation purposes or
    to generate client libraries. It is not necessarily needed,
    but can be helpful in certain scenarios."""
    openapi_data = app.openapi()
    # Change "openapi.json" to desired filename
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)


# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


app.include_router(detection_router, prefix="/analise")
app.include_router(healthcheck_router)
app.include_router(gpt_router)
