# η.ai
η.ai showcases low-precision AI's speed and power efficiency, offering a free 
API for professionals. It explores analogue AI for even greater gains in 
efficiency. The source-code to this platform is provided in this repo.

### Vision
The η.ai (eta.ai) project explores low-precision AI, challenging high-precision 
approaches. It reduces precision requirements, improving efficiency and 
hardware scalability. Low-precision AI (TinyML) swiftly analyzes large 
datasets in finance, cybersecurity, and healthcare. Affordable FPGA hardware, 
in this case Google Coral, supports low-precision AI, providing better 
performance and energy efficiency for edge computing.

### How to use the service
You can create an account and login here: **TBA**

You can then use the API to send pictures to infer the objects that are in 
the picture. The callback is a JSON with the predictions and the power that 
was consumed to make this happen. 

The interactive Swagger API documentation can be found when spinning up the 
service. The default address would be `127.0.0.1:8000`. The OpenAPI 
documentation can then be found under at the `/docs` endpoint: `127.0.0.
1:8000/docs`. 

### Do you wan to setup a service like this one for yourself?
Then you can! Please note that this platform was implemented using:
* Linux (debian)
* Docker (you can deploy instantly) and docker tools like docker compose
* For API-handling FastAPI was used, a python framework
* Python dependencies are managed using Poetry

You can run the code directly on your machine. In that case you need to:
1. Clone the Repo
2. Install dependencies using Poetry and drivers using the 
   `systemdeps/install_and_run.sh` script. This script was install, test 
   and run the API.
4. For running the api seperately, you can use `poetry run uvicorn main:app 
   --reload`
5. You can find the swagger documentation `ip-address:port/docs`

If you want to run the docker-container, please make sure that you:
1. Install the driver on your OS first using `systemdeps/install_drivers.sh`
2. run `docker-compose up`
3. You can find the swagger documentation `ip-address:port/docs`

Note that the docker-compose approach adds additional services and is intended
to be used as 

### License
The license can be found in the [UNLICENSE](UNLICENSE) file. For more 
information, please go to: [unlicense.org](https://unlicense.org/). If you 
want to have more information on the licensing topics as a whole, please 
checkout [choose a license](https://choosealicense.com/).