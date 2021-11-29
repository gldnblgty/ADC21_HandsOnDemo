This repo presents the hands-on demo that we demonstrated in Advanced Developers Conference '21 as a reference point for audience.

Dataset is taken from https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset

Slides can be found in Slideshare: https://www.slideshare.net/guldenbilgutay/

----

To build Windows Image : 

(on the path) 
```
docker build -t <IMAGENAME>:latest -f Dockerfile .
```

----

Local test of the Image  : 

```
docker run -p 5000:80 <IMAGENAME>:latest
```

----

(Alternative to Azure ACI) - To push the Docker Hub: 

```
docker login
docker push <DOCKERUSERNAME>/<IMAGENAME>:latest
```

----

To check Onnx Model - https://netron.app/ OR https://github.com/lutzroeder/netron

----

Local API - JSON Test Data Example :

```
{
    "UDI": 1,
    "Product_ID": "M14860",
    "Type": "M",
    "Air_temperature__K_": 350.1,
    "Process_temperature__K_": 308.6,
    "Rotational_speed__rpm_": 1551,
    "Torque__Nm_": 42.8,
    "Tool_wear__min_": 3
}
```

----


Onnx Model Consumption - JSON Test Data Example :

```
{
    "UDI": 1,
    "Product_ID": "M14860",
    "Type": "M",
    "Air_temperature__K_": 298.1,
    "Process_temperature__K_": 308.6,
    "Rotational_speed__rpm_": 1551,
    "Torque__Nm_": 42.8,
    "Tool_wear__min_": 0,
    "TWF": 0,
    "HDF": 0,
    "PWF": 0,
    "OSF": 0,
    "RNF": 0
}
```
