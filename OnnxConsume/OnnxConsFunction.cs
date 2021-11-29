using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace OnnxConsume
{
    public static class OnnxConsFunction
    {
        [FunctionName("OnnxConsumeFunction")]
        public static async Task<IActionResult> Run([HttpTrigger(AuthorizationLevel.Function, "get", "post", Route = null)] HttpRequest req, ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            var data = (JObject)JsonConvert.DeserializeObject(requestBody);

            string onnxPath = $@"{Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName}\IoTModel.onnx";

            //If you wish - could be merged into tensor array
            var UDITensor = new DenseTensor<float>(new float[] { ((JValue)data["UDI"]).Value<float>() }, new int[] { 1, 1 });
            var TypeTensor = new DenseTensor<string>(new string[] { ((JValue)data["Type"]).Value<string>() }, new int[] { 1, 1 });
            var ProductIdTensor = new DenseTensor<string>(new string[] { ((JValue)data["Product_ID"]).Value<string>() }, new int[] { 1, 1 });
            var airTensor = new DenseTensor<float>(new float[] { ((JValue)data["Air_temperature__K_"]).Value<float>() }, new int[] { 1, 1 });
            var tempTensor = new DenseTensor<float>(new float[] { ((JValue)data["Process_temperature__K_"]).Value<float>() }, new int[] { 1, 1 });
            var rotatTensor = new DenseTensor<float>(new float[] { ((JValue)data["Rotational_speed__rpm_"]).Value<float>() }, new int[] { 1, 1 });
            var torqTensor = new DenseTensor<float>(new float[] { ((JValue)data["Torque__Nm_"]).Value<float>() }, new int[] { 1, 1 });
            var toolWTensor = new DenseTensor<float>(new float[] { ((JValue)data["Tool_wear__min_"]).Value<float>() }, new int[] { 1, 1 });
            var TWFTensor = new DenseTensor<float>(new float[] { ((JValue)data["TWF"]).Value<float>() }, new int[] { 1, 1 });
            var HDFTensor = new DenseTensor<float>(new float[] { ((JValue)data["HDF"]).Value<float>() }, new int[] { 1, 1 });
            var PWFTensor = new DenseTensor<float>(new float[] { ((JValue)data["PWF"]).Value<float>() }, new int[] { 1, 1 });
            var OSFTensor = new DenseTensor<float>(new float[] { ((JValue)data["OSF"]).Value<float>() }, new int[] { 1, 1 });
            var RNFTensor = new DenseTensor<float>(new float[] { ((JValue)data["RNF"]).Value<float>() }, new int[] { 1, 1 });

            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("UDI", UDITensor),
                                                    NamedOnnxValue.CreateFromTensor<float>("Air temperature [K]", airTensor),
                                                    NamedOnnxValue.CreateFromTensor<float>("Process temperature [K]", tempTensor),
                                                    NamedOnnxValue.CreateFromTensor<float>("Rotational speed [rpm]", rotatTensor),
                                                    NamedOnnxValue.CreateFromTensor<float>("Tool wear [min]", toolWTensor),
                                                    NamedOnnxValue.CreateFromTensor<float>("Torque [Nm]", torqTensor),
                                                    NamedOnnxValue.CreateFromTensor<string>("Product ID", ProductIdTensor),
                                                    NamedOnnxValue.CreateFromTensor<string>("Type", TypeTensor),
                                                    NamedOnnxValue.CreateFromTensor<float>("Machine failure", UDITensor),
                                                    NamedOnnxValue.CreateFromTensor<float>("TWF", TWFTensor),
                                                    NamedOnnxValue.CreateFromTensor<float>("HDF", HDFTensor),
                                                    NamedOnnxValue.CreateFromTensor<float>("PWF", PWFTensor),
                                                    NamedOnnxValue.CreateFromTensor<float>("OSF", OSFTensor),
                                                    NamedOnnxValue.CreateFromTensor<float>("RNF", RNFTensor)};
            var session = new InferenceSession(onnxPath);

            var output = session.Run(input);
            var result = output.ToArray();

            var resultMessage = (result[19].Name, result[19].Value, result[18].Name, result[18].Value);
            return new OkObjectResult(resultMessage);
        }
    }
}
