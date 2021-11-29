using Microsoft.AspNetCore;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using System.IO;
using System.Text.Json;
using System;
using Microsoft.Extensions.ML;

namespace IoTMachineAPI
{
    public class Program
    {
        public static void Main(string[] args)
        {
            WebHost.CreateDefaultBuilder()
          .ConfigureServices(services =>
          {
              //Important : Prediction Engine Pool is for getting advantage of ObjectPool pattern.
              services.AddPredictionEnginePool<ModelTrainer.ModelInput, ModelTrainer.ModelOutput>()
                .FromFile($@"{Environment.CurrentDirectory}\Model\IoTModel.zip");

          })
          .Configure(options =>
          {
              options.UseRouting();
              options.UseEndpoints(routes =>
              {
                  // Define prediction endpoint
                  routes.MapPost("/predict", PredictHandler);
              });
          })
          .Build()
          .Run();
        }
        public static async Task PredictHandler(HttpContext http)
        {
            // Get PredictionEnginePool service
            var predictionEnginePool = http.RequestServices.GetRequiredService<PredictionEnginePool<ModelTrainer.ModelInput, ModelTrainer.ModelOutput>>();

            // Deserialize HTTP request JSON body
            var body = http.Request.Body as Stream;
            var input = await JsonSerializer.DeserializeAsync<ModelTrainer.ModelInput>(body);

            // Predict
            ModelTrainer.ModelOutput prediction = predictionEnginePool.Predict(input);

            // Return prediction as response
            await http.Response.WriteAsJsonAsync(prediction);
        }

    }
}
